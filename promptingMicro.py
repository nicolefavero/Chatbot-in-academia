############################################################################################################
#  CHATBOT WITH Llama 3.3 70B FOR QUESTION ANSWERING & Llama-2-7B FOR EXTRACTION OF ENTITIES  #
############################################################################################################

'''
Objective: creation of a Chatbot leveraging the Llama 3.3 70B model for question answering and the Llama-2-7B model for the extraction of entities and relationships 
to build a knowledge graph. The chatbot will be able to answer questions based on the knowledge graph and community summaries generated from the extracted text.
'''

import os
import pickle
import re
import torch
import fitz  
import networkx as nx
import numpy as np
import pandas as pd
import spacy
import copy
import sys
import json
import igraph as ig
import leidenalg
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import ipywidgets as widgets
from IPython.display import display

# Debug flag
DEBUG = True

def debug(msg):
    if DEBUG:
        print("DEBUG:", msg)

# File paths for caching intermediate outputs
ENTITIES_CSV = "extracted_entities.csv"
RELATIONSHIPS_CSV = "extracted_relationships.csv"
GRAPH_PKL = "knowledge_graph.pkl"
COMMUNITY_SUMMARIES_PKL = "community_summaries.pkl"

# -----------------------------------------------------------------------------
# 1. Load Models
# -----------------------------------------------------------------------------
def load_extraction_model():
    debug("Loading extraction model...")
    extraction_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    extraction_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    debug("Extraction model loaded.")
    return extraction_tokenizer, extraction_model

def load_summarization_model():
    debug("Loading summarization model...")
    summ_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    summ_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    debug("Summarization model loaded.")
    return summ_tokenizer, summ_model

# -----------------------------------------------------------------------------
# 2. PDF Processing & Text Chunking
# -----------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    return " ".join(text.strip().split())

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents if sent.text.strip()]

def extract_title_and_authors_from_metadata(meta_text: str) -> str:
    title_pattern = re.compile(r'(?<=\n)([A-Z][^\n]{10,200})(?=\n)')
    author_pattern = re.compile(r'(?:by|authors?:)\s+(.+)', re.IGNORECASE)
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    title_match = title_pattern.search(meta_text)
    author_match = author_pattern.search(meta_text)
    year_match = year_pattern.search(meta_text)
    title = title_match.group(1).strip() if title_match else "Untitled"
    authors = author_match.group(1).strip() if author_match else "Unknown Authors"
    year = year_match.group(0) if year_match else "n.d."
    return f"{authors} ({year}) - \"{title}\""

def process_pdf_for_rag(pdf_path: str, metadata_pages=2):
    debug(f"Processing PDF: {pdf_path}")
    doc = fitz.Document(pdf_path)
    pages = doc.page_count

    # Extract metadata
    metadata_text = []
    for page_num in range(min(metadata_pages, pages)):
        page = doc.load_page(page_num)
        raw_text = preprocess_text(page.get_text("text"))
        metadata_text.append(raw_text)
    full_metadata = "\n".join(metadata_text)
    doc_citation = extract_title_and_authors_from_metadata(full_metadata)

    # Extract main text
    main_sentences = []
    for page_num in range(metadata_pages, pages):
        page = doc.load_page(page_num)
        raw_text = preprocess_text(page.get_text("text"))
        main_sentences.extend(split_into_sentences(raw_text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len
    )
    meta_chunks = [doc.page_content for doc in text_splitter.create_documents([full_metadata])] if full_metadata.strip() else []
    main_chunks = [doc.page_content for doc in text_splitter.create_documents(main_sentences)] if main_sentences else []
    
    debug(f"Extracted {len(meta_chunks)+len(main_chunks)} chunks.")
    return doc_citation, meta_chunks + main_chunks

def process_all_pdfs(pdf_paths_with_idx):
    all_chunks = []
    doc_citations = []
    for (doc_idx, pdf_path) in pdf_paths_with_idx:
        debug(f"Processing PDF #{doc_idx}")
        citation, chunks = process_pdf_for_rag(pdf_path, metadata_pages=2)
        all_chunks.append(chunks)
        doc_citations.append(citation)
    return all_chunks, doc_citations

# -----------------------------------------------------------------------------
# 3. LLM-Based Entity Extraction
# -----------------------------------------------------------------------------
ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities

Format each entity output as a JSON entry with the following format:

{{"name": <entity name>, "type": <type>, "description": <entity description>}}

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: an integer score between 1 to 10, indicating strength of the relationship between the source entity and target entity

Format each relationship as a JSON entry with the following format:

{{"source": <source_entity>, "target": <target_entity>, "relationship": <relationship_description>, "relationship_strength": <relationship_strength>}}

3. Return output in {language} as a single list of all JSON entities and relationships identified in steps 1 and 2.

######################
-Examples-
######################
Example 1:
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
[
  {{"name": "CENTRAL INSTITUTION", "type": "ORGANIZATION", "description": "The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday"}},
  {{"name": "MARTIN SMITH", "type": "PERSON", "description": "Martin Smith is the chair of the Central Institution"}},
  {{"name": "MARKET STRATEGY COMMITTEE", "type": "ORGANIZATION", "description": "The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply"}},
  {{"source": "MARTIN SMITH", "target": "CENTRAL INSTITUTION", "relationship": "Martin Smith is the Chair of the Central Institution and will answer questions at a press conference", "relationship_strength": 9}}
]
######################
-Real Data-
######################
entity_types: {entity_types}
text: {input_text}
######################
Output:
"""

def extract_elements_from_chunk(chunk, tokenizer, model):
    debug("Extracting elements from a chunk using Microsoft prompting style.")
    # Build the prompt using the Microsoft JSON prompt.
    # You can adjust the default entity types and language as needed.
    prompt = ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT.format(
        entity_types="ORGANIZATION, PERSON, GEO",
        input_text=chunk,
        language="English"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        data = json.loads(text)
    except Exception as e:
        debug(f"JSON parsing error: {e}")
        data = {"entities": [], "relationships": []}
    return data

def extract_graph_elements(all_chunks, extraction_tokenizer, extraction_model):
    debug("Beginning extraction over all chunks.")
    entity_rows = defaultdict(list)
    relationship_rows = defaultdict(list)
    
    for doc_idx, chunks in enumerate(all_chunks):
        debug(f"Extracting from document {doc_idx} with {len(chunks)} chunks.")
        for chunk_idx, chunk in enumerate(chunks):
            data = extract_elements_from_chunk(chunk, extraction_tokenizer, extraction_model)
            for ent in data.get("entities", []):
                if len(ent) >= 3:
                    entity_rows["doc_idx"].append(doc_idx)
                    entity_rows["chunk_idx"].append(chunk_idx)
                    entity_rows["Entity"].append(ent[0])
                    entity_rows["Type"].append(ent[1])
                    entity_rows["Description"].append(ent[2])
            for rel in data.get("relationships", []):
                if len(rel) >= 3:
                    relationship_rows["doc_idx"].append(doc_idx)
                    relationship_rows["chunk_idx"].append(chunk_idx)
                    relationship_rows["Source"].append(rel[0])
                    relationship_rows["Target"].append(rel[1])
                    relationship_rows["Description"].append(rel[2])
    
    df_entities = pd.DataFrame(entity_rows)
    df_relationships = pd.DataFrame(relationship_rows)
    debug(f"Extracted {df_entities.shape[0]} entities and {df_relationships.shape[0]} relationships.")
    return df_entities, df_relationships

# -----------------------------------------------------------------------------
# 4. Build a Knowledge Graph 
# -----------------------------------------------------------------------------
def knowledge_graph_from_elements(df_entities, df_relationships):
    debug("Building knowledge graph.")
    G = nx.Graph()
    for _, row in df_entities.iterrows():
        entity = row["Entity"]
        if not G.has_node(entity):
            G.add_node(entity, description=row["Description"], type=row["Type"],
                       doc_idx=row["doc_idx"], chunk_idx=row["chunk_idx"])
    for _, row in df_relationships.iterrows():
        src = row["Source"]
        tgt = row["Target"]
        if G.has_node(src) and G.has_node(tgt):
            if not G.has_edge(src, tgt):
                G.add_edge(src, tgt, description=row["Description"], weight=1.0)
            else:
                G[src][tgt]["weight"] += 1.0
    debug(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# Merge similar nodes using SPECTER2 embeddings
from sklearn.cluster import AgglomerativeClustering
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()[0]

def hierarchical_merge_entities(entities, embeddings, threshold=0.9):
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - threshold, linkage="average")
    labels = clustering.fit_predict(embeddings)
    merged_entities = {}
    for i, label in enumerate(labels):
        merged_entities.setdefault(label, []).append(entities[i])
    return merged_entities

def merge_similar_nodes(G, threshold=0.9, embedding_tokenizer=None, embedding_model=None):
    debug("Merging similar nodes.")
    node_list = list(G.nodes())
    embeddings = np.array([get_embedding(name, embedding_tokenizer, embedding_model) for name in node_list])
    clusters = hierarchical_merge_entities(node_list, embeddings, threshold)
    G_copy = copy.deepcopy(G)
    for cluster_entities in clusters.values():
        if len(cluster_entities) > 1:
            main_entity = cluster_entities[0]
            for entity in cluster_entities[1:]:
                if not G_copy.has_node(entity):
                    continue
                G_copy = nx.contracted_nodes(G_copy, main_entity, entity, self_loops=False)
    debug(f"After merging, graph has {G_copy.number_of_nodes()} nodes.")
    return G_copy

# -----------------------------------------------------------------------------
# 5. Community Detection and Summarization 
# -----------------------------------------------------------------------------
def detect_communities(G):
    debug("Detecting communities using Leiden algorithm.")
    # Convert NetworkX graph to igraph graph
    edges = list(G.edges())
    nodes = list(G.nodes())
    g = ig.Graph.TupleList(edges, directed=False, vertex_name_attr="name")
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition)
    node_names = g.vs["name"]
    part_dict = {}
    for cluster, cluster_nodes in enumerate(partition):
        for idx in cluster_nodes:
            part_dict[node_names[idx]] = cluster
    debug(f"Detected {len(set(part_dict.values()))} communities.")
    return part_dict

def summarize_community(community_text, summ_tokenizer, summ_model):
    debug("Summarizing a community.")
    system_instruction = (
        "You are a university professor tasked with summarizing academic topics. "
        "Summarize the following text into a set of concise, well-connected statements."
    )
    prompt = f"""
{system_instruction}

TEXT:
{community_text}

SUMMARY:
"""
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=False
    )
    summary = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "SUMMARY:" in summary:
        summary = summary.split("SUMMARY:", 1)[-1].strip()
    debug("Community summarized.")
    return summary

def build_community_summaries(G, partition, all_chunks, summ_tokenizer, summ_model):
    debug("Building community summaries.")
    from collections import defaultdict
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    community_summaries = {}
    for comm_id, node_list in communities.items():
        text_for_summary = []
        for node in node_list:
            data = G.nodes[node]
            d_idx = data["doc_idx"]
            c_idx = data["chunk_idx"]
            if d_idx < len(all_chunks) and c_idx < len(all_chunks[d_idx]):
                text_for_summary.append(all_chunks[d_idx][c_idx])
        if not text_for_summary:
            community_summaries[comm_id] = "No textual info available."
        else:
            big_context = "\n".join(text_for_summary)
            summary = summarize_community(big_context, summ_tokenizer, summ_model)
            community_summaries[comm_id] = summary
    debug("Community summaries built.")
    return community_summaries

# -----------------------------------------------------------------------------
# 6. Query Phase: Map–Reduce Over Community Summaries
# -----------------------------------------------------------------------------
def generate_partial_answer(question, summary_text, summ_tokenizer, summ_model):
    debug("Generating partial answer.")
    system_instruction = (
        "You are a research assistant specializing in academic summarization. "
        "Extract key insights from the following community summary that directly answer the user's question."
    )
    prompt = f"""
{system_instruction}

Community Summary:
{summary_text}

Question:
{question}

Academic Answer:
"""
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=False
    )
    ans = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Academic Answer:" in ans:
        ans = ans.split("Academic Answer:", 1)[-1].strip()
    debug("Partial answer generated.")
    return ans if ans else "No relevant information available."

def combine_answers(question, partial_answers, summ_tokenizer, summ_model):
    if not partial_answers:
        return "No relevant information found."
    structured_parts = "\n\n".join([f"- {ans}" for _, ans in partial_answers])
    system_instruction = (
        "You are an AI academic assistant. Merge the following research insights into a coherent, structured response "
        "that directly answers the question."
    )
    prompt = f"""
{system_instruction}

Collected research insights:
{structured_parts}

Question:
{question}

Final Answer:
"""
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=False
    )
    final_ans = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Final Answer:" in final_ans:
        final_ans = final_ans.split("Final Answer:", 1)[-1].strip()
    debug("Combined final answer generated.")
    return final_ans if final_ans else "No final answer could be generated."

def filter_relevant_communities(question, community_summaries, sent_transformer):
    debug("Filtering relevant communities.")
    query_embedding = sent_transformer.encode(question)
    community_texts = list(community_summaries.values())
    community_embeddings = np.array([sent_transformer.encode(text) for text in community_texts])
    sim_scores = cosine_similarity([query_embedding], community_embeddings)[0]
    selected_communities = {}
    for idx, (cid, summary) in enumerate(community_summaries.items()):
        if sim_scores[idx] > 0.75:
            selected_communities[cid] = summary
    debug(f"Filtered down to {len(selected_communities)} communities.")
    return selected_communities

def graph_rag_query_map_reduce(question, community_summaries, summ_tokenizer, summ_model, sent_transformer):
    debug("Starting map-reduce for query.")
    filtered_communities = filter_relevant_communities(question, community_summaries, sent_transformer)
    partials = []
    for cid, summ_text in filtered_communities.items():
        ans = generate_partial_answer(question, summ_text, summ_tokenizer, summ_model)
        if not ans.strip().lower().startswith("no relevant"):
            partials.append((cid, ans))
    final_answer = combine_answers(question, partials, summ_tokenizer, summ_model)
    debug("Map-reduce completed.")
    return final_answer

# -----------------------------------------------------------------------------
# 7. Main & Interactive Query Interface
# -----------------------------------------------------------------------------
def main():
    debug("Loading models...")
    extraction_tokenizer, extraction_model = load_extraction_model()
    summ_tokenizer, summ_model = load_summarization_model()

    from sentence_transformers import SentenceTransformer
    sent_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Set up SPECTER2 for merging similar nodes
    from transformers import AutoTokenizer as SpecTokenizer
    from adapters import AutoAdapterModel as SpecAdapterModel
    global embedding_tokenizer, embedding_model
    embedding_tokenizer = SpecTokenizer.from_pretrained("allenai/specter2_base")
    embedding_model = SpecAdapterModel.from_pretrained("allenai/specter2_base")
    embedding_model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

    pdf_paths = [
        "/work/Chatbot-in-academia/papers-testing/6495.pdf",
        "/work/Chatbot-in-academia/papers-testing/7294.pdf",
        "/work/Chatbot-in-academia/papers-testing/QMOD_ICQSS_2014_CEM_and_business_performance.pdf",
        "/work/Chatbot-in-academia/papers-testing/Wieland_wallenburg_supply_chain_risk_management.pdf",
        "/work/Chatbot-in-academia/papers-testing/allan_hansen_the_purposes_of_performance_management_systems_and_processes_acceptedversion.pdf",
        "/work/Chatbot-in-academia/papers-testing/cbs_forskningsindberetning_smg_30.pdf",
        "/work/Chatbot-in-academia/papers-testing/jan_mouritsen_et_al_performance_risk_and_overflows_acceptedversion.pdf",
        "/work/Chatbot-in-academia/papers-testing/katrine_schr_der_hansen_et_al_performance_management_trends_acceptedversion.pdf",
        "/work/Chatbot-in-academia/papers-testing/linkwp01_27.pdf",
        "/work/Chatbot-in-academia/papers-testing/smg_wp_2008_08.pdf"
    ]
    pdf_paths_with_idx = [(i, path) for i, path in enumerate(pdf_paths)]
    
    debug("Processing PDFs into text chunks...")
    all_chunks, doc_citations = process_all_pdfs(pdf_paths_with_idx)
    
    # --- Extraction Phase ---
    if os.path.exists(ENTITIES_CSV) and os.path.exists(RELATIONSHIPS_CSV):
        debug("Loading cached extracted entities and relationships.")
        df_entities = pd.read_csv(ENTITIES_CSV)
        df_relationships = pd.read_csv(RELATIONSHIPS_CSV)
    else:
        debug("Extracting graph elements using the lightweight model...")
        df_entities, df_relationships = extract_graph_elements(all_chunks, extraction_tokenizer, extraction_model)
        df_entities.to_csv(ENTITIES_CSV, index=False)
        df_relationships.to_csv(RELATIONSHIPS_CSV, index=False)
    debug(f"Extracted {df_entities.shape[0]} entities and {df_relationships.shape[0]} relationships.")
    
    # --- Graph Construction ---
    if os.path.exists(GRAPH_PKL):
        debug("Loading cached knowledge graph.")
        with open(GRAPH_PKL, "rb") as f:
            G = pickle.load(f)
    else:
        debug("Building knowledge graph from extracted elements...")
        G = knowledge_graph_from_elements(df_entities, df_relationships)
        with open(GRAPH_PKL, "wb") as f:
            pickle.dump(G, f)
    debug(f"Knowledge Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # --- Merge Similar Nodes ---
    debug("Merging similar nodes...")
    G_merged = merge_similar_nodes(G, threshold=0.9, embedding_tokenizer=embedding_tokenizer, embedding_model=embedding_model)
    debug(f"Merged Graph has {G_merged.number_of_nodes()} nodes and {G_merged.number_of_edges()} edges.")
    
    # --- Community Detection (using Leiden) ---
    debug("Detecting communities with Leiden algorithm...")
    partition = detect_communities(G_merged)
    
    # --- Community Summaries ---
    if os.path.exists(COMMUNITY_SUMMARIES_PKL):
        debug("Loading cached community summaries.")
        with open(COMMUNITY_SUMMARIES_PKL, "rb") as f:
            community_summaries = pickle.load(f)
    else:
        debug("Building community summaries using Llama 3.3...")
        community_summaries = build_community_summaries(G_merged, partition, all_chunks, summ_tokenizer, summ_model)
        with open(COMMUNITY_SUMMARIES_PKL, "wb") as f:
            pickle.dump(community_summaries, f)
    
    # --- Interactive Query Interface via ipywidgets ---
    input_box = widgets.Text(
        value='Explain Python?',
        placeholder='Type your question here',
        description='Question:',
        disabled=False
    )
    output_area = widgets.Output()

    def on_button_click(b):
        with output_area:
            output_area.clear_output()
            question = input_box.value
            answer = graph_rag_query_map_reduce(question, community_summaries, summ_tokenizer, summ_model, sent_transformer)
            print(answer)

    button = widgets.Button(
        description='Ask',
        disabled=False,
        button_style='',
        tooltip='Ask the question',
        icon='check'
    )
    button.on_click(on_button_click)
    display(input_box, button, output_area)

if __name__ == "__main__":
    main()