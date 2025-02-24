import os
import pickle
import re
import torch
import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import pandas as pd
import spacy
import copy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer  # (still imported; may remove if unused elsewhere)
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from argparse import Namespace
from torch.utils.data import DataLoader, SequentialSampler

# Import the python-louvain library
import community as community_louvain

###############################################################################
# 1. Load LLaMA for response generation
###############################################################################
def load_llama_model():
    print("Loading Llama 3.3 model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model and tokenizer loaded successfully.")
    return tokenizer, model

###############################################################################
# 2. BLINK Entity Linking
###############################################################################
import sys
sys.path.insert(0, "/work/BLINK")
from blink.main_dense import load_models
from blink.biencoder.data_process import process_mention_data

def load_blink_model():
    print("Loading BLINK entity linking model...")
    args_dict = {
        "biencoder_model": "/work/BLINK/models/biencoder_wiki_large.bin",
        "biencoder_config": "/work/BLINK/models/biencoder_wiki_large.json",
        "entity_catalogue": "/work/BLINK/models/entity.jsonl",
        "entity_encoding": "/work/BLINK/models/all_entities_large.t7",
        "crossencoder_model": "/work/BLINK/models/crossencoder_wiki_large.bin",
        "crossencoder_config": "/work/BLINK/models/crossencoder_wiki_large.json",
        "fast": False,
    }
    args = Namespace(**args_dict)
    blink_models = load_models(args)
    print("BLINK loaded successfully.")
    return blink_models

def run_biencoder(biencoder, dataloader, candidate_encoding, top_k=10, indexer=None):
    biencoder.model.eval()  # Set model to eval mode
    labels = []
    nns = []
    all_scores = []

    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        context_input = context_input.to(biencoder.device)

        with torch.no_grad():
            context_embeddings = biencoder.encode_context(context_input)
            context_embeddings = context_embeddings.cpu().numpy()

            if indexer is not None:
                scores, indices = indexer.search_knn(context_embeddings, top_k)
            else:
                scores = np.dot(context_embeddings, candidate_encoding.T)
                indices = np.argsort(-scores, axis=1)[:, :top_k]
                scores = np.sort(scores, axis=1)[:, :top_k]

        labels.extend(label_ids.numpy())
        nns.extend(indices)
        all_scores.extend(scores)

    return labels, nns, all_scores

def run_blink(text, blink_models):
    (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer
    ) = blink_models

    max_length = biencoder_params["max_context_length"] - 2
    mention_text = text.lower()
    if len(mention_text) > max_length:
        mention_text = mention_text[:max_length]

    samples = [{
        "context_left": "",
        "mention": mention_text,
        "context_right": "",
        "label": "unknown",
        "label_id": -1
    }]

    _, tensor_data = process_mention_data(
        samples,
        biencoder.tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params.get("debug", False)
    )

    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )

    labels, nns, scores = run_biencoder(
        biencoder, dataloader, candidate_encoding, top_k=10, indexer=faiss_indexer
    )

    linked_entities = []
    for entity_list in nns:
        for e_id in entity_list:
            entity_title = id2title[e_id]
            entity_description = id2text.get(e_id, "No description available")
            linked_entities.append({
                "entity": entity_title,
                "description": entity_description
            })

    return linked_entities

###############################################################################
# 3. PDF Processing + Metadata Extraction
###############################################################################
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = text.strip()
    text = " ".join(text.split())
    return text

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents if sent.text.strip()]

def extract_title_and_authors_from_metadata(meta_text: str) -> str:
    """
    Example approach with naive regex. Adjust as needed for your PDFs.
    """
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
    doc = fitz.Document(pdf_path)
    pages = doc.page_count

    # 1) Extract metadata text from first N pages
    metadata_text = []
    for page_num in range(min(metadata_pages, pages)):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text")
        raw_text = preprocess_text(raw_text)
        metadata_text.append(raw_text)
    full_metadata = "\n".join(metadata_text)

    # 2) Attempt to parse a user-friendly doc citation
    doc_citation = extract_title_and_authors_from_metadata(full_metadata)

    # 3) Collect main text from the rest
    main_sentences = []
    for page_num in range(metadata_pages, pages):
        page = doc.load_page(page_num)
        raw_text = preprocess_text(page.get_text("text"))
        sents = split_into_sentences(raw_text)
        main_sentences.extend(sents)

    # 4) chunk both metadata & main content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len
    )

    # For metadata
    meta_chunks = []
    if full_metadata.strip():
        meta_docs = text_splitter.create_documents([full_metadata])
        for md in meta_docs:
            meta_chunks.append(md.page_content)

    # For main content
    main_chunks = []
    if main_sentences:
        doc_splits = text_splitter.create_documents(main_sentences)
        for splitdoc in doc_splits:
            main_chunks.append(splitdoc.page_content)

    return doc_citation, meta_chunks, main_chunks

def process_all_pdfs(pdf_paths_with_idx):
    all_chunks = []
    doc_citations = []

    for (doc_idx, pdf_path) in pdf_paths_with_idx:
        print(f"Processing PDF #{doc_idx}: {pdf_path}")
        doc_citation, meta_chunks, main_chunks = process_pdf_for_rag(pdf_path, metadata_pages=2)
        doc_chunks = meta_chunks + main_chunks

        all_chunks.append(doc_chunks)
        doc_citations.append(doc_citation)

    return all_chunks, doc_citations

###############################################################################
# 4. BLINK Linking Over Chunks
###############################################################################
def link_entities_with_blink(text, blink_models):
    linked = run_blink(text, blink_models)
    return linked

###############################################################################
# 5. Build or Load Entities
###############################################################################
ENTITY_CSV = "saved_entities.csv"  # Adjust path as needed

def extract_entities(all_chunks, blink_models):
    entity_rows = defaultdict(list)
    for doc_idx, doc_chunks in enumerate(all_chunks):
        print(f"Extracting entities from doc_idx={doc_idx} ... #chunks={len(doc_chunks)}")
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            entities = link_entities_with_blink(chunk_text, blink_models)
            if not entities:
                continue
            for ent in entities:
                entity_rows["doc_idx"].append(doc_idx)
                entity_rows["chunk_idx"].append(chunk_idx)
                entity_rows["Entity"].append(ent["entity"])
                entity_rows["Description"].append(ent["description"])

    df = pd.DataFrame(entity_rows)
    return df

def load_or_extract_entities(all_chunks, blink_models):
    if os.path.exists(ENTITY_CSV):
        print(f"Loading entities from {ENTITY_CSV}")
        df_entities = pd.read_csv(ENTITY_CSV)
    else:
        print("No existing CSV found. Extracting entities with BLINK (slow) ...")
        df_entities = extract_entities(all_chunks, blink_models)
        df_entities.to_csv(ENTITY_CSV, index=False)
        print(f"Saved entities to {ENTITY_CSV}")
    return df_entities

###############################################################################
# 6. Build a Knowledge Graph & Merge
###############################################################################
GRAPH_PKL = "saved_graph.pkl"  # Adjust path as needed

def knowledge_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        entity = row["Entity"]
        desc = row["Description"]
        d_idx = row["doc_idx"]
        c_idx = row["chunk_idx"]

        if not G.has_node(entity):
            G.add_node(entity, description=desc, doc_idx=d_idx, chunk_idx=c_idx)
        # We do not do else/update because you might have multiple references

    # Now build edges
    for (d_idx, c_idx), group_df in df.groupby(["doc_idx","chunk_idx"]):
        entities_in_that_chunk = list(group_df["Entity"].unique())
        for i in range(len(entities_in_that_chunk)):
            for j in range(i+1, len(entities_in_that_chunk)):
                e1 = entities_in_that_chunk[i]
                e2 = entities_in_that_chunk[j]
                if not G.has_edge(e1, e2):
                    G.add_edge(e1, e2, relationship="related", weight=1.0)
    return G

# Helper function to obtain an embedding using SPECTER2 adapter
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    # Use the [CLS] token as the embedding
    embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()[0]
    return embedding

def merge_similar_nodes(G, threshold=0.9):
    node_list = list(G.nodes(data=True))
    names = [n[0] for n in node_list]
    # Compute embeddings using SPECTER2
    embeddings = np.array([get_embedding(name, embedding_tokenizer, embedding_model) for name in names])

    cos_sim = cosine_similarity(embeddings)
    np.fill_diagonal(cos_sim, 0)

    G_copy = copy.deepcopy(G)
    for i, (node_i, _) in enumerate(node_list):
        # Skip if node_i was contracted away
        if not G_copy.has_node(node_i):
            continue
        for j, (node_j, _) in enumerate(node_list):
            if i != j and cos_sim[i, j] >= threshold:
                # Check both nodes still exist in the current graph
                if not (G_copy.has_node(node_i) and G_copy.has_node(node_j)):
                    continue
                attr_i = G_copy.nodes[node_i]
                attr_j = G_copy.nodes[node_j]
                # Preserve missing attributes from node_j into node_i
                if "doc_idx" not in attr_i and "doc_idx" in attr_j:
                    attr_i["doc_idx"] = attr_j["doc_idx"]
                if "chunk_idx" not in attr_i and "chunk_idx" in attr_j:
                    attr_i["chunk_idx"] = attr_j["chunk_idx"]
                if "description" not in attr_i and "description" in attr_j:
                    attr_i["description"] = attr_j["description"]
                # Contract node_j into node_i
                G_copy = nx.contracted_nodes(G_copy, node_i, node_j, self_loops=False)
    return G_copy

def load_or_build_graph(df_entities, threshold=0.9):
    if os.path.exists(GRAPH_PKL):
        print(f"Loading knowledge graph from {GRAPH_PKL}")
        with open(GRAPH_PKL, "rb") as f:
            G_merged = pickle.load(f)
        print(f"Graph loaded with {G_merged.number_of_nodes()} nodes and {G_merged.number_of_edges()} edges.")
        return G_merged
    else:
        print("Building knowledge graph ...")
        G = knowledge_graph(df_entities)
        print(f"Initial Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

        print("Merging near-duplicate nodes with threshold=", threshold)
        G_merged = merge_similar_nodes(G, threshold=threshold)
        print(f"Merged Graph: {G_merged.number_of_nodes()} nodes, {G_merged.number_of_edges()} edges.")

        with open(GRAPH_PKL, "wb") as f:
            pickle.dump(G_merged, f)
        print(f"Saved merged graph to {GRAPH_PKL}")
        return G_merged

###############################################################################
# 7. Community Detection + Summaries
###############################################################################
def detect_communities_louvain(G):
    """
    Simple function that returns a dict {node: community_id} using Louvain.
    """
    partition = community_louvain.best_partition(G)
    return partition

def summarize_community(community_text, tokenizer, model):
    system_instruction = """\
You are a helpful assistant that summarizes large collections of text into a short statement of main themes, people, relationships.
"""
    prompt = f"""
{system_instruction}

TEXT:
{community_text}

SUMMARY:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(
        **inputs,
        max_length=1024,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "SUMMARY:" in summary:
        summary = summary.split("SUMMARY:", 1)[-1].strip()
    return summary

def build_community_summaries(G, partition, all_chunks, tokenizer, model):
    """
    For each community, gather chunk text from nodes, produce a single summary.
    Return dict: {comm_id: summary_text}
    """
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
            continue

        big_context = "\n".join(text_for_summary)
        summary = summarize_community(big_context, tokenizer, model)
        community_summaries[comm_id] = summary

    return community_summaries

###############################################################################
# 8. Query Phase: Map–Reduce Over Community Summaries
###############################################################################
def generate_partial_answer(question, summary_text, tokenizer, model):
    system_instruction = """\
You have the summary of ONE community. If you see relevant info about the question, produce a partial answer.
Otherwise say "I don't know."
"""
    prompt = f"""
{system_instruction}

Community summary:
{summary_text}

Question:
{question}

Partial Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(
        **inputs,
        max_length=512,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Partial Answer:" in ans:
        ans = ans.split("Partial Answer:", 1)[-1].strip()
    if ans.strip() == "":
        ans = "I don't know."
    return ans

def combine_answers(question, partial_answers, tokenizer, model):
    """
    partial_answers is a list of (community_id, partial_str).
    We'll unify them into one final answer.
    """
    all_parts = []
    for cid, partial in partial_answers:
        all_parts.append(f"Community {cid} partial:\n{partial}\n")

    big_context = "\n".join(all_parts)

    system_instruction = """\
You are a summarizer that merges multiple partial answers into a final integrated answer. 
If contradictory, mention it.
"""
    prompt = f"""
{system_instruction}

Collected partial answers:
{big_context}

Question:
{question}

Final Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(
        **inputs,
        max_length=768,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    final_ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Final Answer:" in final_ans:
        final_ans = final_ans.split("Final Answer:",1)[-1].strip()

    return final_ans

def graph_rag_query_map_reduce(question, community_summaries, tokenizer, model):
    """
    1) Each community tries to answer => partial
    2) combine partial answers => final
    """
    partials = []
    for cid, summ_text in community_summaries.items():
        ans = generate_partial_answer(question, summ_text, tokenizer, model)
        if ans.strip().lower().startswith("i don't know"):
            continue
        partials.append((cid, ans))

    final_answer = combine_answers(question, partials, tokenizer, model)
    return final_answer

###############################################################################
# 9. Main
###############################################################################
def main():
    print("Loading models...")

    # 1) LLaMA for final generation
    llama_tokenizer, llama_model = load_llama_model()

    # 2) BLINK for entity extraction
    blink_models = load_blink_model()

    # 3) SPECTER2 for building/merging
    # Load SPECTER2 adapter model and tokenizer
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    global embedding_tokenizer, embedding_model  # make them available to merge_similar_nodes
    embedding_tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    embedding_model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    embedding_model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

    # 4) PDF paths
    pdf_paths = [
        # Your PDFs
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

    # Process PDFs -> text chunks
    print("Processing PDFs to create text chunks & parse metadata citations...")
    all_chunks, doc_citations = process_all_pdfs(pdf_paths_with_idx)

    # Extract or load BLINK-based entities
    df_entities = load_or_extract_entities(all_chunks, blink_models)
    print(f"DF Entities shape: {df_entities.shape}")

    # Build or load the knowledge graph
    G_merged = load_or_build_graph(df_entities, threshold=0.9)

    # (A) Detect communities (Louvain)
    print("Detecting communities with Louvain...")
    partition = detect_communities_louvain(G_merged)

    # (B) Build community summaries
    print("Building community summaries...")
    community_summaries = build_community_summaries(
        G_merged, partition,
        all_chunks,
        llama_tokenizer,
        llama_model
    )

    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nUser: ")
        if query.lower() == "exit":
            print("Exiting. Goodbye.")
            break

        # For "global" sensemaking queries, do the map–reduce approach:
        answer = graph_rag_query_map_reduce(query, community_summaries, llama_tokenizer, llama_model)
        print("\nBot:", answer)

if __name__ == "__main__":
    main()
