############################################################################################################
#  CHATBOT WITH Llama 3.3 70B FOR QUESTION ANSWERING & MANUAL ENTITIES AND RELATIONSHIPS
############################################################################################################

import os
import pickle
import re
import torch
import fitz
import networkx as nx
import numpy as np
import pandas as pd
import json
import igraph as ig
import leidenalg
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DEBUG = True

def debug(msg):
    if DEBUG:
        print("DEBUG:", msg)

# File paths for storing data
ENTITIES_CSV = "/work/Chatbot-in-academia/llama_chatbot/GraphRAGbot_piloted/entities.csv"
RELATIONSHIPS_CSV = "/work/Chatbot-in-academia/llama_chatbot/GraphRAGbot_piloted/relationships.csv"
GRAPH_PKL = "/work/Chatbot-in-academia/llama_chatbot/GraphRAGbot_piloted/knowledge_graph.pkl"
HIERARCHICAL_SUMMARIES_PKL = "/work/Chatbot-in-academia/llama_chatbot/GraphRAGbot_piloted/hierarchical_community_summaries.pkl"


# ------------------------------------------------------------
# 1. Load Models
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 2. PDF Processing & Text Chunking
# ------------------------------------------------------------

def process_pdf_for_rag(pdf_path, metadata_pages=2):
    debug(f"Processing PDF: {pdf_path}")
    doc = fitz.Document(pdf_path)
    pages = doc.page_count

    # Extract metadata from the first few pages
    metadata_text = []
    for page_num in range(min(metadata_pages, pages)):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text").strip()
        metadata_text.append(raw_text)
    full_metadata = "\n".join(metadata_text)

    # Attempt to extract title/authors/year from the metadata text
    title_pattern = re.compile(r'(?<=\n)([A-Z][^\n]{10,200})(?=\n)')
    author_pattern = re.compile(r'(?:by|authors?:)\s+(.+)', re.IGNORECASE)
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')

    title_match = title_pattern.search(full_metadata)
    author_match = author_pattern.search(full_metadata)
    year_match = year_pattern.search(full_metadata)

    title = title_match.group(1).strip() if title_match else "Untitled"
    authors = author_match.group(1).strip() if author_match else "Unknown Authors"
    year = year_match.group(0) if year_match else "n.d."

    doc_citation = f"{authors} ({year}) - \"{title}\""

    # Extract main text from remaining pages
    main_text = []
    for page_num in range(metadata_pages, pages):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text").strip()
        main_text.append(raw_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len
    )

    meta_chunks = text_splitter.create_documents([full_metadata]) if full_metadata.strip() else []
    main_chunks = text_splitter.create_documents(["\n".join(main_text)]) if main_text else []

    meta_chunks_text = [doc.page_content for doc in meta_chunks]
    main_chunks_text = [doc.page_content for doc in main_chunks]

    all_chunks = meta_chunks_text + main_chunks_text
    debug(f"Extracted {len(all_chunks)} chunks from PDF.")
    return doc_citation, all_chunks


def process_all_pdfs(pdf_paths_with_idx):
    all_chunks = []
    doc_citations = []
    for (doc_idx, pdf_path) in pdf_paths_with_idx:
        debug(f"Processing PDF #{doc_idx}")
        citation, chunks = process_pdf_for_rag(pdf_path)
        all_chunks.append(chunks)
        doc_citations.append(citation)
    return all_chunks, doc_citations


# ------------------------------------------------------------
# 3. Manual Knowledge Graph Creation
# ------------------------------------------------------------

def create_knowledge_graph_from_manual_data(entities_data, relationships_data):
    """
    Create a knowledge graph from manually provided entities and relationships.
    Args: 
        entities_data: List of dictionaries with entity information.
        relationships_data: List of dictionaries with relationship information.
    Returns:
        G: A NetworkX graph object representing the knowledge graph.
    """
    debug("Creating knowledge graph from manual data")

    G = nx.Graph()

    for entity in entities_data:
        G.add_node(
            entity['name'],
            description=entity['description'],
            type=entity.get('type', 'CONCEPT'),
            doc_idx=entity.get('doc_idx', 0),
            chunk_idx=entity.get('chunk_idx', 0)
        )

    for rel in relationships_data:
        source = rel['source']
        target = rel['target']
        if not G.has_node(source) or not G.has_node(target):
            debug(f"Warning: Node not found for relationship: {source} -> {target}")
            continue

        G.add_edge(
            source,
            target,
            description=rel['description'],
            weight=rel.get('strength', 5),
            doc_idx=rel.get('doc_idx', 0),
            chunk_idx=rel.get('chunk_idx', 0)
        )

    debug(f"Knowledge graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


# ------------------------------------------------------------
# 4. Community Detection and Summarization
# ------------------------------------------------------------

def detect_hierarchical_communities(G, max_levels=2):
    """Use Leiden to detect communities at multiple levels.
    Args:
        G: A NetworkX graph object.
        max_levels: Maximum number of hierarchical levels to detect.
    Returns:
        hierarchical_partitions: A dictionary mapping each level to its community partition.
        community_hierarchy: A dictionary mapping child communities to their parent communities.
    """
    debug("Detecting hierarchical communities via Leiden algorithm")

    edges = list(G.edges())
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]

    g = ig.Graph(n=len(nodes), edges=edge_list, directed=False)
    g.vs["name"] = nodes

    edge_weights = None
    if nx.get_edge_attributes(G, 'weight'):
        edge_weights = [G[u][v].get('weight', 1.0) for u, v in edges]
        g.es["weight"] = edge_weights

    hierarchical_partitions = {}

    # Level 0
    resolution0 = 0.5
    partition0 = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution0,
        weights=edge_weights
    )
    modularity0 = g.modularity(partition0.membership, weights=edge_weights)
    debug(f"Level 0: {len(set(partition0.membership))} communities, modularity={modularity0:.4f}")

    level0_dict = {}
    for idx, cluster_id in enumerate(partition0.membership):
        level0_dict[nodes[idx]] = cluster_id
    hierarchical_partitions[0] = level0_dict

    # Other levels
    resolutions = [1.0, 2.0]
    for level, resolution in enumerate(resolutions[:max_levels-1], 1):
        partition = leidenalg.find_partition(
            g, 
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights=edge_weights
        )
        modularity = g.modularity(partition.membership, weights=edge_weights)
        num_communities = len(set(partition.membership))
        debug(f"Level {level}: {num_communities} communities, modularity={modularity:.4f}")

        level_dict = {}
        for idx, cluster_id in enumerate(partition.membership):
            parent_id = hierarchical_partitions[level-1][nodes[idx]]
            hierarchical_id = f"{parent_id}_{cluster_id}"
            level_dict[nodes[idx]] = hierarchical_id

        hierarchical_partitions[level] = level_dict

    # parent-child relationships
    community_hierarchy = {}
    for lvl in range(1, len(hierarchical_partitions)):
        parent_level = lvl - 1
        child_level = lvl
        for node in G.nodes():
            if node in hierarchical_partitions[child_level]:
                child_comm = hierarchical_partitions[child_level][node]
                parent_comm = hierarchical_partitions[parent_level][node]
                if child_comm not in community_hierarchy:
                    community_hierarchy[child_comm] = parent_comm

    debug(f"Built hierarchical community structure with {len(hierarchical_partitions)} levels")
    return hierarchical_partitions, community_hierarchy


def get_community_nodes(partition, community_id):
    """Return all nodes that belong to a given community_id in a partition dictionary.
    Args:
        partition: A dictionary mapping nodes to community IDs.
        community_id: The ID of the community to retrieve nodes for.
    Returns:
        A list of nodes that belong to the specified community.
    """
    return [node for node, comm_id in partition.items() if comm_id == community_id]


def summarize_community(G, community_nodes, all_chunks, summ_tokenizer, summ_model, max_tokens=1800):
    """Generate a summary for a single community of nodes.
    Args:
        G: A NetworkX graph object.
        community_nodes: A list of nodes in the community to summarize.
        all_chunks: A list of all text chunks from the documents.
        summ_tokenizer: The tokenizer for the summarization model.
        summ_model: The summarization model.
        max_tokens: Maximum number of tokens for the summary.
    Returns:
        A string containing the generated summary.
    """
    debug(f"Summarizing community of {len(community_nodes)} nodes")
    if not community_nodes:
        return "No nodes in this community."

    subgraph = G.subgraph(community_nodes)
    community_info = []

    # Node info
    for node in community_nodes:
        node_type = G.nodes[node].get("type", "CONCEPT")
        node_desc = G.nodes[node].get("description", "No description available.")
        community_info.append(f"ENTITY: {node}\nTYPE: {node_type}\nDESCRIPTION: {node_desc}\n")

    # Edge info
    for u, v in subgraph.edges():
        edge_desc = G[u][v].get("description", f"Relationship between {u} and {v}.")
        community_info.append(f"RELATIONSHIP: {u} → {v}\nDESCRIPTION: {edge_desc}\n")

    # attach source text 
    relevant_chunks = set()
    for node in community_nodes:
        d_idx = G.nodes[node].get("doc_idx", 0)
        c_idx = G.nodes[node].get("chunk_idx", 0)
        if d_idx < len(all_chunks) and c_idx < len(all_chunks[d_idx]):
            relevant_chunks.add(all_chunks[d_idx][c_idx])

    for chunk in list(relevant_chunks)[:2]:
        community_info.append(f"SOURCE TEXT: {chunk}\n")

    combined_info = "\n".join(community_info)
    prompt = f"""
You are an academic knowledge synthesizer. Create a comprehensive summary of this research community
based on the entities, relationships, and source text provided.

Your summary should:
1. Identify the main research concepts and how they relate to each other
2. Explain the significance of these concepts in the academic context
3. Highlight any important methodologies, variables, or findings
4. Integrate all information into a coherent narrative

COMMUNITY INFORMATION:
{combined_info}

SUMMARY:
"""

    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.4
    )
    summary = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "SUMMARY:" in summary:
        summary = summary.split("SUMMARY:", 1)[1].strip()

    debug(f"Generated community summary with length {len(summary)} chars")
    return summary


def summarize_hierarchical_communities(G, hierarchical_partitions, community_hierarchy, all_chunks,
                                       summ_tokenizer, summ_model):
    """Build summaries for hierarchical communities at different levels.
    Args:
        G: A NetworkX graph object.
        hierarchical_partitions: A dictionary mapping levels to community partitions.
        community_hierarchy: A dictionary mapping child communities to parent communities.
        all_chunks: A list of all text chunks from the documents.
        summ_tokenizer: The tokenizer for the summarization model.
        summ_model: The summarization model.
    Returns:
        A dictionary mapping each level to its community summaries.
    """
    debug("Building hierarchical community summaries")
    leaf_level = max(hierarchical_partitions.keys())
    level_summaries = {}

    leaf_partitions = hierarchical_partitions[leaf_level]
    unique_leaf_comms = set(leaf_partitions.values())
    debug(f"Summarizing {len(unique_leaf_comms)} leaf communities at level {leaf_level}")

    leaf_summ = {}
    for comm_id in unique_leaf_comms:
        community_nodes = get_community_nodes(leaf_partitions, comm_id)
        summary = summarize_community(G, community_nodes, all_chunks, summ_tokenizer, summ_model)
        leaf_summ[comm_id] = summary
    level_summaries[leaf_level] = leaf_summ

    for lvl in range(leaf_level - 1, -1, -1):
        debug(f"Summarizing communities at level {lvl}")
        partitions = hierarchical_partitions[lvl]
        unique_comms = set(partitions.values())
        level_summaries[lvl] = {}

        for comm_id in unique_comms:
            community_nodes = get_community_nodes(partitions, comm_id)
            child_comms = []
            for child_id, parent_id in community_hierarchy.items():
                if parent_id == comm_id:
                    child_comms.append(child_id)

            if child_comms:
                child_texts = []
                for c_id in child_comms:
                    if c_id in level_summaries[lvl+1]:
                        child_summary = level_summaries[lvl+1][c_id]
                        child_texts.append(f"SUBCOMMUNITY SUMMARY: {child_summary}")

                combined_text = "\n\n".join(child_texts)
                prompt = f"""
You are an academic knowledge synthesizer. Create a comprehensive summary that integrates 
multiple subcommunity summaries into a coherent overview of a research area.

Your summary should:
1. Identify the common themes across subcommunities
2. Highlight the most important concepts, methodologies, and findings
3. Explain the relationships between different subcommunities
4. Present a unified view of this larger research area

SUBCOMMUNITY INFORMATION:
{combined_text}

INTEGRATED SUMMARY:
"""
                inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                outputs = summ_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    pad_token_id=summ_tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.4
                )
                summary = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "INTEGRATED SUMMARY:" in summary:
                    summary = summary.split("INTEGRATED SUMMARY:", 1)[1].strip()
            else:
                # If no child communities, summarize directly
                summary = summarize_community(G, community_nodes, all_chunks, summ_tokenizer, summ_model)

            level_summaries[lvl][comm_id] = summary

    total_summaries = sum(len(s) for s in level_summaries.values())
    debug(f"Generated {total_summaries} community summaries across {len(level_summaries)} levels")
    return level_summaries


# ------------------------------------------------------------
# 5. Query Processing
# ------------------------------------------------------------

def generate_answer_from_community_summary(question, community_summary, summ_tokenizer, summ_model):
    """Generate a partial answer + helpfulness score from a single community summary.
    Args: 
        question: Question posed by the user.
        community_summary: Summary of the community.
        summ_tokenizer: The tokenizer for the summarization model.
        summ_model: The summarization model.
    Returns:
        answer: The generated answer based on the community summary.
        score: A helpfulness score from 0-100.
    """
    debug(f"Generating partial answer from community summary for question: {question}")
    prompt = f"""
You are a specialized academic research assistant. Extract information from the community summary
that directly answers the user's question.

Your response must include:
1. A detailed answer based ONLY on information in the community summary
2. A helpfulness score from 0-100 indicating how useful this information is for answering the question
   (where 0 = completely irrelevant, and 100 = perfectly answers the question)

If the community summary contains no relevant information, assign a score of 0.

COMMUNITY SUMMARY:
{community_summary}

USER QUESTION:
{question}

ANSWER:

HELPFULNESS SCORE (0-100):
"""
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.3
    )
    response = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = ""
    score = 0
    if "ANSWER:" in response:
        parts = response.split("ANSWER:", 1)[1].split("HELPFULNESS SCORE", 1)
        if len(parts) > 0:
            answer = parts[0].strip()

    if "HELPFULNESS SCORE (0-100):" in response:
        score_text = response.split("HELPFULNESS SCORE (0-100):", 1)[1].strip()
        match = re.search(r'\d+', score_text)
        if match:
            try:
                val = int(match.group(0))
                score = max(0, min(100, val))
            except:
                score = 0

    debug(f"Generated partial answer with helpfulness={score}")
    return answer, score


def combine_answers(question, scored_answers, summ_tokenizer, summ_model):
    """Combine multiple partial answers into a final coherent response, prioritizing by helpfulness score.
    Args: 
        question: The original user question.
        scored_answers: A list of tuples (answer, score) from different community summaries.
        summ_tokenizer: The tokenizer for the summarization model.
        summ_model: The summarization model.
    Returns:
        final_response: A final comprehensive answer based on the combined partial answers.
    """
    debug("Combining partial answers by helpfulness score")
    if not scored_answers:
        return "No relevant information found."

    sorted_answers = sorted(scored_answers, key=lambda x: x[1], reverse=True)
    filtered_answers = [(ans, scr) for ans, scr in sorted_answers if scr > 0]
    if not filtered_answers:
        return "No relevant information found in the dataset for this question."

    selected_answers = []
    for ans, scr in filtered_answers[:5]: # combine top 5
        selected_answers.append(f"[Score: {scr}/100] {ans}")

    combined_text = "\n\n".join(selected_answers)
    prompt = f"""
You are an academic research assistant. Synthesize these partial answers into a final comprehensive answer.

Each partial answer has a helpfulness score (0-100). Focus on higher-scored answers but integrate all relevant info.

USER QUESTION:
{question}

PARTIAL ANSWERS:
{combined_text}

FINAL COMPREHENSIVE ANSWER:
"""
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.4
    )
    final_response = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "FINAL COMPREHENSIVE ANSWER:" in final_response:
        final_response = final_response.split("FINAL COMPREHENSIVE ANSWER:", 1)[1].strip()

    return final_response


# ------------------------------------------------------------
# 6. Chunk-Level Entity Retrieval
# ------------------------------------------------------------

def find_chunks_for_entities(query, community_nodes, all_chunks, sent_transformer, top_k=5):
    """
    Perform chunk-level entity-based retrieval.
    Args: 
        query: The user query.
        community_nodes: A list of nodes in the community.
        all_chunks: A list of all text chunks from the documents.
        sent_transformer: The sentence transformer model for encoding.
        top_k: The number of top chunks to return.
    Returns:
        best: A list of the top K relevant chunks based on entity mentions and query similarity.
    """
    debug(f"Searching chunks for entity references + query similarity. Entities: {community_nodes}")
    entity_names = [n.lower() for n in community_nodes if n]  # Lowercase for matching

    candidate_chunks = []
    for doc_idx, doc_chunks in enumerate(all_chunks):
        for c_idx, chunk_text in enumerate(doc_chunks):
            chunk_lower = chunk_text.lower()
            mention_count = 0
            for ent in entity_names:
                if ent in chunk_lower:
                    mention_count += 1
            if mention_count > 0:
                # If chunk mentions at least 1 entity, check similarity
                query_vec = sent_transformer.encode(query)
                chunk_vec = sent_transformer.encode(chunk_text)
                sim_score = cosine_similarity([query_vec], [chunk_vec])[0][0]

                # Weighted approach: final = 0.3 * mention_count + 0.7 * sim
                final_score = 0.3 * mention_count + 0.7 * sim_score
                candidate_chunks.append((chunk_text, final_score))

    candidate_chunks.sort(key=lambda x: x[1], reverse=True)
    best = [c[0] for c in candidate_chunks[:top_k]]
    debug(f"Selected {len(best)} relevant chunks from entity-based scanning.")
    return best


# ------------------------------------------------------------
# 7. Enhanced Query Processing
# ------------------------------------------------------------

def process_query_enhanced(question, G, hierarchical_partitions, level_summaries, all_chunks,
                           sent_transformer, summ_tokenizer, summ_model):
    """
    Enhanced query processing by picking the best community level,
    finding top relevant communities, and collecting their nodes, to then comibe the top chunks with the summaries
    and generate the final answer.
    Args:
        question: The user query.
        G: The knowledge graph.
        hierarchical_partitions: The hierarchical community partitions.
        level_summaries: The community summaries at different levels.
        all_chunks: The text chunks from the documents.
        sent_transformer: The sentence transformer model for encoding.
        summ_tokenizer: The tokenizer for the summarization model.
        summ_model: The summarization model.
    Returns:
        final_answer: The final answer generated based on the query and community information.
    """
    debug(f"Processing query with chunk-level entity retrieval: {question}")
    q_lower = question.lower()

    general_patterns = ["main topics", "overview", "summary", "what is the data about",
                        "high level", "general themes", "main findings"]
    specific_patterns = ["specific", "detail", "exactly", "precisely", "tell me more about",
                         "what is the relationship between", "how does"]

    if any(p in q_lower for p in general_patterns):
        best_level = 0
    elif any(p in q_lower for p in specific_patterns):
        best_level = max(hierarchical_partitions.keys())
    else:
        if 1 in hierarchical_partitions:
            best_level = 1
        else:
            best_level = max(hierarchical_partitions.keys())

    debug(f"Selected community level {best_level} for query")

    if "how many" in q_lower and ("entities" in q_lower or "nodes" in q_lower or "concepts" in q_lower):
        return f"The knowledge graph contains {G.number_of_nodes()} entities and {G.number_of_edges()} edges."

    if "what are the main" in q_lower and ("entities" in q_lower or "concepts" in q_lower):
        c_map = nx.degree_centrality(G)
        top_ents = sorted(c_map.items(), key=lambda x: x[1], reverse=True)[:10]
        response = "The main entities in the knowledge graph are:\n\n"
        for i, (ent, score) in enumerate(top_ents, 1):
            e_type = G.nodes[ent].get("type", "Unknown")
            response += f"{i}. {ent} (Type: {e_type})\n"
        return response

    community_summaries = level_summaries.get(best_level, {})
    if not community_summaries:
        return "No community summaries available at that level."

    query_vec = sent_transformer.encode(question)
    comm_scores = []
    for comm_id, summary in community_summaries.items():
        summ_vec = sent_transformer.encode(summary)
        sim = cosine_similarity([query_vec], [summ_vec])[0][0]
        comm_scores.append((comm_id, summary, sim))

    comm_sorted = sorted(comm_scores, key=lambda x: x[2], reverse=True)
    threshold = 0.4
    top_communities = {}
    for comm_id, summary, score in comm_sorted:
        if score >= threshold or len(top_communities) < 3:
            top_communities[comm_id] = summary
        if len(top_communities) >= 5:
            break

    if not top_communities:
        return "I couldn't find specific information to answer that question in the dataset."

    partitions = hierarchical_partitions[best_level]
    all_community_nodes = []
    for comm_id in top_communities:
        comm_nodes = get_community_nodes(partitions, comm_id)
        all_community_nodes.extend(comm_nodes)

    relevant_chunks = find_chunks_for_entities(question, all_community_nodes, all_chunks, sent_transformer, top_k=5)

    scored_answers = []
    for comm_id, summary_text in top_communities.items():
        ans, s = generate_answer_from_community_summary(question, summary_text, summ_tokenizer, summ_model)
        if s > 0:
            scored_answers.append((ans, s))

    final_answer = combine_answers(question, scored_answers, summ_tokenizer, summ_model)
    return final_answer


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    debug("Starting GraphRAG chatbot with chunk-based entity retrieval...")

    # 1. Load models
    debug("Loading models...")
    summ_tokenizer, summ_model = load_summarization_model()
    sent_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    pdf_paths = ["/work/Chatbot-in-academia/papers-testing/7294.pdf"]
    pdf_paths_with_idx = [(i, path) for i, path in enumerate(pdf_paths)]
    all_chunks, doc_citations = process_all_pdfs(pdf_paths_with_idx)

    debug(f"Loading entities from {ENTITIES_CSV}")
    try:
        entities_df = pd.read_csv(ENTITIES_CSV)
        entities_data = []
        for _, row in entities_df.iterrows():
            e = {
                "name": row["name"].strip().upper(),
                "type": row["type"].strip(),
                "description": row["description"].strip()
            }
            if "doc_idx" in entities_df.columns:
                e["doc_idx"] = int(row["doc_idx"]) if not pd.isna(row["doc_idx"]) else 0
            if "chunk_idx" in entities_df.columns:
                e["chunk_idx"] = int(row["chunk_idx"]) if not pd.isna(row["chunk_idx"]) else 0
            entities_data.append(e)
        debug(f"Loaded {len(entities_data)} entities.")
    except Exception as ex:
        debug(f"Error loading entities CSV: {ex}")
        print(f"Error: Could not load entities from {ENTITIES_CSV}. Check format.")
        return

    debug(f"Loading relationships from {RELATIONSHIPS_CSV}")
    try:
        relationships_df = pd.read_csv(RELATIONSHIPS_CSV)
        relationships_data = []
        for _, row in relationships_df.iterrows():
            r = {
                "source": row["source"].strip().upper(),
                "target": row["target"].strip().upper(),
                "description": row["description"].strip()
            }
            if "strength" in relationships_df.columns:
                r["strength"] = int(row["strength"]) if not pd.isna(row["strength"]) else 5
            if "doc_idx" in relationships_df.columns:
                r["doc_idx"] = int(row["doc_idx"]) if not pd.isna(row["doc_idx"]) else 0
            if "chunk_idx" in relationships_df.columns:
                r["chunk_idx"] = int(row["chunk_idx"]) if not pd.isna(row["chunk_idx"]) else 0
            relationships_data.append(r)
        debug(f"Loaded {len(relationships_data)} relationships.")
    except Exception as ex:
        debug(f"Error loading relationships CSV: {ex}")
        print(f"Error: Could not load relationships from {RELATIONSHIPS_CSV}. Check format.")
        return

    G = create_knowledge_graph_from_manual_data(entities_data, relationships_data)
    if GRAPH_PKL:
        with open(GRAPH_PKL, "wb") as f:
            pickle.dump(G, f)
        debug(f"Knowledge graph saved to {GRAPH_PKL}")

    debug("Detecting hierarchical communities...")
    hierarchical_partitions, community_hierarchy = detect_hierarchical_communities(G, max_levels=2)
    for lvl, partition in hierarchical_partitions.items():
        debug(f"Level {lvl} => {len(set(partition.values()))} communities")

    if os.path.exists(HIERARCHICAL_SUMMARIES_PKL):
        debug(f"Loading summaries from {HIERARCHICAL_SUMMARIES_PKL}")
        with open(HIERARCHICAL_SUMMARIES_PKL, "rb") as f:
            level_summaries = pickle.load(f)
    else:
        debug("Generating hierarchical community summaries...")
        level_summaries = summarize_hierarchical_communities(
            G, hierarchical_partitions, community_hierarchy, all_chunks,
            summ_tokenizer, summ_model
        )
        if HIERARCHICAL_SUMMARIES_PKL:
            with open(HIERARCHICAL_SUMMARIES_PKL, "wb") as f:
                pickle.dump(level_summaries, f)
            debug(f"Saved hierarchical summaries to {HIERARCHICAL_SUMMARIES_PKL}")

    # Interactive loop
    print("\nGraphRAG Chatbot (chunk-based entity retrieval) ready! Type 'exit' to quit.")
    while True:
        query = input("\nUser: ")
        if query.lower() == "exit":
            print("Exiting.")
            break

        answer = process_query_enhanced(
            query,
            G,
            hierarchical_partitions,
            level_summaries,
            all_chunks,
            sent_transformer,
            summ_tokenizer,
            summ_model
        )
        print("\nBot:", answer)


if __name__ == "__main__":
    main()
