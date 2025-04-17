##################################################################
# IMPLEMENTATION OF GRAPH RAG ON OPENAI'S GPT-4o-MINI 
##################################################################

''' This code is a simplified version of the Graph RAG framework on GPT 4o-mini.'''

import os
import pickle
import re
import fitz
import networkx as nx
import pandas as pd
import igraph as ig
import leidenalg
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import gradio as gr

DEBUG = True

def debug(msg):
    if DEBUG:
        print("DEBUG:", msg)

client = OpenAI(api_key="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")

ENTITIES_CSV = "GraphRAGOpenAI/entities.csv"
RELATIONSHIPS_CSV = "GraphRAGOpenAI/relationships.csv"
GRAPH_PKL = "knowledge_graph.pkl"
HIERARCHICAL_SUMMARIES_PKL = "community_summaries.pkl"
PDF_PATHS = ["papers-cleaned/6495.txt"]

sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# PDF Processing
# -------------------------------
def process_pdf_chunks(pdf_path, metadata_pages=2):
    '''
    Process PDF files into chunks of text for further analysis.
    Args: 
        pdf_path: Path to the PDF file.
        metadata_pages: Number of pages to skip for metadata.
    Returns:
        list: List of text chunks.'''
    doc = fitz.open(pdf_path)
    pages = doc.page_count
    text = []
    for i in range(pages):
        page = doc.load_page(i)
        text.append(page.get_text("text"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.create_documents(["\n".join(text)])
    return [chunk.page_content for chunk in chunks]

# -------------------------------
# Graph Construction
# -------------------------------
def create_graph_from_manual_data(entities_df, relationships_df):
    '''
    Create a graph from entities and relationships.
    Args:
        entities_df: DataFrame containing entity information.
        relationships_df: DataFrame containing relationship information.
    Returns:
        G: NetworkX graph object.'''
    G = nx.Graph()
    for _, row in entities_df.iterrows():
        G.add_node(
            row["name"],
            description=row["description"],
            type=row.get("type", "CONCEPT"), # defaulting to concept is the type is missing
            doc_idx=row.get("doc_idx", 0),
            chunk_idx=row.get("chunk_idx", 0)
        )
    for _, row in relationships_df.iterrows():
        if G.has_node(row["source"]) and G.has_node(row["target"]):
            G.add_edge(
                row["source"],
                row["target"],
                description=row["description"],
                weight=row.get("strength", 5),
                doc_idx=row.get("doc_idx", 0),
                chunk_idx=row.get("chunk_idx", 0)
            )
    debug(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

# -------------------------------
# Hierarchical Community Detection
# -------------------------------
def detect_hierarchical_communities(G, levels=2):
    '''
    Detect hierarchical communities in the graph.
    Args:
        G: NetworkX graph object.
        levels: Number of hierarchical levels to detect.
    Returns:
        partitions: Community memberships at different levels.'''
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()] # coversion of edges from name to numeric index
    g = ig.Graph(n=len(nodes), edges=edges)
    g.vs["name"] = nodes # assign original names back to the nodes

    partitions = {}
    resolutions = [0.5, 1.0][:levels]
    for i, res in enumerate(resolutions):
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res
        )
        membership = {nodes[idx]: part for idx, part in enumerate(partition.membership)}
        partitions[i] = membership
    return partitions

# -------------------------------
# Summarization using GPT-4o mini
# -------------------------------
def summarize_community_gpt(community_nodes, G, chunks):
    '''
    Summarize the community using GPT-4o mini.
    Args:
        community_nodes: List of nodes in the community.
        G: NetworkX graph object.
        chunks: List of text chunks.
    Returns:
        response: Summarized text.'''
    parts = []
    for node in community_nodes:
        parts.append(f"ENTITY: {node}\nTYPE: {G.nodes[node].get('type')}\nDESCRIPTION: {G.nodes[node].get('description')}\n")
    for u, v in G.subgraph(community_nodes).edges():
        parts.append(f"RELATIONSHIP: {u} → {v}\nDESCRIPTION: {G[u][v].get('description')}\n")

    seen = set()
    for node in community_nodes:
        d_idx = G.nodes[node].get("doc_idx", 0)
        c_idx = G.nodes[node].get("chunk_idx", 0)
        if d_idx < len(chunks) and c_idx < len(chunks[d_idx]): # check lenght of document and chunk 
            seen.add((d_idx, c_idx, chunks[d_idx][c_idx])) # store in tuple the document index, the chunk index and the actual text chunk

    for d_idx, c_idx, txt in list(seen)[:2]:
        parts.append(f"SOURCE TEXT [doc {d_idx}, chunk {c_idx}]: {txt}\n") # adding 2 examples to ground the summary

    prompt = f"""
You are an academic research assistant. Based on the following information, summarize the main research themes and how they relate:

{''.join(parts)}

SUMMARY:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful academic assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700
    )
    return response.choices[0].message.content.strip()

# -------------------------------
# Chunk Retrieval & Answering
# -------------------------------
def find_relevant_chunks(question, entities, chunks):
    '''Retrieval of the most relevant chunks based on the question and entities.
    Args:
        question: The question to be answered.
        entities: List of entities to consider.
        chunks: List of text chunks.
    Returns: 
        results: List of tuples containing similarity score and chunk text.'''
    results = []
    q_vec = sent_transformer.encode(question)
    for d_idx, doc_chunks in enumerate(chunks):
        for c_idx, ch in enumerate(doc_chunks):
            if any(ent.lower() in ch.lower() for ent in entities): # check match between entities and chunks without case sensitivity
                c_vec = sent_transformer.encode(ch)
                sim = cosine_similarity([q_vec], [c_vec])[0][0] # cosine similarity between question vector and chunk vector
                results.append((sim, f"[Source: doc {d_idx}, chunk {c_idx}]\n{ch}"))
    return sorted(results, reverse=True)[:3]

def synthesize_answers(question, answers):
    '''Synthesize the answers into a response.
    Args: 
        question: The question to be answered.
        answers: List of tuples containing score and answer text.
    Returns: 
        response: syntethized answers put together.'''
    partials = "\n\n".join([f"[Score: {score}]\n{ans}" for score, ans in answers])
    prompt = f"""
You are an academic assistant. Combine the following partial answers into a unified response.

QUESTION:
{question}

PARTIAL ANSWERS:
{partials}

FINAL ANSWER:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful academic assistant."}, {"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response.choices[0].message.content.strip()

def answer_question(question):
    '''Full question-answer loop. 
    Args: 
        question: The question to be answered. 
    Returns: 
        response: The final synthesized answer.'''
    scored_answers = []
    for group, nodes in GLOBAL_COMMUNITIES.items():
        summary = GLOBAL_SUMMARIES[group]
        chunks = find_relevant_chunks(question, nodes, GLOBAL_CHUNKS)
        context = summary + "\n\n" + "\n\n".join([c[1] for c in chunks]) # concatenate summary + chunks for complete context in answer

        prompt = f"""
Based only on the context below, answer the question. Then rate helpfulness 0-100.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:

HELPFULNESS SCORE (0-100):
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful academic assistant."}, {"role": "user", "content": prompt}],
            max_tokens=700
        ).choices[0].message.content

        if "HELPFULNESS SCORE" in response:
            ans, score = response.rsplit("HELPFULNESS SCORE", 1)
            score_val = int(re.findall(r"\d+", score)[0])
            scored_answers.append((score_val, ans.strip()))

    return synthesize_answers(question, sorted(scored_answers, reverse=True)[:3]) # pick 3 most helpful answers and combine them

# -------------------------------
# Gradio UI
# -------------------------------
def launch_ui():
    '''Launch the Gradio UI for the chatbot.'''
    gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(label="Ask a research question"),
        outputs=gr.Textbox(label="Answer with synthesis and citations"),
        title="GraphRAG Chatbot with GPT-4o-mini (Hierarchical + Synthesized + Cited)"
    ).launch()

# -------------------------------
# Main
# -------------------------------
def main():
    debug("Loading data...")
    all_chunks = [process_pdf_chunks(path) for path in PDF_PATHS]
    entities_df = pd.read_csv(ENTITIES_CSV)
    relationships_df = pd.read_csv(RELATIONSHIPS_CSV)
    G = create_graph_from_manual_data(entities_df, relationships_df)
    partitions = detect_hierarchical_communities(G, levels=2)
    leaf_partition = partitions[max(partitions.keys())]

    communities = defaultdict(list)
    for node, group in leaf_partition.items():
        communities[group].append(node)

    summaries = {group: summarize_community_gpt(nodes, G, all_chunks) for group, nodes in communities.items()}

    # setting global variables 
    global GLOBAL_GRAPH, GLOBAL_SUMMARIES, GLOBAL_CHUNKS, GLOBAL_COMMUNITIES
    GLOBAL_GRAPH = G
    GLOBAL_SUMMARIES = summaries
    GLOBAL_CHUNKS = all_chunks
    GLOBAL_COMMUNITIES = communities

    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f)
    with open(HIERARCHICAL_SUMMARIES_PKL, "wb") as f:
        pickle.dump(summaries, f)

    launch_ui()

if __name__ == "__main__":
    main()