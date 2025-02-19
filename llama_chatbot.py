################################################################################
#  CHATBOT WITH LLAMA MODEL   #
################################################################################

'''Reason: use internal chatbot based on Llama model instead of ChatGPT or any other third party model
mainly because of privacy reasons, it's good that we don't have to send the data to any third party server and 
we have full control over the data.'''

import torch
import networkx as nx  
import numpy as np
import fitz
import re
import pandas as pd
import spacy
import json
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydoc import doc
from sklearn.metrics.pairwise import cosine_similarity
# --------------------------------------------------------------------------
# 1.1 Loading Llama 3.3 model from Hugging Face because of storage limitations
# --------------------------------------------------------------------------

def load_llama_model():
    '''Load Llama 3.3 model from Hugging Face'''
    print("Loading Llama 3.3 model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct",
    torch_dtype= torch.float16,
    device_map = "auto")
    print("Model and tokenizer loaded successfully.")
    return tokenizer, model

# --------------------------------------------------------------------------
# 1.2 Load a Fast NER Model for Entity Extraction (using dslim/bert-base-NER)
# --------------------------------------------------------------------------
def load_ner_model():
    '''Load a fast, pre-trained NER model (dslim/bert-base-NER) from Hugging Face'''
    print("Loading NER model (dslim/bert-base-NER)...")
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    print("NER model and tokenizer loaded successfully.")
    return tokenizer, model
# --------------------------------------------------------------------------
# 2. Process PDF Files and Split Text into Chunks
# --------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Preprocess text: clean whitespace and normalize text.
    Args:
        text: The input text to preprocess.
    Returns:
        text: The preprocessed text."""
    text = text.strip()
    text = " ".join(text.split())
    return text

def split_into_sentences(text):
    """Use spaCy to split text into sentences before chunking.
    Args:
        text: The input text to split.
    Returns:
        List of sentences."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# Chunking text of processed PDF with LangChain's recursive character splitter
def process_pdf(pdf_paths):
    '''Reads PDF files, extracts text, and splits it into chunks.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        all_chunks(list): List of chunked text segments.
    '''
    print(f"Starting PDF processing for {len(pdf_paths)} files.")
    # Define the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    # Read the PDF file and extract text
    all_chunks = []  # Stores chunks of each document

    for pdf_idx, pdf_path in pdf_paths:
        print(f"Processing PDF {pdf_idx+1}: {pdf_path}")
        document = fitz.Document(pdf_path)
        pages = document.page_count
        sentences_for_pdf = [] 

        for page_num in range(pages):
            page = document.load_page(page_num)
            raw_text = page.get_text("text")  
            cleaned_text = preprocess_text(raw_text)

            page_sentences = split_into_sentences(cleaned_text)
            sentences_for_pdf.extend(page_sentences)
        print(f"Extracted {len(sentences_for_pdf)} sentences from PDF {pdf_idx+1}.")

        doc_splits = text_splitter.create_documents(sentences_for_pdf)
        doc_chunks = [split.page_content for split in doc_splits]
        all_chunks.append(doc_chunks)
    
    print(f"Finished processing PDFs. Total chunks: {len(all_chunks)}")
    return all_chunks

# --------------------------------------------------------------------------
# 3. Extract Entities from Text Chunks
# --------------------------------------------------------------------------

def extract_entities(all_chunks, ner_tokenizer, ner_model, batch_size=4):
    '''Extract entities using a token classification model (e.g., dslim/bert-base-NER).'''
    print("Starting entity extraction using NER...")
    entity_list = defaultdict(list)
    id2label = ner_model.config.id2label  # e.g., {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', ...}
    
    for doc_idx, doc_chunks in enumerate(all_chunks):
        print(f"Extracting entities from document {doc_idx+1} with {len(doc_chunks)} chunks.")
        for batch_start in range(0, len(doc_chunks), batch_size):
            batch_end = batch_start + batch_size
            chunk_batch = doc_chunks[batch_start:batch_end]
            combined_text = "\n\n".join(chunk_batch)
            inputs = ner_tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                outputs = ner_model(**inputs)
            logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)
            predictions = torch.argmax(logits, dim=2)  # shape: (batch_size, seq_len)
            # Process each sample in the batch
            for i in range(inputs["input_ids"].size(0)):
                input_ids = inputs["input_ids"][i]
                pred = predictions[i]
                tokens = ner_tokenizer.convert_ids_to_tokens(input_ids)
                current_entity = ""
                current_label = ""
                # Optional: track chunk id (using batch_start for now)
                chunk_id = f"Doc_{doc_idx}_Chunk_{batch_start}"
                for token, label_id in zip(tokens, pred):
                    label = id2label[label_id.item()]
                    # If the token is not 'O'
                    if label != "O":
                        # If token is a subword, merge it
                        if token.startswith("##"):
                            current_entity += token[2:]
                        else:
                            # If there's an existing entity, store it before starting a new one
                            if current_entity:
                                entity_list["Document_ID"].append(f"Doc_{doc_idx}")
                                entity_list["Chunk_ID"].append(chunk_id)
                                entity_list["Entity"].append(current_entity)
                                entity_list["Entity_Label"].append(current_label)
                                entity_list["Description"].append(f"Label: {current_label}")
                            current_entity = token
                            current_label = label
                    else:
                        # End of an entity span
                        if current_entity:
                            entity_list["Document_ID"].append(f"Doc_{doc_idx}")
                            entity_list["Chunk_ID"].append(chunk_id)
                            entity_list["Entity"].append(current_entity)
                            entity_list["Entity_Label"].append(current_label)
                            entity_list["Description"].append(f"Label: {current_label}")
                            current_entity = ""
                            current_label = ""
                if current_entity:  # catch any trailing entity
                    entity_list["Document_ID"].append(f"Doc_{doc_idx}")
                    entity_list["Chunk_ID"].append(chunk_id)
                    entity_list["Entity"].append(current_entity)
                    entity_list["Entity_Label"].append(current_label)
                    entity_list["Description"].append(f"Label: {current_label}")
    print(f"Entity extraction complete. Extracted {len(entity_list['Entity'])} entities.")
    return pd.DataFrame(entity_list)

# --------------------------------------------------------------------------
# 4. Build Knowledge Graph from Extracted Entities (GraphRAG)
# -------------------------------------------------------------------------- 
 
def knowledge_graph(df):
    """Build a knowledge graph from the extracted entities.
    Args:
        df: DataFrame containing the extracted entities.
    Returns:
        G: NetworkX graph representing the knowledge graph."""
    print("Building knowledge graph from entities...")
    G = nx.Graph() # Create the graph with nodes (entities) and edges (relationships)

    for _, row in df.iterrows(): 
        G.add_node(row["Entity"], description=row["Description"], chunk=row["Chunk_ID"], document=row["Document_ID"])

    for _, chunk_df in df.groupby("Chunk_ID"):
        entities = list(chunk_df["Entity"])
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)): # every entity in the chunk is connected to every other entity
                G.add_edge(entities[i], entities[j], relationship="related", weight=1.0)
    print(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G

# -----------------------------------------------------------------------------
# 5. Merge near duplicate entities (0.9 SIMILARITY)
# -----------------------------------------------------------------------------

def merge_similar_nodes(G, threshold=0.9):
    """
    Merge near-duplicate entities in the knowledge graph.
    Arg:
        G: NetworkX graph representing the knowledge graph.
        threshold: Similarity threshold for merging entities.
    Returns:
        G_copy: NetworkX graph with merged entities.
    """

    print(f"Merging near-duplicate entities with threshold = {threshold}...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    node_list = list(G.nodes(data=True))
    embeddings = []
    node_labels = []

    for (node_name, data) in node_list:
        text = f"Name: {node_name}\nDescription: {data['description']}"
        emb = embedding_model.encode(text)
        embeddings.append(emb)
        node_labels.append(node_name)

    embeddings = np.array(embeddings)
    cos_sim = cosine_similarity(embeddings)
    np.fill_diagonal(cos_sim, 0)
    adjacency_list = defaultdict(list)

    for i in range(cos_sim.shape[0]):
        for j in range(cos_sim.shape[0]):
            if cos_sim[i, j] >= threshold:
                adjacency_list[i].append(j)
        # if none => self link
        if not adjacency_list[i]:
            adjacency_list[i].append(i)

    visited_global = set()
    connected_components = []
    for idx in adjacency_list.keys():
        if idx not in visited_global:
            queue = [idx]
            visited_local = []
            while queue:
                node = queue.pop(0)
                if node not in visited_local:
                    visited_local.append(node)
                    for nbr in adjacency_list[node]:
                        if nbr not in visited_local:
                            queue.append(nbr)
            for v in visited_local:
                visited_global.add(v)
            connected_components.append(visited_local)

    G_copy = copy.deepcopy(G)
    for comp in connected_components:
        if len(comp) > 1:
            # comp is list of idx, convert to node labels
            nodes_in_comp = [node_labels[i] for i in comp]
            merged_node_name = "\n".join(nodes_in_comp)
            # create new node
            G_copy.add_node(merged_node_name, description=set(), chunk=set())

            # gather descriptions/chunks
            for old_node in nodes_in_comp:
                if G_copy.has_node(old_node):
                    data = G_copy.nodes[old_node]
                    G_copy.nodes[merged_node_name]["description"].add(data.get("description"))
                    G_copy.nodes[merged_node_name]["chunk"].add(data.get("chunk"))
            
            # flatten sets
            desc_combined = "\n".join(list(G_copy.nodes[merged_node_name]["description"]))
            G_copy.nodes[merged_node_name]["description"] = desc_combined
            chunk_combined = list(G_copy.nodes[merged_node_name]["chunk"])
            G_copy.nodes[merged_node_name]["chunk"] = chunk_combined

            # update edges
            for node1, node2, edge_data in list(G.edges(data=True)):
                if node1 in nodes_in_comp or node2 in nodes_in_comp:
                    new_node1 = merged_node_name if node1 in nodes_in_comp else node1
                    new_node2 = merged_node_name if node2 in nodes_in_comp else node2
                    if new_node1 != new_node2:
                        # merge edge
                        if G_copy.has_edge(new_node1, new_node2):
                            existing = G_copy[new_node1][new_node2]
                            existing["relationship"] += f"\n{edge_data['relationship']}"
                            existing["weight"] = max(existing["weight"], edge_data["weight"])
                        else:
                            G_copy.add_edge(new_node1, new_node2, relationship=edge_data["relationship"], weight=edge_data["weight"])

            # remove old nodes
            for old_node in nodes_in_comp:
                if G_copy.has_node(old_node):
                    G_copy.remove_node(old_node)
    print(f"Merged graph: {len(G_copy.nodes)} nodes, {len(G_copy.edges)} edges.")
    return G_copy

# --------------------------------------------------------------------------
# 6. Graph-Based Retrieval 
# --------------------------------------------------------------------------

def graph_retrieval(query, G, embedding_model, top_k=3):
    """Retrieve the most relevant entities using the knowledge graph based on similarity to the query.
    Args:
        query: The query string.
        G: The knowledge graph.
        embedding_model: covert text into numerical representations (vectors).
        top_k: the most relevant entities by similarity scores.
    Returns:
    retrieved_nodes: the most relevant entities based on the query."""
    print(f"Retrieving top {top_k} entities for query: {query}")
    query_embedding = embedding_model.encode(query) # convert the query into a vector
    scores = [] 
    for node, data in G.nodes(data=True):
        node_embedding = embedding_model.encode(data["description"]) # covert the description of the node into a vector
        similarity_score = np.dot(query_embedding, node_embedding)/(np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)) # calculate the cosine similarity, so how similar regardless of magnitude
        scores.append((node,similarity_score))
    scores = sorted(scores, key = lambda x: x[1], reverse = True)[:top_k] # descending order and top 3 k
    retrieved_nodes = [
        {"description": G.nodes[node]["description"], "document": G.nodes[node]["document"]} for node, _ in scores]
    print(f"Retrieved {len(retrieved_nodes)} relevant entities.")
    return retrieved_nodes

# --------------------------------------------------------------------------
# 7. Generate a Response 
# --------------------------------------------------------------------------

def gen_response(query, retrieved_nodes, llama_tokenizer, llama_model):
    '''Generates a response and retrieved context from GN.
    Args:
        query: The user's query.
        retrieved_nodes: Retrieved text segments.
        tokenizer: Hugging Face tokenizer for the Llama model.
        model: Hugging Face model for the Llama model.
    Returns: 
        response: The generated response.'''
    
    context = "\n\n".join(retrieved_nodes)
    prompt = f"""
    Use the following context to answer the question. If the answer is not in the context, say "I don't know.
     When providing information, reference the source document in brackets like this: [Source: Document_ID]"

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    inputs = llama_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
    outputs = llama_model.generate(inputs["input_ids"], attention_mask= inputs["attention_mask"], pad_token_id=llama_tokenizer.eos_token_id, max_length=200)
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --------------------------------------------------------------------------
# 8. Main Function
# --------------------------------------------------------------------------

def main():
    print("\n⏳ Loading model and processing PDFs... (This will take some time)")
    
    # Load model & tokenizer
    llama_tokenizer, llama_model = load_llama_model()
    ner_tokenizer, ner_model = load_ner_model()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    pdf_paths = [
        "papers-testing/6495.pdf",
        "papers-testing/7294.pdf",
        "papers-testing/QMOD_ICQSS_2014_CEM_and_business_performance.pdf",
        "papers-testing/Wieland_wallenburg_supply_chain_risk_management.pdf", 
        "papers-testing/allan_hansen_the_purposes_of_performance_management_systems_and_processes_acceptedversion.pdf",
        "papers-testing/cbs_forskningsindberetning_smg_30.pdf",
        "papers-testing/jan_mouritsen_et_al_performance_risk_and_overflows_acceptedversion.pdf",
        "papers-testing/katrine_schr_der_hansen_et_al_performance_management_trends_acceptedversion.pdf",
        "papers-testing/linkwp01_27.pdf",
        "papers-testing/smg_wp_2008_08.pdf"
    ]
    
    all_chunks = process_pdf([(idx, path) for idx, path in enumerate(pdf_paths)])

    # Extract Entities
    df_entities = extract_entities(all_chunks, ner_tokenizer, ner_model, batch_size=4)
    print("\n✅ Entity extraction complete!")
    
    # Build Graph
    G = knowledge_graph(df_entities)
    print(f"\n📌 Initial Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Merge near-duplicate nodes
    G_merged = merge_similar_nodes(G, threshold=0.9)
    print(f"\n🔗 Merged Graph: {len(G_merged.nodes)} nodes, {len(G_merged.edges)} edges")

    print("\n✅ Chatbot is ready! You can start asking questions. (Type 'exit' to quit)\n")

    # **Interactive Chat Loop so not to have to reload model and graph all the time**
    while True:
        query = input("\n🔍 Your Question: ")
        if query.lower() == "exit":
            print("\n👋 Exiting chatbot. Thanks for playing with me!")
            break

        # Graph retrieval
        retrieved = graph_retrieval(query, G_merged, embedding_model, top_k=3)

        # Prepare context for Llama response
        texts_for_context = [r["description"] for r in retrieved]
        answer = gen_response(query, texts_for_context, llama_tokenizer, llama_model)

        print("\n🧠 Axolbot Answer:\n", answer)

if __name__ == "__main__":
    main()

