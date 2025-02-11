################################################################################
#  CHATBOT WITH LLAMA MODEL   #
################################################################################

'''Reason: use internal chatbot based on Llama model instead of ChatGPT or any other third party model
mainly because of privacy reasons, it's good that we don't have to send the data to any third party server and 
we have full control over the data.'''

import torch
import networkx as nx  # this lib is for graphRAG 
import numpy as np
import fitz
import re
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydoc import doc

# --------------------------------------------------------------------------
# 1. Loading Llama 3.3 model from Hugging Face because of storage limitations
# --------------------------------------------------------------------------

def load_llama_model():
    '''Load Llama 3.3 model from Hugging Face'''
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct",
    torch_dtype= torch.float16,
    device_map = "auto")
    return tokenizer, model

# --------------------------------------------------------------------------
# 2. Process PDF Files and SPlit Text into Chunks
# --------------------------------------------------------------------------

# Chunking text of processed PDF with LangChain's recursive character splitter
def process_pdf(pdf_paths):
    '''Reads PDF files, extracts text, and splits it into chunks.
    
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        full_text(list): Full extracted text for each document.
        all_chunks(list): List of chunked text segments.
    '''
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
        document = fitz.Document(pdf_path)
        pages = document.page_count
        chunks = [] # Stores chunks of a single document

        for page_num in range(pages):
            page = document.load_page(page_num)
            text = page.get_text("text")  
            text = re.sub(r" +", r" ", text)  

            # Split the text into chunks
            text_chunk = text_splitter.create_documents([text])  
            chunks.extend([chunk.page_content for chunk in text_chunk])  
        all_chunks.append(chunks)
    
    return all_chunks

# --------------------------------------------------------------------------
# 3. Extract Entities from Text Chunks
# --------------------------------------------------------------------------

def extract_entities(all_chunks, tokenizer, model):
    '''Extract entities from text chunks using the Llama model.
    Arg:
        chunks: List of text chunks.
        tokenizer: Hugging Face tokenizer for the Llama model.
        model: Hugging Face model for the Llama model.
    Returns:
        DataFrame of entities extracted from the text chunks.
        '''
    entities_prompts = """Extract the entities from the following text and then return the entities as a JSON list.
    Text:
    {text}
    Return the result as a JSON list in this exact format:
[
    {"Name": "Entity Name", "Description": "Description of the entity"},
    {"Name": "Another Entity", "Description": "Its description"},
    ...
]"""
    entity_list = defaultdict(list)

    for doc_idx, doc_chunks in enumerate(all_chunks):
        for chunk_idx, chunk in enumerate(doc_chunks):
            prompt = entities_prompts.replace("{text}", chunk)
            input = tokenizer(prompt, return_tensors="pt").to("cuda") # tokenize the prompt and return a tensor
            output = model.generate(input["input_ids"], max_length= 300) # takes the input and pass it through the model
            response = tokenizer.decode(output[0], skip_special_tokens=True) # decodes the output back from token IDs to text

        # Make sure the JSON format is valid
        try: 
            output_json = json.loads(response)
            if isinstance(output_json, list):
                for entity in output_json:
                    entity_list["Document_ID"].append(f"Doc_{doc_idx}")
                    entity_list["Chunk_ID"].append(f"Doc_{doc_idx}_Chunk_{chunk_idx}")
                    entity_list["Entity"].append(entity.get("Name", "Unknown"))
                    entity_list["Description"].append(entity.get("Description", "Unknown"))
            else:
                print("Warning: LLaMA did not return a valid list.")
        except json.JSONDecodeError:
            print("Warning: Could not parse LLaMA's response.")
        return pd.DataFrame(entity_list)

# --------------------------------------------------------------------------
# 4. Build Knowledge Graph from Extracted Entities (GraphRAG)
# -------------------------------------------------------------------------- 
 
def knowledge_graph(df):
    """Build a knowledge graph from the extracted entities."""
    G = nx.Graph() # Create the graph with nodes (entities) and edges (relationships)

    for _, row in df.iterrows(): 
        G.add_node(row["Entity"], description=row["Description"], chunk=row["Chunk_ID"], document=row["Document_ID"])

    for _, chunk_df in df.groupby("Chunk_ID"):
        entities = list(chunk_df["Entity"])
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)): # every entity in the chunk is connected to every other entity
                G.add_edge(entities[i], entities[j], relationship="related", weight=1.0)
    return G

# --------------------------------------------------------------------------
# 5. Graph-Based Retrieval 
# --------------------------------------------------------------------------

def graph_retrieve(query, G...)

    





def graph_rag_retrieve(query, G, embedding_model, top_k=3):
    '''Retrieves the most relevant entities from the Knowledge Graph based on query similarity.'''
    query_embedding = embedding_model.encode(query)
    scores = []

    for node, data in G.nodes(data=True):
        node_embedding = embedding_model.encode(data['description'])
        sim_score = np.dot(query_embedding, node_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        scores.append((node, sim_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    retrieved_nodes = [G.nodes[node]["description"] for node, _ in scores]

    return retrieved_nodes

def get_graphrag():
    '''Load GraphRAG model for retrieval of data from CBS Archive'''


# --------------------------------------------------------------------------
# 3. Generate a Response 
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 4. Main Function
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 6. Generate a Response Using LLaMA 3.3
# --------------------------------------------------------------------------

def generate_response(query, retrieved_texts, tokenizer, model):
    '''Generates a response using LLaMA 3.3 and retrieved context from the Knowledge Graph.'''
    context = "\n\n".join(retrieved_texts)
    prompt = f"""
    Use the following context to answer the question. If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs["input_ids"], max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# --------------------------------------------------------------------------
# 7. Main Function: Run the GraphRAG Pipeline
# --------------------------------------------------------------------------

def main():
    tokenizer, model = load_llama_model()

    pdf_paths = ["example1.pdf", "example2.pdf"]
    all_chunks = process_pdfs(pdf_paths)

    df_chunk_entities = extract_entities(all_chunks, tokenizer, model)

    G = build_knowledge_graph(df_chunk_entities)

    query = "What are the company's CO2 reduction goals?"
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    retrieved_texts = graph_rag_retrieve(query, G, embedding_model)

    response = generate_response(query, retrieved_texts, tokenizer, model)
    print("\n📝 Response:", response)

if __name__ == "__main__":
    main()