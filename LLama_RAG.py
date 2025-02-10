################################################################################
#  CHATBOT WITH LLAMA MODEL + RAG RETRIEVAL (ACADEMIC)  #
################################################################################

'''Reason: Use an internal chatbot based on the Llama model instead of ChatGPT 
or any other third-party model for privacy reasons. This ensures that we don't 
have to send data to external servers and retain full control over the data.'''

import torch
import fitz  # PyMuPDF for PDF processing
import re
import os
import chromadb
import spacy
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load spaCy NLP model for sentence splitting
nlp = spacy.load("en_core_web_sm")

# --------------------------------------------------------------------------
# 1. Load Llama 7B Model (For Response Generation)
# --------------------------------------------------------------------------

def load_llama_model():
    '''Load Llama 7B model from Hugging Face and move it to GPU if available.'''
    
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    dtype = torch.float16 if device == "cuda" else torch.float32  # Use float16 on GPU

    print(f"Loading model on {device}...")

    HF_TOKEN = "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"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token=HF_TOKEN,
        torch_dtype=dtype
    ).to(device)

    return tokenizer, model, device

# --------------------------------------------------------------------------
# 2. Preprocessing & Chunking Text from PDFs
# --------------------------------------------------------------------------

def preprocess_text(text):
    '''Preprocess text: clean whitespace and normalize text.'''
    text = text.strip()
    text = " ".join(text.split())  # Normalize spaces
    return text

def split_into_sentences(text):
    '''Use spaCy to split text into sentences before chunking.'''
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def process_pdf_with_sentences(pdf_path):
    '''Reads PDF, extracts text, splits it into sentences, 
    and chunks properly to avoid splitting in the middle of sentences.'''
    document = fitz.Document(pdf_path)
    pages = document.page_count
    all_sentences = []

    for page_num in range(pages):
        page = document.load_page(page_num)
        text = page.get_text("text")  
        text = preprocess_text(text)  # Normalize text
        sentences = split_into_sentences(text)  # Sentence splitting
        all_sentences.extend(sentences)

    # Now chunk sentences using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    text_chunks = text_splitter.create_documents(all_sentences)  # Chunk sentence-wise
    return [chunk.page_content for chunk in text_chunks]  # Return only the text content

def process_all_pdfs(folder_path):
    '''Processes all PDFs in a folder and returns all chunks.'''
    all_chunks = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):  # Only process PDFs
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            chunks = process_pdf_with_sentences(pdf_path)
            all_chunks.extend(chunks)
    
    return all_chunks

# --------------------------------------------------------------------------
# 3. Load MSMARCO BERT for Embeddings & Store in ChromaDB
# --------------------------------------------------------------------------

def load_embedding_model():
    '''Load MSMARCO BERT model for embedding'''
    return SentenceTransformer("msmarco-bert-base-dot-v5")

def store_cleaned_embeddings(folder_path, embedding_model):
    '''Processes all PDFs, extracts clean text & stores embeddings in ChromaDB.'''
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(name="academic_papers")

    chunks = process_all_pdfs(folder_path)

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[str(i)], 
            embeddings=[embedding], 
            metadatas=[{"text": chunk}]
        )
    return collection

# --------------------------------------------------------------------------
# 4. Retrieve Documents (Multi-PDF Query Handling)
# --------------------------------------------------------------------------

def retrieve_documents(query, embedding_model, collection, n_results=5):
    '''Retrieve relevant papers using embedding search and return different sources.'''
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    retrieved_docs = []
    seen_texts = set()  # Avoid duplicate chunks

    for doc in results["metadatas"][0]:
        if doc["text"] not in seen_texts:
            retrieved_docs.append(doc)
            seen_texts.add(doc["text"])  # Prevent duplicate chunks
    
    return retrieved_docs

# --------------------------------------------------------------------------
# 5. Generate Response using Llama (Restrict Answers + GPU Optimization)
# --------------------------------------------------------------------------

def generate_response(query, tokenizer, model, embedding_model, collection, device):
    '''Generate a response using retrieved documents and Llama, but only answer based on found papers.'''
    retrieved_docs = retrieve_documents(query, embedding_model, collection)

    # If no relevant documents, return a fallback message
    if not retrieved_docs:
        return "Sorry, I can't find this information in my database, but I might help with another academic topic."

    # Extract relevant text
    context = " ".join([doc["text"] for doc in retrieved_docs])

    # Modify prompt to ensure answer stays within found info
    prompt = f"""
    You are an academic research assistant. Answer the question **strictly** using the provided academic papers.
    **Do not make up any information.** If the information is not in the sources, say: "I cannot find this information in my database."

    Context from papers:
    {context}

    Question: {query}

    Answer:
    """

    # Generate response with no gradient tracking
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# --------------------------------------------------------------------------
# 6. Run Chatbot
# --------------------------------------------------------------------------

if __name__ == "__main__":
    folder_path = "papers-testing"  # Path to your folder containing PDFs

    # Load models
    tokenizer, llama_model, device = load_llama_model()
    embedding_model = load_embedding_model()

    # Process PDFs and store embeddings
    collection = store_cleaned_embeddings(folder_path, embedding_model)

    # Chat loop
    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        response = generate_response(query, tokenizer, llama_model, embedding_model, collection, device)
        print(f"\nBot: {response}")
