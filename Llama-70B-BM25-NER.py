################################################################################
#  CHATBOT WITH LLAMA 3 - 70B + RAG RETRIEVAL (ACADEMIC)  #
################################################################################

import torch
import fitz  # PyMuPDF for PDF processing
import re
import os
import chromadb
import spacy
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Load spaCy NLP model for sentence splitting + NER
nlp = spacy.load("en_core_web_sm")

###############################################################################
# 1. Load Llama 3 - 70B Model
###############################################################################
def load_llama_model():
    """
    Load Llama 3 - 70B model from Hugging Face and distribute across 4 GPUs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model on {device} (4 GPUs)...")
    HF_TOKEN = "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"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-70B",
        token=HF_TOKEN,
        torch_dtype=dtype,
        device_map="auto"
    )
    
    return tokenizer, model, device


###############################################################################
# 2. Preprocessing & Chunking Text from PDFs
###############################################################################
def preprocess_text(text):
    """
    Clean whitespace and normalize text.
    """
    text = text.strip()
    text = " ".join(text.split())
    return text

def split_into_sentences(text):
    """
    Use spaCy to split text into sentences before chunking.
    """
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def process_pdf_with_sentences(pdf_path):
    """
    Reads a PDF, extracts text, splits into sentences, then chunk them.
    Returns a list of chunk strings.
    """
    document = fitz.Document(pdf_path)
    pages = document.page_count
    all_sentences = []

    for page_num in range(pages):
        page = document.load_page(page_num)
        text = page.get_text("text")
        text = preprocess_text(text)
        sentences = split_into_sentences(text)
        all_sentences.extend(sentences)

    # Use RecursiveCharacterTextSplitter to chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len
    )
    chunk_docs = text_splitter.create_documents(all_sentences)
    chunks = [doc.page_content for doc in chunk_docs]
    return chunks

def process_all_pdfs(folder_path):
    """
    Processes all PDFs in folder, returning a list of dicts: 
    {"text": <chunk_text>, "doc_name": <filename_without_pdf>}.
    """
    all_chunks = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            doc_name = os.path.splitext(filename)[0]  # remove .pdf extension

            chunks = process_pdf_with_sentences(pdf_path)
            for chunk_text in chunks:
                all_chunks.append({
                    "text": chunk_text,
                    "doc_name": doc_name
                })
    return all_chunks


###############################################################################
# 3. Embeddings & ChromaDB
###############################################################################
def load_embedding_model():
    """
    Load MSMARCO BERT model for embedding
    """
    return SentenceTransformer("msmarco-bert-base-dot-v5")

def get_or_create_embedding_collection(folder_path, embedding_model):
    """
    Load existing ChromaDB collection if available, otherwise create embeddings.
    """
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(name="academic_papers", metadata={"hnsw:space": "cosine"})

    # If it already has vectors, skip creation
    if collection.count() > 0:
        print("Loaded existing ChromaDB collection. Skipping embedding generation.")
        return collection

    print("Processing PDFs and generating embeddings for the first time...")
    all_chunks = process_all_pdfs(folder_path)

    for i, chunk_data in enumerate(all_chunks):
        chunk_text = chunk_data["text"]
        embedding = embedding_model.encode(chunk_text).tolist()

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            metadatas=[{
                "text": chunk_text,
                "doc_name": chunk_data["doc_name"]
            }]
        )

    return collection


###############################################################################
# 3b. BM25 Index Creation
###############################################################################

def create_bm25_index(chunks):
    """
    Build BM25Okapi index from chunked text.
    """
    tokenized_corpus = []
    for chunk_data in chunks:
        tokens = chunk_data["text"].split()  # basic whitespace splitting
        tokenized_corpus.append(tokens)
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def retrieve_documents_bm25(query, bm25, all_chunks, top_k=5):
    """
    Retrieve top_k chunks using BM25 ranking.
    """
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    ranked_results = sorted(
        zip(range(len(scores)), scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_results = []
    for idx, score in ranked_results[:top_k]:
        top_results.append({
            "text": all_chunks[idx]["text"],
            "doc_name": all_chunks[idx]["doc_name"],
            "score": score
        })
    return top_results

###############################################################################
# 3c. NER-Based Query Refinement
###############################################################################

def extract_entities(query):
    """
    Use spaCy NER on the query to extract named entities.
    """
    doc = nlp(query)
    return [ent.text for ent in doc.ents]

def refine_query_with_ner(query):
    """
    Simple approach: append recognized entity strings to the original query
    to boost their importance in BM25/embedding search.
    """
    entities = extract_entities(query)
    if not entities:
        return query
    refined_query = query + " " + " ".join(entities)
    return refined_query

###############################################################################
# 3d. Hybrid Retrieval (BM25 + Embeddings)
###############################################################################

def hybrid_retrieve(query, bm25, all_chunks, embedding_model, collection, top_k=5):
    """
    1) Refine the query with NER
    2) Retrieve top_k with BM25
    3) Retrieve top_k with embeddings
    4) Merge results (unique by 'text')
    5) Print debug for embedding results (distances, doc name, snippet)
    """
    # Always do query refinement with NER
    refined_q = refine_query_with_ner(query)

    # BM25 retrieval
    bm25_results = retrieve_documents_bm25(refined_q, bm25, all_chunks, top_k=top_k)

    # Embedding-based retrieval
    query_embedding = embedding_model.encode(refined_q).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # --- Debug prints for embedding retrieval ---
    print("\nDEBUG - Embedding-based top-{} results (hybrid path):".format(top_k))
    for i, meta in enumerate(results["metadatas"][0]):
        dist = results["distances"][0][i]
        doc_name = meta["doc_name"]
        snippet = meta["text"][:500].replace("\n", " ")
        print(f"{i+1}) distance={dist:.4f} | doc={doc_name} | snippet={snippet}...")

    embedding_results = []
    for meta in results["metadatas"][0]:
        embedding_results.append({
            "text": meta["text"],
            "doc_name": meta["doc_name"],
        })

    # Merge results
    combined = []
    seen_texts = set()
    
    # Add BM25 results
    for r in bm25_results:
        if r["text"] not in seen_texts:
            combined.append({
                "text": r["text"],
                "doc_name": r["doc_name"],
                "score": r["score"],
                "retrieval_method": "BM25"
            })
            seen_texts.add(r["text"])
    
    # Add embedding results
    for emb_r in embedding_results:
        if emb_r["text"] not in seen_texts:
            combined.append({
                "text": emb_r["text"],
                "doc_name": emb_r["doc_name"],
                "score": 0,  # not merging numeric scores here
                "retrieval_method": "Embedding"
            })
            seen_texts.add(emb_r["text"])

    return combined[:top_k]

# --- NEW CODE END ---

###############################################################################
# 4. Retrieve Documents (Embedding-only)
###############################################################################
def retrieve_documents(query, embedding_model, collection, n_results=5):
    """
    Retrieve relevant chunk texts for the given query using embeddings only.
    NOTE: We won't do thresholding here; we'll let generate_response handle that.
    """
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    # Return both the metadata and the distances, so we can debug elsewhere
    return results


###############################################################################
# 5. Stopping Criteria
###############################################################################
class StopOnTokens(StoppingCriteria):
    """
    If model tries to generate certain tokens, stop.
    """
    def __init__(self, stop_strings, tokenizer):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

    def __call__(self, input_ids, scores):
        for stop_id_seq in self.stop_ids:
            if len(input_ids[0]) >= len(stop_id_seq):
                if list(input_ids[0][-len(stop_id_seq):]) == stop_id_seq:
                    return True
        return False


###############################################################################
# 6. Generate Response - Cite Multiple PDFs if Used
###############################################################################
def generate_response(query, tokenizer, model, embedding_model, collection, device,
                      bm25=None, all_chunks=None):
    """
    If BM25 is provided, do hybrid retrieval; else, embedding-only.
    Also print debug info about embedding retrieval results (distances, doc, snippet).
    """
    # --- 1) Always refine the user query with NER ---
    refined_q = refine_query_with_ner(query)

    if bm25 is not None and all_chunks is not None:
        # --- 2) Hybrid retrieval ---
        retrieved_docs = hybrid_retrieve(refined_q, bm25, all_chunks, embedding_model, collection, top_k=5)
    else:
        # --- 2) Embedding-only retrieval ---
        results = retrieve_documents(refined_q, embedding_model, collection, n_results=5)

        # Debug prints: Show top 5 embedding results
        print("\nDEBUG - Embedding-based top-5 results (no BM25):")
        for i, meta in enumerate(results["metadatas"][0]):
            dist = results["distances"][0][i]
            doc_name = meta["doc_name"]
            snippet = meta["text"][:150].replace("\n", " ")
            print(f"{i+1}) distance={dist:.4f} | doc={doc_name} | snippet={snippet}...")

        # --- 3) Thresholding logic (user's existing code) ---
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        threshold = 0.3
        retrieved_docs = []
        for dist, doc_meta in zip(distances, metadatas):
            if dist < threshold:
                retrieved_docs.append(doc_meta)

    # --- 4) If no relevant docs, fallback
    if not retrieved_docs:
        return "I don’t have information in my database to answer this question."

    # Build context from retrieved docs
    doc_names_used = set()
    context_str = ""
    for doc in retrieved_docs:
        doc_name = doc.get("doc_name", "UnknownDoc")
        chunk_text = doc["text"]
        doc_names_used.add(doc_name)
        context_str += f"[{doc_name}]\n{chunk_text}\n\n"

    # Sort doc names
    all_sources_str = ", ".join(sorted(doc_names_used))
    system_instruction = f"""\
You are an expert academic research assistant, who must help university professors prepare materials for their lessons. 
You must provide **clear and structured expository answers** using the retrieved context.

Follow these rules:
1. Use ONLY the text in the 'Context' below to answer the question.
3. DO NOT copy exact sentences; **rephrase and explain naturally**.
4. Summarize the retrieved information **without redundancy**.
5. If the answer is not in the context, respond: "I don’t have information in my database to answer this question."
6. Do NOT add outside knowledge.
"""

    prompt = f"""
{system_instruction}

Context:
{context_str}

Question: {query}

Answer:
"""

    stop_strings = ["\nQuestion:", "\nYou:"]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_strings, tokenizer)])

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            stopping_criteria=stopping_criteria
        )

    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Answer:" in raw_output:
        raw_output = raw_output.split("Answer:", 1)[-1].strip()

    for delimiter in ["Question:", "Context:", "You:"]:
        if delimiter in raw_output:
            raw_output = raw_output.split(delimiter)[0].strip()

    fallback_phrase = "I don’t have information in my database to answer this question."
    if fallback_phrase in raw_output:
        raw_output = fallback_phrase

    if not raw_output:
        return fallback_phrase

    if fallback_phrase not in raw_output:
        raw_output += f"\nSource: {all_sources_str}"


    return raw_output


###############################################################################
# 7. Main
###############################################################################
if __name__ == "__main__":
    folder_path = "papers-testing"  # Path to your PDFs

    tokenizer, llama_model, device = load_llama_model()
    embedding_model = load_embedding_model()

    # Test the distance for identical text:
    test_text = "some example text"
    embA = embedding_model.encode(test_text)
    embB = embedding_model.encode(test_text)
    dist = np.linalg.norm(embA - embB)
    print(f"DEBUG - Distance for identical text: {dist}")

    collection = get_or_create_embedding_collection(folder_path, embedding_model)

    # --- NEW CODE START ---
    # Build the BM25 index using the same chunk data
    print("Building BM25 index...")
    all_chunks = process_all_pdfs(folder_path)
    bm25 = create_bm25_index(all_chunks)
    # --- NEW CODE END ---

    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        response = generate_response(
            query, tokenizer, llama_model, embedding_model, collection, device,
            bm25=bm25, all_chunks=all_chunks
        )
        print(f"\nBot: {response}")

