################################################################################
#  CHATBOT WITH LLAMA 3 - 70B + RAG RETRIEVAL (ACADEMIC) - MULTI-PDF CITATIONS #
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

# Load spaCy NLP model for sentence splitting
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
# 4. Retrieve Documents
###############################################################################
def retrieve_documents(query, embedding_model, collection, n_results=5):
    """
    Retrieve relevant chunk texts for the given query.
    """
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    retrieved_docs = []
    seen_texts = set()

    # Each doc is a chunk
    for doc in results["metadatas"][0]:
        text = doc["text"]
        if text not in seen_texts:
            retrieved_docs.append(doc)
            seen_texts.add(text)
    return retrieved_docs


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
def generate_response(query, tokenizer, model, embedding_model, collection, device):
    # === 1) Run the raw query ===
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    # The “distances” is a 2D list of floats, one for each result in each query
    # e.g. distances[0] = [0.12, 0.15, 0.23, ...]  (depending on your settings)
    # Check how your Chroma is configured—some use “cosine similarity” (larger = more similar),
    # others use “distance” (smaller = more similar). Adjust threshold logic accordingly.
    distances = results["distances"][0]  # first query’s distances
    metadatas = results["metadatas"][0]  # chunk metadata
    print("DEBUG - Distances returned by Chroma for this query:", distances)


    # === 2) Filter out any chunk whose distance is “too large” or similarity is “too low.” 
    #     Suppose your DB is using “cosine similarity,” i.e. 1.0 is a perfect match, 0.0 is no match.
    #     In that case, you might do if distance >= 0.3 => skip
    #     If it’s “Euclidean distance,” then smaller is better, so you do the inverse.
    #     Below is a typical scenario if your Chroma store returns distances = 1 - cosine_similarity:
    #     ( so smaller = more similar ), you want distance < 0.3 or so:

    relevant_docs = []
    threshold = 0.3  # pick a threshold that fits your embedding scale
    for dist, doc_meta in zip(distances, metadatas):
        if dist < threshold:
            # Only keep docs with distance < threshold
            relevant_docs.append(doc_meta)

    # If we have no relevant docs, fallback:
    if not relevant_docs:
        return "I don’t have information in my database to answer this question."

    # === 3) Build the “context_str” from these filtered docs ===
    doc_names_used = set()
    context_str = ""
    for doc in relevant_docs:
        doc_name = doc.get("doc_name", "UnknownDoc")
        chunk_text = doc["text"]
        doc_names_used.add(doc_name)
        context_str += f"[{doc_name}]\n{chunk_text}\n\n"

    # Sort doc names for consistent output
    all_sources_str = ", ".join(sorted(doc_names_used))

    # === 4) Construct the prompt with the system instruction
    system_instruction = f"""\
You are an academic research assistant.
Follow these rules:
1. Use ONLY the text in the 'Context' below to answer the question.
2. You may use text from multiple documents if needed.
3. If the answer is not in the context, respond: "I don’t have information in my database to answer this question."
4. At the end of your answer, write: "Source: {all_sources_str}" if relevant text is found.
5. Do NOT add outside knowledge.
6. Keep your answer concise.
"""

    prompt = f"""
{system_instruction}

Context:
{context_str}

Question: {query}

Answer:
"""

    # === 5) Generate the text with your stopping criteria
    stop_strings = ["\nQuestion:", "\nYou:"]
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_strings, tokenizer)])

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            stopping_criteria=stopping_criteria
        )

    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # === 6) Post-processing the final text
    if "Answer:" in raw_output:
        raw_output = raw_output.split("Answer:", 1)[-1].strip()

    for delimiter in ["Question:", "Context:", "You:"]:
        if delimiter in raw_output:
            raw_output = raw_output.split(delimiter)[0].strip()

    # === 7) If the model fell back to “I don’t have information...,” remove any leftover source
    fallback_phrase = "I don’t have information in my database to answer this question."
    if fallback_phrase in raw_output:
        # If the fallback phrase appears, we remove everything after it 
        # (or just assign raw_output = fallback_phrase).
        raw_output = fallback_phrase

    if not raw_output:
        return fallback_phrase

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

    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        response = generate_response(
            query, tokenizer, llama_model, embedding_model, collection, device
        )
        print(f"\nBot: {response}")
