################################################################################
#  CHATBOT WITH OPENAI GPT-4o Mini + RAG RETRIEVAL (ACADEMIC) + Gradio UI
################################################################################

import os
import re
import fitz  # PyMuPDF for PDF processing
import chromadb
import spacy
import numpy as np
from collections import Counter
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
import gradio as gr
from DOC_REGISTRY import DOC_REGISTRY
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="Your-OpenAI-API-Key")

# Load spaCy NLP model for sentence splitting
nlp = spacy.load("en_core_web_sm")

###############################################################################
# 0. Intent Classification & Small-Talk Handling
###############################################################################

def is_small_talk(query: str) -> bool:
    """
    Quick rule-based check for greetings / small-talk.
    """
    q = query.lower()
    patterns = [
        r"\bhel+o+\b!*",       # matches hello, hellooo, hello!!, etc.
        r"\bhi+\b!*",          # matches hi, hiiii, hi!!
        r"\bhey+\b!*",         # matches hey, heyyy, hey!!!
        r"\bcan you help\b",
        r"\bhelp me\b",
        r"\bhow are you\b",
        r"\bwhat'?s up\b",
        r"\bthanks?\b",
        r"\bthank you\b"
    ]
    return any(re.search(p, q) for p in patterns)

def handle_small_talk(query: str) -> str:
    """
    Generate a friendly small-talk response using GPT-4o Mini.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful, engaging conversational Q&A assistant. Your scope is to help users answering questions about accademic papers. Don't answer other questions that are not related to your Q&A task."},
            {"role": "user",   "content": query}
        ],
        max_tokens=128,
        temperature=0.7
    )
    return resp.choices[0].message.content

###############################################################################
# 1. OpenAI Embedding Function
###############################################################################

def get_openai_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

###############################################################################
# 2. Text Preprocessing
###############################################################################

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\[\d+\]|\(\w+ et al\., \d+\)", "", text)
    text = re.sub(r"\(see Fig\.\s?\d+\)", "", text)
    text = re.sub(r"[*_#]", "", text)
    text = re.sub(r"-\n", "", text)
    text = " ".join(text.split())
    return text

###############################################################################
# 3. Document Detection & Retrieval
###############################################################################

def detect_target_doc(query: str, registry: dict, threshold=80):
    query_lower = query.lower()
    best_doc = None
    best_score = 0
    for doc_name, data in registry.items():
        for alias in data["aliases"]:
            score = fuzz.partial_ratio(query_lower, alias.lower())
            if score > best_score:
                best_score = score
                best_doc = doc_name
    return best_doc if best_score >= threshold else None

###############################################################################
# 4. PDF Processing
###############################################################################

def split_into_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def process_pdf_and_combine(pdf_path, metadata_pages=2):
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    document = fitz.Document(pdf_path)
    pages = document.page_count

    all_chunks = []
    for page_num in range(pages):
        page = document.load_page(page_num)
        raw_text = preprocess_text(page.get_text("text"))
        sentences = split_into_sentences(raw_text)
        chunk_text = " ".join(sentences)
        all_chunks.append({
            "text": chunk_text,
            "doc_name": doc_name
        })

    return all_chunks

def process_all_pdfs(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            all_chunks.extend(process_pdf_and_combine(pdf_path))
    return all_chunks

###############################################################################
# 5. ChromaDB Setup & Embedding Storage
###############################################################################

def get_or_create_embedding_collection(folder_path, embedding_model):
    client_db = chromadb.PersistentClient(path="db")
    collection = client_db.get_or_create_collection(
        name="academic_papers",
        metadata={"hnsw:space": "cosine", "hnsw:M": 32}
    )

    if collection.count() > 0:
        print("Loaded existing ChromaDB collection. Skipping embedding generation.")
        return collection

    all_chunks = process_all_pdfs(folder_path)

    for i, chunk_data in enumerate(all_chunks):
        chunk_text = chunk_data["text"]
        embedding = get_openai_embedding(chunk_text)

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
# 6. BM25 Index Creation
###############################################################################

def create_bm25_index(chunks):
    tokenized_corpus = [chunk["text"].split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def retrieve_documents_bm25(query, bm25, all_chunks, top_k=5):
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    ranked_results = sorted(zip(range(len(scores)), scores), key=lambda x: x[1], reverse=True)
    return [all_chunks[idx] for idx, _ in ranked_results[:top_k]]

###############################################################################
# 7. Query Generation with OpenAI GPT-4o Mini
###############################################################################

def generate_response(query, collection, bm25, all_chunks, top_k=5, doc_filter=None):
    # Embedding retrieval
    query_embedding = get_openai_embedding(query)
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=doc_filter
    )
    dense_chunks = dense_results["metadatas"][0]

    # BM25 retrieval
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    ranked_bm25 = sorted(zip(range(len(bm25_scores)), bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_chunks = []
    for idx, score in ranked_bm25[:top_k]:
        chunk = all_chunks[idx]
        chunk = dict(chunk)  # Copy to avoid modifying original
        chunk["bm25_score"] = score
        bm25_chunks.append(chunk)

    # Apply BM25 penalties for title match
    for chunk in bm25_chunks:
        text = chunk["text"].lower()
        doc_name = chunk.get("doc_name", "").lower()
        aliases = doc_name.replace("_", " ").split()

        alias_match_count = sum(1 for word in aliases if word in text) / max(1, len(aliases))
        if alias_match_count > 0.5:
            chunk["bm25_score"] *= 0.5

    # Merge BM25 and Dense results
    combined = []
    seen_texts = set()

    for chunk in bm25_chunks:
        if chunk["text"] not in seen_texts:
            combined.append({
                "text": chunk["text"],
                "doc_name": chunk["doc_name"],
                "retrieval_method": "BM25",
                "score": chunk["bm25_score"]
            })
            seen_texts.add(chunk["text"])

    for chunk in dense_chunks:
        if chunk["text"] not in seen_texts:
            combined.append({
                "text": chunk["text"],
                "doc_name": chunk.get("doc_name", "UnknownDoc"),
                "retrieval_method": "Dense",
                "score": 0
            })
            seen_texts.add(chunk["text"])

    # Build context string
    context_str = ""
    for chunk in combined:
        doc_name = chunk["doc_name"]
        context_str += f"[{doc_name}]\n{chunk['text']}\n\n"

    # Build prompt (full prompt preserved exactly as you wrote)
    prompt = (
        f"""You are an expert academic research assistant specializing in supporting professors in business school research. 
        Your expertise lies in analyzing and synthesizing complex information from academic papers, particularly in areas like 
        management, strategy, finance, marketing, and organizational behavior.

        Your role is to provide insightful, clear, and well-structured explanations, that align with the expectations of business academia. 
        Professors may ask questions that require drawing insights from multiple sources, connecting key ideas, and interpreting nuanced academic content.
        Where appropriate, connect insights to practical business applications, theoretical frameworks, or real-world implications.
        You must always add the references of the papers you used to answer the question.
        You are not allowed to make up references or add any outside knowledge.
        DO NOT copy exact sentences; instead, reformulate the retrieved information in your own words.
        DO NOT add outside knowledge unless explicitly provided in the retrieved context.

        End your response with: "**If you have any more questions, I'm here to help**"  
        If no reasonable inference can be made, respond: 
            "I don’t have information in my database to answer this question."

        ### Key Principles for Excellence:
        ✅ Combine insights from multiple chunks when needed.
        ✅ Align responses with a business school mindset, including frameworks, models, and strategic insights.
        ✅ Maintain a clear, academic tone that resonates with university professors.

        Use the following context to answer the query:

        Context: {context_str}

        Question: {query}

        Answer:"""
    )

    # Generate the response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful academic research assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0
    )
    response_text = response.choices[0].message.content

    return response_text

###############################################################################
# 8. Gradio UI
###############################################################################

def gradio_interface(message, chat_history):
    # First check: small talk or real research question?
    if is_small_talk(message):
        return handle_small_talk(message)

    # Otherwise, proceed with RAG pipeline
    folder_path = "papers-testing"
    embedding_model = "text-embedding-ada-002"
    collection = get_or_create_embedding_collection(folder_path, embedding_model)
    all_chunks = process_all_pdfs(folder_path)
    bm25 = create_bm25_index(all_chunks)

    return generate_response(message, collection, bm25, all_chunks)

chatbot = gr.ChatInterface(
    fn=gradio_interface,
    title="🎙️ CBS-bot (Q&A Bot)",
    theme="messages",
    chatbot=gr.Chatbot(show_copy_button=True)
)

chatbot.launch()
