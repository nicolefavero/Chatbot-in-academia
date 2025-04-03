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
from openai import OpenAI  # Using OpenAI client

# Initialize OpenAI client
client = OpenAI(api_key="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")

# Load spaCy NLP model for sentence splitting
nlp = spacy.load("en_core_web_sm")

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
    query_embedding = get_openai_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=doc_filter
    )

        # ------------------------------------------------------------------------
    # DEBUG PRINTS: Check how many docs we got back and print out a snippet
    # ------------------------------------------------------------------------
    # 'results["metadatas"]' is a list of lists: first dimension = query count,
    # second dimension = top_k results.
    retrieved_docs = results["metadatas"][0]  # the top_k metadata for our single query
    unique_doc_names = set(doc_metadata["doc_name"] for doc_metadata in retrieved_docs)
    used_references = []
    for doc_name in unique_doc_names:
        if doc_name in DOC_REGISTRY and "full_reference" in DOC_REGISTRY[doc_name]:
            used_references.append(DOC_REGISTRY[doc_name]["full_reference"])
    print("DEBUG: Number of retrieved docs:", len(retrieved_docs))
    
    # If this is 0, that means we got no documents back from the collection query.
    if len(retrieved_docs) == 0:
        print("DEBUG: No documents found for this query. The context is empty.")
    
    # Loop over each returned doc and show a bit of text
    for i, doc_metadata in enumerate(retrieved_docs):
        print(f"DEBUG: Doc #{i}")
        print("  doc_name:", doc_metadata.get("doc_name"))
        text_snippet = doc_metadata.get("text", "")[:150]  # first 150 chars
        print("  text snippet:", text_snippet, "...")
    # ------------------------------------------------------------------------


    context_str = "\n".join([doc['text'] for doc in results["metadatas"][0]])
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

    # If the user wants references at the end, build a references section
    if used_references:
        references_section = "\n\nCBS papers consulted:\n"
        for ref in used_references:
            references_section += f"- {ref}\n"
        # Append to the original text
        response_text += references_section

    return response_text

###############################################################################
# 8. Gradio UI
###############################################################################

def gradio_interface(message, chat_history):
    folder_path = "papers-testing"
    embedding_model = "text-embedding-ada-002"
    collection = get_or_create_embedding_collection(folder_path, embedding_model)
    all_chunks = process_all_pdfs(folder_path)
    bm25 = create_bm25_index(all_chunks)

    response = generate_response(message, collection, bm25, all_chunks)
    return response

chatbot = gr.ChatInterface(
    fn=gradio_interface,
    title="🎙️ CBS-bot (Q&A Bot)",
    theme="messages",
    chatbot=gr.Chatbot(show_copy_button=True)
)

chatbot.launch()
