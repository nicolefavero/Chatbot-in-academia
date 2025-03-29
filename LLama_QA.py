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
from collections import Counter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from DOC_REGISTRY import DOC_REGISTRY

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

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        token=HF_TOKEN,
        torch_dtype=dtype,
        device_map="auto"
    )
    
    return tokenizer, model, device


###############################################################################
# 2. Preprocessing & Chunking Text from PDFs (NEW CODE)
###############################################################################

def preprocess_text(text: str) -> str:
    """
    Clean whitespace and normalize text.
    """
    text = text.strip()
    # Remove inline citations like [1], (Doe et al., 2020)
    text = re.sub(r"\[\d+\]|\(\w+ et al\., \d+\)", "", text)

    # Remove figure references like (see Fig. 3)
    text = re.sub(r"\(see Fig\.\s?\d+\)", "", text)

    # Remove special characters (common in PDF noise)
    text = re.sub(r"[*_#]", "", text)

    # Fix awkward line breaks in PDFs (like "word-\nnextword")
    text = re.sub(r"-\n", "", text)

    # Normalize spacing
    text = " ".join(text.split())

    return text

def split_into_sentences(text: str):
    """
    Use spaCy to split text into sentences before chunking.
    """
    doc = nlp(text)
    # only keep non-empty sentences
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def filter_repeated_sentences(sentences, min_length=30, max_repetitions=3):
    """
    Filter out short sentences that repeat too many times (likely boilerplate).
    - min_length: only consider filtering lines shorter than this length.
    - max_repetitions: if a short line appears more than this many times, exclude it.
    """
    counter = Counter(sentences)
    filtered = []
    for s in sentences:
        # Exclude repeated short lines
        if len(s) < min_length and counter[s] > max_repetitions:
            continue
        filtered.append(s)
    return filtered

def process_pdf_for_rag(pdf_path: str, metadata_pages=2):
    """
    Reads a PDF, separates metadata pages, extracts text from main pages,
    splits into sentences, filters repeated lines, then chunks them using
    RecursiveCharacterTextSplitter.

    Returns two lists:
      - metadata_chunks: list of raw text strings from metadata
      - main_content_chunks: list of dicts with:
            {"content": <chunk_text>, "metadata": { ... }}
    """
    document = fitz.Document(pdf_path)
    pages = document.page_count

    # Collect metadata text from first 'metadata_pages' pages
    metadata_text = []
    for page_num in range(min(metadata_pages, pages)):
        page = document.load_page(page_num)
        raw_text = page.get_text("text")
        raw_text = preprocess_text(raw_text)
        metadata_text.append(raw_text)
    full_metadata_text = "\n".join(metadata_text)

    # Split metadata into sentences (often short anyway)
    meta_sentences = split_into_sentences(full_metadata_text)

    # Collect main content from the rest
    main_sentences_with_page = []
    for page_num in range(metadata_pages, pages):
        page = document.load_page(page_num)
        raw_text = preprocess_text(page.get_text("text"))
        sentences = split_into_sentences(raw_text)
        for s in sentences:
            main_sentences_with_page.append((s, page_num))

    # Filter out repeated short lines
    all_main_sentences = [s for s, _ in main_sentences_with_page]
    filtered_main_sentences = filter_repeated_sentences(all_main_sentences)

    filtered_main_sentences_with_page = []
    filtered_set = set(filtered_main_sentences)
    for s, page_num in main_sentences_with_page:
        if s in filtered_set:
            filtered_main_sentences_with_page.append((s, page_num))

    # Chunk metadata
    if meta_sentences:
        meta_text = "\n".join(meta_sentences)
        metadata_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            length_function=len
        )
        metadata_docs = metadata_splitter.create_documents([meta_text])
        metadata_chunks = [doc.page_content for doc in metadata_docs]
    else:
        metadata_chunks = []

    # Chunk main content
    main_text = "\n".join(s for s, _ in filtered_main_sentences_with_page)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len
    )
    chunk_docs = text_splitter.create_documents([main_text])

    # Build structured output for main content
    main_content_chunks = []
    for doc in chunk_docs:
        chunk_dict = {
            "content": doc.page_content,
            "metadata": {
                "source_file": os.path.basename(pdf_path),
                # Could track page ranges, etc.
            }
        }
        main_content_chunks.append(chunk_dict)

    return metadata_chunks, main_content_chunks

def process_pdf_and_combine(pdf_path, metadata_pages=2):
    """
    Wrapper that calls process_pdf_for_rag, then merges metadata and main content
    into a single list of dicts with "text" and "doc_name".
    """
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]

    metadata_chunks, main_content_chunks = process_pdf_for_rag(pdf_path, metadata_pages=metadata_pages)

    combined_chunks = []

    # (A) Store metadata chunks if you want them in your DB:
    for meta_chunk in metadata_chunks:
        combined_chunks.append({
            "text": meta_chunk,
            "doc_name": f"{doc_name}_metadata"
        })

    # (B) Store main content chunks
    for chunk_dict in main_content_chunks:
        combined_chunks.append({
            "text": chunk_dict["content"],
            "doc_name": doc_name
        })

    return combined_chunks

def process_all_pdfs(folder_path):
    """
    Processes all PDFs in the folder using the new approach (metadata vs main content).
    Returns a list of dicts: {"text": <chunk_text>, "doc_name": <filename_without_pdf>}.
    """
    all_chunks = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            pdf_chunks = process_pdf_and_combine(pdf_path, metadata_pages=2)
            all_chunks.extend(pdf_chunks)

    return all_chunks

def detect_target_doc(query: str, registry: dict, threshold=90):
    """
    Use fuzzy matching on all doc aliases in registry to find the best match.
    Returns the doc_name (key in registry) if the highest match is above threshold,
    else returns None.
    """
    query_lower = query.lower()
    best_doc = None
    best_score = 0

    for doc_name, data in registry.items():
        aliases = data["aliases"]
        for alias in aliases:
            # Try partial_ratio or token_sort_ratio etc.
            score = fuzz.partial_ratio(query_lower, alias.lower())
            if score > best_score:
                best_score = score
                best_doc = doc_name

    if best_score >= threshold:
        return best_doc
    else:
        return None


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
    collection = client.get_or_create_collection(
        name="academic_papers",
        metadata={"hnsw:space": "cosine", "hnsw:M": 32}
    )

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

def retrieve_documents_bm25(query, bm25, all_chunks, top_k=5, doc_filter=None):
    """
    Retrieve top_k chunks using BM25 ranking while reducing the score of chunks
    that contain too many words from the document title or aliases.
    """
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)

    # Extract document title/aliases if filtering by a specific document
    title_aliases = []
    if doc_filter and doc_filter.get("doc_name") in DOC_REGISTRY:
        title_aliases = DOC_REGISTRY[doc_filter["doc_name"]]["aliases"]
        title_aliases = [alias.lower().split() for alias in title_aliases]  # Split into words

    for i, chunk in enumerate(all_chunks):
        text = chunk["text"].lower()
        chunk_doc_name = chunk["doc_name"]

        # If filtering by a specific document, apply alias-based score penalty
        if doc_filter and chunk_doc_name == doc_filter.get("doc_name"):
            alias_match_count = sum(
                sum(1 for word in alias if word in text) / len(alias)  # Compute % of words matched
                for alias in title_aliases
            )

            if alias_match_count > 0.5:  # If >50% of alias words match, reduce score
                scores[i] *= 0.5  # Reduce relevance by 50%

    ranked_results = sorted(zip(range(len(scores)), scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked_results[:top_k]:
        chunk_doc_name = all_chunks[idx]["doc_name"]

        if doc_filter and chunk_doc_name != doc_filter.get("doc_name"):
            continue  # Skip non-matching docs

        results.append({
            "text": all_chunks[idx]["text"],
            "doc_name": chunk_doc_name,
            "score": score
        })

    return results


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
def hybrid_retrieve(query, bm25, all_chunks, embedding_model, collection, top_k=5, doc_filter=None):
    """
    1) Refine query with NER
    2) Retrieve top_k with BM25 (filtered if doc_filter is set)
    3) Retrieve top_k with embeddings (also filtered by doc_filter)
    4) Merge results
    """
    refined_q = refine_query_with_ner(query)

    # BM25 retrieval (now respects doc_filter)
    bm25_results = retrieve_documents_bm25(refined_q, bm25, all_chunks, top_k=top_k, doc_filter=doc_filter)

    # Embedding-based retrieval (with doc_filter in Chroma)
    query_embedding = embedding_model.encode(refined_q).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=doc_filter
    )
    # Debug prints for embedding retrieval
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

    # Merge BM25 + embedding results
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
                "score": 0,  # no combined scoring here
                "retrieval_method": "Embedding"
            })
            seen_texts.add(emb_r["text"])

    return combined[:top_k]


###############################################################################
# 4. Retrieve Documents (Embedding-only)
###############################################################################
def retrieve_documents(query, embedding_model, collection, n_results=5, doc_filter=None):
    """
    Retrieve relevant chunks for query using embeddings only.
    If doc_filter is not None, e.g. {"doc_name": "..."},
    we restrict the search to that doc.
    """
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=doc_filter,
        ef=300
    )
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
    """

        # (A) Detect if user refers to a specific PDF from the registry
    matched_doc_name = detect_target_doc(query, DOC_REGISTRY, threshold=90)
    if matched_doc_name:
        doc_filter = {"doc_name": matched_doc_name}
        print(f"DEBUG: Matched doc => {matched_doc_name} (score above threshold)")
    else:
        doc_filter = None

    # 1) Refine query with NER
    refined_q = refine_query_with_ner(query)

    if bm25 is not None and all_chunks is not None:
        # 2) Hybrid retrieval
        retrieved_docs = hybrid_retrieve(refined_q, bm25, all_chunks, embedding_model, collection, top_k=5, doc_filter=doc_filter)
    else:
        # 2) Embedding-only retrieval
        results = retrieve_documents(refined_q, embedding_model, collection, n_results=5, doc_filter=doc_filter)

        # Debug prints for embedding retrieval
        print("\nDEBUG - Embedding-based top-5 results (no BM25):")
        for i, meta in enumerate(results["metadatas"][0]):
            dist = results["distances"][0][i]
            doc_name = meta["doc_name"]
            snippet = meta["text"][:150].replace("\n", " ")
            print(f"{i+1}) distance={dist:.4f} | doc={doc_name} | snippet={snippet}...")

        # Optional threshold logic
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        threshold = 0.3
        retrieved_docs = []
        for dist, doc_meta in zip(distances, metadatas):
            if dist < threshold:
                retrieved_docs.append(doc_meta)

    # 3) Fallback if no docs found
    if not retrieved_docs:
        return "I don’t have information in my database to answer this question."

    # 4) Build context from retrieved docs
    doc_names_used = set()
    context_str = ""
    for doc in retrieved_docs:
        doc_name = doc.get("doc_name", "UnknownDoc")
        chunk_text = doc["text"]
        doc_names_used.add(doc_name)
        context_str += f"[{doc_name}]\n{chunk_text}\n\n"

    all_sources_str = ", ".join(sorted(doc_names_used))

    system_instruction = f"""\
You are an expert academic research assistant specializing in supporting professors in business school research. 
Your expertise lies in analyzing and synthesizing complex information from academic papers, particularly in areas like 
management, strategy, finance, marketing, and organizational behavior.

Your role is to provide insightful, clear, and well-structured explanations, within the token limit, that align with the expectations of business academia. 
Professors may ask questions that require drawing insights from multiple sources, connecting key ideas, and interpreting nuanced academic content.

### Response Guidelines:
1. **Synthesize Information Thoughtfully:**
   - Combine insights from multiple retrieved passages to construct a coherent and structured answer.
   - Identify key themes, underlying frameworks, and implications relevant to the query.
   - End your response with: "**If you have any more questions, I'm here to help**"  

2. **Use an Academic Yet Business-Oriented Tone:**
   - Frame responses with clarity, precision, and relevance to business research.
   - Where appropriate, connect insights to practical business applications, theoretical frameworks, or real-world implications.
   - Use concise language while ensuring key terms, frameworks, or methodologies are well-defined.

4. **Rephrase and Summarize Naturally:**
   - DO NOT copy exact sentences; instead, reformulate the retrieved information in your own words.
   - Focus on delivering insights without excessive detail or redundancy.
   - Summarize what’s available in a **concise conclusion**.

5. **Clarify Ambiguities or Missing Information:**
   - If the retrieved content lacks a direct answer, attempt to infer insights from related information.
   - If no reasonable inference can be made, respond: 
     "I don’t have information in my database to answer this question."

6. **No Unverified Knowledge:**
   - DO NOT add outside knowledge unless explicitly provided in the retrieved context.

### Key Principles for Excellence:
✅ Combine insights from multiple chunks when needed.
✅ Align responses with a business school mindset, including frameworks, models, and strategic insights.
✅ Maintain a clear, academic tone that resonates with university professors.
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
            max_new_tokens=700,
            do_sample=False,
            stopping_criteria=stopping_criteria
        )

    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean up prompt text from output
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

    # 1) Load model & tokenizer
    tokenizer, llama_model, device = load_llama_model()

    # 2) Load embedding model
    embedding_model = load_embedding_model()

    # Test identical text distance (debug)
    test_text = "some example text"
    embA = embedding_model.encode(test_text)
    embB = embedding_model.encode(test_text)
    dist = np.linalg.norm(embA - embB)
    print(f"DEBUG - Distance for identical text: {dist}")

    # 3) Get or create ChromaDB collection
    collection = get_or_create_embedding_collection(folder_path, embedding_model)

    # 4) Build the BM25 index
    print("Building BM25 index...")
    all_chunks = process_all_pdfs(folder_path)
    bm25 = create_bm25_index(all_chunks)

    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        response = generate_response(
            query,
            tokenizer,
            llama_model,
            embedding_model,
            collection,
            device,
            bm25=bm25,
            all_chunks=all_chunks
        )
        print(f"\nBot: {response}")
