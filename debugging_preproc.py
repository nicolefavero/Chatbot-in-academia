import fitz  # PyMuPDF for PDF processing
import os
import spacy
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load spaCy NLP model for sentence splitting
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    """
    Clean whitespace and normalize text.
    """
    text = text.strip()
    text = " ".join(text.split())
    return text

def split_into_sentences(text: str):
    """
    Use spaCy to split text into sentences before chunking.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def filter_repeated_sentences(sentences, min_length=30, max_repetitions=3):
    """
    Filter out short sentences that repeat too many times (likely boilerplate).
    - min_length: only consider filtering lines shorter than this length.
    - max_repetitions: if a short line appears more than this many times, exclude it.
    
    Returns a list of filtered sentences.
    """
    # Count occurrences
    counter = Counter(sentences)
    filtered = []
    for s in sentences:
        if len(s) < min_length and counter[s] > max_repetitions:
            # Skip repeated short lines
            continue
        filtered.append(s)
    return filtered

def process_pdf_for_rag(pdf_path: str, metadata_pages=2):
    """
    Reads a PDF, separates metadata pages, extracts text from main pages,
    splits into sentences, filters repeated lines, then chunk them with
    RecursiveCharacterTextSplitter. Returns two lists:
      - metadata_chunks: list of strings (from the first few pages)
      - main_content_chunks: list of dicts with "content" and "metadata"
    """
    document = fitz.Document(pdf_path)
    pages = document.page_count

    # 1) Extract metadata text from first N pages
    metadata_text = []
    for page_num in range(min(metadata_pages, pages)):
        page = document.load_page(page_num)
        metadata_text.append(preprocess_text(page.get_text("text")))
    metadata_text = "\n".join(metadata_text)

    # Split metadata into sentences (optional—often short anyway)
    metadata_sentences = split_into_sentences(metadata_text)

    # 2) Extract main content text from the rest of the pages
    main_sentences_with_page = []
    for page_num in range(metadata_pages, pages):
        page = document.load_page(page_num)
        raw_text = preprocess_text(page.get_text("text"))
        sentences = split_into_sentences(raw_text)
        for s in sentences:
            # Keep track of which page each sentence came from
            main_sentences_with_page.append((s, page_num))

    # Optionally filter out repeated lines (often headers/footers)
    # We only filter short lines for duplicates
    all_main_sentences = [s for s, _ in main_sentences_with_page]
    filtered_main_sentences = filter_repeated_sentences(all_main_sentences)

    # We'll lose the page reference for lines that are filtered out,
    # so let's do a second pass to keep only those sentences that survived:
    filtered_main_sentences_with_page = []
    original_counter = {sent: idx for idx, sent in enumerate(all_main_sentences)}
    
    filtered_set = set(filtered_main_sentences)
    for sent, page_num in main_sentences_with_page:
        if sent in filtered_set:
            filtered_main_sentences_with_page.append((sent, page_num))

    # 3) Chunk metadata separately if you want
    if metadata_sentences:
        # Simple approach: just treat the entire metadata as one chunk
        # or chunk it similarly using the text splitter
        meta_text = "\n".join(metadata_sentences)
        metadata_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            length_function=len
        )
        metadata_docs = metadata_splitter.create_documents([meta_text])
        metadata_chunks = [doc.page_content for doc in metadata_docs]
    else:
        metadata_chunks = []

    # 4) Use RecursiveCharacterTextSplitter on main content
    #    but we need to re-join sentences for the splitter to handle it properly
    #    We'll keep track of pages in metadata.
    main_text = "\n".join(s for s, _ in filtered_main_sentences_with_page)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len
    )
    chunk_docs = text_splitter.create_documents([main_text])

    # 5) Build a structured output with chunk metadata
    #    For example, we can store the PDF file and page ranges.
    main_content_chunks = []
    for doc in chunk_docs:
        # Each doc has doc.page_content
        # doc.metadata is empty by default with create_documents(list_of_texts)
        # You might consider custom splitting that associates specific pages,
        # but that’s more involved (requires chunking page by page).
        chunk_dict = {
            "content": doc.page_content,
            "metadata": {
                "source_file": os.path.basename(pdf_path),
                # Optionally track page range or other info
                # For now, we just store the entire doc as is.
            }
        }
        main_content_chunks.append(chunk_dict)

    return metadata_chunks, main_content_chunks

if __name__ == "__main__":
    # Path to the specific PDF file
    pdf_folder = "papers-testing"
    pdf_filename = "smg_wp_2008_08.pdf"
    pdf_file = os.path.join(pdf_folder, pdf_filename)

    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"The file {pdf_file} was not found.")

    # Process PDF and separate metadata from main content
    metadata_chunks, main_content_chunks = process_pdf_for_rag(pdf_file)

    # Print metadata chunks
    print("\n[METADATA CHUNKS]")
    for i, meta_chunk in enumerate(metadata_chunks, start=1):
        print(f"\nMetadata Chunk {i}:\n{meta_chunk}\n")

    # Print first 5 main content chunks
    print("\n[MAIN CONTENT CHUNKS]")
    for i, chunk_info in enumerate(main_content_chunks[:5], start=1):
        print(f"Chunk {i} (metadata={chunk_info['metadata']}):\n{chunk_info['content']}\n")
