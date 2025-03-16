################################################################################
#  ACADEMIC SUMMARY GENERATOR (WITH PRELOADED LLAMA 3 - 70B MODEL)             #
################################################################################

import os
import fitz  # PyMuPDF for PDF processing
import torch
import warnings
import re
from rapidfuzz import fuzz
from DOC_REGISTRY import DOC_REGISTRY
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_llama_instructor():
    """
    Load a Llama-3 70B *instruction/chat* model from Hugging Face 
    and distribute across GPUs. 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    HF_TOKEN = "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"
    model_name = "meta-llama/Llama-3.3-70B-Instruct"

    print(f"Loading model on {device} (4 GPUs)...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_auth_token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=HF_TOKEN,
        torch_dtype=dtype,
        device_map="auto"
    )

    return tokenizer, model, device

################################################################################
# 1. PDF Preprocessing with Chunking
################################################################################

def preprocess_text(text: str) -> str:
    """
    Clean whitespace and normalize text.
    """
    text = text.strip()
    text = re.sub(r"\[\d+\]|\(\w+ et al\., \d+\)", "", text)  # Remove inline citations
    text = re.sub(r"\(see Fig\.\s?\d+\)", "", text)            # Remove figure refs
    text = re.sub(r"[*_#]", "", text)                          # Remove symbols
    text = re.sub(r"-\n", "", text)                            # Fix broken lines
    text = " ".join(text.split())                              # Normalize spaces
    return text

def read_pdf_full_text(pdf_path: str) -> str:
    """
    Reads the entire PDF in one go, returning a single large string.
    """
    document = fitz.Document(pdf_path)
    all_text = []

    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        raw_text = page.get_text("text")
        cleaned_text = preprocess_text(raw_text)
        all_text.append(cleaned_text)

    return "\n".join(all_text)

def chunk_text(text, chunk_size=2500, chunk_overlap=200):
    """
    Splits text into smaller, coherent chunks while preserving sentence structure.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

################################################################################
# 2. Fuzzy Matching to Detect Target PDF in DOC_REGISTRY
################################################################################

def detect_target_doc(query: str, registry: dict, threshold=80):
    """
    Use fuzzy matching on all doc aliases in registry to find the best match.
    """
    query_lower = query.lower()
    best_doc = None
    best_score = 0

    for doc_name, data in registry.items():
        aliases = data["aliases"]
        for alias in aliases:
            score = fuzz.partial_ratio(query_lower, alias.lower())
            if score > best_score:
                best_score = score
                best_doc = doc_name

    if best_score >= threshold:
        return best_doc
    else:
        return None

################################################################################
# 3. Chunk Summarization Prompt
################################################################################

CHUNK_SUMMARY_PROMPT = """
Here is an excerpt from an academic paper. Summarize it in a continuous text format that condenses the key information into approximately 350 tokens. 
Focus on clarity and coherence, maintaining the essential arguments and insights. Avoid listing points or dividing the text into sections. 
End with a natural conclusion.

Text:
{chunk_text}
"""

def summarize_chunk(chunk, model, tokenizer, device, chunk_idx, total_chunks):
    """
    Summarizes a single chunk with LLaMA 3, displaying progress for better tracking.
    """
    print(f"\n🔎 Processing Chunk {chunk_idx + 1} of {total_chunks}...")

    prompt = CHUNK_SUMMARY_PROMPT.format(chunk_text=chunk)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=350,
        do_sample=False,
        temperature=0.7
    )

    summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Clearer format — Only the summary itself
    result = (
        f"\n{'='*60}\n"
        f"📝 **Summary {chunk_idx + 1} of {total_chunks}:**\n"
        f"{summary}\n"
        f"{'='*60}\n"
    )

    print(result)
    return summary  # <- Return only the clean summary

################################################################################
# 4. Final Summary Merging Prompt
################################################################################

FINAL_SUMMARY_PROMPT = """
You are an expert academic writer. Your task is to refine and rewrite the following content into a clear, coherent, and continuous academic text. 
The result should improve the flow, clarity, and structure without removing key insights. Avoid repetition and unnecessary details. 
Maintain a strong conclusion that leaves no unfinished thoughts. The output should be approximately 4000 tokens.

Content to Rewrite:
{chunk_summaries}
"""

def generate_final_summary(summaries, model, tokenizer, device):
    combined_summary = "\n".join(summaries)

    # Ensure the input size + output tokens stay within the 4096-token limit
    max_output_tokens = max(min(4096 - len(tokenizer(combined_summary)['input_ids']), 2000), 1000)

    final_summary_prompt = FINAL_SUMMARY_PROMPT.format(chunk_summaries=combined_summary)

    inputs = tokenizer(final_summary_prompt, return_tensors="pt").to(device)

    final_summary = model.generate(
        **inputs,
        max_new_tokens=max_output_tokens,
        do_sample=False,
        temperature=0.7
    )

    return tokenizer.decode(final_summary[0], skip_special_tokens=True).strip()

################################################################################
# 5. Full Pipeline: Generate Summary from PDF
################################################################################

def summarize_paper(user_query: str, pdf_folder_path: str, tokenizer, llama_model, device):
    """
    Full pipeline for detecting, reading, chunking, summarizing, and merging summaries.
    """

    target_doc = detect_target_doc(user_query, DOC_REGISTRY, threshold=80)
    if target_doc is None:
        return "I'm not sure which PDF you mean. Please specify the exact title or a known alias."

    pdf_name = f"{target_doc}.pdf"
    pdf_path = os.path.join(pdf_folder_path, pdf_name)

    if not os.path.isfile(pdf_path):
        return f"PDF '{pdf_name}' not found in '{pdf_folder_path}'."

    pdf_text = read_pdf_full_text(pdf_path)
    pdf_chunks = chunk_text(pdf_text, chunk_size=2500, chunk_overlap=200)

    print(f"\n🧩 Total Chunks to Process: {len(pdf_chunks)}\n{'='*60}")

    summaries = [
        summarize_chunk(chunk, llama_model, tokenizer, device, idx, len(pdf_chunks))
        for idx, chunk in enumerate(pdf_chunks)
    ]

    final_summary = generate_final_summary(summaries, llama_model, tokenizer, device)

    return final_summary

################################################################################
# 6. Example Usage
################################################################################

if __name__ == "__main__":
    pdf_folder = "papers-testing"

    # Load model & tokenizer
    tokenizer, llama_model, device = load_llama_instructor()

    print("\nSummary Generator is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        final_summary = summarize_paper(query, pdf_folder, tokenizer, llama_model, device)

        # Save the generated summary to a text file
        with open("generated_summary.txt", "w", encoding="utf-8") as f:
            f.write(final_summary)
        
        print(f"\nSummary saved as: generated_summary.txt")
