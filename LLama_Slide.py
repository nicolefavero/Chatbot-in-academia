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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pptx import Presentation
from pptx.util import Inches
from dataclasses import dataclass
from typing import List
import json

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

def chunk_text(text, chunk_size=4000, chunk_overlap=100):
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


################################################################################
# 3. Chunk Summarization Prompt
################################################################################

CHUNK_SUMMARY_PROMPT = """
Here is an excerpt from an academic paper. Summarize it in a continuous text format that condenses the key information in max 50 tokens, it is fine if it's less. 
Focus on clarity and coherence, maintaining the essential arguments and insights. Avoid listing points or dividing the text into sections. Avoid repetition of the same concept.
End with a self-contained sentence.

DON'T ADD NOTES
DON'T WRITE MORE THEN ONE SUMMARY
END WITH THE END OF THE FIRST SUMMARY YOU GENERATE
DON'T GO OVER THE 50 TOKENS IN THE SUMMARY, Summaries longer than 200 tokens will be cut off, so you must wrap up before reaching that limit. 


Text:
{chunk_text}

Summary:
"""

def summarize_chunk(chunk, model, tokenizer, device, chunk_idx, total_chunks):
    """
    Summarizes a single chunk with LLaMA 3, displaying progress for better tracking.
    """
    print(f"\n🔎 Processing Chunk {chunk_idx + 1} of {total_chunks}...")

    prompt = CHUNK_SUMMARY_PROMPT.format(chunk_text=chunk)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    stop_strings = ["Summary:", "Text:"]  # Prevents prompt leakage
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_strings, tokenizer)])


    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.7,
        stopping_criteria=stopping_criteria
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
    pdf_chunks = chunk_text(pdf_text, chunk_size=4000, chunk_overlap=100)

    print(f"\n🧩 Total Chunks to Process: {len(pdf_chunks)}\n{'='*60}")

    summaries = [
        summarize_chunk(chunk, llama_model, tokenizer, device, idx, len(pdf_chunks))
        for idx, chunk in enumerate(pdf_chunks)
    ]

    final_summary = generate_final_summary(summaries, llama_model, tokenizer, device)

    return final_summary

# -----------------------------
# Slide Content Data Structure
# -----------------------------
class SlideContent:
    title: str
    bullet_points: List[str]

# --------------------------------------
# Function: Convert Text to Slide Format
# --------------------------------------
def generate_slide_prompt(summary_text):
    return f"""
Given the following academic summary, generate a presentation slide with:
- A concise title (max 12 words)
- 3 bullet points (max 20 words each)

Summary:
{summary_text}

Respond with JSON in the format:
{{"title": "Slide Title", "bullet_points": ["point1", "point2", "point3"]}}
"""

# --------------------------------------------------------
# Function: Generate Slide Content via LLM + Format Output
# --------------------------------------------------------
def extract_slide_content(summary_text, tokenizer, model, device):
    prompt = generate_slide_prompt(summary_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        parsed = json.loads(response.strip())
        return SlideContent(title=parsed["title"], bullet_points=parsed["bullet_points"])
    except Exception as e:
        print(f"⚠️ Failed to parse slide content: {e}\nRaw response: {response}")
        return None

# -----------------------------------
# Function: Create PowerPoint Slide
# -----------------------------------
def generate_slide_pptx(slide: SlideContent, filename="chatbot_slide_deck.pptx"):
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]  # Title and Content layout
    ppt_slide = prs.slides.add_slide(slide_layout)

    title_shape = ppt_slide.shapes.title
    content_shape = ppt_slide.placeholders[1]

    title_shape.text = slide.title
    content_shape.text = ""
    for bullet in slide.bullet_points:
        content_shape.text += f"\n• {bullet}"

    prs.save(filename)
    print(f"✅ Slide saved to: {filename}")


if __name__ == "__main__":
    pdf_folder = "papers-cleaned"

    # Load model & tokenizer
    tokenizer, llama_model, device = load_llama_instructor()

    print("\nSummary Generator is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        wants_slides = any(kw in query.lower() for kw in ["slides", "powerpoint", "presentation"])

        final_summary = summarize_paper(query, pdf_folder, tokenizer, llama_model, device)

        # Save the generated summary to a text file
        with open("generated_summary.txt", "w", encoding="utf-8") as f:
            f.write(final_summary)

        print(f"\nSummary saved as: generated_summary.txt")

        if wants_slides:
            slide = extract_slide_content(final_summary, tokenizer, llama_model, device)
            if slide:
                generate_slide_pptx(slide)
