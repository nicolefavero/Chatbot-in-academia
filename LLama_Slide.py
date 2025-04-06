################################################################################
#  ACADEMIC SUMMARY GENERATOR + SLIDE CREATOR (LLAMA 3 - 70B, TWO LLM CALLS)   #
################################################################################

import os
import torch
import re
import subprocess
from rapidfuzz import fuzz
from DOC_REGISTRY import DOC_REGISTRY
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pptx import Presentation
from pptx.util import Inches
from typing import List
from pydantic import BaseModel, Field
import json
import ast

################################################################################
# 0. Load LLaMA-3 Model
################################################################################

def load_llama_instructor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    HF_TOKEN = os.getenv("HF_TOKEN") or "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"
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=HF_TOKEN,
        torch_dtype=dtype,
        device_map="auto"
    )
    return tokenizer, model, device

################################################################################
# 1. TXT Reading and Chunking
################################################################################

def preprocess_text(text: str) -> str:
    text = text.strip()
    # remove references, figure refs, special chars
    text = re.sub(r"\\[\\d+\\]|\\(\\w+ et al\\., \\d+\\)", "", text)
    text = re.sub(r"\\(see Fig\\.\\s?\\d+\\)", "", text)
    text = re.sub(r"[*_#]", "", text)
    text = re.sub(r"-\\n", "", text)
    return " ".join(text.split())

def read_txt_full_text(txt_path: str) -> str:
    with open(txt_path, 'r', encoding='utf-8') as f:
        return preprocess_text(f.read())

def chunk_text(text, chunk_size=4000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

################################################################################
# 2. Match Query to File
################################################################################

def detect_target_doc(query: str, registry: dict, threshold=80):
    query = query.lower()
    best_score, best_doc = 0, None
    for name, data in registry.items():
        for alias in data["aliases"]:
            score = fuzz.partial_ratio(query, alias.lower())
            if score > best_score:
                best_score, best_doc = score, name
    return best_doc if best_score >= threshold else None

################################################################################
# 3. Stopping Criteria
################################################################################

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

    def __call__(self, input_ids, scores):
        return any(
            list(input_ids[0][-len(stop):]) == stop
            for stop in self.stop_ids
        )

################################################################################
# 4. Chunk Summarization
################################################################################

CHUNK_SUMMARY_PROMPT = """
Here is an excerpt from an academic paper. Summarize it in a continuous text format (max 50 tokens).
Focus on clarity and coherence, maintaining the essential arguments and insights.
Avoid repetition. End with a self-contained sentence.

DON'T ADD NOTES
DON'T WRITE MORE THAN ONE SUMMARY
END BEFORE 50 TOKENS

Text:
{chunk_text}

Summary:
"""

def summarize_chunk(chunk, model, tokenizer, device, chunk_idx, total_chunks):
    print(f"\n🔎 Processing Chunk {chunk_idx + 1} of {total_chunks}...")
    prompt = CHUNK_SUMMARY_PROMPT.format(chunk_text=chunk)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stopping = StoppingCriteriaList([StopOnTokens(["Summary:"], tokenizer)])
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.7,
        stopping_criteria=stopping
    )
    summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    print(f"📝 Summary {chunk_idx + 1}:\n{summary}\n{'='*60}")
    return summary

################################################################################
# 5. Merge Summaries
################################################################################

FINAL_SUMMARY_PROMPT = """
You are an expert academic writer. Refine the following content into a clear, coherent academic text:
- Improve flow, clarity
- Avoid repetition
- Maintain a strong conclusion

Content:
{chunk_summaries}

Refined Text:
"""

def generate_final_summary(summaries, model, tokenizer, device):
    combined = "\n".join(summaries)
    max_output_tokens = max(min(4096 - len(tokenizer(combined)['input_ids']), 2000), 1000)
    prompt = FINAL_SUMMARY_PROMPT.format(chunk_summaries=combined)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_output_tokens,
        do_sample=False,
        temperature=0.7
    )
    final_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return final_text

################################################################################
# 6. Multiple-Prompt Slide Generation
################################################################################

SLIDE_TITLES_PROMPT = """
You are an academic presentation assistant. The user wants EXACTLY 5 concise, descriptive slide titles as a valid Python list of strings.

Constraints:
1. NO explanation or commentary
2. Output only a bracketed Python list of 5 short strings
3. The content must come from the summary below
4. Example valid answer: ["Intro","Key Concepts","Methods","Results","Conclusion"]

Summary:
{summary_text}

Now produce your bracketed list:
"""

BULLETS_PROMPT = """
You are an academic presentation assistant. 
Given the summary below and the slide title, return EXACTLY 3 bullet points as a Python list of short strings.
No commentary. Example valid answer: ["Key theme","Insight","Implication"]

Summary:
{summary_text}

Slide Title: "{slide_title}"

Now produce your bracketed list:
"""

def simple_llm_call(prompt, model, tokenizer, device, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

def get_text_after_keyword(raw_output: str, keyword: str = "Now produce your bracketed list:") -> str:
    """
    Return only the substring that comes AFTER the given keyword (ignoring case).
    If not found, return raw_output unchanged.
    """
    lower_raw = raw_output.lower()
    lower_key = keyword.lower()
    idx = lower_raw.find(lower_key)
    if idx == -1:
        return raw_output  # not found
    return raw_output[idx + len(keyword):].strip()

def bracketed_list_only(raw_output: str) -> str:
    """
    Extract from the FIRST '[' to the LAST ']' in raw_output.
    """
    start = raw_output.find("[")
    end = raw_output.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw_output[start:end+1].strip()

def parse_python_list(raw_output: str) -> list:
    """
    1) Get only content after "Now produce your bracketed list:"
    2) Then find [ ... ] inside it
    3) parse with ast.literal_eval
    """
    text_after = get_text_after_keyword(raw_output, "Now produce your bracketed list:")
    bracketed = bracketed_list_only(text_after)
    if not bracketed:
        return None
    try:
        data = ast.literal_eval(bracketed)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return None

def generate_slides_multiple_calls(summary_text, model, tokenizer, device):
    # 1) Slide titles
    raw_titles = simple_llm_call(
        SLIDE_TITLES_PROMPT.format(summary_text=summary_text),
        model, tokenizer, device
    )
    titles = parse_python_list(raw_titles)
    if not titles or len(titles) < 5:
        print("⚠️ Could not parse 5 slide titles. Using fallback.")
        titles = [f"Slide {i+1}" for i in range(5)]
    
    # 2) bullet points for each title
    slides = []
    for title in titles[:5]:
        raw_bullets = simple_llm_call(
            BULLETS_PROMPT.format(summary_text=summary_text, slide_title=title),
            model, tokenizer, device
        )
        bullet_points = parse_python_list(raw_bullets)
        if not bullet_points or len(bullet_points) < 3:
            print(f"⚠️ Could not parse bullet points for '{title}'. Using fallback.")
            bullet_points = [
                "Key insight related to the slide",
                "Important supporting detail",
                "Another notable point"
            ]
        bullet_points = bullet_points[:3]
        slides.append({
            "title": title,
            "bullet_points": bullet_points
        })
    return slides

################################################################################
# 7. Prompt for Python PPTX Code
################################################################################

def extract_code_snippet(raw_code: str) -> str:
    """
    Look for code in triple backticks (```python ... ```).
    If found, return only that block. Else return entire string.
    """
    pattern = r"```python\s*([\s\S]*?)```"
    match = re.search(pattern, raw_code)
    if match:
        return match.group(1).strip()
    return raw_code

PYTHON_PPTX_PROMPT = """
You are a Python programmer. Generate python code using 'python-pptx' to create a PowerPoint file 'presentation.pptx'.

Slides Data (Python list of dicts):
{slides_json}

Constraints:
1. Output ONLY valid Python code (no commentary).
2. ASCII only (no en dash, em dash, curly quotes).
3. from pptx import Presentation, from pptx.util import Inches.
4. Title slide first, then slides from 'slides_data' with 'title' + 'bullet_points'.
5. Save as 'presentation.pptx'.
"""

def prompt_for_pptx_code(slides_data, model, tokenizer, device):
    slides_json_str = json.dumps(slides_data, indent=2)
    prompt = PYTHON_PPTX_PROMPT.format(slides_json=slides_json_str)
    code_response = simple_llm_call(prompt, model, tokenizer, device, max_tokens=800)
    
    # 1) Extract code from triple backticks
    code_snippet = extract_code_snippet(code_response)
    return code_snippet.strip()

def clean_code_for_python(code_str):
    # remove fancy dashes/quotes
    replacements = {
        '–': '-',
        '—': '-',
        '‘': "'",
        '’': "'",
        '“': '"',
        '”': '"'
    }
    for bad, good in replacements.items():
        code_str = code_str.replace(bad, good)
    return code_str

################################################################################
# 8. Summarize Text, Then Create Slides, Then Code
################################################################################

def summarize_text_doc(user_query: str, txt_folder: str, tokenizer, model, device):
    doc = detect_target_doc(user_query, DOC_REGISTRY)
    if not doc:
        return None, "Document not found."
    txt_name = f"{doc}.txt"
    possible_files = os.listdir(txt_folder)
    matched = next((f for f in possible_files if f.lower() == txt_name.lower()), None)
    if not matched:
        return None, f"Text file '{txt_name}' not found in '{txt_folder}'."
    
    txt_path = os.path.join(txt_folder, matched)
    full_text = read_txt_full_text(txt_path)
    chunks = chunk_text(full_text)
    print(f"\n🧩 Total Chunks: {len(chunks)}")
    print("=" * 60)
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        s = summarize_chunk(chunk, model, tokenizer, device, i, len(chunks))
        chunk_summaries.append(s)
    
    final_summary = generate_final_summary(chunk_summaries, model, tokenizer, device)
    return final_summary, None

################################################################################
# 9. CLI Runner
################################################################################

if __name__ == "__main__":
    folder = "/work/Chatbot-in-academia/papers-cleaned"
    tokenizer, model, device = load_llama_instructor()
    print("\n📚 Academic Slide Generator is ready. Type your paper title. Type 'exit' to quit.")
    
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        
        generate_slide = any(k in query.lower() for k in ["slide", "deck", "presentation"])
        
        try:
            print(f"🔍 Searching for document matching: '{query}'...")
            final_summary, err = summarize_text_doc(query, folder, tokenizer, model, device)
            if err:
                print(f"❌ {err}")
                continue
            
            with open("generated_summary.txt", "w", encoding="utf-8") as f:
                f.write(final_summary)
            print("✅ Final summary saved to 'generated_summary.txt'")
            
            if generate_slide:
                print("\n🖼️ Generating slides in multiple LLM calls...\n")
                
                slides_data = generate_slides_multiple_calls(final_summary, model, tokenizer, device)
                print(f"✅ Created {len(slides_data)} slides.\n")
                
                pptx_code = prompt_for_pptx_code(slides_data, model, tokenizer, device)
                
                # 2) Clean the code to remove fancy punctuation
                pptx_code = clean_code_for_python(pptx_code)
                
                if not pptx_code.strip():
                    print("❌ Slide code generation returned empty.")
                    continue
                
                # 3) Save & run
                with open("generated_slide.py", "w", encoding="utf-8") as f:
                    f.write(pptx_code)
                
                print("🚀 Running generated_slide.py to create 'presentation.pptx'...\n")
                try:
                    subprocess.run(["python3", "generated_slide.py"], check=True)
                    print("✅ Presentation created: presentation.pptx")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error running slide code: {e}")
                    print("⚠️ Could not generate slides.")
            else:
                print("No slide creation requested. Done.")
                
        except Exception as e:
            print(f"❌ An error occurred: {repr(e)}")
            print("⚠️ Try again or try a different query.")
