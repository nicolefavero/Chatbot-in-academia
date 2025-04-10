###############################################################################
#  ACADEMIC SUMMARY GENERATOR (WITH PRELOADED LLAMA 3 - 70B MODEL)           #
###############################################################################

import os
import torch
import re
from rapidfuzz import fuzz
from DOC_REGISTRY import DOC_REGISTRY
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

###############################################################################
# 0. Load LLaMA 3 Model + Tokenizer
###############################################################################

def load_llama_instructor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    HF_TOKEN = "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"
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

###############################################################################
# 1. Preprocessing and File Handling
###############################################################################

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\[\d+\]|\(\w+ et al\., \d+\)", "", text)
    text = re.sub(r"\(see Fig\.\s?\d+\)", "", text)
    text = re.sub(r"[*_#]", "", text)
    text = re.sub(r"-\n", "", text)
    return " ".join(text.split())

def read_txt_full_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return preprocess_text(f.read())

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
# 2. Prompt and Final Summary Generator
###############################################################################

FINAL_SUMMARY_PROMPT = """
You are an expert academic writer. Your task is to generate a comprehensive and structured summary of the content below. 
Your summary should be clear, well-organized, and suitable for a graduate student or professor in business and social sciences.

Please follow this structure:

1. **Introduction** – Briefly present the topic and research objective of the paper.
2. **Core Contributions** – Identify key arguments, theoretical frameworks, or questions addressed.
3. **Methods** – Mention the methodology and approach, if present.
4. **Findings and Discussion** – Present the main results and what they imply.
5. **Conclusion** – Sum up the significance of the paper and its broader relevance.

Avoid direct quotations. Instead, rewrite the ideas in your own words. Keep the tone formal, informative, and suitable for academic discussion.

Below is the content to summarize:

{chunk_summaries}

Your Summary:
"""

def generate_final_summary(summaries, model, tokenizer, device):
    combined_summary = "\n".join(summaries)
    prompt = FINAL_SUMMARY_PROMPT.format(chunk_summaries=combined_summary)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    max_output_tokens = max(min(4096 - len(inputs['input_ids'][0]), 2000), 1000)

    output = model.generate(
        **inputs,
        max_new_tokens=max_output_tokens,
        do_sample=False,
        temperature=0.7
    )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Only extract what's after "Your Summary:"
    if "Your Summary:" in full_output:
        summary_only = full_output.split("Your Summary:")[-1].strip()
    else:
        summary_only = full_output  # fallback

    return summary_only

###############################################################################
# 3. Full Pipeline to Summarize Text File (Chunking not yet included)
###############################################################################

def summarize_text_doc(user_query: str, txt_folder_path: str, tokenizer, llama_model, device):
    target_doc = detect_target_doc(user_query, DOC_REGISTRY, threshold=80)
    if target_doc is None:
        return "I'm not sure which text file you mean. Please specify the exact title or a known alias."

    txt_path = os.path.join(txt_folder_path, f"{target_doc}.txt")
    if not os.path.isfile(txt_path):
        return f"Text file '{target_doc}.txt' not found in '{txt_folder_path}'."

    doc_text = read_txt_full_text(txt_path)

    # Placeholder chunking: treat whole text as a single chunk
    summaries = [doc_text]  # You can add real chunking logic here later

    final_summary = generate_final_summary(summaries, llama_model, tokenizer, device)
    return final_summary

###############################################################################
# 4. CLI Runner
###############################################################################

if __name__ == "__main__":
    txt_folder = "papers-cleaned"

    tokenizer, llama_model, device = load_llama_instructor()

    print("\nSummary Generator is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        final_summary = summarize_text_doc(query, txt_folder, tokenizer, llama_model, device)

        with open("generated_summary.txt", "w", encoding="utf-8") as f:
            f.write(final_summary)
        
        print(f"\n✅ Summary saved to: generated_summary.txt")
