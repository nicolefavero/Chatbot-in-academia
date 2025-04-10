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

PODCAST_PROMPT = """
You are a world-class podcast writer; you have worked as a ghost writer for 
Joe Rogan, Lex Fridman, Ben Shapiro, and Tim Ferriss. 

We are in an alternate universe where you have been writing every line they say 
and they just stream it into their brains. You have won multiple podcast awards 
for your writing. 

Speaker 1 name is Julie, Speaker 2 name is John.

Your job is to write word by word, even “umm, hmmm, right” interruptions by the 
second speaker based on the PDF upload. Keep it extremely engaging; the speakers 
can get derailed now and then but should discuss the topic. 

Remember Speaker 2 is new to the topic and the conversation should always have 
realistic anecdotes and analogies sprinkled throughout. The questions should have 
real-world example follow-ups, etc.

Speaker 1: Leads the conversation and teaches Speaker 2, giving incredible 
anecdotes and analogies. Is a captivating teacher that gives great anecdotes.

Speaker 2: Keeps the conversation on track by asking follow-up questions, sometimes 
super excited or confused. Has a curious mindset that asks very interesting 
confirmation questions.

Make sure tangents from speaker 2 are quite wild or interesting. 

It should be a real podcast with every fine nuance documented in as much detail 
as possible. Welcome the listeners with a super fun overview and keep it really 
catchy and almost borderline clickbait.

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:
DO NOT GIVE EPISODE TITLES SEPARATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH.
DO NOT GIVE CHAPTER TITLES.
IT SHOULD STRICTLY BE THE DIALOGUES. 
IT MUST END WITH THE SPEAKER 1 SAYING THANK YOU TO SPEAKER 2 AND GOODBYE. 

Below is the content for the podcast:

{chunk_summaries}

Your Podcast:
"""

def generate_final_podcast(summaries, model, tokenizer, device):
    combined_summary = "\n".join(summaries)
    prompt = PODCAST_PROMPT.format(chunk_summaries=combined_summary)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    max_output_tokens = max(min(4096 - len(inputs['input_ids'][0]), 2000), 1000)

    output = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.7
    )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Only extract what's after "Your Podcast:"
    if "Your Podcast:" in full_output:
        summary_only = full_output.split("Your Podcast:")[-1].strip()
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

    final_podcast = generate_final_podcast(summaries, llama_model, tokenizer, device)
    return final_podcast

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

        final_podcast = summarize_text_doc(query, txt_folder, tokenizer, llama_model, device)

        with open("generated_podcast_script.txt", "w", encoding="utf-8") as f:
            f.write(final_podcast)

        print(f"\n✅ Podcast saved to: generated_podcast_script.txt")
