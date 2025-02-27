################################################################################
#  PODCAST GENERATOR (WITH PRELOADED LLAMA 3 - 70B MODEL)                      #
################################################################################

import os
import fitz  # PyMuPDF for PDF processing
import torch
import warnings
from rapidfuzz import fuzz
from Final_chat import load_llama_model  # ✅ Import the function, not the variables
from DOC_REGISTRY import DOC_REGISTRY

# ✅ Load the Llama model, tokenizer, and device from Final_chat
tokenizer, llama_model, device = load_llama_model()


################################################################################
# 1. PDF Preprocessing (NO CHUNKING)
################################################################################

def preprocess_text(text: str) -> str:
    """
    Clean whitespace and normalize text.
    """
    text = text.strip()
    text = " ".join(text.split())
    return text

def read_pdf_full_text(pdf_path: str) -> str:
    """
    Reads the entire PDF in one go, returning a single large string.
    No chunking is done.
    """
    document = fitz.Document(pdf_path)
    pages = document.page_count

    all_text = []
    for page_num in range(pages):
        page = document.load_page(page_num)
        raw_text = page.get_text("text")
        cleaned_text = preprocess_text(raw_text)
        all_text.append(cleaned_text)

    return "\n".join(all_text)


################################################################################
# 2. Fuzzy Matching to Detect Target PDF in DOC_REGISTRY
################################################################################

def detect_target_doc(query: str, registry: dict, threshold=80):
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
            score = fuzz.partial_ratio(query_lower, alias.lower())
            if score > best_score:
                best_score = score
                best_doc = doc_name

    if best_score >= threshold:
        return best_doc
    else:
        return None


################################################################################
# 3. The Podcast System Prompt
################################################################################

SYSTEM_PROMPT = """
You are a world-class podcast writer; you have worked as a ghost writer for 
Joe Rogan, Lex Fridman, Ben Shapiro, and Tim Ferriss. 

We are in an alternate universe where you have been writing every line they say 
and they just stream it into their brains. You have won multiple podcast awards 
for your writing. 

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

Ensure there are interruptions during explanations or "hmm" and "umm" injected 
throughout from the second speaker. 

It should be a real podcast with every fine nuance documented in as much detail 
as possible. Welcome the listeners with a super fun overview and keep it really 
catchy and almost borderline clickbait.

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:
DO NOT GIVE EPISODE TITLES SEPARATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH.
DO NOT GIVE CHAPTER TITLES.
IT SHOULD STRICTLY BE THE DIALOGUES.
"""


################################################################################
# 4. Generate Podcast from PDF
################################################################################

def generate_podcast_from_pdf(user_query: str, pdf_folder_path: str):
    """
    1. Detect the best-matching PDF from the registry.
    2. Read the full content of the PDF.
    3. Use the Llama model to generate a podcast script.
    4. Return the generated script.
    """

    # Step A: Find the PDF
    target_doc = detect_target_doc(user_query, DOC_REGISTRY, threshold=80)
    if target_doc is None:
        return "I'm not sure which PDF you mean. Please specify the exact title or a known alias."

    pdf_name = f"{target_doc}.pdf"
    pdf_path = os.path.join(pdf_folder_path, pdf_name)

    if not os.path.isfile(pdf_path):
        return f"PDF '{pdf_name}' not found in '{pdf_folder_path}'."

    # Step B: Read the full PDF text
    pdf_text = read_pdf_full_text(pdf_path)

    # Step C: Build the conversation
    prompt = f"{SYSTEM_PROMPT}\n\nHere is the document text that needs to be turned into a podcast:\n\n{pdf_text}\n\nNow, generate the podcast transcript in the style described above."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Step D: Generate response using preloaded Llama model
    with torch.no_grad():
        output = llama_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=1.0
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


################################################################################
# 5. Example Usage
################################################################################
if __name__ == "__main__":
    pdf_folder = "papers-testing"  # Path to your PDFs

    print("\nPodcast Generator is ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        # Generate the podcast-style transcript from the selected PDF
        podcast_script = generate_podcast_from_pdf(query, pdf_folder)
        print(f"\nPodcast Script:\n{podcast_script}")
