import gradio as gr
from DOC_REGISTRY import DOC_REGISTRY
from openai import OpenAI
import re
from fuzzywuzzy import fuzz

client = OpenAI(api_key="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")

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\[\d+\]|\(\w+ et al\., \d+\)", "", text)
    text = re.sub(r"\(see Fig\.\s?\d+\)", "", text)
    text = re.sub(r"[*_#]", "", text)
    text = re.sub(r"-\n", "", text)
    text = " ".join(text.split())
    return text

def read_txt_full_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return preprocess_text(raw_text)

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

def generate_summary_response(message, chat_history):
    matched_doc = detect_target_doc(message, DOC_REGISTRY)
    if matched_doc:
        try:
            file_path = f"./papers-cleaned/{matched_doc}.txt"
            full_text = read_txt_full_text(file_path)

            prompt = (
            """You are an expert academic writer. Your task is to write a comprehensive and detailed summary of the following academic text. 
            Highlight the key arguments, methods, findings, and conclusions in depth. Make sure the summary is informative enough for someone 
            who hasn’t read the original paper."""
                + full_text
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional summary writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"⚠️ Error generating podcast: {e}"
    else:
        # If no paper is found, delegate to a general conversation handling
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful, engaging conversational summary assistant. Your scope is to help users generate summary of on accademic papers. Don't answer other questions that are not related to you summarization task."},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"⚠️ Error handling your message: {e}"

# Gradio UI
chatbot = gr.ChatInterface(
    fn=generate_summary_response,
    title="🎙️ CBS-bot (Summary Bot)",
    theme="messages",
    chatbot=gr.Chatbot(show_copy_button=True)
)

chatbot.launch()
