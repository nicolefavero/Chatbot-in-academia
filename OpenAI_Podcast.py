import gradio as gr
from DOC_REGISTRY import DOC_REGISTRY
from openai import OpenAI
import re
from fuzzywuzzy import fuzz
from pathlib import Path
from pydub import AudioSegment

client = OpenAI(api_key="Your-OpenAI-API-Key")

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\[\d+\]|\(\w+ et al\., \d+\)", "", text)
    text = re.sub(r"\(see Fig\.\s?\d+\)", "", text)
    text = re.sub(r"[*_#]", "", text)
    text = re.sub(r"-\n", "", text)
    text = " ".join(text.split())
    return text

def clean_for_tts(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]", r"<break time='500ms'/> (\1)", text)
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

def parse_script(script_text):
    """
    Parses the podcast script into [(speaker, line), ...], skipping narration and metadata.
    Handles **Julie**:, [John]:, or plain Julie:
    """
    lines = []
    for match in re.finditer(r"(?:\*\*|\[)?(Julie|John)(?:\*\*|\])?:\s(.+?)(?=\n(?:\*\*|\[)?(?:Julie|John)(?:\*\*|\])?:|\Z)", script_text, re.DOTALL):
        speaker = match.group(1).strip()
        line = match.group(2).strip().replace("\n", " ")
        lines.append((speaker, line))
    return lines

def generate_podcast_response(message, chat_history):
    matched_doc = detect_target_doc(message, DOC_REGISTRY)
    if matched_doc:
        try:
            file_path = f"./papers-cleaned/{matched_doc}.txt"
            full_text = read_txt_full_text(file_path)

            prompt = (
            """You are a world-class podcast writer; you have worked as a ghost writer for 
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
            DO NOT include any ** or [] in the output. 
            DO NOT include an intro or outro music in the output.
            DO NOT include any comment, only the podcast script with Julie and John's lines.
            To distinguish between speakers, use the format: "Julie:" and "John:"."""
                + full_text
            )

            # 1. Generate the podcast script
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional podcast scriptwriter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1500
            )
            script_text = response.choices[0].message.content

            # 2. Parse the script
            script_lines = parse_script(script_text)

            # 3. Generate TTS per speaker
            voices = {
                "Julie": "coral",
                "John": "onyx"
            }

            instructions = {
                "Julie": "Speak in a friendly, captivating podcast host tone.",
                "John": "Speak in a curious, engaged and warm tone."
            }

            output_dir = Path("generated_audio")
            segment_dir = output_dir / "segments"
            segment_dir.mkdir(parents=True, exist_ok=True)

            audio_segments = []
            errors = []

            for idx, (speaker, text) in enumerate(script_lines):
                audio_path = segment_dir / f"{idx:02d}_{speaker}.mp3"
                preview = text[:60].strip() + ("..." if len(text) > 60 else "")
                print(f"🔊 Generating segment {idx:02d}: {speaker} - \"{preview}\"")

                try:
                    response = client.audio.speech.create(
                        model="tts-1",
                        voice=voices[speaker],
                        input=text,
                        speed=1.0
                    )
                    with open(audio_path, "wb") as f:
                        f.write(response.content)
                    audio_segments.append(audio_path)

                except Exception as e:
                    print(f"❌ Error on segment {idx:02d} ({speaker}): {e}")
                    errors.append((idx, speaker, str(e)))

            # 4. Merge audio segments
            final_audio = AudioSegment.empty()
            for seg in audio_segments:
                final_audio += AudioSegment.from_mp3(seg)

            final_path = output_dir / f"podcast_{matched_doc}.mp3"
            final_audio.export(final_path, format="mp3")

            if errors:
                print(f"\n⚠️ Podcast generated with {len(errors)} segment error(s).")
                for idx, speaker, err in errors:
                    print(f" - Segment {idx:02d} ({speaker}) failed: {err}")
            else:
                print("✅ All segments generated successfully.")

            return script_text, str(final_path)

        except Exception as e:
            return f"⚠️ Error generating podcast: {e}", None

    else:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful, engaging conversational podcast assistant. Your scope is to help users generate podcast scripts based on academic papers. Don't answer other questions that are not related to your podcasting task."},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            raw_output = response.choices[0].message.content
            cleaned_output = clean_for_tts(raw_output)
            return cleaned_output, None
        except Exception as e:
            return f"⚠️ Error handling your message: {e}", None

# Gradio UI
def gradio_wrapper(message, history):
    script, audio_path = generate_podcast_response(message, history)
    if audio_path:
        return [script, audio_path]
    else:
        return [script, None]

gr.Interface(
    fn=gradio_wrapper,
    inputs=[gr.Textbox(label="Ask about a paper")],
    outputs=[
        gr.Textbox(label="Podcast Script"),
        gr.Audio(label="Download Podcast Audio", type="filepath")
    ],
    title="🎙️ CBS-bot (Podcast Bot)"
).launch()
