import re
from pathlib import Path
from openai import OpenAI

# Setup
client = OpenAI(api_key="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")
input_file = Path("generated_podcast_script.txt")
output_dir = Path("audio_segments")
output_dir.mkdir(exist_ok=True)

# Voice Mapping 
voices = {
    "Julie": "coral",  # Female voice
    "John": "onyx",    # Male voice
}

instructions = {
    "Julie": "Speak in a friendly, informative tone.",
    "John": "Speak in a curious and engaged tone.",
}

# Step 1: Read and Parse Transcript
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Automatically extract speaker lines, including the intro line (assumed to be Julie)
lines = []

# Add intro if it doesn’t start with a name
if not re.match(r"^\w+:", text.strip()):
    first_line, rest = text.strip().split("\n", 1)
    lines.append(("Julie", first_line.strip()))
    text = rest

# Match lines like: "John: Hello!"
for match in re.finditer(r"(\w+):\s(.+?)(?=\n\w+:|\Z)", text, re.DOTALL):
    speaker = match.group(1).strip()
    line = match.group(2).strip().replace("\n", " ")
    if speaker in voices:
        lines.append((speaker, line))

# Step 2: Generate Audio 
for idx, (speaker, line) in enumerate(lines):
    speech_file_path = output_dir / f"{idx:02d}_{speaker}.mp3"
    print(f"Generating audio for: {speaker}: {line[:50]}...")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voices[speaker],
        input=line,
        instructions=instructions[speaker],
    ) as response:
        response.stream_to_file(speech_file_path)

print("✅ All audio segments generated.")

from pydub import AudioSegment

final_audio = AudioSegment.empty()

# Make sure they're sorted by filename (e.g., 00_Julie.mp3, 01_John.mp3, ...)
audio_files = sorted(output_dir.glob("*.mp3"))

for file in audio_files:
    segment = AudioSegment.from_mp3(file)
    final_audio += segment

# Export final audio
final_output_path = Path("final_podcast_episode.mp3")
final_audio.export(final_output_path, format="mp3")
print(f"✅ Final audio saved to: {final_output_path}")

