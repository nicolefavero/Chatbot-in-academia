import os
import re
from TTS.api import TTS
from pydub import AudioSegment

# Load TTS models for two different voices
tts_speaker1 = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)
tts_speaker2 = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=False, gpu=True)

# Sample podcast script for testing
sample_script = """
Speaker 1: Welcome back to the show, everyone! Today, we're diving into something fascinating—the surprising science behind why we remember some things so vividly yet forget others almost instantly. And joining me, as always, is my incredibly curious co-host.

Speaker 2: Hey there! Honestly, this topic has been on my mind for weeks. Like, why do I still remember my childhood phone number, but I can never remember where I left my keys?

Speaker 1: Exactly! And that has everything to do with how our brain filters information. So, neuroscientists have found that our brain prioritizes information that's emotionally charged or tied to strong visual cues.

Speaker 2: Oh! So that's why I can vividly remember that time I accidentally spilled coffee on my boss's laptop in my first week on the job?

Speaker 1: Yep! Emotional moments get a VIP pass in your memory system. It's called "flashbulb memory." But here's where it gets interesting—our brain doesn't just record facts; it reconstructs memories like a movie editor filling in the gaps.

Speaker 2: Wait, so... my brain might be making up parts of my memories?

Speaker 1: Exactly. Studies show that even confident memories can be surprisingly inaccurate. There's one experiment where participants were shown a video of a car crash, and when researchers asked how fast the cars were going, the way they phrased the question actually changed how people remembered the event!

Speaker 2: That's wild! So... does that mean I can't trust my own memories?

Speaker 1: Not entirely, but it's important to know that memories are flexible. The best way to improve memory recall is through association—linking new information to things you already know.

Speaker 2: So, like using mnemonics or visualization tricks?

Speaker 1: Exactly! Mnemonics are fantastic because they give your brain "mental hooks" to grab onto. For example, if you're trying to remember a grocery list, you could imagine walking through your house and placing each item in different rooms. That taps into your brain's spatial memory, which is incredibly powerful.

Speaker 2: Oh, I'm definitely trying that! Next time I forget my keys, I'll imagine myself hanging them from a chandelier or something.

Speaker 1: That's the spirit! Memory isn’t just about facts—it’s about storytelling. The more creative and emotional the associations, the stronger the memory.

Speaker 2: Love it! Alright, folks, if you've ever wondered why you can't remember names but never forget embarrassing moments, now you know why.

Speaker 1: Thanks for joining us, and as always, stay curious!

Speaker 2: Catch you next time!
"""

def read_sample_script():
    dialogues = []
    current_speaker = None
    current_text = []

    for line in sample_script.strip().split("\n"):
        if line.startswith("Speaker 1:"):
            if current_text:
                dialogues.append((current_speaker, " ".join(current_text)))
                current_text = []
            current_speaker = "Speaker 1"
            current_text.append(line.replace("Speaker 1:", "").strip())
        elif line.startswith("Speaker 2:"):
            if current_text:
                dialogues.append((current_speaker, " ".join(current_text)))
                current_text = []
            current_speaker = "Speaker 2"
            current_text.append(line.replace("Speaker 2:", "").strip())
        else:
            current_text.append(line.strip())

    if current_text:
        dialogues.append((current_speaker, " ".join(current_text)))

    return dialogues

def generate_audio(dialogues):
    audio_parts = []

    for idx, (speaker, text) in enumerate(dialogues):
        print(f"Generating audio for {speaker} (Segment {idx + 1})...")

        file_path = f"temp_part_{idx}.wav"

        if speaker == "Speaker 1":
            tts_speaker1.tts_to_file(text=text, file_path=file_path)
        elif speaker == "Speaker 2":
            tts_speaker2.tts_to_file(text=text, file_path=file_path)

        audio_parts.append(file_path)

    return audio_parts  # <-- Return the list of generated audio files

def merge_audio(audio_parts, output_path):
    combined_audio = AudioSegment.empty()

    for part in audio_parts:
        segment = AudioSegment.from_wav(part)
        combined_audio += segment

    combined_audio.export(output_path, format="wav")

    # Cleanup temporary files
    for part in audio_parts:
        os.remove(part)

    print(f"🎧 Podcast audio saved as: {output_path}")

if __name__ == "__main__":
    dialogues = read_sample_script()
    audio_parts = generate_audio(dialogues)  # <-- Capture audio_parts here
    merge_audio(audio_parts, "podcast_output.wav")  # <-- Pass audio_parts here
