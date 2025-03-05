import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import os

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Parler TTS model & tokenizer
    model_name = "parler-tts/parler-tts-mini-v1"
    print("Loading model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # The text you want to synthesize
    text_prompt = """
Welcome to our podcast, Uncovering the Secrets of a Firm's Environmental Performance: 
A Deep Dive into the Research of Thorbjørn Knudsen. I'm your host, 
and I'm excited to dive into the world of environmental management and corporate greening with my guest, 
who has a deep understanding of this topic. Today, we're going to explore the factors that influence a firm's 
environmental performance, and I have to say, it's a topic that's near and dear to my heart. 
So, let's get started. Can you tell us a bit about the research of Thorbjørn Knudsen 
and what inspired him to study this topic?"""

    # A free-form description of the voice style
    description = """
Laura's voice is expressive and dramatic in delivery, speaking at a moderate pace. 
The recording is very clean, with almost no background noise."""

    # Encode both description and text
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)

    print("Generating audio...")
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids, 
            prompt_input_ids=prompt_input_ids
            )

    # Convert tensor to numpy
    audio_arr = generation.cpu().numpy().squeeze()

    # Save to WAV file
    output_file = "output.wav"
    sf.write(output_file, audio_arr, model.config.sampling_rate)
    print(f"Audio saved to {output_file}.")

if __name__ == "__main__":
    main()
