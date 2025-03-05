from TTS.api import TTS

# Attempt to load a TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

text = """Welcome to our podcast, Uncovering the Secrets of a Firm's Environmental Performance: 
A Deep Dive into the Research of Thorbjørn Knudsen. I'm your host, 
and I'm excited to dive into the world of environmental management and corporate greening with my guest, 
who has a deep understanding of this topic. Today, we're going to explore the factors that influence a firm's 
environmental performance, and I have to say, it's a topic that's near and dear to my heart. 
So, let's get started. Can you tell us a bit about the research of Thorbjørn Knudsen 
and what inspired him to study this topic?"""

tts.tts_to_file(text=text, file_path="test.wav")

print("Done! Check test.wav")
