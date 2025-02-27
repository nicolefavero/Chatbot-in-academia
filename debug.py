import torch
from Final_chat import load_llama_model  # ✅ Load Llama model and tokenizer

# ✅ Load the Llama model, tokenizer, and device
tokenizer, llama_model, device = load_llama_model()

################################################################################
#  The Podcast System Prompt
################################################################################
SYSTEM_PROMPT = """[INST] <<SYS>>
You are a world-class podcast writer; you have worked as a ghost writer for 
Joe Rogan, Lex Fridman, Ben Shapiro, and Tim Ferriss. 

We are in an alternate universe where you have been writing every line they say 
and they just stream it into their brains. You have won multiple podcast awards 
for your writing. 

Your job is to write word by word, even “umm, hmmm, right” interruptions by the 
second speaker based on the input topic. Keep it extremely engaging; the speakers 
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
<</SYS>> [/INST]
"""

################################################################################
#  🔎 Test Input for Debugging (Instead of PDF)
################################################################################

TEST_INPUT_TEXT = """
The AI Revolution in Companies: How businesses are adopting artificial intelligence 
to automate processes, enhance decision-making, and personalize customer experiences.
From supply chain optimization to AI-powered marketing strategies, this conversation 
explores real-world examples and the future of AI in the workplace.
"""

################################################################################
#  Generate Podcast-Style Script
################################################################################

def generate_podcast_from_text(input_text: str):
    """
    Takes a text input and generates a podcast-style script.
    """

    # Step A: Manually format the chat input (since `apply_chat_template()` is unavailable)
    user_prompt = f"[INST] Here is the topic that needs to be turned into a podcast:\n\n{input_text}\n\nNow, generate the podcast transcript in the style described above. [/INST]"

    full_prompt = SYSTEM_PROMPT + user_prompt

    # Step B: Tokenize and send to model
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True).to(device)

    # Step C: Generate response using preloaded Llama model
    with torch.no_grad():
        output = llama_model.generate(
            **inputs,
            max_new_tokens=2048,  # ✅ Increased to allow full podcast
            do_sample=True,
            temperature=0.8,  # ✅ Balanced creativity
            top_p=0.9,
            repetition_penalty=1.1,  # ✅ Reduce looping issues
            eos_token_id=tokenizer.eos_token_id
        )

    # Step D: Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


################################################################################
#  🚀 Run the Test (No User Input Required)
################################################################################
if __name__ == "__main__":
    print("\n📢 Running Debug Test for Podcast Generation...\n")

    # Run the debug test with the test input
    podcast_script = generate_podcast_from_text(TEST_INPUT_TEXT)
    
    print(f"\n🎙️ Podcast Script:\n{podcast_script}")
