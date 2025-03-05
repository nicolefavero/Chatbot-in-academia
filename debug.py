import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

################################################################################
# 1. Load Llama-2 70B (Instruction-Finetuned or Chat Variant)
################################################################################

def load_llama_instructor():
    """
    Load a Llama-2 70B *instruction/chat* model from Hugging Face 
    and distribute across 4 GPUs. 
    Adjust 'model_name' if you want a different instruct-finetuned version.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Replace with your Hugging Face token that has permission for Llama 2
    HF_TOKEN = "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"

    # This is an example name for a chat/instruct-finetuned variant:
    model_name = "meta-llama/Llama-3.3-70B-Instruct"

    print(f"Loading model on {device} (4 GPUs)...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_auth_token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=HF_TOKEN,
        torch_dtype=dtype,
        device_map="auto"
    )

    return tokenizer, model, device

################################################################################
# 2. The Podcast System Prompt (Instruction)
################################################################################

SYSTEM_PROMPT = """
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
"""

################################################################################
# 3. Test Input for Debugging
################################################################################

TEST_INPUT_TEXT = """
The AI Revolution in Companies: How businesses are adopting artificial intelligence 
to automate processes, enhance decision-making, and personalize customer experiences.
From supply chain optimization to AI-powered marketing strategies, this conversation 
explores real-world examples and the future of AI in the workplace.
"""

################################################################################
# 4. Generate Podcast-Style Script
################################################################################

def generate_podcast_from_text(tokenizer, model, device, input_text: str):
    """
    Takes a text input and generates a podcast-style script using an 
    instruction-finetuned or chat-finetuned Llama-2 70B model.
    """

    prompt = f"""{SYSTEM_PROMPT}

Now here's the conversation topic:
{input_text}

Please write the podcast script now:
"""

    # Tokenize and move to the model's device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,       # Enough room for a full answer
            do_sample=True,            # Sampling for creativity
            temperature=0.8,           # Balanced creativity
            top_p=0.9,                 # Typical nucleus sampling
            repetition_penalty=1.1,    # Helps avoid looping
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

################################################################################
# 5. Run the Test (No User Input Required)
################################################################################
if __name__ == "__main__":
    print("\n📢 Running Debug Test for Podcast Generation with Llama-2 70B Instruct...\n")

    # Load the chat/instruction model
    tokenizer, llama_model, device = load_llama_instructor()

    # Generate the podcast script
    podcast_script = generate_podcast_from_text(tokenizer, llama_model, device, TEST_INPUT_TEXT)

    print(f"\n🎙️ Podcast Script:\n{podcast_script}")
