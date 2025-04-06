################################################################################
#  ACADEMIC SUMMARY GENERATOR + SLIDE CREATOR (LLAMA 3 - 70B, TWO LLM CALLS)     #
################################################################################

import os
import torch
import re
import subprocess
from rapidfuzz import fuzz
from DOC_REGISTRY import DOC_REGISTRY
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pptx import Presentation
from pptx.util import Inches
from typing import List
from pydantic import BaseModel, Field
import json

################################################################################
# 0. Load LLaMA-3 Model
################################################################################

def load_llama_instructor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    HF_TOKEN = os.getenv("HF_TOKEN") or "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"
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN, torch_dtype=dtype, device_map="auto")
    return tokenizer, model, device

################################################################################
# 1. TXT Reading and Chunking
################################################################################

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\[\\d+\\]|\\(\\w+ et al\\., \\d+\\)", "", text)
    text = re.sub(r"\\(see Fig\\.\\s?\\d+\\)", "", text)
    text = re.sub(r"[*_#]", "", text)
    text = re.sub(r"-\\n", "", text)
    return " ".join(text.split())

def read_txt_full_text(txt_path: str) -> str:
    with open(txt_path, 'r', encoding='utf-8') as f:
        return preprocess_text(f.read())

def chunk_text(text, chunk_size=4000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    return splitter.split_text(text)

################################################################################
# 2. Match Query to File
################################################################################

def detect_target_doc(query: str, registry: dict, threshold=80):
    query = query.lower()
    best_score, best_doc = 0, None
    for name, data in registry.items():
        for alias in data["aliases"]:
            score = fuzz.partial_ratio(query, alias.lower())
            if score > best_score:
                best_score, best_doc = score, name
    return best_doc if best_score >= threshold else None

################################################################################
# 3. Stopping Criteria
################################################################################
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
    def __call__(self, input_ids, scores):
        return any(list(input_ids[0][-len(stop):]) == stop for stop in self.stop_ids)

################################################################################
# 4. Chunk Summarization
################################################################################
CHUNK_SUMMARY_PROMPT = """
Here is an excerpt from an academic paper. Summarize it in a continuous text format that condenses the key information in max 50 tokens, it is fine if it's less. 
Focus on clarity and coherence, maintaining the essential arguments and insights. Avoid listing points or dividing the text into sections. Avoid repetition of the same concept.
End with a self-contained sentence.

DON'T ADD NOTES
DON'T WRITE MORE THEN ONE SUMMARY
END WITH THE END OF THE FIRST SUMMARY YOU GENERATE
DON'T GO OVER THE 50 TOKENS IN THE SUMMARY, Summaries longer than 200 tokens will be cut off, so you must wrap up before reaching that limit. 


Text:
{chunk_text}

Summary:
"""

def summarize_chunk(chunk, model, tokenizer, device, chunk_idx, total_chunks):
    print(f"\n🔎 Processing Chunk {chunk_idx + 1} of {total_chunks}...")
    prompt = CHUNK_SUMMARY_PROMPT.format(chunk_text=chunk)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stopping = StoppingCriteriaList([StopOnTokens(["Summary:"], tokenizer)])
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=0.7, stopping_criteria=stopping)
    summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    print(f"\n📝 Summary {chunk_idx + 1} of {total_chunks}::\n{summary}\n{'='*60}")
    return summary

################################################################################
# 5. Merge Summaries
################################################################################
FINAL_SUMMARY_PROMPT = """
You are an expert academic writer. Your task is to refine and rewrite the following content into a clear, coherent, and continuous academic text. 
The result should improve the flow, clarity, and structure without removing key insights. Avoid repetition and unnecessary details. 
Maintain a strong conclusion that leaves no unfinished thoughts. The output should be approximately 4000 tokens.

Content to Rewrite:
{chunk_summaries}
"""

def generate_final_summary(summaries, model, tokenizer, device):
    combined = "\n".join(summaries)
    max_output_tokens = max(min(4096 - len(tokenizer(combined)['input_ids']), 2000), 1000)
    prompt = FINAL_SUMMARY_PROMPT.format(chunk_summaries=combined)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_output_tokens, do_sample=False, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

################################################################################
# 6. Slide Structuring Prompt (Call 1) + Python Code Generator (Call 2)
################################################################################
class SlideSchema(BaseModel):
    title: str = Field(..., description="Slide title")
    bullet_points: List[str] = Field(..., description="3 bullet points")

SLIDE_STRUCTURE_PROMPT = """
You are an expert presentation creator. Based on the academic summary below, generate a well-structured PowerPoint presentation with 5-7 slides in JSON format.

Summary:
{summary_text}

Your output must be a valid JSON array of slide objects following this exact format:
[
  {{
    "title": "Meaningful title based on content",
    "bullet_points": [
      "First key point from the summary - be specific and informative",
      "Second key point from the summary - be specific and informative",
      "Third key point from the summary - be specific and informative"
    ]
  }},
  {{
    "title": "Another meaningful title",
    "bullet_points": [
      "Another important point",
      "Another specific detail",
      "Another relevant insight"
    ]
  }}
]

Rules:
1. Return ONLY the JSON array - no explanations, no markdown, just the JSON
2. Each slide must have a descriptive title that accurately reflects its content
3. Each slide must have exactly 3 bullet points
4. Bullet points must contain actual content from the summary, not placeholders
5. Make the content academically rigorous and precise
6. Ensure the JSON is properly formatted with double quotes for all strings

IMPORTANT: Output only valid JSON that can be parsed by Python's json.loads()
"""

PYTHON_PPTX_PROMPT = """
You are a Python programmer. Generate `python-pptx` code that creates multiple PowerPoint slide using the structure below.

Title: {title}
Bullet Points:
{bullet_points}

⚠️ Constraints:
- Output ONLY valid Python code (no markdown, no comments, no text before/after).
- Use `pptx` and `pptx.util.Inches`.
- Do NOT specify font sizes or styles.
- Save output to 'presentation.pptx'.

Only return the executable code:
"""

def extract_first_code_block(text):
    if "```" in text:
        code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
    lines = text.splitlines()
    clean_lines = [line for line in lines if not line.strip().startswith("#") and "Title:" not in line]
    return "\n".join(clean_lines).strip()

def extract_slide_code(summary_text, model, tokenizer, device):
    prompt1 = SLIDE_STRUCTURE_PROMPT.format(summary_text=summary_text)
    inputs1 = tokenizer(prompt1, return_tensors="pt").to(device)
    output1 = model.generate(**inputs1, max_new_tokens=1000, do_sample=False)
    raw1 = tokenizer.decode(output1[0], skip_special_tokens=True).strip()

    # Save the raw output for debugging
    with open("llm_output_raw.txt", "w", encoding="utf-8") as f:
        f.write(raw1)
    
    print("\n🔍 Checking for JSON array in LLM output...")
    
    try:
        # Try multiple JSON extraction patterns
        found_json = None
        
        # Pattern 1: Look for content between triple backticks with json
        json_matches = re.findall(r'```json\s*([\s\S]*?)\s*```', raw1)
        if json_matches:
            print("✅ Found JSON in code block format")
            for i, match in enumerate(json_matches):
                try:
                    # Clean and validate JSON
                    clean_json = match.strip()
                    json.loads(clean_json)  # Test if it's valid JSON
                    found_json = clean_json
                    print(f"✅ JSON match {i+1} is valid")
                    break
                except json.JSONDecodeError as e:
                    print(f"❌ JSON match {i+1} is invalid: {e}")
        
        # Pattern 2: Look for array starting with [ and ending with ]
        if not found_json:
            array_matches = re.findall(r'\[\s*\{\s*"title"[\s\S]*?\}\s*\]', raw1)
            if array_matches:
                print("✅ Found JSON in array format")
                for i, match in enumerate(array_matches):
                    try:
                        # Clean and validate JSON
                        clean_json = match.strip()
                        json.loads(clean_json)  # Test if it's valid JSON
                        found_json = clean_json
                        print(f"✅ JSON match {i+1} is valid")
                        break
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON match {i+1} is invalid: {e}")
        
        # If still not found, try a more aggressive extraction
        if not found_json:
            print("⚠️ No valid JSON found with standard patterns, trying aggressive extraction...")
            # Look for anything that remotely resembles a JSON array
            aggressive_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', raw1)
            if aggressive_match:
                potential_json = aggressive_match.group(0)
                # Try to fix common JSON issues
                potential_json = re.sub(r',\s*\}', '}', potential_json)  # Remove trailing commas in objects
                potential_json = re.sub(r',\s*\]', ']', potential_json)  # Remove trailing commas in arrays
                potential_json = potential_json.replace("'", '"')  # Replace single quotes with double quotes
                
                try:
                    json.loads(potential_json)  # Test if it's valid JSON
                    found_json = potential_json
                    print("✅ Found and fixed JSON with aggressive pattern")
                except json.JSONDecodeError:
                    print("❌ Aggressive JSON extraction failed")
        
        if not found_json:
            raise ValueError("No valid JSON found in LLM output")
        
        # Save the extracted JSON for debugging
        with open("extracted_json.txt", "w", encoding="utf-8") as f:
            f.write(found_json)
        
        # Parse the JSON
        slides_data = json.loads(found_json)
        print(f"✅ Successfully parsed JSON with {len(slides_data)} slides")
        
        # Generate the PowerPoint code
        pptx_code = [
            "from pptx import Presentation",
            "from pptx.util import Inches",
            "",
            "# Create presentation",
            "prs = Presentation()",
            "",
            "# Add title slide",
            "title_slide_layout = prs.slide_layouts[0]",
            "slide = prs.slides.add_slide(title_slide_layout)",
            'slide.shapes.title.text = "Academic Paper Summary"',
            'slide.placeholders[1].text = "Generated from research analysis"',
            ""
        ]
        
        # Add content slides
        for i, slide in enumerate(slides_data):
            title = slide["title"]
            bullets = slide["bullet_points"]
            
            pptx_code.extend([
                f"# Content slide {i+1}",
                "slide = prs.slides.add_slide(prs.slide_layouts[1])",  # Title and content layout
                f"slide.shapes.title.text = {repr(title)}",
                "",
                "# Get content placeholder",
                "content = slide.placeholders[1]",
                "tf = content.text_frame",
                ""
            ])
            
            # Add bullet points
            for bullet in bullets:
                pptx_code.extend([
                    "p = tf.add_paragraph()",
                    f"p.text = {repr(bullet)}",
                    "p.level = 0",
                    ""
                ])
        
        # Add save command
        pptx_code.append("prs.save('presentation.pptx')")
        
        return "\n".join(pptx_code)

    except Exception as e:
        print(f"❌ Failed to generate slide code: {repr(e)}")
        print("⚠️ Using fallback slide generation method")
        return create_fallback_slide_code(summary_text)

def create_fallback_slide_code(summary_text):
    """Create a basic slide with extracted content when JSON extraction fails"""
    print("📊 Creating fallback slides...")
    
    # First, try to create a clean summary by aggressively removing prompts
    # List of prompt fragments that should be removed
    prompt_fragments = [
        "You are an expert academic writer",
        "Your task is to refine",
        "The result should improve",
        "Maintain a strong conclusion",
        "Content to Rewrite:",
        "Here is an excerpt from an academic paper",
        "Summarize it in a continuous text format",
        "Focus on clarity and coherence",
        "End with a self-contained sentence",
        "DON'T ADD NOTES",
        "DON'T WRITE MORE THEN ONE SUMMARY",
        "END WITH THE END OF THE FIRST SUMMARY",
        "DON'T GO OVER THE 50 TOKENS IN THE SUMMARY",
        "Summaries longer than 200 tokens will be cut off",
        "Text:",
        "Summary:"
    ]
    
    # Try to load cleaned content from generated summary file first
    try:
        with open("generated_summary.txt", "r", encoding="utf-8") as f:
            clean_content = f.read()
        print("📝 Loaded content from generated_summary.txt")
    except:
        clean_content = summary_text
        print("⚠️ Using provided summary text")
    
    # Remove all prompt fragments from the content
    for fragment in prompt_fragments:
        clean_content = clean_content.replace(fragment, "")
    
    # Split into paragraphs and filter out very short ones and those that look like prompts
    paragraphs = []
    for p in clean_content.split('\n'):
        p = p.strip()
        if len(p) < 15:  # Skip very short lines
            continue
        # Skip lines that appear to be part of a prompt
        if any(keyword in p.lower() for keyword in ["token", "don't", "note:", "the end"]):
            continue
        paragraphs.append(p)
    
    # If we have no valid paragraphs after filtering, use default content
    if not paragraphs:
        print("⚠️ No valid paragraphs found, using default content")
        paragraphs = [
            "This academic paper explores key concepts in the research field.",
            "The methodology involves analysis of relevant data and findings.",
            "Results indicate significant implications for theory and practice.",
            "Conclusions suggest directions for future research in this area.",
            "The paper contributes to the existing literature by offering new insights."
        ]
    
    # Find meaningful title from paragraphs
    potential_titles = [p for p in paragraphs if 20 < len(p) < 100]
    title = potential_titles[0] if potential_titles else "Academic Paper Summary"
    if len(title) > 50:
        title = title[:47] + "..."
    
    # Generate slide code
    pptx_code = [
        "from pptx import Presentation",
        "from pptx.util import Inches",
        "",
        "# Create presentation",
        "prs = Presentation()",
        ""
    ]
    
    # Add title slide
    pptx_code.extend([
        "# Title slide",
        "title_slide = prs.slides.add_slide(prs.slide_layouts[0])",
        f"title_slide.shapes.title.text = {repr(title)}",
        "title_slide.placeholders[1].text = 'Academic Paper Summary'",
        ""
    ])
    
    # Create content slides - max 5 slides, 3 paragraphs per slide
    num_slides = min(5, (len(paragraphs) + 2) // 3)
    
    for slide_idx in range(num_slides):
        start_idx = slide_idx * 3
        end_idx = min(start_idx + 3, len(paragraphs))
        slide_content = paragraphs[start_idx:end_idx]
        
        # If we don't have enough content for 3 bullet points, add generic ones
        while len(slide_content) < 3:
            slide_content.append("Additional research findings support the paper's conclusions.")
        
        # Create a slide title
        slide_title = f"Key Points {slide_idx + 1}"
        
        pptx_code.extend([
            f"# Slide {slide_idx + 1}",
            "slide = prs.slides.add_slide(prs.slide_layouts[1])",
            f"slide.shapes.title.text = {repr(slide_title)}",
            "",
            "# Get content placeholder",
            "content = slide.placeholders[1]",
            "tf = content.text_frame",
            ""
        ])
        
        # Add bullet points
        for content in slide_content:
            # Limit length for readability
            if len(content) > 200:
                content = content[:197] + "..."
                
            pptx_code.extend([
                "p = tf.add_paragraph()",
                f"p.text = {repr(content)}",
                "p.level = 0",
                ""
            ])
    
    # Save the presentation
    pptx_code.append("prs.save('presentation.pptx')")
    
    return "\n".join(pptx_code)

################################################################################
# 7. Full Pipeline
################################################################################

def summarize_text_doc(user_query: str, txt_folder: str, tokenizer, model, device):
    doc = detect_target_doc(user_query, DOC_REGISTRY)
    if not doc:
        return None, "Document not found."
    txt_name = f"{doc}.txt"
    possible_files = os.listdir(txt_folder)
    matched = next((f for f in possible_files if f.lower() == txt_name.lower()), None)
    if not matched:
        return None, f"Text file '{txt_name}' not found in '{txt_folder}'."
    txt_path = os.path.join(txt_folder, matched)
    full_text = read_txt_full_text(txt_path)
    chunks = chunk_text(full_text)
    print(f"\n🧩 Total Chunks: {len(chunks)}\n{'='*60}")
    summaries = [summarize_chunk(c, model, tokenizer, device, i, len(chunks)) for i, c in enumerate(chunks)]
    final_summary = generate_final_summary(summaries, model, tokenizer, device)
    return final_summary, None

################################################################################
# 8. CLI Runner
################################################################################

if __name__ == "__main__":
    folder = "/work/Chatbot-in-academia/papers-cleaned"
    tokenizer, model, device = load_llama_instructor()
    print("\n📚 Academic Slide Generator is ready. Type your paper title. Type 'exit' to quit.")
    
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        
        generate_slide = any(k in query.lower() for k in ["slide", "deck", "presentation"])
        
        try:
            print(f"🔍 Searching for document matching: {query}")
            summary, err = summarize_text_doc(query, folder, tokenizer, model, device)
            
            if err:
                print(f"❌ {err}")
                continue
                
            print("✅ Generated summary successfully")
            with open("generated_summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)
            print("📝 Summary saved to generated_summary.txt")
            
            if generate_slide:
                print("\n🖼️ Generating PowerPoint presentation...")
                
                # Check for prompt contamination
                prompt_indicators = [
                    "You are an expert academic writer", 
                    "Your task is to refine",
                    "DON'T ADD NOTES",
                    "DON'T WRITE MORE THEN ONE SUMMARY"
                ]
                
                if any(indicator in summary for indicator in prompt_indicators):
                    print("⚠️ Warning: Summary contains prompt text. Creating clean fallback slides.")
                    # Clean summary before processing
                    cleaned_summary = summary
                    for indicator in prompt_indicators:
                        cleaned_summary = cleaned_summary.replace(indicator, "")
                    
                    # Use direct JSON generation for cleaner slides
                    try:
                        # Try to generate slides directly
                        print("🔄 Attempting direct slide generation...")
                        code = create_direct_slides(cleaned_summary)
                    except:
                        print("⚠️ Direct generation failed, using fallback...")
                        code = create_fallback_slide_code(cleaned_summary)
                else:
                    # Normal processing path
                    print("🔄 Processing summary to extract presentation structure...")
                    code = extract_slide_code(summary, model, tokenizer, device)
                
                if code:
                    with open("generated_slide.py", "w", encoding="utf-8") as f:
                        f.write(code)
                    print("📝 Slide generation code saved to generated_slide.py")
                    
                    try:
                        print("🚀 Executing slide generation code...")
                        subprocess.run(["python3", "generated_slide.py"], check=True)
                        print("✅ Presentation created: presentation.pptx")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Error running slide code: {e}")
                        
                        # Attempt to rescue the situation with direct slide generation
                        print("🔄 Attempting emergency slide generation...")
                        emergency_code = """
from pptx import Presentation
from pptx.util import Inches

prs = Presentation()

# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Academic Paper Summary"
slide.placeholders[1].text = "Generated from text analysis"

# Content slide
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "Key Findings"
content = slide.placeholders[1]
tf = content.text_frame

p = tf.add_paragraph()
p.text = "The academic paper discusses important research findings."
p.level = 0

p = tf.add_paragraph()
p.text = "The methodology and results contribute to the existing literature."
p.level = 0

p = tf.add_paragraph()
p.text = "Conclusions suggest implications for theory and practice in the field."
p.level = 0

prs.save('presentation.pptx')
"""
                        with open("emergency_slide.py", "w", encoding="utf-8") as f:
                            f.write(emergency_code)
                        try:
                            subprocess.run(["python3", "emergency_slide.py"], check=True)
                            print("✅ Emergency presentation created: presentation.pptx")
                        except:
                            print("❌ Failed to create slides. Please try again.")
                else:
                    print("❌ Slide code generation failed completely.")
        except Exception as e:
            print(f"❌ An error occurred: {repr(e)}")
            print("⚠️ Please try again with a different query")


def create_direct_slides(summary_text):
    """Create slides directly without using LLM for JSON generation"""
    # Extract key sentences from the summary
    sentences = []
    for paragraph in summary_text.split('\n'):
        if paragraph.strip():
            # Split paragraph into sentences
            para_sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sentences.extend([s.strip() for s in para_sentences if len(s.strip()) > 20])
    
    # If we have fewer than 6 good sentences, extract more aggressively
    if len(sentences) < 6:
        sentences = []
        for paragraph in summary_text.split('\n'):
            if paragraph.strip():
                # Split by any reasonable sentence boundary
                para_sentences = re.split(r'(?<=[.!?;:])\s+', paragraph)
                sentences.extend([s.strip() for s in para_sentences if len(s.strip()) > 15])
    
    # Create slide data
    slides_data = []
    
    # Title slide - use first sentence as title
    title = "Academic Paper Summary"
    if sentences and len(sentences[0]) < 100:
        title = sentences[0]
    
    # Content slides - organize in groups of 3
    for i in range(0, min(15, len(sentences)), 3):
        group = sentences[i:i+3]
        while len(group) < 3:  # Ensure we have 3 bullet points
            group.append("Further research is needed to explore these findings in depth.")
        
        # Create a title based on the first sentence in the group
        slide_title = "Key Findings"
        if i // 3 + 1 <= 5:  # Limit to 5 content slides
            slides_data.append({
                "title": f"Key Points {i // 3 + 1}",
                "bullet_points": group
            })
    
    # If we couldn't extract enough content, use default slides
    if len(slides_data) < 1:
        slides_data = [
            {
                "title": "Introduction",
                "bullet_points": [
                    "The academic paper explores key concepts in the research field.",
                    "The study applies rigorous methodology to investigate the research questions.",
                    "Findings contribute to the existing body of knowledge in this domain."
                ]
            },
            {
                "title": "Methodology",
                "bullet_points": [
                    "Data collection involved comprehensive analysis of relevant sources.",
                    "Multiple analytical techniques were employed to ensure robust results.",
                    "The research design addressed potential limitations and biases."
                ]
            },
            {
                "title": "Conclusions",
                "bullet_points": [
                    "The study offers valuable insights for both theory and practice.",
                    "Results indicate several implications for future research directions.",
                    "The findings emphasize the importance of continuing work in this area."
                ]
            }
        ]
    
    # Generate presentation code
    pptx_code = [
        "from pptx import Presentation",
        "from pptx.util import Inches",
        "",
        "# Create presentation",
        "prs = Presentation()",
        "",
        "# Title slide",
        "title_slide = prs.slides.add_slide(prs.slide_layouts[0])",
        f"title_slide.shapes.title.text = {repr(title)}",
        "title_slide.placeholders[1].text = 'Academic Paper Analysis'",
        ""
    ]
    
    # Add content slides
    for i, slide in enumerate(slides_data):
        pptx_code.extend([
            f"# Slide {i+1}",
            "slide = prs.slides.add_slide(prs.slide_layouts[1])",
            f"slide.shapes.title.text = {repr(slide['title'])}",
            "",
            "content = slide.placeholders[1]",
            "tf = content.text_frame",
            ""
        ])
        
        for bullet in slide["bullet_points"]:
            if len(bullet) > 200:
                bullet = bullet[:197] + "..."
            pptx_code.extend([
                "p = tf.add_paragraph()",
                f"p.text = {repr(bullet)}",
                "p.level = 0",
                ""
            ])
    
    # Add save command
    pptx_code.append("prs.save('presentation.pptx')")
    
    return "\n".join(pptx_code)