################################################################################
#  ACADEMIC SUMMARY GENERATOR + SLIDE CREATOR (LLAMA 3 - 70B, TWO LLM CALLS)   #
################################################################################

import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import summary functions from LLama_Summary.py
from LLama_Summary import (
    load_llama_instructor,
    summarize_text_doc as llama_summarize_text_doc
)
from DOC_REGISTRY import DOC_REGISTRY

################################################################################
# 1. Summary Processing and Slide Content Generation
################################################################################

def extract_sections_from_summary(summary_text):
    """Extract sections from the summary text based on bold headings."""
    # Pattern to match a bold section heading and its content
    section_pattern = r'\*\*([^*]+)\*\*\s*(.*?)(?=\*\*|$)'
    
    # Find all sections in the summary
    sections = re.findall(section_pattern, summary_text, re.DOTALL)
    
    # If no sections found, try to split by newlines and look for section-like titles
    if not sections:
        lines = summary_text.split('\n')
        current_section = None
        current_content = []
        sections = []
        
        for line in lines:
            if line.strip().isupper() or line.strip().endswith(':'):
                # This looks like a section heading
                if current_section:
                    sections.append((current_section, ' '.join(current_content)))
                current_section = line.strip().rstrip(':')
                current_content = []
            elif current_section and line.strip():
                current_content.append(line.strip())
        
        # Add the last section
        if current_section and current_content:
            sections.append((current_section, ' '.join(current_content)))
    
    # Create dictionaries for each section
    section_dict = {}
    for heading, content in sections:
        section_dict[heading.strip()] = content.strip()
    
    return section_dict

def generate_bullet_points_for_section(section_title, section_content, model, tokenizer, device):
    """Generate bullet points for a section."""
    prompt = f"""
You are creating bullet points for a PowerPoint slide titled "{section_title}" for a presentation on knowledge management in multinational corporations.

The slide will be based on this section from an academic paper summary:
"{section_content}"

Create EXACTLY 3 bullet points that:
1. Highlight the key insights from this section
2. Are written as complete sentences (about 10-15 words each - KEEP THEM SHORT)
3. Are clear, concise, and academically sound
4. Do not repeat the instructions or bullet point criteria

ONLY output the 3 bullet points, one per line, with no numbering, no bullet symbols, and no quotation marks.
"""
    
    response = simple_llm_call(prompt, model, tokenizer, device)
    
    # Split by newlines and clean up each line
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    # Remove any bullet point markers or numbering at the beginning of lines
    bullet_points = []
    for line in lines:
        # Clean bullet point markers and numbering
        cleaned_line = re.sub(r'^[-•*\d\.]+\s*', '', line)
        if cleaned_line:
            # Keep bullet points shorter - limit to 120 characters
            if len(cleaned_line) > 120:
                # Try to find a logical break point
                break_point = cleaned_line.rfind('.', 0, 120)
                if break_point > 50:  # If we can find a sentence end
                    cleaned_line = cleaned_line[:break_point+1]
                else:
                    # Otherwise just truncate at a reasonable length
                    break_point = cleaned_line.rfind(' ', 90, 120)
                    if break_point > 0:
                        cleaned_line = cleaned_line[:break_point] + "."
            bullet_points.append(cleaned_line)
    
    # If we didn't get exactly 3 bullet points, extract sentences from the section content
    if len(bullet_points) != 3:
        print(f"⚠️ Didn't get exactly 3 bullet points for '{section_title}'. Using sentences from section content.")
        sentences = re.split(r'(?<=[.!?])\s+', section_content)
        bullet_points = []
        for sentence in sentences:
            if len(sentence.split()) >= 5 and len(bullet_points) < 3:
                # Truncate long sentences
                if len(sentence) > 120:
                    # Try to find a logical break point
                    break_point = sentence.rfind('.', 0, 120)
                    if break_point > 50:  # If we can find a sentence end
                        sentence = sentence[:break_point+1]
                    else:
                        # Otherwise just truncate
                        break_point = sentence.rfind(' ', 90, 120)
                        if break_point > 0:
                            sentence = sentence[:break_point] + "."
                bullet_points.append(sentence.strip())
    
    # Fallback if we still don't have 3 bullet points
    while len(bullet_points) < 3:
        if section_title == "Introduction":
            bullet_points.append("The study examines knowledge management's impact on MNC subsidiary performance.")
        elif section_title == "Core Contributions":
            bullet_points.append("Absorptive capacity enables knowledge inflows and enhances performance.")
        elif section_title == "Methods":
            bullet_points.append("Data collected from a German MNC with 222 questionnaires.")
        elif section_title == "Findings" or section_title == "Findings and Discussion":
            bullet_points.append("Knowledge management tools positively influence absorptive capacity.")
        elif section_title == "Conclusion":
            bullet_points.append("Knowledge tools enhance performance through absorptive capacity.")
        else:
            bullet_points.append(f"Key aspects of {section_title.lower()} in knowledge management.")
    
    # Ensure we have exactly 3 bullet points
    return bullet_points[:3]

def simple_llm_call(prompt, model, tokenizer, device, max_tokens=1000):
    """Make a simple LLM call with the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

def generate_slides_from_summary(summary_text, model, tokenizer, device):
    """Generate slides directly from the summary sections."""
    print("\nExtracting sections from summary...")
    sections = extract_sections_from_summary(summary_text)
    
    if not sections:
        print("⚠️ No sections found in summary. Using default sections.")
        sections = {
            "Introduction": summary_text[:500],
            "Core Contributions": summary_text[500:1000] if len(summary_text) > 500 else summary_text,
            "Methods": summary_text[1000:1500] if len(summary_text) > 1000 else summary_text,
            "Findings": summary_text[1500:2000] if len(summary_text) > 1500 else summary_text,
            "Conclusion": summary_text[2000:] if len(summary_text) > 2000 else summary_text
        }
    
    slides = []
    for title, content in sections.items():
        print(f"Generating bullet points for: {title}")
        bullet_points = generate_bullet_points_for_section(title, content, model, tokenizer, device)
        
        slides.append({
            "title": title,
            "bullet_points": bullet_points
        })
    
    return slides

################################################################################
# 2. PowerPoint Creation
################################################################################

def create_powerpoint(slides_data, paper_title, output_filename='presentation.pptx'):
    """Create a PowerPoint presentation from the slides data."""
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.enum.text import PP_ALIGN
        
        # Create presentation
        prs = Presentation()
        
        # Add title slide
        slide_layout = prs.slide_layouts[0]  # Title Slide layout
        slide = prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        
        # Set title slide content with smaller font
        title.text = paper_title
        title_font = title.text_frame.paragraphs[0].font
        title_font.size = Pt(40)  # Slightly smaller title
        
        subtitle.text = "Knowledge Management Impact Summary"
        subtitle_font = subtitle.text_frame.paragraphs[0].font
        subtitle_font.size = Pt(24)  # Smaller subtitle
        
        # Add content slides
        for slide_data in slides_data:
            slide_layout = prs.slide_layouts[1]  # Title and Content layout
            slide = prs.slides.add_slide(slide_layout)
            
            # Set title
            title = slide.shapes.title
            title.text = slide_data["title"]
            # Make title font a bit smaller
            title_font = title.text_frame.paragraphs[0].font
            title_font.size = Pt(36)
            
            # Add bullet points with smaller font
            content = slide.placeholders[1]
            tf = content.text_frame
            
            # Clear any existing paragraphs (sometimes there's a default one)
            if tf.paragraphs:
                for _ in range(len(tf.paragraphs) - 1):
                    tf._p.remove(tf._p.xpath('./a:p')[-1])
                
                # Use the first paragraph for the first bullet point
                if slide_data["bullet_points"]:
                    p = tf.paragraphs[0]
                    p.text = slide_data["bullet_points"][0]
                    p.level = 0
                    p.font.size = Pt(20)  # Smaller font size for bullets
                
                # Add remaining bullet points
                for bullet in slide_data["bullet_points"][1:]:
                    p = tf.add_paragraph()
                    p.text = bullet
                    p.level = 0
                    p.font.size = Pt(20)  # Smaller font size for bullets
            
        prs.save(output_filename)
        print(f"\n✅ PowerPoint presentation saved as '{output_filename}'")
        return True
    
    except Exception as e:
        print(f"\n❌ Error creating PowerPoint: {e}")
        return False

################################################################################
# 3. Main Runner
################################################################################

if __name__ == "__main__":
    folder = "/work/Chatbot-in-academia/papers-cleaned"
    
    print("Loading LLaMA model...")
    tokenizer, model, device = load_llama_instructor()
    
    print("\n📚 Summary-Based Slide Generator is ready. Type your paper title. Type 'exit' to quit.")
    
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        
        try:
            print(f"🔍 Searching for document matching: '{query}'...")
            
            final_summary = llama_summarize_text_doc(query, folder, tokenizer, model, device)
            
            if isinstance(final_summary, str) and final_summary.startswith("I'm not sure"):
                print(f"❌ {final_summary}")
                continue
            
            with open("generated_summary.txt", "w", encoding="utf-8") as f:
                f.write(final_summary)
            
            print("✅ Final summary saved to 'generated_summary.txt'")
            
            # Extract paper title
            paper_title = "Knowledge Management Tools"
            if "knowledge management" in query.lower() and "subsidiary performance" in query.lower():
                paper_title = "The Impact of Knowledge Management on MNC Subsidiary Performance"
            
            # Generate slides directly from the summary
            slides_data = generate_slides_from_summary(final_summary, model, tokenizer, device)
            
            # Create PowerPoint
            create_powerpoint(slides_data, paper_title)
        
        except Exception as e:
            print(f"❌ An error occurred: {repr(e)}")
            print("⚠️ Try again or try a different query.")