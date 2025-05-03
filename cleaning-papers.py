import os
import re
import fitz

def extract_text_between_sections(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    # Normalize text (lowercase and strip extra spaces)
    text = re.sub(r'\s+', ' ', text)

    # Extract text between "Abstract" and "References" or "Bibliography"
    match = re.search(r'(?i)abstract(.*?)(?=references|bibliography)', text)

    if match:
        return match.group(1).strip()
    return None

def process_papers(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            extracted_text = extract_text_between_sections(pdf_path)

            if extracted_text:
                output_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(extracted_text)
                print(f"Extracted text saved for: {filename}")
            else:
                print(f"No valid text found in: {filename}")

input_folder = "papers-testing"
output_folder = "papers-cleaned"
process_papers(input_folder, output_folder)
