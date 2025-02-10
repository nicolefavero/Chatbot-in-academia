################################################################################
#  CHATBOT WITH LLAMA MODEL   #
################################################################################

'''Reason: use internal chatbot based on Llama model instead of ChatGPT or any other third party model
mainly because of privacy reasons, it's good that we don't have to send the data to any third party server and 
we have full control over the data.'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import networkx as nx  # this lib is for graphRAG 
import numpy as np
import fitz
import re
import pandas as pd
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydoc import doc

# --------------------------------------------------------------------------
# 1. Loading Llama 3.3 model from Hugging Face because of storage limitations
# --------------------------------------------------------------------------

def load_llama_model():
    '''Load Llama 3.3 model from Hugging Face'''
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct",
    torch_dtype= torch.float16,
    device_map = "auto")
    return tokenizer, model

# --------------------------------------------------------------------------
# 2. Setting up GraphRAG retrieval
# --------------------------------------------------------------------------

# 2.1 Chunking text of processed PDF with LangChain's recursive character splitter
def processed_pdf(pdf_path):
    ''' Reads a PDF file, extracts text, and splits it into chunks.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        List[str]: List of chunked text segments.
    '''
    # Define the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    # Read the PDF file and extract text
    document = fitz.Document(pdf_path)
    pages = document.page_count
    chunks = []

    for page_num in range(pages):
        page = document.load_page(page_num)
        text = page.get_text("text")  
        text = re.sub(r" +", r" ", text)  

        # Split the text into chunks
        text_chunk = text_splitter.create_documents([text])  
        chunks.extend([chunk.page_content for chunk in text_chunk])  
    return chunks

# 2.2 Extract Entities from chunks using Llama model
def extract_entities(chunks, tokenizer, model):
    '''Extract entities from text chunks using the Llama model.
    Arg:
        chunks: List of text chunks.
        tokenizer: Hugging Face tokenizer for the Llama model.
        model: Hugging Face model for the Llama model.
    Returns:
        DataFrame of entities extracted from the text chunks.
        '''
    entities_prompts = """Extract the entities from the following text and then return the entities as a JSON list.
    Text:
    {text}
    Entities (JSON format):"""
    entity_list = defaultdict(list)

    for idx, chunk in enumerate(chunks):
        prompt = entities_prompts.replace("{text}", chunks)
        input = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(input["input_ids"], max_length= 256)
        response = tokenizer.decode(output[0], skip_special_tokens=True)




def get_graphrag():
    '''Load GraphRAG model for retrieval of data from CBS Archive'''

