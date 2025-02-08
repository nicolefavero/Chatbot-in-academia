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

# Loading Llama 3.3 model from Hugging Face because of storage limitations

def load_llama_model():
    '''Load Llama 3.3 model from Hugging Face'''
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    return tokenizer, model

# Setting up GraphRAG retrieval

def get_graphrag():
    '''Load GraohRAG model for retrieval of data from CBS Archive'''
    
