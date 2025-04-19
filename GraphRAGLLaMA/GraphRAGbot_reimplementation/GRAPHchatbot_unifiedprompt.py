############################################################################################################
#  CHATBOT WITH Llama 3.3 70B FOR QUESTION ANSWERING USING GRAPH RAG WITH UNIFIED PROMPTING
############################################################################################################

'''
Creation of a Chatbot leveraging the Llama 3.3 70B model for question answering and for the extraction of entities and relationships 
to build a knowledge graph. The chatbot will be able to answer questions based on the knowledge graph and community summaries generated from the extracted text.
'''

import os
import pickle
import re
import torch
import fitz  
import networkx as nx
import numpy as np
import pandas as pd
import spacy
import copy
import sys
import json
import igraph as ig
import leidenalg
import random 
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer as SpecTokenizer
from adapters import AutoAdapterModel as SpecAdapterModel

DEBUG = True

def debug(msg):
    if DEBUG:
        print("DEBUG:", msg)

# File paths 
ENTITIES_CSV = "GraphRAGLLaMA/GraphRAGbot_reimplementation/extracted_entities.csv"
RELATIONSHIPS_CSV = "GraphRAGLLaMA/GraphRAGbot_reimplementation/extracted_relationships.csv"
GRAPH_PKL = "GraphRAGLLaMA/GraphRAGbot_reimplementation/knowledge_graph.pkl"
COMMUNITY_SUMMARIES_PKL = "GraphRAGLLaMA/GraphRAGbot_reimplementation/community_summaries.pkl"

# -----------------------------------------------------------------------------
# 1. Load Models
# -----------------------------------------------------------------------------

def load_summarization_model():
    '''Loading Llama 3.3 70B model'''
    debug("Loading summarization model...")
    summ_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    summ_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    debug("Summarization model loaded.")
    return summ_tokenizer, summ_model

# -----------------------------------------------------------------------------
# 2. PDF Processing & Text Chunking
# -----------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    return " ".join(text.strip().split())

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents if sent.text.strip()]

def extract_title_and_authors_from_metadata(meta_text: str) -> str:
    title_pattern = re.compile(r'(?<=\n)([A-Z][^\n]{10,200})(?=\n)')
    author_pattern = re.compile(r'(?:by|authors?:)\s+(.+)', re.IGNORECASE)
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    title_match = title_pattern.search(meta_text)
    author_match = author_pattern.search(meta_text)
    year_match = year_pattern.search(meta_text)
    title = title_match.group(1).strip() if title_match else "Untitled"
    authors = author_match.group(1).strip() if author_match else "Unknown Authors"
    year = year_match.group(0) if year_match else "n.d."
    return f"{authors} ({year}) - \"{title}\""

def process_pdf_for_rag(pdf_path: str, metadata_pages=2):
    debug(f"Processing PDF: {pdf_path}")
    doc = fitz.Document(pdf_path)
    pages = doc.page_count

    # Extract metadata
    metadata_text = []
    for page_num in range(min(metadata_pages, pages)):
        page = doc.load_page(page_num)
        raw_text = preprocess_text(page.get_text("text"))
        metadata_text.append(raw_text)
    full_metadata = "\n".join(metadata_text)
    doc_citation = extract_title_and_authors_from_metadata(full_metadata)

    # Extract main text
    main_sentences = []
    for page_num in range(metadata_pages, pages):
        page = doc.load_page(page_num)
        raw_text = preprocess_text(page.get_text("text"))
        main_sentences.extend(split_into_sentences(raw_text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len
    )
    meta_chunks = [doc.page_content for doc in text_splitter.create_documents([full_metadata])] if full_metadata.strip() else []
    main_chunks = [doc.page_content for doc in text_splitter.create_documents(main_sentences)] if main_sentences else []
    
    debug(f"Extracted {len(meta_chunks)+len(main_chunks)} chunks.")
    return doc_citation, meta_chunks + main_chunks

def process_all_pdfs(pdf_paths_with_idx):
    all_chunks = []
    doc_citations = []
    for (doc_idx, pdf_path) in pdf_paths_with_idx:
        debug(f"Processing PDF #{doc_idx}")
        citation, chunks = process_pdf_for_rag(pdf_path, metadata_pages=2)
        all_chunks.append(chunks)
        doc_citations.append(citation)
    return all_chunks, doc_citations

# -----------------------------------------------------------------------------
# 3. LLM-Based Entity Extraction with Multi-Round "Gleanings"
# -----------------------------------------------------------------------------

# Prompt for extracting academic entities and relationships
ACADEMIC_ENTITY_EXTRACTION_PROMPT = """
-Goal-
Given an academic text, identify all relevant academic entities and their relationships.

-Steps-
1. Identify all academic entities. For each entity, extract:
- entity_name: Name of the entity, using standard academic terminology (capitalized)
- entity_type: One of the following types: [{entity_types}]
- entity_description: Precise academic definition or explanation

Format each entity as a JSON entry:
{{"name": <entity name>, "type": <type>, "description": <entity description>}}

2. From the identified entities, determine all pairs of entities that have a clear academic relationship.
For each relationship, extract:
- source_entity: First entity in the relationship
- target_entity: Second entity in the relationship
- relationship_description: Explanation of how these academic concepts relate (builds on, measures, contradicts, etc.)
- relationship_strength: Score from 1-10 indicating relationship strength (higher for direct, explicit relationships)

Format each relationship as:
{{"source": <source_entity>, "target": <target_entity>, "relationship": <relationship_description>, "relationship_strength": <relationship_strength>}}

3. Return all entities and relationships in a single JSON list.

IMPORTANT: Focus ONLY on entities explicitly present in THIS text, using precise academic terminology.

######################
-Examples-
######################
Example:
Text:
The study examined patterns of biodiversity loss in marine ecosystems affected by climate change. Researchers found that rising ocean temperatures correlated with decreased species richness in coral reef ecosystems. Statistical modeling indicated a 0.8% decline in biodiversity for each 0.1°C increase in average water temperature.
######################
Output:
[
  {{"name": "BIODIVERSITY LOSS", "type": "RESEARCH_CONCEPT", "description": "The decrease in variety and variability of living organisms in marine ecosystems"}},
  {{"name": "OCEAN TEMPERATURE RISE", "type": "VARIABLE", "description": "The increase in average water temperature in oceans due to climate change"}},
  {{"name": "SPECIES RICHNESS", "type": "VARIABLE", "description": "A measure of biodiversity representing the number of different species in coral reef ecosystems"}},
  {{"name": "STATISTICAL MODELING", "type": "METHODOLOGY", "description": "Mathematical approach used to quantify the relationship between temperature and biodiversity"}},
  {{"name": "CORAL REEF ECOSYSTEMS", "type": "RESEARCH_CONCEPT", "description": "Marine habitats formed by coral colonies that support high biodiversity"}},
  {{"source": "OCEAN TEMPERATURE RISE", "target": "BIODIVERSITY LOSS", "relationship": "Rising ocean temperatures cause biodiversity loss in marine ecosystems", "relationship_strength": 9}},
  {{"source": "OCEAN TEMPERATURE RISE", "target": "SPECIES RICHNESS", "relationship": "Rising temperatures correlate with decreased species richness", "relationship_strength": 8}},
  {{"source": "STATISTICAL MODELING", "target": "BIODIVERSITY LOSS", "relationship": "Statistical modeling quantified the rate of biodiversity decline per temperature increase", "relationship_strength": 6}}
]

######################
-Input Text-
######################
entity_types: {entity_types}
text: {input_text}
######################
Output:
"""

# Gleaning assessment prompt to determine if more entities need to be extracted
GLEANING_ASSESSMENT_PROMPT = """
-Goal-
Determine if there are additional academic entities or relationships that should be extracted from the text.

-Input-
1. Original academic text
2. Entities and relationships already extracted

-Question-
Are there ADDITIONAL entities or relationships in the text that haven't been extracted yet? 
Answer with ONLY "YES" or "NO".

######################
Original Text:
{input_text}

Already Extracted:
{extracted_items}
######################
"""

# Gleaning extraction prompt to find missed entities
GLEANING_EXTRACTION_PROMPT = """
-Goal-
MANY entities and relationships were missed in the previous extraction. Identify ADDITIONAL academic entities and relationships that weren't captured in the first round.

-Instructions-
1. Review the original text and already extracted items
2. Focus ONLY on identifying NEW entities and relationships not in the previous extraction
3. Format output as a JSON list using the same format as before

######################
Original Text:
{input_text}

Already Extracted:
{extracted_items}
######################
Output:
"""

def extract_elements_with_gleanings(chunk, tokenizer, model, max_gleanings=2):
    """
    Extract academic entities and relationships with multiple rounds of "gleanings"
    to improve recall.
    Args: 
        chunk: Text chunk to process
        tokenizer: Tokenizer for the model
        model: Model used for generation
        max_gleanings: Maximum number of rounds for extracting additional entities
    Returns: 
        A dictionary with entities and relationships
    """
    debug("Extracting elements with multi-round gleanings approach")
    
    prompt = ACADEMIC_ENTITY_EXTRACTION_PROMPT.format(
        entity_types="RESEARCH_CONCEPT, METHODOLOGY, VARIABLE, FINDING, ORGANIZATION",
        input_text=chunk
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.3  # Lower temperature for more precise extraction
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from result
    json_pattern = r'\[\s*\{.*?\}\s*\]'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if not match:
        debug("No JSON found in initial extraction")
        return {"entities": [], "relationships": []}
    
    try:
        initial_data = json.loads(match.group(0))
        debug(f"Initial extraction found {len(initial_data)} items")
    except json.JSONDecodeError:
        debug("Failed to decode initial JSON")
        return {"entities": [], "relationships": []}
    
    # Separate into entities and relationships since prompt was unified
    entities = []
    relationships = []
    
    for item in initial_data:
        if "source" in item and "target" in item:
            relationships.append(item)
        elif "name" in item and "type" in item:
            entities.append(item)
    
    debug(f"Initial extraction: {len(entities)} entities, {len(relationships)} relationships")
    
    # Multiple rounds of gleanings
    gleaning_round = 0
    
    while gleaning_round < max_gleanings:
        extracted_json = json.dumps(initial_data, indent=2)
        assessment_prompt = GLEANING_ASSESSMENT_PROMPT.format(
            input_text=chunk,
            extracted_items=extracted_json
        )
        
        # Force yes/no decision with logit bias
        inputs = tokenizer(assessment_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        
        yes_token_id = tokenizer.encode(" YES")[0] # YES token
        no_token_id = tokenizer.encode(" NO")[0]  # NO token
        
        logit_bias = {} # setting logit bias
        for i in range(tokenizer.vocab_size):
            if i not in [yes_token_id, no_token_id]:
                logit_bias[i] = -100 
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            logits_processor=logit_bias
        )
        
        assessment = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If YES, more entities need to be extracted
        if "YES" in assessment.upper():
            debug(f"Gleaning round {gleaning_round+1}: More entities identified")
            
            # Extract additional entities
            gleaning_prompt = GLEANING_EXTRACTION_PROMPT.format(
                input_text=chunk,
                extracted_items=extracted_json
            )
            
            inputs = tokenizer(gleaning_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.4  # Slightly higher temperature for creativity
            )
            
            gleaning_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            match = re.search(json_pattern, gleaning_text, re.DOTALL)
            
            if match:
                try:
                    additional_data = json.loads(match.group(0))
                    
                    # Add new entities and relationships
                    for item in additional_data:
                        if "source" in item and "target" in item:
                            is_new = True # check if it existed before
                            for existing in relationships:
                                if (item["source"] == existing["source"] and 
                                    item["target"] == existing["target"]):
                                    is_new = False
                                    break
                            
                            if is_new:
                                relationships.append(item)
                        
                        elif "name" in item and "type" in item:
                            is_new = True
                            for existing in entities:
                                if item["name"] == existing["name"]:
                                    is_new = False
                                    break
                            
                            if is_new:
                                entities.append(item)
                    
                    initial_data = entities + relationships
                    
                    debug(f"After gleaning {gleaning_round+1}: {len(entities)} entities, {len(relationships)} relationships")
                    
                except json.JSONDecodeError:
                    debug(f"Failed to decode JSON in gleaning round {gleaning_round+1}")
            
            gleaning_round += 1
        else:
            debug("No more entities to extract")
            break
    
    return {"entities": entities, "relationships": relationships}

def extract_graph_elements_improved(all_chunks, summ_tokenizer, summ_model, max_chunks=None):
    """
    Extract graph elements from all chunks with multi-round gleaning approach.
    Args:
        all_chunks: List of document chunks
        summ_tokenizer: Tokenizer for the model
        summ_model: Model used for generation
        max_chunks: Maximum number of chunks to process per document
    Returns:
        DataFrames for entities and relationships
    """
    debug("Beginning extraction over all chunks with gleaning approach")
    entity_rows = defaultdict(list)
    relationship_rows = defaultdict(list)
    
    for doc_idx, chunks in enumerate(all_chunks):
        debug(f"Extracting from document {doc_idx} with {len(chunks)} chunks")
        
        chunk_count = len(chunks) if max_chunks is None else min(max_chunks, len(chunks))
        processed_chunks = chunks[:chunk_count]
        
        for chunk_idx, chunk in enumerate(processed_chunks):
            debug(f"Processing chunk {chunk_idx+1}/{chunk_count}")
            
            data = extract_elements_with_gleanings(chunk, summ_tokenizer, summ_model)
            
            # Process entities
            for ent in data.get("entities", []):
                name = ent.get("name", "").strip()
                typ = ent.get("type", "").strip()
                desc = ent.get("description", "").strip()
                
                if name and typ and desc:
                    entity_rows["doc_idx"].append(doc_idx)
                    entity_rows["chunk_idx"].append(chunk_idx)
                    entity_rows["Entity"].append(name)
                    entity_rows["Type"].append(typ)
                    entity_rows["Description"].append(desc)
            
            # Process relationships
            for rel in data.get("relationships", []):
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                description = rel.get("relationship", "").strip()
                strength = rel.get("relationship_strength", 5)
                
                if source and target and description:
                    relationship_rows["doc_idx"].append(doc_idx)
                    relationship_rows["chunk_idx"].append(chunk_idx)
                    relationship_rows["Source"].append(source)
                    relationship_rows["Target"].append(target)
                    relationship_rows["Description"].append(description)
                    relationship_rows["Strength"].append(strength)
    
    df_entities = pd.DataFrame(entity_rows)
    df_relationships = pd.DataFrame(relationship_rows)
    
    # Validation check for entities
    if not df_entities.empty:
        unique_entities = df_entities['Entity'].nunique()
        debug(f"Extracted {df_entities.shape[0]} total entities with {unique_entities} unique names")
        
        if unique_entities < 5 and df_entities.shape[0] > 10:
            debug("WARNING: Very low entity diversity. Extraction may still have issues.")
    
    return df_entities, df_relationships

# -----------------------------------------------------------------------------
# 4. Element Summaries
# -----------------------------------------------------------------------------

def summarize_element_instances(df_entities, df_relationships, summ_tokenizer, summ_model):
    """
    Create a descriptive text for each graph element using instance summaries.
    Args: 
        df_entities: DataFrame of extracted entities
        df_relationships: DataFrame of extracted relationships
        summ_tokenizer: Tokenizer for the model
        summ_model: Model used for generation
    Returns:
        entity_summaries: Dictionary of summarized entities
        relationship_summaries: Dictionary of summarized relationships
    """
    debug("Summarizing element instances to create element summaries")
    
    entity_groups = df_entities.groupby('Entity') # summarize duplicates
    entity_summaries = {}
    
    for entity_name, group in entity_groups:
        if group.empty:
            continue
            
        entity_type = group['Type'].mode()[0]
        descriptions = group['Description'].tolist()
        
        if len(descriptions) == 1: # if one description, use that
            entity_summaries[entity_name] = {
                'type': entity_type,
                'description': descriptions[0],
                'doc_idx': group['doc_idx'].iloc[0],
                'chunk_idx': group['chunk_idx'].iloc[0]
            }
        else:
            combined_text = "\n".join([f"- {desc}" for desc in descriptions]) # otherwise if multiple, summarize
            
            prompt = f"""
            You are an academic knowledge summarizer. Create a unified, comprehensive description 
            of the following academic entity from these possibly redundant descriptions.
            
            ENTITY: {entity_name}
            TYPE: {entity_type}
            
            DESCRIPTIONS:
            {combined_text}
            
            UNIFIED DESCRIPTION:
            """
            
            inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            outputs = summ_model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=summ_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3
            )
            
            summary = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated description
            if "UNIFIED DESCRIPTION:" in summary:
                summary = summary.split("UNIFIED DESCRIPTION:", 1)[1].strip()
            
            # Store summarized entity
            entity_summaries[entity_name] = {
                'type': entity_type,
                'description': summary,
                'doc_idx': group['doc_idx'].iloc[0],  
                'chunk_idx': group['chunk_idx'].iloc[0]
            }
    
    debug(f"Created {len(entity_summaries)} entity summaries")
    
    # Same process for relationships
    relationship_keys = []
    for _, row in df_relationships.iterrows():
        source = row['Source']
        target = row['Target']
        rel_key = tuple(sorted([source, target]))
        relationship_keys.append((rel_key, row))
    
    rel_groups = {}
    for rel_key, row in relationship_keys:
        if rel_key not in rel_groups:
            rel_groups[rel_key] = []
        rel_groups[rel_key].append(row)
    
    relationship_summaries = {}
    
    for rel_key, rows in rel_groups.items():
        source, target = rel_key
        if source not in entity_summaries or target not in entity_summaries:
            continue
        
        if len(rows) == 1: # only 1
            row = rows[0]
            rel_id = f"{row['Source']}|{row['Target']}"
            relationship_summaries[rel_id] = {
                'source': row['Source'],
                'target': row['Target'],
                'description': row['Description'],
                'doc_idx': row['doc_idx'],
                'chunk_idx': row['chunk_idx'],
                'strength': row.get('Strength', 5)
            }
        else:
            descriptions = [row['Description'] for row in rows] # multiples
            combined_text = "\n".join([f"- {desc}" for desc in descriptions])
            
            first_row = rows[0]
            
            actual_source = first_row['Source']
            actual_target = first_row['Target']
            
            prompt = f"""
            You are an academic knowledge summarizer. Create a unified description of the relationship 
            between these entities based on possibly redundant or complementary descriptions.
            
            SOURCE ENTITY: {actual_source}
            TARGET ENTITY: {actual_target}
            
            SOURCE DESCRIPTION: {entity_summaries[source]['description'][:100]}...
            TARGET DESCRIPTION: {entity_summaries[target]['description'][:100]}...
            
            RELATIONSHIP DESCRIPTIONS:
            {combined_text}
            
            UNIFIED RELATIONSHIP DESCRIPTION:
            """
            
            inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            outputs = summ_model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=summ_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3
            )
            
            summary = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "UNIFIED RELATIONSHIP DESCRIPTION:" in summary:
                summary = summary.split("UNIFIED RELATIONSHIP DESCRIPTION:", 1)[1].strip()
            
            avg_strength = 5  
            if 'Strength' in rows[0]:
                strengths = [row.get('Strength', 5) for row in rows]
                avg_strength = sum(strengths) / len(strengths)
            
            # summarized relationship
            rel_id = f"{actual_source}|{actual_target}"
            relationship_summaries[rel_id] = {
                'source': actual_source,
                'target': actual_target,
                'description': summary,
                'doc_idx': first_row['doc_idx'],  
                'chunk_idx': first_row['chunk_idx'],
                'strength': avg_strength
            }
    
    debug(f"Created {len(relationship_summaries)} relationship summaries")
    
    return entity_summaries, relationship_summaries

# -----------------------------------------------------------------------------
# 5. Build a Knowledge Graph 
# -----------------------------------------------------------------------------

def build_enriched_knowledge_graph(entity_summaries, relationship_summaries):
    """
    Build a knowledge graph with the summarized elements.
    Args:
        entity_summaries: Dictionary of summarized entities
        relationship_summaries: Dictionary of summarized relationships
    Returns:
        G: graph
    """
    debug("Building enriched knowledge graph from element summaries")
    
    G = nx.Graph()
    
    # Add nodes from entity summaries
    for entity_name, entity_data in entity_summaries.items():
        G.add_node(
            entity_name,
            description=entity_data['description'],
            type=entity_data['type'],
            doc_idx=entity_data['doc_idx'],
            chunk_idx=entity_data['chunk_idx']
        )
    
    # Add edges from relationship summaries
    for rel_id, rel_data in relationship_summaries.items():
        source = rel_data['source']
        target = rel_data['target']
        
        if G.has_node(source) and G.has_node(target): # need to make sure the node exists on both sides of the edge
            G.add_edge(
                source,
                target,
                description=rel_data['description'],
                weight=rel_data['strength'],
                doc_idx=rel_data['doc_idx'],
                chunk_idx=rel_data['chunk_idx']
            )
    
    debug(f"Enriched knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def improved_merge_similar_nodes(G, embedding_tokenizer, embedding_model, threshold=0.85):
    """
    Improved algorithm for merging similar nodes based on semantic similarity.
    Args: 
        G: Graph
        embedding_tokenizer: Tokenizer for the embedding model
        embedding_model: Model used for generating embeddings
        threshold: Similarity threshold for merging nodes
    Returns:
        G_merged: Merged graph
    """
    debug("Merging similar nodes with improved algorithm")
    
    node_list = list(G.nodes())
    
    if len(node_list) <= 1:
        debug("Not enough nodes to merge")
        return G
        
    node_texts = []
    for node in node_list:
        node_data = G.nodes[node]
        node_text = f"{node} - {node_data.get('type', 'UNKNOWN')} - {node_data.get('description', '')}"
        node_texts.append(node_text)
    
    embeddings = []
    batch_size = 16  # process in batches for limitations
    
    for i in range(0, len(node_texts), batch_size):
        batch = node_texts[i:i+batch_size]
        batch_embeddings = []
        
        for text in batch:
            inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = embedding_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()[0]
            batch_embeddings.append(embedding)
            
        embeddings.extend(batch_embeddings)
        
    embeddings = np.array(embeddings)
    
    similarity_matrix = cosine_similarity(embeddings)
    
    merge_candidates = [] # pairs to merge because too similar
    
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            if similarity_matrix[i, j] >= threshold:
                node_i = node_list[i]
                node_j = node_list[j]
                type_i = G.nodes[node_i].get('type', '')
                type_j = G.nodes[node_j].get('type', '')
                
                if type_i == type_j or type_i == '' or type_j == '':
                    score_i = G.degree(node_i) * G.nodes[node_i].get('frequency', 1)
                    score_j = G.degree(node_j) * G.nodes[node_j].get('frequency', 1)
                    
                    if score_i >= score_j: # keep the node with higher score
                        merge_candidates.append((node_i, node_j))
                    else:
                        merge_candidates.append((node_j, node_i))
    
    # Merge most similar pairs first by similarity score
    merge_candidates.sort(key=lambda pair: similarity_matrix[node_list.index(pair[0]), node_list.index(pair[1])], 
                          reverse=True)
    
    G_merged = copy.deepcopy(G)
    merged_nodes = set()
    
    for keep_node, merge_node in merge_candidates:
        if keep_node in merged_nodes or merge_node in merged_nodes:
            continue
            
        if G_merged.has_node(keep_node) and G_merged.has_node(merge_node):
            keep_attrs = G_merged.nodes[keep_node]
            merge_attrs = G_merged.nodes[merge_node]
            
            if 'description' in merge_attrs and 'description' in keep_attrs:
                if merge_attrs['description'] not in keep_attrs['description']:
                    keep_attrs['description'] = f"{keep_attrs['description']}; {merge_attrs['description']}"
            
            keep_attrs['frequency'] = keep_attrs.get('frequency', 1) + merge_attrs.get('frequency', 1)
            
            G_merged = nx.contracted_nodes(G_merged, keep_node, merge_node, self_loops=False)
            merged_nodes.add(merge_node)
    
    debug(f"After improved merging, graph has {G_merged.number_of_nodes()} nodes")
    debug(f"Merged {len(merged_nodes)} nodes")
    
    return G_merged

# -----------------------------------------------------------------------------
# 6. Community Detection and Summarization 
# -----------------------------------------------------------------------------
def detect_hierarchical_communities(G, max_levels=2):
    """
    Detect hierarchical communities using the Leiden algorithm.
    Args: 
        G: Graph
        max_levels: Maximum number of levels to detect
    Returns: 
        hierarchical_partitions: Dictionary of communities at different levels
        community_hierarchy: Dictionary of parent-child relationships between communities
    """
    debug("Detecting hierarchical communities using Leiden algorithm")
    
    # Convert NetworkX graph to igraph
    edges = list(G.edges())
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
    
    g = ig.Graph(n=len(nodes), edges=edge_list, directed=False)
    g.vs["name"] = nodes
    
    edge_weights = None
    if nx.get_edge_attributes(G, 'weight'):
        edge_weights = [G[u][v].get('weight', 1.0) for u, v in edges]
        g.es["weight"] = edge_weights
    
    hierarchical_partitions = {}
    
    resolution0 = 0.5 # lower resolution for smaller communitites
    partition0 = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution0,
        weights=edge_weights
    )
    
    modularity0 = g.modularity(partition0.membership, weights=edge_weights)
    debug(f"Level 0: {len(set(partition0.membership))} communities, modularity={modularity0:.4f}")
    
    level0_dict = {}
    for idx, cluster_id in enumerate(partition0.membership):
        level0_dict[nodes[idx]] = cluster_id
    
    hierarchical_partitions[0] = level0_dict
    
    resolutions = [1.0, 2.0] # increase resolution for larger communities
    
    for level, resolution in enumerate(resolutions[:max_levels-1], 1):
        partition = leidenalg.find_partition(
            g, 
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights=edge_weights
        )
        
        modularity = g.modularity(partition.membership, weights=edge_weights)
        num_communities = len(set(partition.membership))
        debug(f"Level {level}: {num_communities} communities, modularity={modularity:.4f}")
        
        level_dict = {}
        for idx, cluster_id in enumerate(partition.membership):
            node = nodes[idx]
            parent_id = hierarchical_partitions[level-1][node] # hierarchical community ID: parentID_childID
            hierarchical_id = f"{parent_id}_{cluster_id}"
            level_dict[node] = hierarchical_id
        
        hierarchical_partitions[level] = level_dict
    
    community_hierarchy = {}
    
    for level in range(1, len(hierarchical_partitions)):
        parent_level = level - 1
        child_level = level
        
        parent_communities = set(hierarchical_partitions[parent_level].values())
        child_communities = set(hierarchical_partitions[child_level].values())
        
        for node in G.nodes():
            if node in hierarchical_partitions[child_level]:
                child_comm = hierarchical_partitions[child_level][node] # mapping child and parent
                parent_comm = hierarchical_partitions[parent_level][node]
                
                if child_comm not in community_hierarchy:
                    community_hierarchy[child_comm] = parent_comm
    
    debug(f"Built hierarchical community structure with {len(hierarchical_partitions)} levels")
    
    return hierarchical_partitions, community_hierarchy

def get_community_nodes(partition, community_id):
    """
    Get all nodes belonging to a specific community.
    Args: 
        partition: Dictionary of node to community ID mapping
        community_id: Community ID to filter by
    Returns:    
        nodes: List of nodes in the specified community
    """
    nodes = []
    for node, comm_id in partition.items():
        if comm_id == community_id:
            nodes.append(node)
    return nodes

def summarize_community_prioritized(G, community_nodes, entity_summaries, relationship_summaries, 
                                    all_chunks, summ_tokenizer, summ_model, max_tokens=1800):
    """
    Summarize a community using prioritized element summaries.
    Args:
        G: Graph
        community_nodes: List of nodes in the community
        entity_summaries: Dictionary of summarized entities
        relationship_summaries: Dictionary of summarized relationships
        all_chunks: List of document chunks
        summ_tokenizer: Tokenizer for the model
        summ_model: Model used for generation
        max_tokens: Maximum number of tokens for the summary
    Returns:
        summary: Prioritized summary of the community
    """
    debug(f"Generating prioritized summary for community with {len(community_nodes)} nodes")
    
    if not community_nodes:
        return "No nodes in this community."
    
    subgraph = G.subgraph(community_nodes)
    node_importance = {node: G.degree(node) for node in community_nodes}
    
    community_edges = list(subgraph.edges())
    edge_importance = {}
    for u, v in community_edges:
        edge_importance[(u, v)] = node_importance[u] + node_importance[v]
    
    sorted_edges = sorted(community_edges, key=lambda e: edge_importance[e], reverse=True)
    
    prioritized_info = []
    current_tokens = 0
    
    added_nodes = set()
    added_edges = set()
    
    top_nodes = sorted(community_nodes, key=lambda n: node_importance[n], reverse=True)[:3] # top 3 nodes
    
    for node in top_nodes:
        if node in entity_summaries:
            node_info = f"ENTITY: {node}\nTYPE: {entity_summaries[node]['type']}\nDESCRIPTION: {entity_summaries[node]['description']}\n\n"
            node_tokens = len(summ_tokenizer.encode(node_info))
            
            if current_tokens + node_tokens <= max_tokens:
                prioritized_info.append(node_info)
                current_tokens += node_tokens
                added_nodes.add(node)
    
    for u, v in sorted_edges:
        if current_tokens >= max_tokens:
            break
            
        # Create relationship ID
        rel_id1 = f"{u}|{v}"
        rel_id2 = f"{v}|{u}"
        rel_id = rel_id1 if rel_id1 in relationship_summaries else rel_id2 if rel_id2 in relationship_summaries else None
        
        if rel_id:
            rel_data = relationship_summaries[rel_id]
            relationship_info = f"RELATIONSHIP: {rel_data['source']} → {rel_data['target']}\nDESCRIPTION: {rel_data['description']}\n\n"
            rel_tokens = len(summ_tokenizer.encode(relationship_info))
            
            source_info = ""
            target_info = ""
            source_tokens = 0
            target_tokens = 0
            
            if rel_data['source'] not in added_nodes and rel_data['source'] in entity_summaries:
                source_info = f"ENTITY: {rel_data['source']}\nTYPE: {entity_summaries[rel_data['source']]['type']}\nDESCRIPTION: {entity_summaries[rel_data['source']]['description']}\n\n"
                source_tokens = len(summ_tokenizer.encode(source_info))
            
            if rel_data['target'] not in added_nodes and rel_data['target'] in entity_summaries:
                target_info = f"ENTITY: {rel_data['target']}\nTYPE: {entity_summaries[rel_data['target']]['type']}\nDESCRIPTION: {entity_summaries[rel_data['target']]['description']}\n\n"
                target_tokens = len(summ_tokenizer.encode(target_info))
            
            total_tokens = rel_tokens + source_tokens + target_tokens
            
            if current_tokens + total_tokens <= max_tokens:
                if source_info:
                    prioritized_info.append(source_info)
                    added_nodes.add(rel_data['source'])
                
                if target_info:
                    prioritized_info.append(target_info)
                    added_nodes.add(rel_data['target'])
                
                prioritized_info.append(relationship_info)
                added_edges.add((u, v))
                
                current_tokens += total_tokens
            else:
                if current_tokens + rel_tokens <= max_tokens:
                    prioritized_info.append(relationship_info)
                    added_edges.add((u, v))
                    current_tokens += rel_tokens
                else:
                    break
    
    # source text chunks added if we have space
    if current_tokens < max_tokens:
        chunk_texts = set()
        
        for node in community_nodes:
            if node in entity_summaries:
                d_idx = entity_summaries[node]['doc_idx']
                c_idx = entity_summaries[node]['chunk_idx']
                
                if d_idx < len(all_chunks) and c_idx < len(all_chunks[d_idx]):
                    chunk = all_chunks[d_idx][c_idx]
                    chunk_texts.add(chunk)
        
        for chunk in list(chunk_texts)[:3]:  # limit to 3 chunks
            chunk_info = f"SOURCE TEXT:\n{chunk}\n\n"
            chunk_tokens = len(summ_tokenizer.encode(chunk_info))
            
            if current_tokens + chunk_tokens <= max_tokens:
                prioritized_info.append(chunk_info)
                current_tokens += chunk_tokens
            else:
                break
    
    community_info = "\n".join(prioritized_info)
    
    prompt = """
    You are an academic knowledge synthesizer. Create a comprehensive summary of this research community
    based on the entities, relationships, and source text provided.
    
    Your summary should:
    1. Identify the main research concepts and how they relate to each other
    2. Explain the significance of these concepts in the academic context
    3. Highlight any important methodologies, variables, or findings
    4. Integrate all information into a coherent narrative
    
    COMMUNITY INFORMATION:
    {}
    
    SUMMARY:
    """
    
    prompt = prompt.format(community_info)
    
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.4
    )
    
    summary = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "SUMMARY:" in summary:
        summary = summary.split("SUMMARY:", 1)[1].strip()
    
    debug(f"Generated community summary of {len(summary)} characters")
    return summary

def summarize_hierarchical_communities(G, hierarchical_partitions, community_hierarchy, entity_summaries, 
                                       relationship_summaries, all_chunks, summ_tokenizer, summ_model):
    """
    Build summaries for hierarchical communities at different levels.
    Args:
        G: Graph
        hierarchical_partitions: Dictionary of communities at different levels
        community_hierarchy: Dictionary of parent-child relationships between communities
        entity_summaries: Dictionary of summarized entities
        relationship_summaries: Dictionary of summarized relationships
        all_chunks: List of document chunks
        summ_tokenizer: Tokenizer for the model
        summ_model: Model used for generation
    Returns:
        level_summaries: Dictionary of community summaries at different levels
    """
    debug("Building hierarchical community summaries")
    
    leaf_level = max(hierarchical_partitions.keys())
    
    level_summaries = {}
    
    leaf_partitions = hierarchical_partitions[leaf_level]
    unique_leaf_communities = set(leaf_partitions.values())
    
    debug(f"Summarizing {len(unique_leaf_communities)} leaf-level communities at level {leaf_level}")
    
    leaf_summaries = {}
    
    for community_id in unique_leaf_communities:
        community_nodes = get_community_nodes(leaf_partitions, community_id)
        
        summary = summarize_community_prioritized(
            G, community_nodes, entity_summaries, relationship_summaries, 
            all_chunks, summ_tokenizer, summ_model
        )
        
        leaf_summaries[community_id] = summary
    
    level_summaries[leaf_level] = leaf_summaries
    
    for level in range(leaf_level - 1, -1, -1):
        debug(f"Summarizing communities at level {level}")
        
        partitions = hierarchical_partitions[level]
        unique_communities = set(partitions.values())
        
        level_summaries[level] = {}
        
        for community_id in unique_communities:
            community_nodes = get_community_nodes(partitions, community_id)
            child_communities = []
            
            for child_id, parent_id in community_hierarchy.items():
                if parent_id == community_id:
                    child_communities.append(child_id)
            
            community_subgraph = G.subgraph(community_nodes)
            total_tokens = 0
            
            for node in community_nodes:
                if node in entity_summaries:
                    node_info = f"ENTITY: {node}\nTYPE: {entity_summaries[node]['type']}\nDESCRIPTION: {entity_summaries[node]['description']}\n\n"
                    total_tokens += len(summ_tokenizer.encode(node_info))
            
            for u, v in community_subgraph.edges(): # control edges 
                rel_id1 = f"{u}|{v}"
                rel_id2 = f"{v}|{u}"
                rel_id = rel_id1 if rel_id1 in relationship_summaries else rel_id2 if rel_id2 in relationship_summaries else None
                
                if rel_id:
                    rel_data = relationship_summaries[rel_id]
                    rel_info = f"RELATIONSHIP: {rel_data['source']} → {rel_data['target']}\nDESCRIPTION: {rel_data['description']}\n\n"
                    total_tokens += len(summ_tokenizer.encode(rel_info))
            
            if total_tokens <= 1800: # if the community is small enough direct summarization
                summary = summarize_community_prioritized(
                    G, community_nodes, entity_summaries, relationship_summaries, 
                    all_chunks, summ_tokenizer, summ_model
                )
            else:
                child_texts = [] # otherwise summarize the children
                
                child_tokens = {}
                for child_id in child_communities:
                    child_summary = level_summaries[level+1].get(child_id, "")
                    child_tokens[child_id] = len(summ_tokenizer.encode(child_summary))
                
                sorted_children = sorted(child_communities, key=lambda c: child_tokens.get(c, 0), reverse=True)
                
                current_tokens = 0
                max_tokens = 1800
                
                for child_id in sorted_children:
                    child_summary = level_summaries[level+1].get(child_id, "")
                    child_token_count = child_tokens.get(child_id, 0)
                    
                    if current_tokens + child_token_count <= max_tokens:
                        child_texts.append(f"SUBCOMMUNITY SUMMARY: {child_summary}")
                        current_tokens += child_token_count
                    else:
                        break
                
                combined_text = "\n\n".join(child_texts)
                
                prompt = f"""
                You are an academic knowledge synthesizer. Create a comprehensive summary that integrates 
                multiple subcommunity summaries into a coherent overview of a research area.
                
                Your summary should:
                1. Identify the common themes across subcommunities
                2. Highlight the most important concepts, methodologies, and findings
                3. Explain the relationships between different subcommunities
                4. Present a unified view of this larger research area
                
                SUBCOMMUNITY INFORMATION:
                {combined_text}
                
                INTEGRATED SUMMARY:
                """
                
                inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                outputs = summ_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    pad_token_id=summ_tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.4
                )
                
                summary = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if "INTEGRATED SUMMARY:" in summary:
                    summary = summary.split("INTEGRATED SUMMARY:", 1)[1].strip()
            
            level_summaries[level][community_id] = summary
    
    debug(f"Generated summaries for {sum(len(s) for s in level_summaries.values())} communities across {len(level_summaries)} levels")
    
    return level_summaries

# -----------------------------------------------------------------------------
# 7. Query Phase: Map–Reduce Over Community Summaries
# -----------------------------------------------------------------------------
def generate_partial_answer_with_score(question, summary_text, summ_tokenizer, summ_model):
    """
    Generate a partial answer with score to the query based on a community summary.
    Args:
        question: User's question
        summary_text: Community summary text
        summ_tokenizer: Tokenizer for the model
        summ_model: Model used for generation
    Returns:
        answer: Generated partial answer
        score: Helpfulness score (0-100)
    """
    debug("Generating partial answer with helpfulness score")
    
    system_instruction = """
    You are a specialized academic research assistant. Extract information from the community 
    summary that directly answers the user's question.
    
    Your response must include:
    1. A detailed answer based ONLY on information in the community summary
    2. A helpfulness score from 0-100 indicating how useful this information is for answering the question
       (where 0 = completely irrelevant, and 100 = perfectly answers the question)
    
    If the community summary contains no relevant information, assign a score of 0.
    """
    
    prompt = f"""
    {system_instruction}
    
    COMMUNITY SUMMARY:
    {summary_text}
    
    USER QUESTION:
    {question}
    
    ANSWER:
    
    HELPFULNESS SCORE (0-100):
    """
    
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.3
    )
    
    response = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer = ""
    score = 0
    
    if "ANSWER:" in response:
        parts = response.split("ANSWER:", 1)[1].split("HELPFULNESS SCORE", 1)
        if len(parts) > 0:
            answer = parts[0].strip()
    
    if "HELPFULNESS SCORE (0-100):" in response:
        score_text = response.split("HELPFULNESS SCORE (0-100):", 1)[1].strip()
        score_match = re.search(r'\d+', score_text)
        if score_match:
            try:
                score = int(score_match.group(0))
                score = max(0, min(100, score)) # valide range of score
            except:
                score = 0
    
    debug(f"Generated partial answer with helpfulness score: {score}")
    return answer, score

def shuffle_and_chunk_community_summaries(community_summaries, chunk_size=3):
    """
    Randomly shuffle community summaries and divide into chunks of specified size.
    Args:
        community_summaries: Dictionary of community summaries
        chunk_size: Size of each chunk
    Returns:
        summary_chunks: List of chunks of community summaries
    """
    debug("Shuffling and chunking community summaries")
    
    summary_items = list(community_summaries.items())
    
    random.shuffle(summary_items)
    
    summary_chunks = []
    for i in range(0, len(summary_items), chunk_size):
        chunk = summary_items[i:i+chunk_size]
        summary_chunks.append(dict(chunk))
    
    debug(f"Created {len(summary_chunks)} chunks from {len(summary_items)} community summaries")
    return summary_chunks

def combine_answers_with_scores(question, scored_answers, summ_tokenizer, summ_model):
    """
    Combine partial answers into a final answer, prioritizing the most helpful answers.
    Args:
        question: User's question
        scored_answers: List of tuples (answer, score)
        summ_tokenizer: Tokenizer for the model
        summ_model: Model used for generation
    Returns:
        final_answer: Combined answer
    """
    debug("Combining partial answers based on helpfulness scores")
    
    if not scored_answers:
        return "No relevant information found."
    
    sorted_answers = sorted(scored_answers, key=lambda x: x[1], reverse=True)
    
    filtered_answers = [(ans, score) for ans, score in sorted_answers if score > 0]
    
    if not filtered_answers:
        return "No relevant information found in the dataset for this question."
    
    selected_answers = []
    current_tokens = 0
    max_tokens = 1800
    
    for answer, score in filtered_answers:
        answer_info = f"[Score: {score}/100] {answer}"
        tokens = len(summ_tokenizer.encode(answer_info))
        
        if current_tokens + tokens <= max_tokens:
            selected_answers.append(answer_info)
            current_tokens += tokens
        else:
            break
    
    combined_text = "\n\n".join(selected_answers)
    
    prompt = f"""
    You are an academic research assistant tasked with synthesizing multiple partial answers 
    into a comprehensive final answer.
    
    Each partial answer has a helpfulness score (0-100) indicating its relevance to the question.
    Focus more on information from higher-scored answers but integrate all relevant information.
    
    Your final answer should:
    1. Present a complete, coherent response to the question
    2. Prioritize information from the most relevant sources
    3. Avoid contradictions and resolve any conflicting information
    4. Use academic language appropriate to the subject matter
    
    USER QUESTION:
    {question}
    
    PARTIAL ANSWERS (with relevance scores):
    {combined_text}
    
    FINAL COMPREHENSIVE ANSWER:
    """
    
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.4
    )
    
    final_answer = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "FINAL COMPREHENSIVE ANSWER:" in final_answer:
        final_answer = final_answer.split("FINAL COMPREHENSIVE ANSWER:", 1)[1].strip()
    
    debug("Generated final combined answer")
    return final_answer

def improved_query_processing(question, level_summaries, summ_tokenizer, summ_model, sent_transformer, G=None):
    """
    Process a query using the hierarchical community summaries with shuffling
    and map-reduce approach.
    Args:
        question: User's question
        level_summaries: Dictionary of community summaries at different levels
        summ_tokenizer: Tokenizer for the model
        summ_model: Model used for generation
        sent_transformer: Sentence transformer for semantic similarity
        G: Graph (optional)
    Returns:
        final_answer: Combined answer to the query
    """
    debug(f"Processing query with shuffling and map-reduce: {question}")
    
    question_lower = question.lower()
    
    general_patterns = [
        "main topics", "overview", "summary", "what is the data about",
        "high level", "general themes", "main findings"
    ]
    
    specific_patterns = [
        "specific", "detail", "exactly", "precisely", "tell me more about",
        "what is the relationship between", "how does"
    ]
    
    if any(pattern in question_lower for pattern in general_patterns):
        best_level = 0  # Most general level
    elif any(pattern in question_lower for pattern in specific_patterns):
        best_level = max(level_summaries.keys())  # Most specific level
    else:
        if 1 in level_summaries:
            best_level = 1
        else:
            best_level = max(level_summaries.keys())
    
    debug(f"Selected community level {best_level} for query")
    
    # top topics
    top_words = ["top", "main", "key", "primary", "important"]
    topic_words = ["topic", "topics", "entity", "entities", "subject", "theme", "concept"]
    
    is_top_topics_question = (
        any(word in question_lower for word in top_words) and 
        any(word in question_lower for word in topic_words)
    )
    
    if is_top_topics_question and G is not None:
        debug("Generating direct answer for top topics question")
        
        type_weights = {
            "RESEARCH_CONCEPT": 2.0,
            "FINDING": 1.7,
            "VARIABLE": 1.5,
            "METHODOLOGY": 1.3,
            "ORGANIZATION": 1.0
        }
        
        centrality = {}
        base_centrality = nx.degree_centrality(G)
        
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "ORGANIZATION")
            weight = type_weights.get(node_type, 1.0)
            centrality[node] = base_centrality[node] * weight
        
        top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        entity_types = {}
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            entity_types[node_type] = entity_types.get(node_type, 0) + 1
        
        community_counts = {level: len(summaries) for level, summaries in level_summaries.items()}
        
        direct_answer = f"""
        Based on my analysis of the knowledge graph, the top topics/entities in the dataset (by centrality) are:
        
        1. {top_entities[0][0] if len(top_entities) > 0 else 'N/A'}
        2. {top_entities[1][0] if len(top_entities) > 1 else 'N/A'} 
        3. {top_entities[2][0] if len(top_entities) > 2 else 'N/A'}
        4. {top_entities[3][0] if len(top_entities) > 3 else 'N/A'}
        5. {top_entities[4][0] if len(top_entities) > 4 else 'N/A'}
        
        These represent the most central concepts in the knowledge graph, with {G.number_of_nodes()} total entities 
        organized into {community_counts.get(best_level, 0)} communities at the selected hierarchy level.
        
        The dataset contains these entity types: 
        {', '.join([f"{t} ({c})" for t, c in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)])}
        
        The knowledge graph has a hierarchical community structure with {len(level_summaries)} levels, 
        from general topics (level 0) to specific research areas (level {max(level_summaries.keys())}).
        """
        
        debug("Direct answer generated for top topics question")
        return direct_answer.strip()
    
    community_summaries = level_summaries.get(best_level, {})
    
    if not community_summaries:
        return "No community summaries available at the selected hierarchy level."
    
    query_embedding = sent_transformer.encode(question)
    scored_communities = []
    
    for comm_id, summary in community_summaries.items():
        summary_embedding = sent_transformer.encode(summary)
        similarity = cosine_similarity([query_embedding], [summary_embedding])[0][0] # similarity
        scored_communities.append((comm_id, summary, similarity))
    
    sorted_communities = sorted(scored_communities, key=lambda x: x[2], reverse=True)
    
    # Filter communities based on a threshold
    threshold = 0.4
    selected_communities = {}
    
    for comm_id, summary, score in sorted_communities:
        if score >= threshold or len(selected_communities) < 3:  # at least 3 communities
            selected_communities[comm_id] = summary
        
        if len(selected_communities) >= 10:
            break
    
    if not selected_communities:
        return "I couldn't find specific information to answer that question in the dataset."
    
    summary_chunks = shuffle_and_chunk_community_summaries(selected_communities)
    
    all_scored_answers = []
    for chunk in summary_chunks:
        for comm_id, summary in chunk.items():
            answer, score = generate_partial_answer_with_score(question, summary, summ_tokenizer, summ_model)
            if score > 0: 
                all_scored_answers.append((answer, score))
    
    final_answer = combine_answers_with_scores(question, all_scored_answers, summ_tokenizer, summ_model)
    
    debug("Map-reduce query processing completed")
    return final_answer

# -----------------------------------------------------------------------------
# 8. Main & Interactive Query Interface
# -----------------------------------------------------------------------------
def main():
    debug("Starting enhanced Graph RAG chatbot with hierarchical communities...")
    
    debug("Loading models...")
    summ_tokenizer, summ_model = load_summarization_model()
    sent_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    global embedding_tokenizer, embedding_model
    embedding_tokenizer = SpecTokenizer.from_pretrained("allenai/specter2_base")
    embedding_model = SpecAdapterModel.from_pretrained("allenai/specter2_base")
    embedding_model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

    pdf_paths = [
        "papers-testing/7294.pdf"
    ]
    pdf_paths_with_idx = [(i, path) for i, path in enumerate(pdf_paths)]
    
    debug("Processing PDFs into text chunks...")
    all_chunks, doc_citations = process_all_pdfs(pdf_paths_with_idx)
    
    # 1
    if os.path.exists(ENTITIES_CSV) and os.path.exists(RELATIONSHIPS_CSV):
        debug("Loading cached extracted entities and relationships.")
        df_entities = pd.read_csv(ENTITIES_CSV)
        df_relationships = pd.read_csv(RELATIONSHIPS_CSV)
    else:
        debug("Extracting graph elements with multi-round gleanings approach...")
        df_entities, df_relationships = extract_graph_elements_improved(all_chunks, summ_tokenizer, summ_model)
        df_entities.to_csv(ENTITIES_CSV, index=False)
        df_relationships.to_csv(RELATIONSHIPS_CSV, index=False)
    debug(f"Extracted {df_entities.shape[0]} entities and {df_relationships.shape[0]} relationships.")
    
    # 2
    debug("Creating element summaries from element instances...")
    entity_summaries, relationship_summaries = summarize_element_instances(
        df_entities, df_relationships, summ_tokenizer, summ_model
    )
    
    # 3
    if os.path.exists(GRAPH_PKL):
        debug("Loading cached knowledge graph.")
        with open(GRAPH_PKL, "rb") as f:
            G = pickle.load(f)
    else:
        debug("Building knowledge graph from element summaries...")
        G = build_enriched_knowledge_graph(entity_summaries, relationship_summaries)
        with open(GRAPH_PKL, "wb") as f:
            pickle.dump(G, f)
    debug(f"Knowledge Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # 4
    debug("Merging similar nodes with improved algorithm...")
    G_merged = improved_merge_similar_nodes(G, embedding_tokenizer, embedding_model, threshold=0.85)
    debug(f"Merged Graph has {G_merged.number_of_nodes()} nodes and {G_merged.number_of_edges()} edges.")
    
    # 5
    debug("Detecting hierarchical communities...")
    hierarchical_partitions, community_hierarchy = detect_hierarchical_communities(G_merged, max_levels=2)
    
    for level, partition in hierarchical_partitions.items():
        unique_communities = set(partition.values())
        debug(f"Level {level}: {len(unique_communities)} communities")
    
    # 6
    community_level_summaries_file = "/work/Chatbot-in-academia/hierarchical_community_summaries.pkl"
    
    if os.path.exists(community_level_summaries_file):
        debug("Loading cached hierarchical community summaries.")
        with open(community_level_summaries_file, "rb") as f:
            level_summaries = pickle.load(f)
    else:
        debug("Building hierarchical community summaries with prioritization...")
        level_summaries = summarize_hierarchical_communities(
            G_merged, hierarchical_partitions, community_hierarchy,
            entity_summaries, relationship_summaries, all_chunks,
            summ_tokenizer, summ_model
        )
        
        with open(community_level_summaries_file, "wb") as f:
            pickle.dump(level_summaries, f)
    
    for level, summaries in level_summaries.items():
        debug(f"Level {level}: {len(summaries)} community summaries")
    
    # Interface
    print("\nEnhanced Graph RAG Chatbot is ready! Type 'exit' to quit.")
    print("This implementation includes:")
    print("- Multi-round gleanings for improved entity extraction")
    print("- Element summarization before community summarization")
    print("- Hierarchical community detection and utilization")
    print("- Prioritized community summarization mechanism")
    print("- Random shuffling of summaries for query processing")
    print("- Map-reduce approach for combining partial answers")
    
    while True:
        query = input("\nUser: ")
        if query.lower() == "exit":
            print("Exiting. Goodbye.")
            break
        
        answer = improved_query_processing(
            query, level_summaries, summ_tokenizer, summ_model, 
            sent_transformer, G_merged
        )
        
        print("\nBot:", answer)

if __name__ == "__main__":
    main()