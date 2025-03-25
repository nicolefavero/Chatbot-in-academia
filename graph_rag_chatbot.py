############################################################################################################
#  CHATBOT WITH Llama 3.3 70B FOR QUESTION ANSWERING & Llama-2-7B FOR EXTRACTION OF ENTITIES  #
############################################################################################################

'''
Objective: creation of a Chatbot leveraging the Llama 3.3 70B model for question answering and the Llama-2-7B model for the extraction of entities and relationships 
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
import random # Added for shuffling summaries
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Debug flag
DEBUG = True

def debug(msg):
    if DEBUG:
        print("DEBUG:", msg)

# File paths for caching intermediate outputs
ENTITIES_CSV = "/work/Chatbot-in-academia/extracted_entities.csv"
RELATIONSHIPS_CSV = "/work/Chatbot-in-academia/extracted_relationships.csv"
GRAPH_PKL = "/work/Chatbot-in-academia/knowledge_graph.pkl"
COMMUNITY_SUMMARIES_PKL = "/work/Chatbot-in-academia/community_summaries.pkl"

def extract_and_fix_json(text):
    """
    Extract JSON from text and fix common JSON formatting issues.
    Returns parsed data or empty lists for entities and relationships.
    """
    import re
    import json
    
    debug("Attempting to extract and fix JSON")
    
    # Try to find a JSON array in the text
    json_pattern = r'\[\s*\{.*?\}\s*\]'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if not matches:
        debug("No JSON array found in response, printing raw output")
        debug(f"Raw output first 500 chars: {text[:500]}")
        return {"entities": [], "relationships": []}
    
    # Try each match until we find valid JSON
    for json_str in matches:
        try:
            # Try to parse the JSON directly
            data = json.loads(json_str)
            debug(f"Successfully parsed JSON with {len(data)} items")
            
            # Separate entities and relationships
            entities = []
            relationships = []
            
            for item in data:
                if isinstance(item, dict):
                    if "source" in item and "target" in item:
                        relationships.append(item)
                    elif "name" in item and "type" in item:
                        entities.append(item)
            
            return {"entities": entities, "relationships": relationships}
            
        except json.JSONDecodeError as e:
            debug(f"JSON parse error: {e}")
            try:
                # Fix common JSON errors
                fixed_str = json_str
                
                # 1. Fix trailing commas
                fixed_str = re.sub(r',\s*}', '}', fixed_str)
                fixed_str = re.sub(r',\s*\]', ']', fixed_str)
                
                # 2. Fix missing quotes around keys
                fixed_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', fixed_str)
                
                # 3. Replace single quotes with double quotes
                fixed_str = fixed_str.replace("'", '"')
                
                # Try parsing again
                data = json.loads(fixed_str)
                debug(f"Successfully parsed fixed JSON with {len(data)} items")
                
                # Separate entities and relationships
                entities = []
                relationships = []
                
                for item in data:
                    if isinstance(item, dict):
                        if "source" in item and "target" in item:
                            relationships.append(item)
                        elif "name" in item and "type" in item:
                            entities.append(item)
                
                return {"entities": entities, "relationships": relationships}
                
            except Exception as e2:
                debug(f"Failed to fix JSON: {e2}")
    
    debug("All JSON extraction attempts failed")
    return {"entities": [], "relationships": []}

# -----------------------------------------------------------------------------
# 1. Load Models
# -----------------------------------------------------------------------------

def load_summarization_model():
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

# STAGE 1: Entity Extraction Prompt - Focuses ONLY on entity extraction
ENTITY_EXTRACTION_PROMPT = r"""
-Goal-
Extract ONLY academic entities that are EXPLICITLY present in the INPUT TEXT provided below. 
DO NOT extract any relationships between entities at this stage - focus solely on identifying entities.

-Extraction Instructions-
1. First, carefully read the INPUT TEXT at the bottom of this prompt.
2. Identify academic entities that are ACTUALLY present in the INPUT TEXT (not from examples).

-Expected Output Format-
For each entity found in the INPUT TEXT, extract:
- entity_name: Academic term as it appears in the text (capitalized)
- entity_type: Categorize as one of: {entity_types}
- entity_description: Brief definition based ONLY on how it's described in the INPUT TEXT

Format each entity as:
{{"name": "ENTITY_NAME", "type": "ENTITY_TYPE", "description": "DESCRIPTION_FROM_TEXT"}}

-CRITICAL INSTRUCTION-
YOU MUST ONLY extract entities that are EXPLICITLY mentioned in the INPUT TEXT below. 
DO NOT copy entities from format examples unless they genuinely appear in the actual input text.
DO NOT extract relationships between entities at this stage - that will be done separately.

-JSON Output Format Requirements-
- Use ONLY double quotes (") for strings, never single quotes (')
- Do not include trailing commas
- Ensure all brackets and braces are properly closed
- Output ONLY a JSON array with entities, like: [{{"name": "ENTITY1", "type": "TYPE1", "description": "DESC1"}}, {{"name": "ENTITY2", "type": "TYPE2", "description": "DESC2"}}]

-Domain Specific Examples-
These brief examples show the expected format, but DO NOT EXTRACT THESE ENTITIES unless they appear in your INPUT TEXT:

Example 1:
Input: "Machine learning algorithms like neural networks require large datasets for training."
[
  {{"name": "MACHINE LEARNING", "type": "RESEARCH_CONCEPT", "description": "Algorithms that can learn from data"}},
  {{"name": "NEURAL NETWORKS", "type": "METHODOLOGY", "description": "A type of machine learning algorithm"}},
  {{"name": "LARGE DATASETS", "type": "RESEARCH_CONCEPT", "description": "Collections of data required for training machine learning algorithms"}}
]

-INPUT TEXT TO ANALYZE-
entity_types: {entity_types}
text: {input_text}

-OUTPUT (only entities from THIS input text)-
"""

# STAGE 2: Relationship Extraction Prompt - Takes entities from Stage 1 as input
RELATIONSHIP_EXTRACTION_PROMPT = r"""
-Goal-
This is the SECOND STAGE of knowledge extraction. Given the INPUT TEXT and a list of ALREADY IDENTIFIED ENTITIES from the first stage, 
determine all relationships between these entities that are EXPLICITLY stated in the text.

-Extraction Instructions-
1. Review both the INPUT TEXT and the IDENTIFIED ENTITIES from the first stage.
2. For each pair of entities, determine if they have a clear relationship in the text.
3. ONLY create relationships between entities in the provided list.

-Expected Output Format-
For each relationship between entities, extract:
- source_entity: Name of the source entity (must match one of the IDENTIFIED ENTITIES exactly)
- target_entity: Name of the target entity (must match one of the IDENTIFIED ENTITIES exactly)
- relationship_description: How these concepts relate according to the INPUT TEXT
- relationship_strength: Score from 1-10 based on how explicitly the relationship is stated

Format each relationship as:
{{"source": "SOURCE_ENTITY", "target": "TARGET_ENTITY", "relationship": "RELATIONSHIP_DESCRIPTION", "relationship_strength": NUMBER}}

-CRITICAL INSTRUCTION-
ONLY create relationships between entities that are in the provided IDENTIFIED ENTITIES list.
ONLY identify relationships that are EXPLICITLY stated in the INPUT TEXT.
DO NOT create new entities that weren't identified in the first stage.

-JSON Output Format Requirements-
- Use ONLY double quotes (") for strings, never single quotes (')
- Do not include trailing commas
- Ensure all brackets and braces are properly closed
- Output ONLY a JSON array with relationships, like: [{{"source": "ENTITY1", "target": "ENTITY2", "relationship": "DESC", "relationship_strength": 7}}]

-Domain Specific Examples-
These brief examples show the expected format, but DO NOT EXTRACT THESE RELATIONSHIPS unless they appear in your INPUT TEXT:

Example 1:
Input: "Machine learning algorithms like neural networks require large datasets for training."
Identified Entities: ["MACHINE LEARNING", "NEURAL NETWORKS", "LARGE DATASETS"]
[
  {{"source": "NEURAL NETWORKS", "target": "LARGE DATASETS", "relationship": "Neural networks require large datasets for training", "relationship_strength": 9}}
]

-INPUT TEXT TO ANALYZE-
text: {input_text}

-IDENTIFIED ENTITIES FROM FIRST STAGE-
{identified_entities}

-OUTPUT (only relationships from THIS input text between the identified entities)-
"""

# Gleaning assessment prompt to determine if more entities need to be extracted
GLEANING_ASSESSMENT_PROMPT = """
-Goal-
Determine if there are additional academic entities that should be extracted from the text.

-Input-
1. Original academic text
2. Entities already extracted

-Question-
Are there ADDITIONAL entities in the text that haven't been extracted yet? 
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
MANY entities were missed in the previous extraction. Identify ADDITIONAL academic entities that weren't captured in the first round.
Focus ONLY on identifying entities, NOT relationships between them.

-Instructions-
1. Review the original text and already extracted items
2. Focus ONLY on identifying NEW entities not in the previous extraction
3. Format output as a JSON list using the same format as before

-Expected Output Format-
For each NEW entity found in the INPUT TEXT, extract:
- entity_name: Academic term as it appears in the text (capitalized)
- entity_type: Categorize as one of: {entity_types}
- entity_description: Brief definition based ONLY on how it's described in the INPUT TEXT

######################
Original Text:
{input_text}

Already Extracted:
{extracted_items}

entity_types: {entity_types}
######################
Output:
"""

def extract_entities_with_gleanings(chunk, tokenizer, model, entity_types, max_gleanings=2):
    """
    Extract academic entities with multiple rounds of "gleanings" to improve recall.
    This is the first stage of the Microsoft GraphRAG approach, focusing ONLY on entity extraction.
    """
    debug("Extracting entities with multi-round gleanings approach (Stage 1)")
    
    # Initial extraction
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        entity_types=entity_types,
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

    # Use robust JSON extraction function
    parsed_data = extract_and_fix_json(text)
    entities = parsed_data["entities"]
    
    debug(f"Initial entity extraction: {len(entities)} entities")
    
    # Multiple rounds of gleanings
    gleaning_round = 0
    
    while gleaning_round < max_gleanings:
        # Construct the assessment prompt with the entities we've already extracted
        extracted_json = json.dumps(entities, indent=2)
        assessment_prompt = GLEANING_ASSESSMENT_PROMPT.format(
            input_text=chunk,
            extracted_items=extracted_json
        )
        
        # Force yes/no decision 
        inputs = tokenizer(assessment_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

        # Create token_ids for "YES" and "NO"
        yes_token_id = tokenizer.encode(" YES")[0]  # Using space prefix to get token
        no_token_id = tokenizer.encode(" NO")[0]    # Using space prefix to get token

        # All tokens except YES, NO, and EOS
        allowed_tokens = [yes_token_id, no_token_id, tokenizer.eos_token_id]
        suppress_tokens = [i for i in range(tokenizer.vocab_size) if i not in allowed_tokens]

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Need to set this to true for temperature
            temperature=0.7,
            suppress_tokens=suppress_tokens  # This suppresses all tokens except YES/NO/EOS
        )
        
        assessment = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if more entities need extraction
        if "YES" in assessment.upper():
            debug(f"Gleaning round {gleaning_round+1}: More entities identified")
            
            # Extract additional entities
            gleaning_prompt = GLEANING_EXTRACTION_PROMPT.format(
                input_text=chunk,
                extracted_items=extracted_json,
                entity_types=entity_types
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
            
            # Extract JSON from gleaning result
            additional_data_dict = extract_and_fix_json(gleaning_text)
            additional_entities = additional_data_dict["entities"]

            if additional_entities:
                # Add new entities
                for item in additional_entities:
                    # Check if this entity is new
                    is_new = True
                    for existing in entities:
                        if item["name"] == existing["name"]:
                            is_new = False
                            break
                    
                    if is_new:
                        entities.append(item)
                
                debug(f"After gleaning {gleaning_round+1}: {len(entities)} entities")
            
            gleaning_round += 1
        else:
            debug("No more entities to extract")
            break
    
    return entities

def extract_relationships(chunk, entities, tokenizer, model):
    """
    Extract relationships between identified entities.
    This is the second stage of the Microsoft GraphRAG approach, focused exclusively on relationship extraction
    between the entities identified in the first stage.
    """
    debug(f"Extracting relationships between {len(entities)} entities (Stage 2)")
    
    if not entities:
        debug("No entities to extract relationships for")
        return []
    
    # Format entities as a list of entity names for the prompt
    entity_names = [entity["name"] for entity in entities]
    entities_formatted = json.dumps(entity_names)
    
    # Relationship extraction
    prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
        input_text=chunk,
        identified_entities=entities_formatted
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.3
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse relationships
    parsed_data = extract_and_fix_json(text)
    relationships = parsed_data["relationships"]
    
    debug(f"Extracted {len(relationships)} relationships")
    return relationships

def extract_graph_elements_improved(all_chunks, summ_tokenizer, summ_model, max_chunks=None):
    """
    Extract graph elements from all chunks with improved two-stage approach.
    First extract entities, then extract relationships between those entities.
    This follows Microsoft's GraphRAG approach with clear separation between the stages.
    """
    debug("Beginning extraction over all chunks with Microsoft-style two-stage approach")
    entity_rows = defaultdict(list)
    relationship_rows = defaultdict(list)
    
    entity_types = "RESEARCH_CONCEPT, METHODOLOGY, VARIABLE, FINDING, ORGANIZATION"
    
    for doc_idx, chunks in enumerate(all_chunks):
        debug(f"Extracting from document {doc_idx} with {len(chunks)} chunks")
        
        # Limit processing to max_chunks if specified
        chunk_count = len(chunks) if max_chunks is None else min(max_chunks, len(chunks))
        processed_chunks = chunks[:chunk_count]
        
        for chunk_idx, chunk in enumerate(processed_chunks):
            debug(f"Processing chunk {chunk_idx+1}/{chunk_count}")
            
            # STAGE 1: Extract entities with gleanings approach
            entities = extract_entities_with_gleanings(chunk, summ_tokenizer, summ_model, entity_types)
            
            # Process extracted entities
            for ent in entities:
                name = ent.get("name", "").strip()
                typ = ent.get("type", "").strip()
                desc = ent.get("description", "").strip()
                
                if name and typ and desc:
                    entity_rows["doc_idx"].append(doc_idx)
                    entity_rows["chunk_idx"].append(chunk_idx)
                    entity_rows["Entity"].append(name)
                    entity_rows["Type"].append(typ)
                    entity_rows["Description"].append(desc)
            
            # STAGE 2: Extract relationships between identified entities
            # Only proceed if we have entities to work with
            if entities:
                relationships = extract_relationships(chunk, entities, summ_tokenizer, summ_model)
                
                # Process relationships
                for rel in relationships:
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
    
    # Create dataframes
    df_entities = pd.DataFrame(entity_rows)
    df_relationships = pd.DataFrame(relationship_rows)
    
    # Validation check
    if not df_entities.empty:
        unique_entities = df_entities['Entity'].nunique()
        debug(f"Extracted {df_entities.shape[0]} total entities with {unique_entities} unique names")
        
        if unique_entities < 5 and df_entities.shape[0] > 10:
            debug("WARNING: Very low entity diversity. Extraction may have issues.")
    
    # Check if we have relationships but no entities - this shouldn't happen with the two-stage approach
    if df_entities.empty and not df_relationships.empty:
        debug("WARNING: Found relationships but no entities. This is unexpected with the two-stage approach.")
    
    return df_entities, df_relationships

# -----------------------------------------------------------------------------
# 4. Element Summaries
# -----------------------------------------------------------------------------

def summarize_element_instances(df_entities, df_relationships, summ_tokenizer, summ_model):
    """
    Convert instance-level summaries into single blocks of descriptive text for each graph element.
    This creates enriched element descriptions before community summarization.
    """
    debug("Summarizing element instances to create element summaries")
    
    # Group entities by name to summarize duplicate entries
    entity_groups = df_entities.groupby('Entity')
    entity_summaries = {}
    
    for entity_name, group in entity_groups:
        # Skip if empty
        if group.empty:
            continue
            
        # Get the most common type for this entity
        entity_type = group['Type'].mode()[0]
        
        # Combine all descriptions of this entity
        descriptions = group['Description'].tolist()
        
        # If only one description, use it directly
        if len(descriptions) == 1:
            entity_summaries[entity_name] = {
                'type': entity_type,
                'description': descriptions[0],
                'doc_idx': group['doc_idx'].iloc[0],
                'chunk_idx': group['chunk_idx'].iloc[0]
            }
        else:
            # For multiple descriptions, summarize them
            combined_text = "\n".join([f"- {desc}" for desc in descriptions])
            
            # Create summarization prompt
            prompt = f"""
            You are an academic knowledge summarizer. Create a unified, comprehensive description 
            of the following academic entity from these possibly redundant descriptions.
            
            ENTITY: {entity_name}
            TYPE: {entity_type}
            
            DESCRIPTIONS:
            {combined_text}
            
            UNIFIED DESCRIPTION:
            """
            
            # Generate summary
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
                'doc_idx': group['doc_idx'].iloc[0],  # Use first occurrence
                'chunk_idx': group['chunk_idx'].iloc[0]
            }
    
    debug(f"Created {len(entity_summaries)} entity summaries")
    
    # Now do the same for relationships
    # Group relationships by source and target
    relationship_keys = []
    for _, row in df_relationships.iterrows():
        # Create a standardized key for each relationship
        source = row['Source']
        target = row['Target']
        # Sort to treat A->B and B->A as the same relationship
        rel_key = tuple(sorted([source, target]))
        relationship_keys.append((rel_key, row))
    
    # Group by relationship key
    rel_groups = {}
    for rel_key, row in relationship_keys:
        if rel_key not in rel_groups:
            rel_groups[rel_key] = []
        rel_groups[rel_key].append(row)
    
    relationship_summaries = {}
    
    for rel_key, rows in rel_groups.items():
        # Skip if entity doesn't exist in entity summaries
        source, target = rel_key
        if source not in entity_summaries or target not in entity_summaries:
            continue
        
        # If only one relationship description, use it directly
        if len(rows) == 1:
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
            # For multiple relationships, summarize them
            # Get all descriptions
            descriptions = [row['Description'] for row in rows]
            combined_text = "\n".join([f"- {desc}" for desc in descriptions])
            
            # First row for reference
            first_row = rows[0]
            
            # Determine actual source and target (not the sorted version)
            actual_source = first_row['Source']
            actual_target = first_row['Target']
            
            # Create summarization prompt
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
            
            # Generate summary
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
            if "UNIFIED RELATIONSHIP DESCRIPTION:" in summary:
                summary = summary.split("UNIFIED RELATIONSHIP DESCRIPTION:", 1)[1].strip()
            
            # Calculate average strength
            avg_strength = 5  # Default
            if 'Strength' in rows[0]:
                strengths = [row.get('Strength', 5) for row in rows]
                avg_strength = sum(strengths) / len(strengths)
            
            # Store summarized relationship
            rel_id = f"{actual_source}|{actual_target}"
            relationship_summaries[rel_id] = {
                'source': actual_source,
                'target': actual_target,
                'description': summary,
                'doc_idx': first_row['doc_idx'],  # Use first occurrence
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
        
        # Check if both nodes exist
        if G.has_node(source) and G.has_node(target):
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
    Uses entity types, descriptions, and connections for better merging decisions.
    """
    debug("Merging similar nodes with improved algorithm")
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx
    import copy
    
    # Get all nodes and prepare for embedding
    node_list = list(G.nodes())
    
    if len(node_list) <= 1:
        debug("Not enough nodes to merge")
        return G
        
    # Create rich node representations for embedding
    node_texts = []
    for node in node_list:
        node_data = G.nodes[node]
        # Combine name, type and description for richer embedding
        node_text = f"{node} - {node_data.get('type', 'UNKNOWN')} - {node_data.get('description', '')}"
        node_texts.append(node_text)
    
    # Get embeddings for all nodes
    embeddings = []
    batch_size = 16  # Process in batches to avoid OOM
    
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
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Identify pairs to merge
    merge_candidates = []
    
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            if similarity_matrix[i, j] >= threshold:
                node_i = node_list[i]
                node_j = node_list[j]
                type_i = G.nodes[node_i].get('type', '')
                type_j = G.nodes[node_j].get('type', '')
                
                # Only merge if node types match or one is unknown
                if type_i == type_j or type_i == '' or type_j == '':
                    # Calculate node importance scores (based on connections and metadata)
                    score_i = G.degree(node_i) * G.nodes[node_i].get('frequency', 1)
                    score_j = G.degree(node_j) * G.nodes[node_j].get('frequency', 1)
                    
                    # Determine which node to keep (higher score)
                    if score_i >= score_j:
                        merge_candidates.append((node_i, node_j))
                    else:
                        merge_candidates.append((node_j, node_i))
    
    # Sort by similarity score to merge most similar pairs first
    merge_candidates.sort(key=lambda pair: similarity_matrix[node_list.index(pair[0]), node_list.index(pair[1])], 
                          reverse=True)
    
    # Perform merging
    G_merged = copy.deepcopy(G)
    merged_nodes = set()
    
    for keep_node, merge_node in merge_candidates:
        # Skip if either node has already been merged
        if keep_node in merged_nodes or merge_node in merged_nodes:
            continue
            
        if G_merged.has_node(keep_node) and G_merged.has_node(merge_node):
            # Merge node attributes
            keep_attrs = G_merged.nodes[keep_node]
            merge_attrs = G_merged.nodes[merge_node]
            
            # Update description
            if 'description' in merge_attrs and 'description' in keep_attrs:
                if merge_attrs['description'] not in keep_attrs['description']:
                    keep_attrs['description'] = f"{keep_attrs['description']}; {merge_attrs['description']}"
            
            # Update frequency
            keep_attrs['frequency'] = keep_attrs.get('frequency', 1) + merge_attrs.get('frequency', 1)
            
            # Contract nodes
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
    Detect hierarchical communities using the Leiden algorithm with different resolution parameters.
    Returns a dictionary of partitions at different hierarchy levels.
    """
    debug("Detecting hierarchical communities using Leiden algorithm")
    
    import igraph as ig
    import leidenalg
    
    # Convert NetworkX graph to igraph
    edges = list(G.edges())
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create mapping of edge list with node indices
    edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
    
    # Create igraph graph
    g = ig.Graph(n=len(nodes), edges=edge_list, directed=False)
    g.vs["name"] = nodes
    
    # Add edge weights if available
    edge_weights = None
    if nx.get_edge_attributes(G, 'weight'):
        edge_weights = [G[u][v].get('weight', 1.0) for u, v in edges]
        g.es["weight"] = edge_weights
    
    # Dictionary to store partitions at different levels
    hierarchical_partitions = {}
    
    # Level 0: Coarsest level (smallest number of communities)
    resolution0 = 0.5
    partition0 = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution0,
        weights=edge_weights
    )
    
    modularity0 = g.modularity(partition0.membership, weights=edge_weights)
    debug(f"Level 0: {len(set(partition0.membership))} communities, modularity={modularity0:.4f}")
    
    # Convert to node dictionary
    level0_dict = {}
    for idx, cluster_id in enumerate(partition0.membership):
        level0_dict[nodes[idx]] = cluster_id
    
    hierarchical_partitions[0] = level0_dict
    
    # Generate subsequent levels with increasing resolution (more fine-grained communities)
    resolutions = [1.0, 2.0]
    
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
        
        # Convert to node dictionary
        level_dict = {}
        for idx, cluster_id in enumerate(partition.membership):
            # Create hierarchical community ID: parentID_childID
            node = nodes[idx]
            parent_id = hierarchical_partitions[level-1][node]
            # Create unique ID within parent community
            hierarchical_id = f"{parent_id}_{cluster_id}"
            level_dict[node] = hierarchical_id
        
        hierarchical_partitions[level] = level_dict
    
    # Create parent-child relationships between communities
    community_hierarchy = {}
    
    for level in range(1, len(hierarchical_partitions)):
        parent_level = level - 1
        child_level = level
        
        # Get unique communities at each level
        parent_communities = set(hierarchical_partitions[parent_level].values())
        child_communities = set(hierarchical_partitions[child_level].values())
        
        # Map each child to its parent
        for node in G.nodes():
            if node in hierarchical_partitions[child_level]:
                child_comm = hierarchical_partitions[child_level][node]
                parent_comm = hierarchical_partitions[parent_level][node]
                
                if child_comm not in community_hierarchy:
                    community_hierarchy[child_comm] = parent_comm
    
    debug(f"Built hierarchical community structure with {len(hierarchical_partitions)} levels")
    
    return hierarchical_partitions, community_hierarchy

def get_community_nodes(partition, community_id):
    """
    Get all nodes belonging to a specific community.
    """
    nodes = []
    for node, comm_id in partition.items():
        if comm_id == community_id:
            nodes.append(node)
    return nodes

def summarize_community_prioritized(G, community_nodes, entity_summaries, relationship_summaries, 
                                    all_chunks, summ_tokenizer, summ_model, max_tokens=1800):
    """
    Summarize a community using prioritized element summaries as described in the paper.
    
    For each community edge in decreasing order of combined source and target node degree,
    add descriptions of the source node, target node, and the edge itself until token limit is reached.
    """
    debug(f"Generating prioritized summary for community with {len(community_nodes)} nodes")
    
    if not community_nodes:
        return "No nodes in this community."
    
    # Create subgraph of community
    subgraph = G.subgraph(community_nodes)
    
    # Calculate node importance (degree in the original graph)
    node_importance = {node: G.degree(node) for node in community_nodes}
    
    # Get edges in the community
    community_edges = list(subgraph.edges())
    
    # Calculate edge importance (sum of endpoint degrees)
    edge_importance = {}
    for u, v in community_edges:
        edge_importance[(u, v)] = node_importance[u] + node_importance[v]
    
    # Sort edges by importance
    sorted_edges = sorted(community_edges, key=lambda e: edge_importance[e], reverse=True)
    
    # Prioritized information for the summary
    prioritized_info = []
    current_tokens = 0
    
    # Track which nodes and edges have been added
    added_nodes = set()
    added_edges = set()
    
    # First, add the most important nodes (top 3)
    top_nodes = sorted(community_nodes, key=lambda n: node_importance[n], reverse=True)[:3]
    
    for node in top_nodes:
        if node in entity_summaries:
            node_info = f"ENTITY: {node}\nTYPE: {entity_summaries[node]['type']}\nDESCRIPTION: {entity_summaries[node]['description']}\n\n"
            node_tokens = len(summ_tokenizer.encode(node_info))
            
            if current_tokens + node_tokens <= max_tokens:
                prioritized_info.append(node_info)
                current_tokens += node_tokens
                added_nodes.add(node)
    
    # Then add edges in order of importance
    for u, v in sorted_edges:
        # Skip if token limit reached
        if current_tokens >= max_tokens:
            break
            
        # Create relationship ID
        rel_id1 = f"{u}|{v}"
        rel_id2 = f"{v}|{u}"
        
        # Get the right relationship ID
        rel_id = rel_id1 if rel_id1 in relationship_summaries else rel_id2 if rel_id2 in relationship_summaries else None
        
        if rel_id:
            rel_data = relationship_summaries[rel_id]
            relationship_info = f"RELATIONSHIP: {rel_data['source']} → {rel_data['target']}\nDESCRIPTION: {rel_data['description']}\n\n"
            rel_tokens = len(summ_tokenizer.encode(relationship_info))
            
            # Add source and target nodes if not already added
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
            
            # Check if we can add all information
            total_tokens = rel_tokens + source_tokens + target_tokens
            
            if current_tokens + total_tokens <= max_tokens:
                # Add source if needed
                if source_info:
                    prioritized_info.append(source_info)
                    added_nodes.add(rel_data['source'])
                
                # Add target if needed
                if target_info:
                    prioritized_info.append(target_info)
                    added_nodes.add(rel_data['target'])
                
                # Add relationship
                prioritized_info.append(relationship_info)
                added_edges.add((u, v))
                
                current_tokens += total_tokens
            else:
                # If we can't add everything, try to add just the relationship
                if current_tokens + rel_tokens <= max_tokens:
                    prioritized_info.append(relationship_info)
                    added_edges.add((u, v))
                    current_tokens += rel_tokens
                else:
                    # We've reached the token limit
                    break
    
    # Add relevant source text chunks if we have space
    if current_tokens < max_tokens:
        chunk_texts = set()
        
        # Find relevant chunks for the community nodes
        for node in community_nodes:
            if node in entity_summaries:
                d_idx = entity_summaries[node]['doc_idx']
                c_idx = entity_summaries[node]['chunk_idx']
                
                if d_idx < len(all_chunks) and c_idx < len(all_chunks[d_idx]):
                    chunk = all_chunks[d_idx][c_idx]
                    chunk_texts.add(chunk)
        
        # Add chunks until we reach the token limit
        for chunk in list(chunk_texts)[:3]:  # Limit to 3 chunks
            chunk_info = f"SOURCE TEXT:\n{chunk}\n\n"
            chunk_tokens = len(summ_tokenizer.encode(chunk_info))
            
            if current_tokens + chunk_tokens <= max_tokens:
                prioritized_info.append(chunk_info)
                current_tokens += chunk_tokens
            else:
                break
    
    # Combine all information
    community_info = "\n".join(prioritized_info)
    
    # Generate the community summary
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
    
    # Generate summary
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.4
    )
    
    summary = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated summary
    if "SUMMARY:" in summary:
        summary = summary.split("SUMMARY:", 1)[1].strip()
    
    debug(f"Generated community summary of {len(summary)} characters")
    return summary

def summarize_hierarchical_communities(G, hierarchical_partitions, community_hierarchy, entity_summaries, 
                                       relationship_summaries, all_chunks, summ_tokenizer, summ_model):
    """
    Build summaries for hierarchical communities at different levels.
    Implementation follows the paper's approach for leaf-level vs. higher-level communities.
    """
    debug("Building hierarchical community summaries")
    
    # Start with leaf-level communities (highest level in the hierarchy)
    leaf_level = max(hierarchical_partitions.keys())
    
    # Community summaries at each level
    level_summaries = {}
    
    # First, summarize leaf-level communities
    leaf_partitions = hierarchical_partitions[leaf_level]
    unique_leaf_communities = set(leaf_partitions.values())
    
    debug(f"Summarizing {len(unique_leaf_communities)} leaf-level communities at level {leaf_level}")
    
    leaf_summaries = {}
    
    for community_id in unique_leaf_communities:
        # Get nodes in this community
        community_nodes = get_community_nodes(leaf_partitions, community_id)
        
        # Generate summary
        summary = summarize_community_prioritized(
            G, community_nodes, entity_summaries, relationship_summaries, 
            all_chunks, summ_tokenizer, summ_model
        )
        
        leaf_summaries[community_id] = summary
    
    level_summaries[leaf_level] = leaf_summaries
    
    # Now, build higher-level community summaries
    for level in range(leaf_level - 1, -1, -1):
        debug(f"Summarizing communities at level {level}")
        
        partitions = hierarchical_partitions[level]
        unique_communities = set(partitions.values())
        
        level_summaries[level] = {}
        
        for community_id in unique_communities:
            # Get nodes in this community
            community_nodes = get_community_nodes(partitions, community_id)
            
            # Get child communities of this community
            child_communities = []
            
            for child_id, parent_id in community_hierarchy.items():
                if parent_id == community_id:
                    child_communities.append(child_id)
            
            # Check if all element summaries fit within token limit
            community_subgraph = G.subgraph(community_nodes)
            total_tokens = 0
            
            # Estimate tokens from element summaries
            for node in community_nodes:
                if node in entity_summaries:
                    node_info = f"ENTITY: {node}\nTYPE: {entity_summaries[node]['type']}\nDESCRIPTION: {entity_summaries[node]['description']}\n\n"
                    total_tokens += len(summ_tokenizer.encode(node_info))
            
            # Check edges
            for u, v in community_subgraph.edges():
                rel_id1 = f"{u}|{v}"
                rel_id2 = f"{v}|{u}"
                rel_id = rel_id1 if rel_id1 in relationship_summaries else rel_id2 if rel_id2 in relationship_summaries else None
                
                if rel_id:
                    rel_data = relationship_summaries[rel_id]
                    rel_info = f"RELATIONSHIP: {rel_data['source']} → {rel_data['target']}\nDESCRIPTION: {rel_data['description']}\n\n"
                    total_tokens += len(summ_tokenizer.encode(rel_info))
            
            # If all element summaries fit, use standard approach
            if total_tokens <= 1800:
                summary = summarize_community_prioritized(
                    G, community_nodes, entity_summaries, relationship_summaries, 
                    all_chunks, summ_tokenizer, summ_model
                )
            else:
                # Otherwise, substitute child community summaries
                child_texts = []
                
                # Sort child communities by estimated token count (descending)
                child_tokens = {}
                for child_id in child_communities:
                    child_summary = level_summaries[level+1].get(child_id, "")
                    child_tokens[child_id] = len(summ_tokenizer.encode(child_summary))
                
                sorted_children = sorted(child_communities, key=lambda c: child_tokens.get(c, 0), reverse=True)
                
                # Collect child summaries until we reach token limit
                current_tokens = 0
                max_tokens = 1800
                
                for child_id in sorted_children:
                    child_summary = level_summaries[level+1].get(child_id, "")
                    child_token_count = child_tokens.get(child_id, 0)
                    
                    if current_tokens + child_token_count <= max_tokens:
                        child_texts.append(f"SUBCOMMUNITY SUMMARY: {child_summary}")
                        current_tokens += child_token_count
                    else:
                        # If we can't add more summaries, break
                        break
                
                # Combine child summaries and generate higher-level summary
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
                
                # Generate summary
                inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                outputs = summ_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    pad_token_id=summ_tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.4
                )
                
                summary = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the generated summary
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
    Generate a partial answer to the query based on a community summary,
    and also return a helpfulness score from 0-100.
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
    
    # Generate answer with score
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.3
    )
    
    response = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer and score
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
                # Ensure score is in valid range
                score = max(0, min(100, score))
            except:
                score = 0
    
    debug(f"Generated partial answer with helpfulness score: {score}")
    return answer, score

def shuffle_and_chunk_community_summaries(community_summaries, chunk_size=3):
    """
    Randomly shuffle community summaries and divide into chunks of specified size.
    As described in the paper, this ensures relevant information is distributed across chunks.
    """
    debug("Shuffling and chunking community summaries")
    
    # Convert to list of (community_id, summary) pairs
    summary_items = list(community_summaries.items())
    
    # Randomly shuffle
    random.shuffle(summary_items)
    
    # Divide into chunks
    summary_chunks = []
    for i in range(0, len(summary_items), chunk_size):
        chunk = summary_items[i:i+chunk_size]
        summary_chunks.append(dict(chunk))
    
    debug(f"Created {len(summary_chunks)} chunks from {len(summary_items)} community summaries")
    return summary_chunks

def combine_answers_with_scores(question, scored_answers, summ_tokenizer, summ_model):
    """
    Combine partial answers into a final answer, prioritizing the most helpful answers.
    """
    debug("Combining partial answers based on helpfulness scores")
    
    if not scored_answers:
        return "No relevant information found."
    
    # Sort by helpfulness score (descending)
    sorted_answers = sorted(scored_answers, key=lambda x: x[1], reverse=True)
    
    # Filter out answers with score 0
    filtered_answers = [(ans, score) for ans, score in sorted_answers if score > 0]
    
    if not filtered_answers:
        return "No relevant information found in the dataset for this question."
    
    # Combine answers until we reach token limit
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
    
    # Generate final answer
    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = summ_model.generate(
        **inputs,
        max_new_tokens=1024,
        pad_token_id=summ_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.4
    )
    
    final_answer = summ_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated answer
    if "FINAL COMPREHENSIVE ANSWER:" in final_answer:
        final_answer = final_answer.split("FINAL COMPREHENSIVE ANSWER:", 1)[1].strip()
    
    debug("Generated final combined answer")
    return final_answer

def improved_query_processing(question, level_summaries, summ_tokenizer, summ_model, sent_transformer, G=None):
    """
    Process a query using the hierarchical community summaries with shuffling
    and map-reduce approach as described in the paper.
    """
    debug(f"Processing query with shuffling and map-reduce: {question}")
    
    # Determine best community level to use for this question
    question_lower = question.lower()
    
    # For general questions about dataset overview, use highest level (0)
    general_patterns = [
        "main topics", "overview", "summary", "what is the data about",
        "high level", "general themes", "main findings"
    ]
    
    # For specific detailed questions, use lower level (leaf level)
    specific_patterns = [
        "specific", "detail", "exactly", "precisely", "tell me more about",
        "what is the relationship between", "how does"
    ]
    
    # Determine the best level
    if any(pattern in question_lower for pattern in general_patterns):
        best_level = 0  # Most general level
    elif any(pattern in question_lower for pattern in specific_patterns):
        best_level = max(level_summaries.keys())  # Most specific level
    else:
        # For other questions, use intermediate level if available, otherwise most specific
        if 1 in level_summaries:
            best_level = 1
        else:
            best_level = max(level_summaries.keys())
    
    debug(f"Selected community level {best_level} for query")
    
    # Special handling for "top topics" questions
    top_words = ["top", "main", "key", "primary", "important"]
    topic_words = ["topic", "topics", "entity", "entities", "subject", "theme", "concept"]
    
    is_top_topics_question = (
        any(word in question_lower for word in top_words) and 
        any(word in question_lower for word in topic_words)
    )
    
    if is_top_topics_question and G is not None:
        debug("Generating direct answer for top topics question")
        
        # Calculate weighted centrality
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
        
        # Get top entities
        top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Group by type
        entity_types = {}
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            entity_types[node_type] = entity_types.get(node_type, 0) + 1
        
        # Get community counts for each level
        community_counts = {level: len(summaries) for level, summaries in level_summaries.items()}
        
        # Formulate direct answer
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
    
    # Get community summaries at the selected level
    community_summaries = level_summaries.get(best_level, {})
    
    if not community_summaries:
        return "No community summaries available at the selected hierarchy level."
    
    # Filter relevant communities using semantic similarity
    query_embedding = sent_transformer.encode(question)
    scored_communities = []
    
    for comm_id, summary in community_summaries.items():
        summary_embedding = sent_transformer.encode(summary)
        similarity = cosine_similarity([query_embedding], [summary_embedding])[0][0]
        scored_communities.append((comm_id, summary, similarity))
    
    # Sort by similarity score
    sorted_communities = sorted(scored_communities, key=lambda x: x[2], reverse=True)
    
    # Select top communities (either by threshold or fixed number)
    threshold = 0.4
    selected_communities = {}
    
    for comm_id, summary, score in sorted_communities:
        if score >= threshold or len(selected_communities) < 3:  # Ensure at least 3 communities
            selected_communities[comm_id] = summary
        
        # Cap at maximum 10 communities
        if len(selected_communities) >= 10:
            break
    
    if not selected_communities:
        return "I couldn't find specific information to answer that question in the dataset."
    
    # Randomly shuffle and chunk community summaries
    summary_chunks = shuffle_and_chunk_community_summaries(selected_communities)
    
    # Map phase: Generate partial answers for each chunk
    all_scored_answers = []
    
    for chunk in summary_chunks:
        # Process each community in the chunk
        for comm_id, summary in chunk.items():
            answer, score = generate_partial_answer_with_score(question, summary, summ_tokenizer, summ_model)
            if score > 0:  # Only keep non-zero scores
                all_scored_answers.append((answer, score))
    
    # Reduce phase: Combine partial answers into final answer
    final_answer = combine_answers_with_scores(question, all_scored_answers, summ_tokenizer, summ_model)
    
    debug("Map-reduce query processing completed")
    return final_answer

# -----------------------------------------------------------------------------
# 7. Main & Interactive Query Interface
# -----------------------------------------------------------------------------
def main():
    debug("Starting enhanced Graph RAG chatbot with hierarchical communities...")
    
    # Load models
    debug("Loading models...")
    summ_tokenizer, summ_model = load_summarization_model()

    from sentence_transformers import SentenceTransformer
    sent_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Set up SPECTER2 for merging similar nodes
    from transformers import AutoTokenizer as SpecTokenizer
    from adapters import AutoAdapterModel as SpecAdapterModel
    global embedding_tokenizer, embedding_model
    embedding_tokenizer = SpecTokenizer.from_pretrained("allenai/specter2_base")
    embedding_model = SpecAdapterModel.from_pretrained("allenai/specter2_base")
    embedding_model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

    # Define PDF paths
    pdf_paths = [
        "/work/Chatbot-in-academia/papers-testing/6495.pdf"
    ]
    pdf_paths_with_idx = [(i, path) for i, path in enumerate(pdf_paths)]
    
    # Process PDFs into chunks
    debug("Processing PDFs into text chunks...")
    all_chunks, doc_citations = process_all_pdfs(pdf_paths_with_idx)
    
    # --- Extraction Phase with Microsoft-style Two-Stage Approach ---
    if os.path.exists(ENTITIES_CSV) and os.path.exists(RELATIONSHIPS_CSV):
        debug("Loading cached extracted entities and relationships.")
        df_entities = pd.read_csv(ENTITIES_CSV)
        df_relationships = pd.read_csv(RELATIONSHIPS_CSV)
    else:
        debug("Extracting graph elements with Microsoft-style two-stage approach...")
        df_entities, df_relationships = extract_graph_elements_improved(all_chunks, summ_tokenizer, summ_model)
        df_entities.to_csv(ENTITIES_CSV, index=False)
        df_relationships.to_csv(RELATIONSHIPS_CSV, index=False)
    debug(f"Extracted {df_entities.shape[0]} entities and {df_relationships.shape[0]} relationships.")
    
    # --- Element Instances → Element Summaries Step (New) ---
    debug("Creating element summaries from element instances...")
    entity_summaries, relationship_summaries = summarize_element_instances(
        df_entities, df_relationships, summ_tokenizer, summ_model
    )
    
    # --- Graph Construction from Element Summaries ---
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
    
    # --- Merge Similar Nodes ---
    debug("Merging similar nodes with improved algorithm...")
    G_merged = improved_merge_similar_nodes(G, embedding_tokenizer, embedding_model, threshold=0.85)
    debug(f"Merged Graph has {G_merged.number_of_nodes()} nodes and {G_merged.number_of_edges()} edges.")
    
    # --- Hierarchical Community Detection (New) ---
    debug("Detecting hierarchical communities...")
    hierarchical_partitions, community_hierarchy = detect_hierarchical_communities(G_merged, max_levels=2)
    
    # Print hierarchy stats
    for level, partition in hierarchical_partitions.items():
        unique_communities = set(partition.values())
        debug(f"Level {level}: {len(unique_communities)} communities")
    
    # --- Hierarchical Community Summaries (New) ---
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
    
    # Print summary counts at each level
    for level, summaries in level_summaries.items():
        debug(f"Level {level}: {len(summaries)} community summaries")
    
    # --- Interactive Query Interface ---
    print("\nEnhanced Graph RAG Chatbot is ready! Type 'exit' to quit.")
    print("This implementation includes:")
    print("- Microsoft-style two-stage approach: entities first, then relationships")
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
        
        # Process the query using improved hierarchical approach with shuffling
        answer = improved_query_processing(
            query, level_summaries, summ_tokenizer, summ_model, 
            sent_transformer, G_merged
        )
        
        print("\nBot:", answer)

if __name__ == "__main__":
    main()