# 🤖 Chatbot-in-Academia

A Retrieval-Augmented Generation (RAG) chatbot that generates academically grounded outputs such as summaries, slide decks, and podcast scripts from research papers. This project compares Meta’s LLaMA 3.3 70B and OpenAI GPT-4o-mini across two retrieval strategies for QA Advanced RAG and Graph RAG and then evaluating non-traditional outputs. 

---

## 🎓 Thesis Information

- **Degree**: MSc in Business Administration & Data Science  
- **Institution**: Copenhagen Business School  
- **Authors**: Francesca Salute & Nicole Favero 
- **Supervisor**: Professor Daniel Hardt  
- **Date**: May 2025

---

## 🔍 Key Features

- ✅ Retrieval-Augmented Q&A with Advanced and Graph RAG  
- 🧾 Academic paper summarization  
- 🎞️ Slide deck creation using `python-pptx`  
- 🎙️ Podcast generation with Text-to-Speech (TTS)  
- 🧠 Model evaluation using human and LLM judges  
- 📊 Comparative analysis of LLaMA 3.3 vs. GPT-4o-mini

---

## 🗂️ Repository Structure

```text
CHATBOT-IN-ACADEMIA/
├── __pycache__/
├── .gradio/
├── audio_segments/
├── db/
├── generated_audio/
├── GraphRAGLLaMA/
├── GraphRAGOpenAI/
├── papers-cleaned/
├── papers-testing/
├── Results_Multinational Performance and Risk Management Effects Capital Structure Contingencies
├── Results_The impact of knowledge management on MNC subsidiary performance
├── cleaning-papers.py
├── DOC_REGISTRY.py
├── generated_podcast_script.txt
├── generated_summary.txt
├── LLama_Podcast_audio.py
├── LLama_Podcast.py
├── LLama_QA.py
├── LLama_Slide.py
├── LLama_Summary.py
├── OpenAI_Podcast.py
├── OpenAI_QA.py
├── OpenAI_Slidedeck.py
├── OpenAI_SlidedeckImproved.py
├── OpenAI_Summary.py
├── README.md
├── requirements.txt
└── scraping.py

## ⚙️ Install required packages

```bash
pip install -r requirements.txt

## 📁 Dataset

Academic papers are sourced from the **CBS Research Archive**.

### To scrape papers:

```bash
python scraping.py
# Cleaned papers: papers-cleaned/
# Test set: papers-testing/

---

## 🧠 Model Architecture

| Model         | Type             | Access        | Max Context | Used In                             |
|---------------|------------------|---------------|-------------|-------------------------------------|
| LLaMA 3.3 70B | Open-source LLM  | Hugging Face  | 128K tokens | QA, Summary, Slides, Podcast        |
| GPT-4o-mini   | Proprietary API  | OpenAI        | 128K tokens | QA, Summary, Slides, Podcast        |

### Retrieval Strategies

- **Advanced RAG**: Hybrid (BM25 + Dense Embeddings via ChromaDB)  
- **Graph RAG**: Manual entities and relationships + Knowledge Graph 

---

## 📦 Outputs

- `generated_summary.txt`: Summary of an academic paper  
- `generated_podcast_script.txt`: Scripted podcast dialogue  
- `Podcast_*.mp3`: TTS-generated podcast audio  
- `.pptx` files: PowerPoint slide decks
- different `py.` for QA

---
## 📈 Evaluation Setup

- **Human Evaluators**: 11 CBS students  
- **LLM Judges**: Claude & DeepSeek (no self-evaluation by GPT/LLaMA)

### Output Files

- `Results_Multinational Performance and Risk Management Effects Capital Structure Contingencies`  
- `Results_The impact of knowledge management on MNC subsidiary performance`  
---

## 💬 Use Cases

- 👨‍🏫 **Professors**: Generate lecture-ready slides from academic sources  
- 🎓 **Students**: Listen to podcasts or read summaries for study  

---

## 📜 Citation

If you use this repository or its results in your research, please cite:

```text
Francesca Salute 
Nicole Favero  
Department of Digitalization, Copenhagen Business School
or you can also reference this GitHub Repository:  
Chatbot-in-Academia


