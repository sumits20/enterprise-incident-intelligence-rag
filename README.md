# Enterprise Incident Intelligence (RAG)
**Author:** Sumit Sardar  
**Role:** Enterprise / SAP / AI Engineer  

Lightweight, low-token RAG system for enterprise incident intelligence (FAISS + MiniLM + Streamlit + GPT-4o mini)

This project demonstrates how structured incident data can be:
- semantically searched using embeddings + FAISS
- grounded with evidence
- and summarized using a small, cost-efficient LLM

# Why this project

RAG is often discussed at massive scale, but many enterprise use cases are:
- bounded
- structured
- latency and cost sensitive

This project intentionally uses:
- a small dataset
- simple Python modules
- clear failure modes

to show when RAG works — and when it doesn’t.

# Architecture

**Flow:**
1. Incident data loaded from Excel
2. Rows converted into text documents
3. MiniLM embeddings generated
4. FAISS used for similarity search
5. GPT-4o mini used only for synthesis (not retrieval)

# Tech Stack

- Python
- Streamlit
- Pandas
- SentenceTransformers (MiniLM)
- FAISS
- OpenAI GPT-4o mini
- Hugging Face Spaces

# Live Demo

Hugging Face Space:  
https://huggingface.co/spaces/sumitsardar/Enterprise_Incident_Intelligence

# Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
