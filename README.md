# RAG Using ORCA-mini 

## 📌 Overview

Financial Data Q&A with Orca-Mini is an AI-powered Streamlit app that lets you upload financial statement CSV files and ask questions in natural language. It uses LangChain for retrieval-augmented generation (RAG), FAISS for vector storage, and Ollama’s orca-mini model for generating answers.

## ✨ Features

- 📄 CSV Upload – Upload financial statements in CSV format.
- 🧠 Conversational AI – Uses Ollama’s orca-mini model for intelligent responses.
- 🗂 Vector Search – Stores document embeddings locally in FAISS for quick retrieval.
- 🔍 RAG Pipeline – Retrieves relevant data chunks before answering.
- 💬 Conversation History – Maintains context across multiple questions.
- 🖥 Simple UI – Built with Streamlit for quick deployment and interaction.

## 🛠 Tech Stack

- Frontend: Streamlit
- LLM: Ollama (orca-mini) via LangChain
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Vector Store: FAISS
- Orchestration: LangChain
- Deployment: Local or server-based Streamlit

 ## 🚀 Getting Started
### 1️⃣ Clone the repository  
```bash
git clone [https://github.com/Osama9588/.git](https://github.com/Osama9588/RAG_CSV.git)
cd google-gemini-chatbot
```

### 2️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```
