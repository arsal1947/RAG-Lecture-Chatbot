# 📚 Lecture Chatbot

An AI-powered chatbot that lets you upload your lecture PDFs and ask questions about them using RAG (Retrieval-Augmented Generation), LangChain, Groq, and Streamlit.

---

## 📌 Overview

This app allows students to upload their lecture PDFs and have a natural conversation about the content. It uses HuggingFace embeddings to index the documents into a ChromaDB vector store, retrieves the most relevant chunks using similarity search, and passes them to **LLaMA 3.3 70B** via Groq to generate accurate, context-based answers. The chatbot strictly answers from the uploaded documents only — no hallucination from outside knowledge.

---

## ✨ Features

- 📄 Upload multiple PDF lecture files at once
- 🔍 Semantic search using HuggingFace embeddings + ChromaDB
- 🤖 AI answers powered by LLaMA 3.3 70B (Groq)
- 💬 Full chat interface with conversation history
- 🚫 Answers strictly from your documents — no outside knowledge
- ♻️ Automatically resets chat when new PDFs are uploaded

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | LLaMA 3.3 70B via Groq |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Framework | LangChain (LCEL) |
| PDF Loader | PyPDFLoader |

---

## 📁 Project Structure

```
Lecture-Chatbot/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not pushed to GitHub)
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/arsal1947/RAG-Lecture-Chatbot.git
cd Lecture-Chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🚀 How to Use

1. Upload one or more PDF lecture files from the sidebar
2. Click **⚡ Process PDFs** to index the content
3. Type your question in the chat input at the bottom
4. The chatbot will answer based strictly on your uploaded documents
5. Chat history is maintained throughout the session

---

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-huggingface
langchain-chroma
langchain-groq
langchain-core
langchain-text-splitters
pypdf
chromadb
sentence-transformers
python-dotenv
```

---

## 🔐 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key for LLaMA 3.3 70B access |

---

## 👤 Author

**Arsal** — AI/ML Student  
GitHub: [@arsal1947](https://github.com/arsal1947)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
