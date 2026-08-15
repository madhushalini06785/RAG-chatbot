# 📚 RAG AI Assistant

> **Intelligent document Q&A chatbot** — Ask questions, get answers from your PDFs using AI

A production-ready Retrieval-Augmented Generation (RAG) system combining **Streamlit** UI, **LangChain** orchestration, **Pinecone** vector search, and **Groq** LLM for accurate, context-aware responses.

---

## ⚡ Quick Start (5 minutes)

### 1️⃣ Prerequisites
```bash
✓ Python 3.8+
✓ API Keys: Pinecone, Groq (both have free tiers)
```

### 2️⃣ Setup
```bash
# Clone & enter directory
git clone <https://github.com/madhushalini06785/RAG-chatbot.git>
cd RAGCHATBOT

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Configure
Create `.env` file in project root:
```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=your_index_name
GROQ_API_KEY=your_key_here
```

### 4️⃣ Run
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` 🎉

---

## 🎯 How It Works

```
Your PDF → Extract Text → Split into Chunks → Generate Embeddings
    ↓                                                 ↓
    └─────────────────────────────────────────► Pinecone DB
                                                    ↓
User Question ──→ Find Similar Chunks ──→ Send to Groq LLM ──→ Answer
```

**On first run:** App automatically ingests your PDF (2-4 minutes)  
**On subsequent runs:** Uses cached vectors, instant queries

---

## 📁 Project Structure

```
RAGCHATBOT/
├── streamlit_app.py       ← Run this to start
├── config.py              ← API keys & environment
├── ingest.py              ← Document processing
├── rag_chain.py           ← LLM & retrieval setup
├── requirements.txt       ← Dependencies
├── .env                   ← Create this (your secrets)
├── data/
│   └── notes.pdf         ← Your document
└── README.md             ← This file
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Streamlit 1.32.2 | Chat interface |
| **Orchestration** | LangChain 0.1.16 | RAG pipeline |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Vector generation |
| **Vector DB** | Pinecone 3.2.2 | Semantic search |
| **LLM** | Groq (gpt-oss-20b) | Answer generation |

---

## 🔧 Configuration Guide

### config.py
Loads environment variables from `.env`:
```python
PINECONE_API_KEY       # Authentication
PINECONE_INDEX         # Vector database index name
GROQ_API_KEY          # LLM API key
```

### ingest.py
Processes your PDF once:
- Extracts text from PDF
- Splits into 500-char chunks (100 overlap)
- Generates embeddings
- Stores in Pinecone
- **Skips if vectors already exist**

Key function: `ingest_document()`

### rag_chain.py
Configures the retrieval chain:
```python
embedding_model      # all-MiniLM-L6-v2 (lightweight)
vector_store         # Pinecone index
retriever            # Returns top 4 similar chunks (k=4)
llm                  # Groq gpt-oss-20b, temperature=0
```

### streamlit_app.py
Interactive chat UI:
1. Check if vectors indexed on startup
2. Auto-ingest if database empty
3. Display chat history
4. Stream LLM responses

---

## ⚙️ Tuning & Customization

### Performance Optimization

**Faster responses?** (Edit in `rag_chain.py`)
```python
search_kwargs={"k": 2}   # Return 2 chunks instead of 4
```

**More accurate?** (Edit in `rag_chain.py`)
```python
search_kwargs={"k": 8}   # Return 8 chunks for more context
```

**Change chunk size** (Edit in `ingest.py`)
```python
chunk_size = 300         # Smaller = more specific answers
chunk_overlap = 50       # Smaller = faster processing
```

**Different LLM?** (Edit in `rag_chain.py`)
```python
model_name="mixtral-8x7b-32768"  # Available on Groq
```

### Advanced: Embedding Models
```python
# In rag_chain.py, options include:
"sentence-transformers/all-MiniLM-L6-v2"      # ⭐ Current (fast)
"sentence-transformers/all-mpnet-base-v2"     # Better quality, slower
"BAAI/bge-small-en"                           # Excellent quality
```

---

## 🚀 Deployment

### Streamlit Cloud (Easiest)
1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect repository
4. Add secrets (Settings):
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX`
   - `GROQ_API_KEY`

Done! ✅

### Local Machine
```bash
streamlit run streamlit_app.py
```

### Docker (Optional)
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

---

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| **"notes.pdf not found"** | Ensure `data/notes.pdf` exists |
| **"PINECONE_API_KEY not found"** | Create `.env` with API keys |
| **"Index not found"** | Create index in Pinecone Console, wait 2-3 min |
| **Slow responses (>5 sec)** | Reduce `k=2` in `rag_chain.py` |
| **Rate limit exceeded** | Groq free tier has limits; upgrade or wait 1 hour |
| **"module not found"** | Run: `pip install -r requirements.txt` |

### Reset Everything
```bash
# Delete Pinecone index in console, then:
rm -rf __pycache__
python -m venv venv  # Fresh venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📊 Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| PDF Ingestion | 2-4 min | One-time, 50-100 pages |
| Vector Search | 50-200 ms | Pinecone retrieval |
| LLM Response | 1-3 sec | Groq API call |
| **Total E2E** | 2-4 sec | User perceives as smooth |

---

## ❓ FAQ

**Q: How do I use a different PDF?**
- Delete Pinecone index in console
- Replace `data/notes.pdf`
- Restart app → auto-ingest

**Q: Can I process multiple PDFs?**
- Modify `ingest.py` to loop through `data/` folder
- See inline comments for implementation

**Q: Why does first run take 2-4 minutes?**
- PDF extraction: 30 sec
- Text chunking: 20 sec
- Embedding generation: 60-90 sec ⬅️ Largest step
- Uploading to Pinecone: 20-30 sec

**Q: How much does this cost?**
- Pinecone: Free (up to 100K vectors)
- Groq: Free tier (rate limited)
- HuggingFace: Free (runs locally)
- Streamlit: Free tier available

**Q: Can I run this offline?**
- Not with current setup (needs APIs)
- Alternative: Use Ollama (local LLM) + Chroma (local vectors)

**Q: How do I improve answer quality?**
- Increase `k=8` (retrieve more context)
- Decrease `chunk_size` (more granular chunks)
- Use better embedding model: `all-mpnet-base-v2`

**Q: How do I add authentication?**
- Wrap in Streamlit Cloud auth
- Or add: `@st.cache_resource` decorator
- See Streamlit docs for details

---

## 🔐 Security

✅ **DO:**
- Use `.env` for all secrets
- Add `.env` to `.gitignore`
- Rotate API keys regularly
- Use free tiers for testing

❌ **DON'T:**
- Commit `.env` to Git
- Hardcode API keys in code
- Share `.env` file
- Use production keys for testing

---

## 📚 Resources

- [LangChain Docs](https://python.langchain.com/) — RAG framework
- [Streamlit Docs](https://docs.streamlit.io/) — UI framework
- [Pinecone Docs](https://docs.pinecone.io/) — Vector DB
- [Groq Docs](https://console.groq.com/docs) — LLM API
- [HuggingFace](https://huggingface.co/models) — Embedding models

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/improvement`
3. Make changes & test
4. Submit pull request

Report bugs/ideas: Create a GitHub Issue

---

## 📄 License

MIT License — See LICENSE file

---

## 📊 Quick Commands

```bash
# Start the app
streamlit run streamlit_app.py

# Check Pinecone status
python -c "from config import *; print(index.describe_index_stats())"

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Activate virtual env (Windows)
venv\Scripts\activate

# Activate virtual env (macOS/Linux)
source venv/bin/activate

# Stop Streamlit (Ctrl+C)
```

---

**Built for Document Intelligence** 🚀 | Made with ❤️ | Last Updated: 2026-08-15

---

## 🎯 Next Steps

### Beginner
1. ✅ Set up locally (follow Quick Start)
2. ✅ Ask a question → see it work
3. ✅ Read your first answer

### Intermediate
4. ✅ Modify system prompt (in `rag_chain.py`)
5. ✅ Tune chunk size
6. ✅ Deploy to Streamlit Cloud


### Advanced
7. ✅ Add multiple document support
8. ✅ Implement caching layer
9. ✅ Add analytics/logging
10. ✅ Deploy to production with scaling

---

**Need help?** Check Troubleshooting → FAQ → Resources sections above ⬆️
