# 📚 AI Document Assistant - RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **Pinecone** that allows users to ask questions about uploaded documents and get intelligent answers using AI.

## ✨ Features

- 🤖 **AI-Powered Q&A**: Ask questions about your documents and get accurate answers
- 📄 **PDF Support**: Seamlessly ingest and process PDF documents
- 🔍 **Vector Search**: Uses Pinecone vector database for semantic search
- 💬 **Chat Interface**: Interactive Streamlit-based chat UI
- ⚡ **Smart Caching**: Automatic knowledge base initialization with session management
- 🧠 **Context-Aware**: Retrieves relevant document chunks to answer questions accurately
- 🚀 **Scalable**: Built with production-ready technologies

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.32.2 |
| **LLM Framework** | LangChain 0.1.16 |
| **Vector Database** | Pinecone 3.2.2 |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **LLM** | Groq (OpenAI GPT-OSS-20B) |
| **PDF Processing** | PyPDF 4.2.0 |
| **Environment** | Python 3.x with venv |

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- API Keys for:
  - **Pinecone** (vector database) - [Get API Key](https://www.pinecone.io/)
  - **Groq** (LLM provider) - [Get API Key](https://console.groq.com/)
  - **HuggingFace** (for embeddings) - [Get API Key](https://huggingface.co/settings/tokens)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <https://github.com/madhushalini06785/RAG-chatbot.git>
cd RAGCHATBOT
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name
GROQ_API_KEY=your_groq_api_key
```

### 5. Prepare Your Document
- Place your PDF file in the `data/` folder
- Name it `notes.pdf` (or update the path in `ingest.py`)

### 6. Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
RAGCHATBOT/
├── streamlit_app.py        # Main Streamlit application
├── config.py              # Configuration and API key management
├── ingest.py              # Document ingestion and indexing
├── rag_chain.py           # RAG chain and LLM configuration
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python runtime version (for deployment)
├── .env                   # Environment variables (create this)
├── data/                  # Directory for documents
│   └── notes.pdf         # Your document to index
└── README.md             # This file
```

## 🔧 Configuration

### config.py
Loads environment variables and configures API keys:
```python
PINECONE_API_KEY      # Your Pinecone API key
PINECONE_INDEX        # Your Pinecone index name
GROQ_API_KEY          # Your Groq API key
```

### ingest.py
Handles document processing:
- Loads PDF files from `data/notes.pdf`
- Splits text into chunks (500 chars with 100 char overlap)
- Generates embeddings using HuggingFace
- Uploads vectors to Pinecone

### rag_chain.py
Configures the RAG pipeline:
- Initializes embeddings (sentence-transformers/all-MiniLM-L6-v2)
- Sets up Pinecone vector retriever (k=4 most similar chunks)
- Configures Groq LLM with zero temperature for consistent responses
- Defines system prompts for accurate, context-aware answers

### streamlit_app.py
Main application interface:
- Checks if knowledge base is indexed on first run
- Automatically initiates ingestion if needed
- Manages chat message history using Streamlit session state
- Displays AI responses with streaming capability

## 🎯 How It Works

### 1. **First Run Setup**
```
App starts → Check Pinecone database
            ↓ (if empty)
         Ingest PDF
            ↓
      Index vectors
            ↓
   Ready for chat
```

### 2. **Chat Flow**
```
User asks question
        ↓
    Retrieve relevant document chunks from Pinecone
        ↓
   Send to Groq LLM with context
        ↓
      Return answer
        ↓
   Display in chat & save to history
```

## 📚 API Endpoints & Services Used

- **Pinecone API**: Vector storage and similarity search
- **Groq API**: LLM inference (gpt-oss-20b model)
- **HuggingFace**: Pre-trained embeddings model

## 🔐 Security Notes

- ✅ Never commit `.env` file to version control
- ✅ Use environment variables for all API keys
- ✅ Add `.env` to `.gitignore`
- ✅ Rotate API keys periodically
- ✅ Keep dependencies updated for security patches

## 🚀 Deployment

### Deploy to Streamlit Cloud
1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add secrets in Settings:
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX`
   - `GROQ_API_KEY`

### Deploy to Heroku/AWS/Azure
See `runtime.txt` for Python version. Ensure all dependencies are in `requirements.txt`.

## 🐛 Troubleshooting

### Error: "notes.pdf not found"
- Ensure PDF is placed in `data/` folder
- Check file name matches `notes.pdf`

### Error: "PINECONE_API_KEY not found"
- Create `.env` file in project root
- Verify API keys are correctly set

### Knowledge base not updating
- Delete the index in Pinecone and run again
- Or modify `ingest.py` to force re-ingestion

### Slow responses
- Reduce `chunk_size` in `ingest.py` for faster retrieval
- Check Pinecone index size and optimize as needed

## 📊 Performance Tips

- **Chunk Size**: Default 500 chars provides good balance
- **Retrieval K**: Returns top 4 chunks (adjust in `rag_chain.py`)
- **Temperature**: Set to 0 for consistent, factual responses
- **Embeddings**: Using lightweight model (6M params) for fast inference

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

## 📄 License

MIT License - see LICENSE file for details

## 🎓 Learning Resources

- [LangChain Docs](https://python.langchain.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Pinecone Docs](https://docs.pinecone.io/)
- [Groq Docs](https://console.groq.com/docs)

## 📧 Support

For issues and questions:
1. Check the troubleshooting section
2. Review your API keys and environment setup
3. Check application logs in terminal

---

**Happy chatting! 🚀**
