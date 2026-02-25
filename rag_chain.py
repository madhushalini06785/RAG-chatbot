import os
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# -------------------- ENV --------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- EMBEDDINGS --------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------- PINECONE --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding,
    namespace="default"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# -------------------- LLM --------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# -------------------- PROMPT --------------------
template = """
You are an AI assistant that answers questions strictly using the document.

Rules:
1. Only use the provided context
2. Do NOT use outside knowledge
3. If the answer is missing say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer clearly and briefly.
"""

prompt = ChatPromptTemplate.from_template(template)

# -------------------- RAG CHAIN --------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# -------------------- ASK FUNCTION --------------------
def ask_question(query: str) -> str:

    result = qa_chain.invoke({"query": query})

    answer = result["result"]
    docs = result["source_documents"]

    # extract page numbers
    pages = set()
    for doc in docs:
        if "page" in doc.metadata:
            pages.add(doc.metadata["page"] + 1)

    if pages:
        answer += "\n\n📄 Source Pages: " + ", ".join(map(str, sorted(pages)))

    return answer