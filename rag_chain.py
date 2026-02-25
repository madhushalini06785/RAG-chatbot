import os
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------- LLM --------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# -------------------- MEMORY --------------------
# simple conversation memory (stable on Streamlit)
chat_history = []

# -------------------- ASK FUNCTION --------------------
def ask_question(query: str) -> str:
    global chat_history

    # 1️⃣ Retrieve relevant chunks
    docs = retriever.invoke(query)

    context_parts = []
    sources = set()

    for doc in docs:
        # collect context
        context_parts.append(doc.page_content)

        # collect citations
        if "page" in doc.metadata:
            sources.add(f"Page {doc.metadata['page'] + 1}")
        else:
            sources.add("Unknown page")

    context = "\n\n".join(context_parts)

    # 2️⃣ Build chat history text
    history_text = ""
    for q, a in chat_history:
        history_text += f"User: {q}\nAssistant: {a}\n"

    # 3️⃣ Strict anti-hallucination prompt
    prompt = f"""
You are a document question answering AI.

Follow these STRICT rules:
1. Answer ONLY from the provided context
2. Do NOT use your own knowledge
3. If answer is missing say exactly:
"I could not find the answer in the document."

Conversation History:
{history_text}

Context:
{context}

Question: {query}

Answer:
"""

    # 4️⃣ LLM call
    response = llm.invoke(prompt)
    answer = response.content

    # 5️⃣ Attach citations
    source_text = "\n\nSources:\n" + "\n".join([f"• {s}" for s in sources])
    final_answer = answer + source_text

    # 6️⃣ Save memory
    chat_history.append((query, answer))

    return final_answer


# -------------------- TERMINAL TEST (optional) --------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer = ask_question(query)
        print("\nAnswer:\n", answer)