import os
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Embedding Model ----
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---- Pinecone ----
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding,
    namespace="default"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---- Groq LLM ----
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ---- Manual Memory ----
chat_history = []

def ask_question(query):
    global chat_history

    # Retrieve relevant docs
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build conversation history text
    history_text = ""
    for q, a in chat_history:
        history_text += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer.
If the answer is not present, say:
"I could not find the answer in the document."

Conversation History:
{history_text}

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content

    # Save to memory
    chat_history.append((query, answer))

    return answer


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer = ask_question(query)
        print("\nAnswer:\n", answer)