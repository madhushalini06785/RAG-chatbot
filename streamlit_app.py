import streamlit as st
from ingest import ingest_document
from rag_chain import ask_question
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖")

st.title("📚 AI Document Assistant")
st.write("Ask questions from your uploaded document")

# ---------------- DATABASE CHECK (VERY IMPORTANT) ----------------
# Connect Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Only run ingestion if database is empty
if "db_checked" not in st.session_state:
    stats = index.describe_index_stats()

    if stats["total_vector_count"] == 0:
        with st.spinner("First time setup: Creating AI knowledge base (2-4 minutes) ⏳"):
            ingest_document()
        st.success("Knowledge base created successfully!")
    else:
        st.info("Knowledge base already prepared. Ready to chat!")

    st.session_state.db_checked = True


# ---------------- CHAT SYSTEM ----------------
# Store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_prompt = st.chat_input("Ask something about the document...")

if user_prompt:

    # Show user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤖"):
            response = ask_question(user_prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})