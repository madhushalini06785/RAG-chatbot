import streamlit as st
from rag_chain import ask_question, chat_history

st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖")

st.title("📚 AI Document Assistant")
st.write("Ask questions from your uploaded document")

# Initialize session chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_prompt = st.chat_input("Ask something about the document...")

if user_prompt:

    # show user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_question(user_prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})