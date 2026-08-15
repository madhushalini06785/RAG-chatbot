import uuid
import streamlit as st

from ingest import ingest_document
from rag_chain import ask_question


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# SESSION ID
# =========================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = (
        "session_" + uuid.uuid4().hex
    )


SESSION_NAMESPACE = st.session_state.session_id


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# DOCUMENT STATUS
# =========================================================

if "documents_processed" not in st.session_state:

    st.session_state.documents_processed = False


# =========================================================
# TITLE
# =========================================================

st.title("📚 AI Document Assistant with multiple documents")

st.write(
    "Upload up to 10 documents and ask questions about them."
)


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_files = st.file_uploader(
    "📂 Upload Documents",
    type=[
        "pdf",
        "docx",
        "txt",
        "csv",
        "xlsx",
        "xls"
    ],
    accept_multiple_files=True,
    help=(
        "Maximum 10 documents. "
        "Supported: PDF, DOCX, TXT, CSV, XLSX, XLS."
    )
)


# =========================================================
# SHOW FILE COUNT
# =========================================================

if uploaded_files:

    if len(uploaded_files) > 10:

        st.error(
            "⚠️ You can upload a maximum of 10 documents."
        )

    else:

        st.info(
            f"📄 {len(uploaded_files)} document(s) selected."
        )


# =========================================================
# PROCESS DOCUMENTS
# =========================================================

if uploaded_files:

    if len(uploaded_files) <= 10:

        if st.button(
            "🚀 Process Documents",
            use_container_width=True
        ):

            progress_bar = st.progress(0)

            status_text = st.empty()

            try:

                result = ingest_document(
                    uploaded_files,
                    namespace=SESSION_NAMESPACE,
                    progress_bar=progress_bar,
                    status_text=status_text
                )

                processed_files = result[
                    "processed_files"
                ]

                failed_files = result[
                    "failed_files"
                ]


                # -----------------------------------------
                # Success
                # -----------------------------------------

                if processed_files > 0:

                    st.success(
                        f"🎉 Successfully processed "
                        f"{processed_files} document(s)."
                    )

                    st.session_state.documents_processed = True


                # -----------------------------------------
                # Failed files
                # -----------------------------------------

                if failed_files:

                    st.warning(
                        "⚠️ Some documents could not be processed:"
                    )

                    for filename, error in failed_files:

                        st.write(
                            f"• **{filename}**: {error}"
                        )


                # -----------------------------------------
                # Complete
                # -----------------------------------------

                progress_bar.progress(100)

                status_text.success(
                    "Document processing completed! 🎉"
                )


            except Exception as e:

                st.error(
                    f"❌ Error while processing documents: {e}"
                )


# =========================================================
# DIVIDER
# =========================================================

st.divider()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

user_prompt = st.chat_input(
    "Ask something about your documents..."
)


# =========================================================
# USER QUESTION
# =========================================================

if user_prompt:

    # -----------------------------------------
    # Display user question
    # -----------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_prompt
        )


    st.session_state.messages.append({

        "role": "user",

        "content": user_prompt

    })


    # -----------------------------------------
    # Generate answer
    # -----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching your documents... 🤖"
        ):

            try:

                response = ask_question(
                    user_prompt,
                    namespace=SESSION_NAMESPACE
                )

                st.markdown(
                    response
                )


            except Exception as e:

                response = f"❌ Error: {e}"

                st.error(
                    response
                )


    # -----------------------------------------
    # Save response
    # -----------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": response

    })