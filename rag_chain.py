import os

from dotenv import load_dotenv

from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX = os.getenv(
    "PINECONE_INDEX"
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# PINECONE
# =========================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


index = pc.Index(
    PINECONE_INDEX
)


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(

    groq_api_key=GROQ_API_KEY,

    model_name="openai/gpt-oss-20b",

    temperature=0

)


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(
    query: str,
    namespace: str
):

    # -----------------------------------------------------
    # Vector store for CURRENT SESSION ONLY
    # -----------------------------------------------------

    vectorstore = PineconeVectorStore(

        index=index,

        embedding=embedding,

        namespace=namespace

    )


    # -----------------------------------------------------
    # Retriever
    # -----------------------------------------------------

    retriever = vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": 4
        }

    )


    # -----------------------------------------------------
    # Strict RAG Prompt
    # -----------------------------------------------------

    template = """

You are an AI document question-answering assistant.

Your job is to answer the user's question ONLY from
the information contained in the provided context.

STRICT RULES:

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT use information from previous sessions.
4. Do NOT assume missing information.
5. Do NOT invent facts.
6. If the answer is not present in the context, respond exactly:

"I could not find the answer in the uploaded documents."

7. You may combine information from multiple uploaded
   documents if that information is present in the context.

Context:
{context}

Question:
{question}

Answer:
"""


    prompt = ChatPromptTemplate.from_template(
        template
    )


    # -----------------------------------------------------
    # RAG Chain
    # -----------------------------------------------------

    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        retriever=retriever,

        chain_type="stuff",

        return_source_documents=True,

        chain_type_kwargs={
            "prompt": prompt
        }

    )


    # -----------------------------------------------------
    # Ask question
    # -----------------------------------------------------

    result = qa_chain.invoke({

        "query": query

    })


    answer = result[
        "result"
    ]


    documents = result[
        "source_documents"
    ]


    # =====================================================
    # SOURCES
    # =====================================================

    sources = set()

    pages = set()


    for document in documents:

        metadata = document.metadata


        # Source filename

        if "source" in metadata:

            sources.add(
                metadata["source"]
            )


        # PDF page

        if "page" in metadata:

            pages.add(
                metadata["page"] + 1
            )


    # =====================================================
    # ADD SOURCE FILES
    # =====================================================

    if sources:

        answer += (

            "\n\n📄 **Sources:** "

            + ", ".join(
                sorted(sources)
            )

        )


    # =====================================================
    # ADD PDF PAGES
    # =====================================================

    if pages:

        answer += (

            "\n📖 **Source Pages:** "

            + ", ".join(

                map(
                    str,
                    sorted(pages)
                )

            )

        )


    return answer