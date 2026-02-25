import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX


NAMESPACE = "default"


def index_already_exists(index):
    """
    Check if vectors already exist in Pinecone.
    Prevents re-ingesting every time Streamlit restarts.
    """
    stats = index.describe_index_stats()
    total_vectors = stats.get("namespaces", {}).get(NAMESPACE, {}).get("vector_count", 0)
    return total_vectors > 0


def ingest_document():

    # ---------- Connect Pinecone ----------
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    # ---------- Check if already indexed ----------
    if index_already_exists(index):
        print("Pinecone already contains vectors. Skipping ingestion.")
        return

    # ---------- Load PDF ----------
    print("Loading document...")
    pdf_path = os.path.join("data", "notes.pdf")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            "notes.pdf not found. Make sure data/notes.pdf is pushed to GitHub."
        )

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # ---------- Split text ----------
    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = splitter.split_documents(documents)

    # ---------- Embedding model ----------
    print("Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [doc.page_content for doc in splits]
    embeddings = embedding_model.embed_documents(texts)

    # ---------- Prepare vectors ----------
    print("Preparing vectors...")

    vectors = []
    for i, (doc, emb) in enumerate(zip(splits, embeddings)):
        metadata = {
            "text": doc.page_content,
            "page": doc.metadata.get("page", "unknown"),
            "source": "notes.pdf"
        }

        vectors.append({
            "id": f"chunk-{i}",
            "values": emb,
            "metadata": metadata
        })

    # ---------- Upload ----------
    print("Uploading vectors to Pinecone...")
    index.upsert(vectors=vectors, namespace=NAMESPACE)

    print("SUCCESS: Knowledge base created in Pinecone!")


if __name__ == "__main__":
    ingest_document()