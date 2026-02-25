from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX

def ingest_document():

    print("Loading document...")
    loader = PyPDFLoader("data/notes.pdf")
    docs = loader.load()

    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    splits = splitter.split_documents(docs)

    print("Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [doc.page_content for doc in splits]
    embeddings = embedding_model.embed_documents(texts)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    print("Uploading vectors to Pinecone...")

    vectors = []
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": emb,
            "metadata": {"text": text}
        })

    index.upsert(vectors=vectors, namespace="default")

    print("SUCCESS: Vectors uploaded to Pinecone!")

if __name__ == "__main__":
    ingest_document()