# RAG-based QA Bot using Pinecone, HuggingFace & LangChain
This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** Question Answering (QA) Bot using:
1.LangChain
2.Pinecone (Vector Database)
3.HuggingFace (Embeddings & LLMs)
4.Text Documents
5.Sentence Transformers

## Features
✅ Document loading and chunking  
✅ Text embeddings using `sentence-transformers/all-MiniLM-L6-v2`  
✅ Pinecone vector storage and retrieval  
✅ Local HuggingFace model (`FLAN-T5`) for question answering  
✅ Fully functional Retrieval-Augmented QA pipeline

## Installation
pip install -U langchain langchain-community sentence-transformers pinecone-client==3.0.0 transformers

## Environment Setup
import os
os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key"
os.environ["PINECONE_ENV"] = "us-east-1"

## Document Preparation
1.Upload your .txt file (e.g., Business_info.txt) in Google Colab or local environment.
2.Load and split the document:

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = TextLoader("Business_info.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

## Generate Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = [doc.page_content for doc in split_docs]
metadatas = [doc.metadata for doc in split_docs]
embeddings = embedding_model.embed_documents(texts)

## Upload to Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("index")  # Use your Pinecone index name
all_vectors = [
    {
        "id": f"chunk-{i}",
        "values": embeddings[i],
        "metadata": {**metadatas[i], "text": texts[i]}
    }
    for i in range(len(embeddings))
]
index.upsert(vectors=all_vectors)

## Ask Questions with QA Bot
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
vectorstore = LangchainPinecone.from_existing_index(index_name="index", embedding=embedding_model)
retriever = vectorstore.as_retriever()
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_length=512)
llm = HuggingFacePipeline(pipeline=qa_pipeline)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
# Ask a question
response = qa_chain.invoke("What services does the company offer?")
print("Answer:", response)

## Notes
This project uses a local model (FLAN-T5-small), so it doesn't require an OpenAI API key.
You can scale this up with larger HuggingFace models or integrate OpenAI for better results.

## License
MIT License

## Author
K.Madhu Shalini – B.Tech 3rd Year Student | AI/ML Enthusiast
Connect on LinkedIn (http://www.linkedin.com/in/kanneboina-madhushalini-8884632b6)
