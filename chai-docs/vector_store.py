import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Changed import
from langchain_qdrant import Qdrant
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document  # Import Document class


load_dotenv()

# Configuration
PDF_PATH = "C:\\Users\\Gautam Kumar Mahato\\Desktop\\apps\\chai\\RAG\\chai-docs\\chai_docs.pdf"
COLLECTION_NAME = "chai_docs_collection"  # Changed collection name

google_api_key = os.getenv("GOOGLE_API_KEY")

# Create the LLM with the correct model and key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",  # Use the correct model name
    google_api_key=google_api_key
)

# Initialize Google Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Gemini embedding model
    google_api_key=os.getenv("GOOGLE_API_KEY")  # Changed to Google API key
)

# Load and process PDF
loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()

print("[info] Document loaded successfully")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

print(f"[info] Number of document chunks created: {len(docs)}")

# Create Qdrant vector store with force_recreate=True
print(f"[info] Creating the vector store {COLLECTION_NAME}...")
vector_store = Qdrant.from_documents(
    documents=docs,
    embedding=embeddings,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=COLLECTION_NAME,
    force_recreate=True,  # This will delete and recreate the collection
    prefer_grpc=True
) 

print("[info] Successfully created the vector collection")
