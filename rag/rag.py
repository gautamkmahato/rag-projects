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

load_dotenv()

# Configuration
PDF_PATH = "C:\\Users\\Gautam Kumar Mahato\\Desktop\\apps\\chai\\node.pdf"
COLLECTION_NAME = "node_pdf_collection"  # Changed collection name

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
# loader = PyMuPDFLoader(PDF_PATH)
# documents = loader.load()

# print("document loaded successfully")

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )
# docs = text_splitter.split_documents(documents)

# print(f"Number of document chunks created: {len(docs)}")

# Create Qdrant vector store with force_recreate=True
# vector_store = Qdrant.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY"),
#     collection_name=COLLECTION_NAME,
#     force_recreate=True,  # This will delete and recreate the collection
#     prefer_grpc=True
# ) 

# print("successfully created the vector collection")

def load_vector_store():
    client=QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True
        )
    return QdrantVectorStore(
        client=client,
        collection_name="node_pdf_collection",
        embedding=embeddings
    )

# Example usage
print("========= start chat ===========")

while(True):
    query = input("User: ")
    if(query == "quit"):
        break
    
    # Step 1: Embed the user query
    embedded_query = embeddings.embed_query(query)

    vector_store = load_vector_store()
    results = vector_store.similarity_search(
        query=query,
        k=3
    )

    # Step 2: Do similarity search by vector
    result = vector_store.similarity_search_by_vector(embedded_query, k=3)

    # After similarity search
    retrieved_text = ""

    for doc in results:
        retrieved_text = retrieved_text + doc.page_content

    #print(retrieved_text)

    final_prompt = f"""
    You are an expert assistant. Based on the following retrieved information, answer the user's question.

    Context:
    {retrieved_text}

    Question:
    {query}
    """

    # Now use LLM
    output = llm.invoke(final_prompt)
    print("\nAssistant:", output.content)
