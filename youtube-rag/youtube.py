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
import yt_dlp
import whisper
import re
from langchain.schema import Document


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

# split docs to chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


def split_to_chunks():
    print("[info] chunking started...")
    with open("transcript_1745568637.txt", "r", encoding="utf-8") as f:
        transcript_data = f.read()
    # Wrap the text in a Document object
    docs = [Document(page_content=transcript_data)]
    split_docs = text_splitter.split_documents(docs)
    print("[info] Successfully splitted the chunks: ", len(split_docs))
    return split_docs

def add_chunks_to_vector_store(chunks): 
    COLLECTION_NAME = 'hitesh_video_collection'
    chunks = chunks
    print("[info] Creating Vector DB: ", COLLECTION_NAME)
    vector_store = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME,
        force_recreate=True,  # This will delete and recreate the collection
        prefer_grpc=True
    ) 
    print("[info] Successfully stored the chunks in ", COLLECTION_NAME)

def load_vector_store():
    client=QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True
        )

    return QdrantVectorStore(
        client=client,
        collection_name="hitesh_video_collection",
        embedding=embeddings
    )

def search_from_vectorDB(query):
    vector_store = load_vector_store()
    # result = vector_store.similarity_search_with_relevance_scores
    results = vector_store.similarity_search_with_score(
        query=query,
        k=5
    )
    return results

def create_context_for_llm(results):
    context = []
    for document, score in results:
        retrieved_text = {}
        retrieved_text["content"] = document.page_content
        retrieved_text["score"] = score
        context.append(retrieved_text)

    # print(context)
    return context

def search_using_llm(context, query):
    final_prompt = f"""
        You are an AI assistant that helps users understand the content of a specific YouTube video by answering questions based on its transcript.

        You have access to a set of context chunks from the video’s transcript, which may include dialogue, narration, and descriptions of actions. These context chunks are highly relevant to the user's query.

        Based on the user query first analyze the query then find the relevent information from the given context, club all the context and then 
        use your own knowledge and answer the query. Provide most relevent timestamp


        Transcript Context:
        {context}

        Question:
        {query}
    """

    # Now use LLM
    output = llm.invoke(final_prompt)
    return output

def start_chat():
    print("========= chat start ===========")
    while True:
        query = input("\nUser: ")
        if(query == 'quit' or query == 'exit'):
            break
        results = search_from_vectorDB(query)
        # print("results: ", results)
        context = create_context_for_llm(results)
        # print("context: ", context)
        output = search_using_llm(context, query)
        print("Assistant:", output.content)

def run():
    # chunks = split_to_chunks()
    # add_chunks_to_vector_store(chunks)
    start_chat()

run()