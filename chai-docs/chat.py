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

def load_vector_store():
    client=QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True
        )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

def search_from_vectorDB(query):
    vector_store = load_vector_store()
    # result = vector_store.similarity_search_with_relevance_scores
    results = vector_store.similarity_search_with_score(
        query=query,
        k=5
    )
    # print(results)
    return results

def create_context_for_llm(results):
    context = []
    for document, score in results:
        retrieved_text = {}
        retrieved_text["content"] = document.page_content
        retrieved_text["score"] = score
        retrieved_text["page"] = document.metadata["page"]
        context.append(retrieved_text)

    # print(context)
    return context

def search_using_llm(context, query):
    final_prompt = f"""

        You are a helpful and precise assistant. 
        Using only the retrieved context below, answer the user's question.

        - Do not assume anything beyond the provided context.
        - If no relevant answer is found, reply: "Sorry, I don't have any clear information on that topic."

        IMPORTANT - Your response MUST include:
            1. The "page number" from the context with the highest relevance score.
            2. ALWAYS include ALL source URLs that appear in the format "[URL]: https://docs.chaicode.com/..." in the context.
            3. Present these URLs exactly as they appear, without modification or abbreviation.
            4. List ALL relevant URLs in a "Sources:" section at the end of your response.

        Context:
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
    start_chat()

run()