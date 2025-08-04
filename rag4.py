import asyncio
import getpass
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()

# --- Setup ---
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load and split documents (this part is synchronous and fine)
print("Loading documents...")
loader = WebBaseLoader(
    web_paths=["https://www.casact.org/"]
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Add documents to the vector store
print("Adding documents to Chroma...")
vectorstore = Chroma(
    collection_name="educosys_genai_info",
    embedding_function=embeddings,
    persist_directory="./chroma_genai"
)
vectorstore.add_documents(documents=all_splits)
print(f"Total stored chunks: {vectorstore._collection.count()}")

# --- Fix for the 'no current event loop' error ---
# The original error occurred because `create_react_agent` was trying to call a synchronous tool (`retrieve_context`)
# in an asynchronous streaming context, which can cause issues with thread pools.
# The correct approach is to make the tool itself asynchronous and use `await` with the agent's stream.

@tool
async def retrieve_context(query: str):
    """Search for info related to the CAS"""
    try:
        # Re-initialize the vector store for the tool's context
        vector_store = Chroma(
            collection_name="educosys_genai_info",
            embedding_function=embeddings,
            persist_directory="./chroma_genai",
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        print(f"Querying retrieve_context with: {query}")
        print("--------------------------------------------------------------")

        # The key change: Use `await retriever.ainvoke(query)` for asynchronous retrieval
        results = await retriever.ainvoke(query)

        print(f"Retrieved documents: {len(results)} matches found")
        for i, doc in enumerate(results):
            print(f"Document {i + 1}: {doc.page_content[:100]}...")
        
        print("--------------------------------------------------------------")

        content = "\n".join([doc.page_content for doc in results])
        if not content:
            print(f"No content retrieved for query: {query}")
            return f"No reviews found for '{query}'."
        
        print("--------------------------------------------------------------")
        print(f"Returning content: {content[:200]}...")
        return content
    except Exception as e:
        print(f"Error in retrieve_context: {e}")
        # Return a more descriptive error message
        return f"Error retrieving context for '{query}'. Please try again. Error: {e}"

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Create the agent with the asynchronous tool
agent_executor = create_react_agent(llm, [retrieve_context])

async def main():
    """Main asynchronous function to run the agent."""
    input_message = "give me the founding year of cas?"
    
    print("Starting agent stream...")
    # The agent's stream method should be awaited
    async for event in agent_executor.astream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values"
    ):
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    # The entire script must be run within an asyncio event loop
    asyncio.run(main())