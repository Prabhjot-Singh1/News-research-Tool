import os
import streamlit as st
from dotenv import load_dotenv

os.environ["USER_AGENT"] = "RockyBot/1.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (like GROQ_API_KEY) from .env file
load_dotenv()

# App title
st.title("News Research Tool 📈")

# Sidebar for URL inputs
st.sidebar.title("News Article URLs")

# Collect up to 3 URLs from user
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# Define where to store the FAISS index locally
faiss_index_path = "faiss_store"

# Create a custom robust wrapper around HuggingFace's Inference API natively
class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str):
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # HuggingFace hub dynamic routers successfully return arrays without 410 errors
        res = self.client.feature_extraction(texts, model=self.model_name)
        return res.tolist()
        
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

# Initialize Hugging Face API Embeddings (Bypasses local CPU AVX limits)
hf_token = os.getenv("HUGGINGFACE_API_KEY")
if not hf_token:
    st.sidebar.error("Hardware Alert: Your computer lacks AVX instruction support for local AI. Please grab a free Hugging Face API token, add `HUGGINGFACE_API_KEY=your_token` to your `.env` file, and refresh!")
    st.stop()
    
embeddings = CustomHuggingFaceEmbeddings(
    api_key=hf_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if process_url_clicked:
    # Filter out empty URLs
    valid_urls = [url.strip() for url in urls if url.strip()]
    
    # Error handling: check if any URLs were entered
    if not valid_urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        st.sidebar.info("Loading article content...")
        try:
            # Step 1: Load data from the provided URLs
            loader = WebBaseLoader(web_paths=valid_urls)
            data = loader.load()
            
            # Error handling: check if content was loaded
            if not data:
                st.sidebar.error("Failed to load content. Please ensure the URLs are valid and accessible.")
            else:
                st.sidebar.info("Splitting text into chunks...")
                
                # Step 2: Split the extracted text into smaller, manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                docs = text_splitter.split_documents(data)
                
                st.sidebar.info("Building Embedding Vector Database...")
                
                # Step 3 & 4: Create embeddings for chunks and store them in FAISS vector database
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                # Save the FAISS index locally so we don't have to re-compute
                vectorstore.save_local(faiss_index_path)
                
                st.sidebar.success("URL processing complete! You can now ask questions.")
        except Exception as e:
            st.sidebar.error(f"Error Context: {repr(e)}")
            if isinstance(e, KeyError) and str(e) in ["0", "'0'"]:
                st.sidebar.warning("⚠️ Hugging Face API Error: The embedding model is currently cold-booting on their servers (or your Token is invalid). Please wait exactly 20 seconds, and carefully click 'Process URLs' again!")
            else:
                st.sidebar.error("If this persists, carefully double-check your URLs and API keys in your .env file.")


# Main area: Text input for the user's question
query = st.text_input("Ask a question about the processed news articles:")

if query:
    # Check if the FAISS index exists (meaning URLs have been processed)
    if os.path.exists(faiss_index_path):
        try:
            # Step 5: Load the local FAISS index
            # allow_dangerous_deserialization=True is required in newer FAISS versions when loading local pickle files
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            
            # Create a retriever to get the most relevant chunks
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Step 6: Setup the LLM using Groq API
            # Note: The GROQ_API_KEY must be in the .env file
            llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
            
            # Step 7: Define the system prompt for the QA generation
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know based on the context. "
                "Use three sentences maximum and keep the answer concise.\n\n"
                "{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Step 8: Build the retrieval chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # Step 9: Get the answer
            result = rag_chain.invoke({"input": query})
            
            # Display the generated answer
            st.subheader("Answer:")
            st.write(result["answer"])
            
            # Display the source URLs for transparency
            st.subheader("Sources:")
            # Use a Set to avoid duplicate sources if multiple chunks came from the same URL
            sources = set([doc.metadata.get("source", "Unknown") for doc in result["context"]])
            for source in sources:
                st.write(f"- {source}")
                
        except Exception as e:
            st.error(f"An error occurred while answering your question: {str(e)}")
    else:
        # Error handling: if user asks a question before processing any URLs
        st.error("Please process some URLs first before asking a question.")
