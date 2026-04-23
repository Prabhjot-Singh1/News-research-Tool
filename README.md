#  News Research Tool 📈

RockyBot is a beginner-friendly Retrieval-Augmented Generation (RAG) web application. It allows you to input news article URLs, process their content, and ask questions based strictly on the extracted information. 

This project uses **LangChain**, a local **FAISS** index, **Hugging Face embeddings**, and the powerful **Groq API** to provide fast and accurate answers. No paid OpenAI dependencies are used.

## Project Overview
As a fresher-friendly AI project, the goal is to demonstrate a clean, understandable pipeline for processing text data from the web and interacting with it via a Large Language Model (LLM).

## How It Works
1. **Input URLs:** Users can provide up to 3 news article URLs through the sidebar.
2. **Data Extraction:** The app uses LangChain's `UnstructuredURLLoader` to read the text content of the articles.
3. **Chunking Text:** The long article text is split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
4. **Vector Embeddings:** The chunks are converted into vector embeddings using free Hugging Face models.
5. **Vector Database:** These embeddings are stored locally using a FAISS vector index.
6. **Query & Retrieval:** When the user asks a question, the app searches the FAISS database for the most relevant chunks.
7. **Answer Generation:** The retrieved context is sent to a conversational LLM via the Groq API to formulate a concise and accurate answer. Article sources are also displayed.

## How to Run Locally in VS Code

### 1. Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### 2. Clone the Repository
Open your VS Code terminal and navigate to or clone the project folder.

### 3. Create a Virtual Environment
It's standard practice to use a virtual environment for Python projects:
```bash
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 4. Install Dependencies
Install all required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 5. Setup Environment Variables
1. Create a copy of the `.env.example` file and rename it to `.env`.
2. Open the `.env` file and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

### 6. Run the Application
Start the Streamlit development server by running:
```bash
streamlit run app.py
```
This will automatically open the web application in your default browser at `http://localhost:8501`.
