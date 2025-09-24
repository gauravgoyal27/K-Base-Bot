

import sys
import os
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- System Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- Now import from your src module ---
from src.ingestion.pdf_loader import load_and_process_pdfs

def get_or_create_vector_store(config: dict):
    """
    Checks if the vector store exists. If so, loads it.
    If not, builds it, saves it, and returns the store object directly from memory.
    This function is now completely decoupled from Streamlit.
    """
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    api_key = config['gemini']['api_key']
    
    # --- 1. Check if store exists, and load it ---
    if os.path.exists(vector_store_path):
        print("Vector store found. Loading from disk...")
        embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
        return vector_store

    # --- 2. If it doesn't exist, build it ---
    else:
        # UI messages like st.info() are now handled by the calling script (app.py)
        print("Knowledge base not found. Triggering build process...")
        
        pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
        documents = load_and_process_pdfs(pdf_path, config)
        if not documents:
            # Error messages are now simple prints; app.py will show the st.error()
            print("ERROR: No documents were loaded to build the knowledge base.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)
        
        print("Building and saving FAISS vector store...")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(vector_store_path)
        print(f"Knowledge base built and saved successfully at {vector_store_path}")
        # Return the newly created object directly from memory
        return vector_store

# This block allows you to still run this script directly from the command line for local building
if __name__ == '__main__':
    # When run directly, it loads its own config from the standard path
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
        main_config = yaml.safe_load(f)
    get_or_create_vector_store(main_config)




