# src/vector_store/retriever.py

import nest_asyncio
nest_asyncio.apply()

import yaml
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import streamlit as st # Import streamlit to access secrets

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- DEFINE PROJECT ROOT for reliable file paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def get_retriever():
    """
    Loads the FAISS vector store using absolute paths for deployment compatibility.
    """
    try:
        # --- Load Config using absolute path and hybrid secrets logic ---
        config = {}
        settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        try:
            with open(settings_path, 'r') as f:
                config = yaml.safe_load(f)
            if "API_KEY" in st.secrets:
                config['gemini']['api_key'] = st.secrets["API_KEY"]
        except FileNotFoundError:
            if "API_KEY" in st.secrets:
                config = {
                    "gemini": {"api_key": st.secrets["API_KEY"], "embedding_model": "models/embedding-001"},
                    "data": {"vector_store_path": "vector_store/faiss_index"}
                }
            else:
                raise ValueError("API Key not found in Streamlit secrets.")

        api_key = config['gemini']['api_key']
        embedding_model = config['gemini']['embedding_model']
        
        # --- Use absolute path for the vector store ---
        vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])

        # 2. Initialize embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model, 
            google_api_key=api_key
        )
        
        # 3. Load the local FAISS vector store
        log.info(f"Loading vector store from {vector_store_path}...")
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True 
        )
        
        log.info("Vector store loaded successfully.")
        return vector_store.as_retriever(search_kwargs={"k": 7})

    except FileNotFoundError:
        log.error(f"Vector store not found at {vector_store_path}. Please ensure it is built.")
        st.error(f"Vector store not found at {vector_store_path}. Build process may have failed.")
        return None
    except Exception as e:
        log.error(f"An error occurred while loading the retriever: {e}")
        st.error(f"An error occurred while loading the retriever: {e}")
        return None

# src/vector_store/retriever.py

