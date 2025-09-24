
# src/ui/app.py

import streamlit as st
import yaml
import sys
import os
from thefuzz import process

# --- System Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- Backend Imports ---
from src.ingestion.excel_parser import parse_excel_qa
from src.bot_engine.gemini_responder import get_rag_chain
# We now only need this one function for the vector store
from src.vector_store.vector_builder import get_or_create_vector_store

# --- Page Configuration ---
st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
st.title("IRCTC Chatbot: Ask all your queries")
st.subheader("CENTER FOR RAILWAY INFORMATION SYSTEMS")
st.write("Ask a question about your documents, or check our FAQs!")

@st.cache_resource
def load_all_resources():
    """
    Loads all necessary resources using the robust get_or_create_vector_store function.
    """
    print("\n--- INITIATING RESOURCE LOADING ---")

    # --- 1. Load Config (Hybrid Approach) ---
    config = {}
    settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    try:
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        print("1. Loaded config from local 'settings.yaml' file.")
        if "API_KEY" in st.secrets:
            config['gemini']['api_key'] = st.secrets["API_KEY"]
    except FileNotFoundError:
        print("1. 'settings.yaml' not found. Loading config from Streamlit secrets.")
        if "API_KEY" in st.secrets:
            config = {
                "gemini": {
                    "api_key": st.secrets["API_KEY"],
                    "embedding_model": "models/embedding-001",
                    "llm_model": "models/gemini-1.5-flash-latest"
                },
                "data": {
                    "pdf_path": "data/pdf",
                    "excel_path": "data/excelfile.xlsx",
                    "vector_store_path": "vector_store/faiss_index"
                },
                "ingestion": {
                    "parsing_strategy": "fast"
                }
            }
        else:
            st.error("API Key not found in Streamlit secrets.")
            st.stop()

    # --- 2. Load or Build the Vector Store and Create Retriever ---
    vector_store = get_or_create_vector_store(config)
    if vector_store is None:
        st.error("Failed to load or build the vector store. App cannot continue.")
        st.stop()
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    print("Retriever created successfully.")

    # --- 3. Load other resources ---
    faq_data = None
    rag_chain = None

    try:
        excel_path = os.path.join(PROJECT_ROOT, config['data']['excel_path'])
        faq_data = parse_excel_qa(excel_path)
        print(f"FAQ Data Loaded: {'SUCCESS' if faq_data is not None else 'FAILED'}")
    except Exception as e:
        print(f"FAQ Data Loaded: FAILED with an exception: {e}")

    try:
        rag_chain = get_rag_chain(retriever)
        print(f"RAG Chain Loaded: {'SUCCESS' if rag_chain is not None else 'FAILED'}")
    except Exception as e:
        print(f"RAG Chain Loaded: FAILED with an exception: {e}")
    
    # --- Final Check ---
    if faq_data is None or retriever is None or rag_chain is None:
        st.error("Failed to load one or more resources. Please check terminal logs for details.")
        st.stop()
        
    print("--- ALL RESOURCES LOADED SUCCESSFULLY ---\n")
    return faq_data, retriever, rag_chain

# --- Load all resources and assign them to variables ---
faq_data, retriever, rag_chain = load_all_resources()

# --- [The rest of your app.py (Chat Logic, UI State, Main Interaction) is correct and can remain the same] ---
def get_faq_answer(query: str, faqs: list[dict]) -> str or None:
    if not faqs: return None
    faq_questions = [item['user_desc'] for item in faqs]
    best_match = process.extractOne(query, faq_questions, score_cutoff=90)
    
    if best_match:
        best_matching_question_text = best_match[0]
        for item in faqs:
            if item['user_desc'] == best_matching_question_text:
                print(f"FAQ Match Found: '{query}' -> '{best_matching_question_text}' (Score: {best_match[1]})")
                return item['user_reply_desc']
    return None

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            faq_answer = get_faq_answer(prompt, faq_data)
            
            if faq_answer:
                response = f"**From FAQ:**\n\n{faq_answer}"
            else:
                st.info("No FAQ match found. Searching documents...")
                response = rag_chain.invoke(prompt)
                response = response.replace("<br><br>", "\n\n")
            # Replace any single <br> with a single newline
                response = response.replace("<br>", "\n")
            
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})

