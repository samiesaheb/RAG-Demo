import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import ollama

# --- Load data ---
@st.cache_resource
def load_index_and_texts():
    index = faiss.read_index("rag_index.faiss")
    with open("rag_chunks.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["texts"]

index, texts = load_index_and_texts()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- Retrieval function ---
def retrieve_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    query_embedding = normalize(query_embedding, axis=1)
    D, I = index.search(query_embedding, k)
    return [texts[i] for i in I[0]]

# --- LLM generation function ---
def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    messages = [{'role': 'user', 'content': prompt}]
    response = ollama.chat(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest', messages=messages)
    return response['message']['content']

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chat", page_icon="🤖")
st.title("💬 RAG Chat with HuggingFace Docs")

query = st.text_input("Ask a question:", placeholder="e.g. What is Hugging Face used for?")

if query:
    with st.spinner("Retrieving and answering..."):
        chunks = retrieve_chunks(query, k=3)
        answer = generate_answer(query, chunks)
        st.markdown("### Answer")
        st.success(answer)

        with st.expander("🔍 Retrieved Chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}:** {chunk}")
