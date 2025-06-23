import datasets

ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")
documents = [doc["text"] for doc in ds]

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = [Document(page_content=t) for t in documents]
chunks = splitter.split_documents(docs)
texts = [c.page_content for c in chunks]

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

# Normalize embeddings for cosine similarity
embeddings = normalize(embeddings, axis=1)

import faiss
import numpy as np

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product = cosine if normalized

index.add(np.array(embeddings))
print(f"Indexed {index.ntotal} vectors.")

faiss.write_index(index, "rag_index.faiss")

import pickle
with open("rag_chunks.pkl", "wb") as f:
    pickle.dump({"texts": texts}, f)

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np

# Load FAISS index
index = faiss.read_index("rag_index.faiss")

# Load chunked texts
with open("rag_chunks.pkl", "rb") as f:
    data = pickle.load(f)
texts = data["texts"]

# Load embedder again
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    query_embedding = normalize(query_embedding, axis=1)
    D, I = index.search(query_embedding, k)
    return [texts[i] for i in I[0]]

def generate_answer(query, context_chunks):
    import ollama
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    messages = [{'role': 'user', 'content': prompt}]
    response = ollama.chat(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest', messages=messages)
    return response['message']['content']

query = "What is Hugging Face used for?"

top_chunks = retrieve_chunks(query, k=3)
answer = generate_answer(query, top_chunks)

print("Answer:\n", answer)