import datasets

ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")
documents = [doc["text"] for doc in ds]

print(f"Loaded {len(documents)} documents.")

def chunk_examples(examples):
    chunks = []
    for text in examples['text']:  # replace 'text' with your dataset's text field name
        # Split text into 200-character chunks
        chunks += [text[i:i + 200] for i in range(0, len(text), 200)]
    return {'chunks': chunks}

chunked_ds = ds.map(chunk_examples, batched=True, remove_columns=ds.column_names)

print(chunked_ds[:10])

print(ds.column_names)

texts = chunked_ds['chunks']

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight and effective

embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

import faiss
import numpy as np

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance metric

index.add(np.array(embeddings))
print(f"Indexed {index.ntotal} vectors.")

faiss.write_index(index, "rag_index.faiss")

import pickle
with open("rag_chunks.pkl", "wb") as f:
    pickle.dump(texts, f)

query = "Tell me about cat agility."

query_embedding = embedder.encode([query])

D, I = index.search(np.array(query_embedding), k=3)  # retrieve top 3 chunks

retrieved_chunks = [texts[i] for i in I[0]]

import ollama

context = "\n".join(retrieved_chunks)
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

messages = [{'role': 'user', 'content': prompt}]

response = ollama.chat(model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest', messages=messages)

print(response['message']['content'])
