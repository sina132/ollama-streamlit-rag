import ollama
#import nltk
import chromadb
import hashlib
import os
import json
import numpy


MODEL = "llama3.1:8b"
FILE_DIR = "./files"
DB_DIR = "./chroma_db"
HASH_DIR = "./files.json"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
N_RESULTS = 3

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="docs")

def embed(text):
    return ollama.embed(model=MODEL, input=text)["embeddings"]

def hash(filepath):
    with open(filepath,"rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_hashed_files():
    if os.path.exists(HASH_DIR):
        with open(HASH_DIR,"r") as f:
            return json.load(f)
    return {}

def set_hashed_files(data):
    with open(HASH_DIR,"w") as f:
        json.dump(data,f)

# Sentence-aware chunking to preserve semantic context
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    words = text.split(' ')
    for i in range(0,len(words),chunk_size):
        chunks.append(' '.join(words[i:i+chunk_size-overlap]))
    return chunks

def process_files():
    old_files = get_hashed_files()
    new_files = os.listdir(FILE_DIR)
    deleted_files = list(set(old_files) - set(new_files))
    
    if deleted_files:
        collection.delete(ids=deleted_files)
        for file in deleted_files:
            del old_files[file]

    for file in new_files:
        filepath = os.path.join(FILE_DIR,file)
        new_hash = hash(filepath)
        if file in old_files and new_hash == old_files[file]:
            continue

        with open(filepath,"r",encoding="utf8") as f:
            content = f.read()

        chunks = chunk_text(content)
        for i,chunk in enumerate(chunks):
            collection.add(
            ids=[f"{filepath}_chunk_{i}"],
            embeddings=embed(chunk),
            documents=[chunk],
            metadatas=[{
                "filename": filepath,
                "hash": new_hash,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }])
        old_files[file] = new_hash
    
    set_hashed_files(old_files)



process_files()
running = True
messages =[
    {
        "role": "system",
        "content": "You are a helpful assistant. Use the provided context to answer questions accurately."
    },
]
while running:

    prompt = input(":")
    if prompt.strip() == "bye":
        running = False
        break
    embedded_prompt = embed(prompt)

    results = collection.query(
        query_embeddings=embedded_prompt,
        n_results=N_RESULTS,
        include=["documents","metadatas"]
    )
    data = [f"content: {c}\nfile:{m['filename']}" for c, m in zip(results["documents"][0], results["metadatas"][0])]

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{'\n'.join(data)}\n\nQuestion: {prompt}"
        }
    )

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={
            "temperature": 1, #0:most #1:most
            "num_ctx": 4096
        },
        stream=True
    )
    
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)

    messages.append(
        {
            "role":"assistant",
            "content":response.message.content
        }
    )
