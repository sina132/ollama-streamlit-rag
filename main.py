import ollama
import chromadb
import hashlib
import os
import json
import streamlit as st
from unstructured.partition.auto import partition

MODEL = "llama3.1:8b"
FILE_DIR = "./files"
DB_DIR = "./chroma_db"
HASH_DIR = "./files.json"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
N_RESULTS = 3
INSTRUCTIONS = "You are a helpful assistant. Use the provided context to answer questions accurately."
TEMPRETURE = 1 #0 = deterministic, 1 = creative
TOKENS = 4096

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

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    words = text.split(' ')
    for i in range(0,len(words),chunk_size):
        chunks.append(' '.join(words[i:i+chunk_size-overlap]))
    return chunks

def extract_text(filepath):
    elements = partition(filename=filepath)
    return "\n".join(str(el) for el in elements)

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

        content = extract_text(filepath)

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

#states:
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": INSTRUCTIONS
        },
    ]
if "running" not in st.session_state:
    st.session_state.running = False

#page config:
st.set_page_config(layout="wide", page_title="Ollama Chat")

def get_context(prompt):
    embedded_prompt = embed(prompt)
    results = collection.query(
        query_embeddings=embedded_prompt,
        n_results=N_RESULTS,
        include=["documents","metadatas"]
    )
    data = [f"content: {c}\nfile:{m['filename']}" for c, m in zip(results["documents"][0], results["metadatas"][0])]
    content = f"Context:\n{'\n'.join(data)}\n\nQuestion: {prompt}"
    return content
def ask_ollama():
    stream = ollama.chat(
        model=MODEL,
        messages=st.session_state.messages,
        stream=True
    )
    for chunk in stream:
        yield chunk['message']['content']

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    if message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])


prompt = st.chat_input("ask AI...",accept_file=True,disabled=st.session_state.running)
if prompt:
    st.session_state.running = True
    if hasattr(prompt, 'files') and prompt.files:
        for file in prompt.files:
            try:
                # Get file information
                file_name = file.name
                file_data = file.read()
                
                # Save file to directory
                file_path = os.path.join(FILE_DIR, file_name)
                
                # Write file to disk
                with open(file_path, "wb") as f:
                    f.write(file_data)
                
                st.success(f"File '{file_name}' saved successfully!")
                
                # Display file info (optional)
                st.write(f"File type: {file.type}")
                st.write(f"File size: {len(file_data)} bytes")
                
            except Exception as e:
                st.error(f"Error saving file: {e}")
        process_files()
        
    context = get_context(prompt.text)
    st.chat_message("user").write(context)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": context
        }
    )

    res = st.chat_message("assistant").write_stream(ask_ollama())
    st.session_state.messages.append({"role": "assistant", "content": res})

    st.session_state.running = False