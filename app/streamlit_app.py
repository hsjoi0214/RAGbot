import os
import streamlit as st
import pathlib
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# === Load environment variables ===
load_dotenv()

secrets_path = pathlib.Path(".streamlit/secrets.toml")

if secrets_path.exists():
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Constants ===
TEXT_FILE = "data/crime_and_punishment.txt"
PERSIST_DIR = "storage/vector_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# === Embedding setup ===
@st.cache_resource
def build_or_load_index(top_k):
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        docs = SimpleDirectoryReader(input_files=[TEXT_FILE]).load_data()
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, embed_model=embed_model)

    return index.as_retriever(similarity_top_k=top_k)

# === LLM + Chat Engine ===
@st.cache_resource
def init_chat_engine_with_top_k(top_k):
    retriever = build_or_load_index(top_k=top_k)

    llm = Groq(
        model=GROQ_MODEL,
        token=GROQ_API_KEY
    )

    memory = ChatMemoryBuffer.from_defaults()
    prefix_messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.SYSTEM, content="Answer using only the provided context and chat history."),
    ]

    return ContextChatEngine(
        llm=llm,
        retriever=retriever,
        memory=memory,
        prefix_messages=prefix_messages
    )

# === UI Config ===
st.set_page_config(page_title="📖 Chat with Dostoevsky", layout="centered")

st.title(" RAGbot — Conversations with *Crime and Punishment*")
st.caption("Ask anything about Dostoevsky's masterpiece and let LLaMA 3 guide you.")

# Step 1: Welcome and explanation
with st.expander(" How it works", expanded=True):
    st.markdown("""
        Welcome to **RAGbot** — a Retrieval-Augmented Generation chatbot based on *Crime and Punishment*.

        **How it works:**
        - Retrieves relevant passages from the book.
        - Uses LLaMA 3 (via Groq) to generate responses.
        - Memory-aware: keeps track of your conversation.
        
        ---  
        """)
    st.markdown("👇 Start by choosing how deep the model should search for context:")

# Step 2: Configure top_k
top_k = st.slider("🔍 Retrieve how many relevant passages?", min_value=1, max_value=10, value=2, step=1)

# Step 3: Initialize the chat engine
chat_engine = init_chat_engine_with_top_k(top_k)

# Step 4: Chat UI
st.divider()
st.subheader("🗨️ Ask away")

# Show history
for message in chat_engine.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.content)

# Input
if prompt := st.chat_input(" Ask a philosophical or narrative question..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🧠 Thinking deeply like Raskolnikov..."):
        response = chat_engine.chat(prompt).response

    with st.chat_message("assistant"):
        st.markdown(response)
