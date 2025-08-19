# --- path bootstrap: keep at very top ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

# app/streamlit_app.py
import os, pathlib, time, uuid
import streamlit as st
from dotenv import load_dotenv
import time
from app.tracing import get_tracer

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Observability bits
# NEW (absolute package imports)
from app.config import cfg
from app.tracing import get_tracer
from app.metrics import REQS, ERRS, LAT_E2E, TOK_IN, TOK_OUT
from app.logging_utils import log_event


load_dotenv()

# === Secrets ===
secrets_path = pathlib.Path(".streamlit/secrets.toml")
if secrets_path.exists():
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Streamlit page ===
st.set_page_config(page_title="📖 Chat with Dostoevsky", layout="centered")
st.title("RAGbot — Conversations with *Crime and Punishment*")
st.caption("Ask anything about Dostoevsky's masterpiece and let LLaMA 3 guide you.")
if st.button("Emit test trace"):
    with get_tracer().start_as_current_span("ui.smoke") as s:
        s.set_attribute("clicked_at", int(time.time()))
    st.success("Sent a test trace — check Jaeger (service: rag-streamlit).")

# Stable session id for correlating requests
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"s_{uuid.uuid4().hex[:8]}"

tracer = get_tracer()

# === Embedding + Index ===
@st.cache_resource
def build_or_load_index(top_k: int):
    t0 = time.perf_counter()
    with tracer.start_as_current_span("index.load_or_build") as span:
        embed_model = HuggingFaceEmbedding(model_name=cfg.EMBEDDING_MODEL)
        if not os.path.exists(cfg.PERSIST_DIR) or not os.listdir(cfg.PERSIST_DIR):
            docs = SimpleDirectoryReader(input_files=[cfg.TEXT_FILE]).load_data()
            index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
            index.storage_context.persist(persist_dir=cfg.PERSIST_DIR)
            span.set_attribute("built", True)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=cfg.PERSIST_DIR)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            span.set_attribute("built", False)
        span.set_attribute("top_k_default", top_k)
        span.set_attribute("embedding_model", cfg.EMBEDDING_MODEL)
    _ = time.perf_counter() - t0
    return index.as_retriever(similarity_top_k=top_k)

@st.cache_resource
def init_chat_engine_with_top_k(top_k: int):
    with tracer.start_as_current_span("engine.init") as span:
        retriever = build_or_load_index(top_k=top_k)
        llm = Groq(model=cfg.GROQ_MODEL, token=GROQ_API_KEY)
        memory = ChatMemoryBuffer.from_defaults()
        prefix_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.SYSTEM, content="Answer using only the provided context and chat history. Keep your response concise."),
        ]
        span.set_attribute("model", cfg.GROQ_MODEL)
        span.set_attribute("top_k", top_k)
        return ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages)

# === How it works panel ===
with st.expander(" How it works", expanded=True):
    st.markdown("""
        **RAG flow**: retrieve relevant chunks → build prompt with context → generate with LLaMA 3 (Groq) → show answer.
        Below, you control how many chunks to retrieve (`top_k`).
    """)

# === Controls ===
top_k = st.slider("🔍 Retrieve how many relevant passages?", min_value=1, max_value=10, value=2, step=1)
chat_engine = init_chat_engine_with_top_k(top_k)

# === Show history ===
st.divider()
st.subheader("🗨️ Ask away")
for message in chat_engine.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.content)

# === Input ===
if prompt := st.chat_input(" Ask a philosophical or narrative question..."):
    REQS.labels(route="/chat").inc()
    req_id = f"r_{uuid.uuid4().hex[:8]}"
    start = time.perf_counter()
    log_event("query_received",
              request_id=req_id,
              session_id=st.session_state["session_id"],
              prompt=prompt,
              top_k=top_k)

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("🧠 Thinking deeply like Raskolnikov..."):
            with tracer.start_as_current_span("rag.request") as root:
                root.set_attribute("request_id", req_id)
                root.set_attribute("session_id", st.session_state["session_id"])
                root.set_attribute("top_k", top_k)

                # We can’t easily split retriever vs LLM inside ContextChatEngine,
                # so we time the full generate call here.
                with tracer.start_as_current_span("engine.chat") as s:
                    s.set_attribute("model", cfg.GROQ_MODEL)
                    answer = chat_engine.chat(prompt)
                    response = answer.response or ""

        # Metrics
        LAT_E2E.labels(route="/chat").observe(time.perf_counter() - start)
        # If you later get token counts, call:
        # TOK_IN.labels(model=cfg.GROQ_MODEL).observe(prompt_tokens)
        # TOK_OUT.labels(model=cfg.GROQ_MODEL).observe(completion_tokens)

        log_event("query_answered",
                  request_id=req_id,
                  response_len=len(response),
                  model=cfg.GROQ_MODEL)

        with st.chat_message("assistant"):
            st.markdown(response)

    except Exception as e:
        ERRS.labels(route="/chat", type=type(e).__name__).inc()
        log_event("error", request_id=req_id, error=str(e))
        st.error(f"Something went wrong: {e}")

