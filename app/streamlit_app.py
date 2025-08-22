# --- path bootstrap: keep at very top ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import os, time, uuid, json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from shutil import rmtree

# LlamaIndex
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Observability bits
from app.config import cfg
from app.tracing import get_tracer
from app.metrics import REQS, ERRS, LAT_E2E, TOK_IN, TOK_OUT
from app.logging_utils import log_event

# === ENV: force-load the nearest .env from the current working dir upward
ENV_PATH = find_dotenv(usecwd=True)
load_dotenv(ENV_PATH)

# Silence HF tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Quiet Streamlit file-watcher tip if you don’t want to install watchdog
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# --- helpers: corpus + index mgmt ---
def corpus_exists() -> bool:
    return os.path.exists(cfg.TEXT_FILE) and os.path.isfile(cfg.TEXT_FILE)

def clear_index_dir():
    if os.path.exists(cfg.PERSIST_DIR):
        try:
            rmtree(cfg.PERSIST_DIR)
        except Exception:
            pass
    os.makedirs(cfg.PERSIST_DIR, exist_ok=True)

# ----------------- SECRETS / API KEY (robust) -----------------
def _mask(k: str) -> str:
    return k[:4] + "…" + k[-4:] if k and len(k) >= 10 else "<missing>"

secrets_path = pathlib.Path(".streamlit/secrets.toml")

# Prefer Streamlit secrets if present; fall back to env
secrets_key = None
if secrets_path.exists():
    try:
        secrets_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secrets_key = None

env_key = os.environ.get("GROQ_API_KEY")  # env is populated by load_dotenv above
GROQ_API_KEY = (secrets_key or env_key or "").strip()

# Fail fast with a friendly message
if not GROQ_API_KEY:
    where = f"env at {ENV_PATH}" if ENV_PATH else "env (not found)"
    st.error(
        "GROQ_API_KEY is not set. Put it in `.env` or `.streamlit/secrets.toml` and restart.\n\n"
        "Examples:\n"
        "  .env\n    GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXX\n\n"
        "  .streamlit/secrets.toml\n    GROQ_API_KEY = \"gsk_XXXXXXXXXXXXXXXX\"\n\n"
        f"Debug: tried secrets.toml ({'present' if secrets_path.exists() else 'absent'}) and {where}."
    )
    st.stop()
# ---------------------------------------------------------------

# --- helper: best score from retrieved nodes (if available) ---
def _best_score(nodes):
    try:
        return getattr(nodes[0], "score", None) if nodes else None
    except Exception:
        return None

# --- DIY trace storage (local JSON) ---
TRACE_FILE = "local_traces.json"

def store_trace(span_name: str, start_s: float, end_s: float, attributes: dict | None = None):
    """Persist a simple trace event locally for the DIY observability panel."""
    rec = {
        "span_name": span_name,
        "start_ts_ms": int(start_s * 1000),
        "end_ts_ms": int(end_s * 1000),
        "duration_ms": round((end_s - start_s) * 1000, 3),
        "attributes": attributes or {},
        "session_id": st.session_state.get("session_id"),
        "request_id": (attributes or {}).get("request_id"),
    }
    try:
        existing = []
        if os.path.exists(TRACE_FILE):
            with open(TRACE_FILE, "r") as f:
                existing = json.load(f)
        existing.append(rec)
        with open(TRACE_FILE, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        st.warning(f"Could not write {TRACE_FILE}: {e}")

def load_traces_df() -> pd.DataFrame:
    if not os.path.exists(TRACE_FILE):
        return pd.DataFrame()
    try:
        return pd.DataFrame(json.load(open(TRACE_FILE)))
    except Exception:
        return pd.DataFrame()

# --- UI header ---
st.set_page_config(page_title="📖 Chat with Dostoevsky", layout="centered")
st.title("Conversations with *Crime and Punishment*")
st.caption("Ask anything about Dostoevsky's masterpiece and let LLaMA 3 (Groq) guide you.")

# Show where .env was loaded from + masked key (helpful debug)
st.caption(f".env loaded from: {ENV_PATH or 'not found'} · GROQ key: {_mask(GROQ_API_KEY)} · source: {'secrets' if secrets_key else 'env'}")

# Stable session id
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"s_{uuid.uuid4().hex[:8]}"

tracer = get_tracer()

# Emit a test trace (timed by us → stored locally)
if st.button("Emit test trace"):
    t0 = time.perf_counter()
    with get_tracer().start_as_current_span("ui.smoke") as s:
        s.set_attribute("clicked_at", int(time.time()))
    t1 = time.perf_counter()
    store_trace("ui.smoke", t0, t1, {"clicked_at": int(time.time())})
    st.success("Emitted a test trace locally (local_traces.json).")

# === How it works + controls ===
with st.expander(" How it works", expanded=True):
    st.markdown("""
**RAG flow:** retrieve relevant chunks → build prompt with context → generate with LLaMA 3 (Groq) → show answer.  
You can control how many chunks to retrieve (`top_k`). “Rebuild index” clears the on‑disk vector index so it’s rebuilt from the book on the next query.
""")
    st.code(
        f"TEXT_FILE = {cfg.TEXT_FILE}\n"
        f"PERSIST_DIR = {cfg.PERSIST_DIR}\n"
        f"EMBEDDING_MODEL = {cfg.EMBEDDING_MODEL}\n"
        f"GROQ_MODEL = {cfg.GROQ_MODEL}",
        language="bash",
    )
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Rebuild index"):
            clear_index_dir()
            st.success("Cleared index storage. It will rebuild on the next request.")
    with cols[1]:
        st.write("✅ Corpus present" if corpus_exists() else "❌ Corpus missing")

# --- guard: stop early if corpus missing ---
if not corpus_exists():
    st.error(
        f"Corpus file not found at:\n**{cfg.TEXT_FILE}**\n\n"
        "Fix it by either moving the book there or setting `TEXT_FILE` in `.env` "
        "to the correct path (e.g., `assets/crime_and_punishment.txt`)."
    )
    st.stop()

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
def init_chat_engine_with_top_k(top_k: int, _key_version: str = _mask(GROQ_API_KEY)):
    """Return BOTH the chat engine and the underlying retriever for DIY timing."""
    with tracer.start_as_current_span("engine.init") as span:
        retriever = build_or_load_index(top_k=top_k)
        llm = Groq(model=cfg.GROQ_MODEL, api_key=GROQ_API_KEY)  # explicit key
        memory = ChatMemoryBuffer.from_defaults()
        prefix_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.SYSTEM, content="Answer using only the provided context and chat history. Keep your response concise."),
        ]
        span.set_attribute("model", cfg.GROQ_MODEL)
        span.set_attribute("top_k", top_k)
        engine = ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages)
        return engine, retriever

# === Controls ===
top_k = st.slider("🔍 Retrieve how many relevant passages?", min_value=1, max_value=10, value=2, step=1)
chat_engine, retriever = init_chat_engine_with_top_k(top_k)

# === Chat history ===
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
    log_event(
        "query_received",
        request_id=req_id,
        session_id=st.session_state["session_id"],
        prompt=prompt,
        top_k=top_k,
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("🧠 Thinking deeply like Raskolnikov..."):
            # 1) Retrieval timing (explicit, using the retriever we returned)
            t_ret0 = time.perf_counter()
            nodes = retriever.retrieve(prompt)
            t_ret1 = time.perf_counter()
            store_trace("retrieve.topk", t_ret0, t_ret1, {
                "k": top_k, "hits": len(nodes), "best_score": _best_score(nodes), "request_id": req_id
            })

            # 2) Generation timing
            t_gen0 = time.perf_counter()
            with tracer.start_as_current_span("engine.chat") as s:
                s.set_attribute("model", cfg.GROQ_MODEL)
                answer = chat_engine.chat(prompt)
                response = answer.response or ""
            t_gen1 = time.perf_counter()
            store_trace("engine.chat", t_gen0, t_gen1, {"model": cfg.GROQ_MODEL, "request_id": req_id})

            # 3) End-to-end timing (user-perceived)
            t_e2e1 = time.perf_counter()
            store_trace("rag.e2e", start, t_e2e1, {"top_k": top_k, "request_id": req_id})

        LAT_E2E.labels(route="/chat").observe(time.perf_counter() - start)
        log_event("query_answered", request_id=req_id, response_len=len(response), model=cfg.GROQ_MODEL)

        with st.chat_message("assistant"):
            st.markdown(response)

    except Exception as e:
        ERRS.labels(route="/chat", type=type(e).__name__).inc()
        log_event("error", request_id=req_id, error=str(e))
        st.error(f"Something went wrong: {e}")

# === DIY observability panel ===
st.divider()
st.subheader("🔎 DIY Observability")

df = load_traces_df()
if df.empty:
    st.info("No local traces yet. Ask a question to record retrieval, generation, and end‑to‑end timings.")
else:
    df = df.sort_values("start_ts_ms")
    recent = df.tail(20)
    recent["duration_ms"] = pd.to_numeric(recent["duration_ms"], errors="coerce")

    def _avg(span: str) -> float:
        ser = recent.loc[recent["span_name"] == span, "duration_ms"]
        ser = pd.to_numeric(ser, errors="coerce").dropna()
        return float(ser.mean()) if not ser.empty else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Retrieval avg (ms)", f"{_avg('retrieve.topk'):.1f}")
    k2.metric("Generation avg (ms)", f"{_avg('engine.chat'):.1f}")
    k3.metric("End‑to‑end avg (ms)", f"{_avg('rag.e2e'):.1f}")

    show = recent[["span_name","duration_ms","request_id","session_id","start_ts_ms"]].rename(
        columns={"span_name":"stage","duration_ms":"duration (ms)","start_ts_ms":"start (ms)"}
    )
    st.dataframe(show, use_container_width=True)

    chart_df = recent[["start_ts_ms","span_name","duration_ms"]].copy()
    chart_df = chart_df.pivot_table(index="start_ts_ms", columns="span_name", values="duration_ms", aggfunc="last").fillna(0.0)
    st.line_chart(chart_df)
