# streamlit_app.py
"""
================================================================================
Streamlit RAGbot — "Conversations with Crime and Punishment"
================================================================================

WHAT THIS APP DOES (plain English)
----------------------------------
This is a tiny "RAG" chatbot (Retrieval-Augmented Generation). It helps you ask
questions about a single book (Fyodor Dostoevsky's *Crime and Punishment*).

At a high level, when you ask a question:

1) Retrieve: it looks up a few relevant passages ("chunks") from the book.
2) Augment: it builds a prompt that includes those passages + your chat history.
3) Generate: it asks an LLM (LLaMA 3 served by Groq) to answer using only that context.
4) Observe: it records timings (retrieve, generate, end-to-end) so you can see simple
   "trace" metrics in a DIY observability tab.

WHAT YOU NEED (env & files)
---------------------------
- Your Groq API key must be available as GROQ_API_KEY (in .env or Streamlit secrets).
- The book text file must exist where `cfg.TEXT_FILE` points (e.g. assets/crime_and_punishment.txt).
- The app builds/loads a small local index in `cfg.PERSIST_DIR` the first time it runs.

PRIMARY LIBRARIES (just the essentials)
---------------------------------------
- Streamlit: UI and session state.
- LlamaIndex: document reading, embedding, vector index, retrieval, and chat engine.
- Groq (via LlamaIndex): the actual LLM for generation.
- HuggingFace embedding model: to convert book text into vectors for similarity search.

OBSERVABILITY (what we show)
----------------------------
- Local JSON traces (very simple) that capture durations per step.
- A few counters/histograms via `app.metrics` (for example requests/errors).
- A "DIY Observability" tab that shows averages and a timeline chart.

NOTE
----
Everything below is carefully annotated for non-programmers, but the operational
logic is unchanged. Only comments and docstrings were added.
"""

# --- path bootstrap: keep at very top ---
# (Technical: ensures imports like `from app.config import cfg` work even when
# Streamlit launches from a different working directory.)
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

# LlamaIndex: pieces that handle reading files, building a vector index,
# embedding chunks, talking to the LLM, and maintaining chat memory.
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Observability bits (your own small helpers)
from app.config import cfg                 # central configuration (paths, model names, etc.)
from app.tracing import get_tracer         # tracer for recording spans (timed operations)
from app.metrics import REQS, ERRS, LAT_E2E, TOK_IN, TOK_OUT  # prometheus-style counters/histograms
from app.logging_utils import log_event    # lightweight structured logging

# === ENV: load .env explicitly from repo root, fallback to search upward (UNCHANGED) ===
# We load environment variables (like GROQ_API_KEY) from a local ".env" file.
# If that exact file isn't present, we search upward for any .env.
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)           # preferred: exact file
else:
    load_dotenv(find_dotenv(usecwd=True))       # fallback

# Silence HF tokenizers parallelism warning (cosmetic; avoids noisy console messages)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Quiet Streamlit file-watcher tip if you don’t want to install watchdog
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# ----------------- SECRETS / API KEY (robust) -----------------
def _mask(k: str) -> str:
    """Small helper to partially hide the API key when displaying errors."""
    if not k or len(k) < 10:
        return "<missing>"
    return k[:4] + "..." + k[-4:]

# Streamlit has a built-in place for secrets: .streamlit/secrets.toml
# We prefer that if present; otherwise we use .env (already loaded above).
secrets_path = pathlib.Path(".streamlit/secrets.toml")

# Prefer Streamlit secrets if present; else env loaded above
secrets_key = None
if secrets_path.exists():
    try:
        secrets_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secrets_key = None

# Pull the key from (secrets OR env). If missing or malformed, we stop with a friendly error.
raw_key = (secrets_key or os.environ.get("GROQ_API_KEY") or "")
# normalize: strip + remove accidental newline / carriage returns
GROQ_API_KEY = raw_key.strip().replace("\n", "").replace("\r", "")

if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    # This is a user-facing message that explains how to fix the key.
    where = f"{ENV_PATH}" if ENV_PATH.exists() else "search(fallback)"
    st.error(
        "GROQ_API_KEY is missing or malformed. Put it in `.env` (repo root) or `.streamlit/secrets.toml` and restart.\n\n"
        "Examples:\n"
        "  .env                 -> GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXX\n"
        "  .streamlit/secrets.toml -> GROQ_API_KEY = \"gsk_XXXXXXXXXXXXXXXX\"\n\n"
        f"Debug: .env path tried: {where} · secrets.toml: {'present' if secrets_path.exists() else 'absent'}"
    )
    st.stop()

# Force-export so any code reading os.environ later will see the key.
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Quick probe: we make a tiny request to Groq via LlamaIndex to confirm the key works.
try:
    _probe_llm = Groq(model=cfg.GROQ_MODEL, api_key=GROQ_API_KEY)
    _ = _probe_llm.complete("ping")  # raises on bad auth
except Exception as e:
    # Friendly, specific guidance for a bad/expired key or other errors.
    msg = str(e)
    if "Invalid API Key" in msg or "401" in msg:
        st.error(
            f"Groq rejected your API key (401). Loaded key: {_mask(GROQ_API_KEY)}\n"
            "Double-check the value in `.env` or `secrets.toml` (no quotes, no spaces, no line breaks), "
            "or regenerate a new key in the Groq console."
        )
    else:
        st.error(f"Groq probe failed: {e}")
    st.stop()
# ---------------------------------------------------------------

# --- helpers: corpus + index mgmt ---
def corpus_exists() -> bool:
    """Return True if the configured text file (the book) exists."""
    return os.path.exists(cfg.TEXT_FILE) and os.path.isfile(cfg.TEXT_FILE)

def clear_index_dir():
    """
    Remove and recreate the index storage directory.

    Why this exists:
    - If you ever change embedding/model settings or the corpus, you may want to
      clear the old index. (Not used by the main flow here, but useful during dev.)
    """
    if os.path.exists(cfg.PERSIST_DIR):
        try:
            rmtree(cfg.PERSIST_DIR)
        except Exception:
            pass
    os.makedirs(cfg.PERSIST_DIR, exist_ok=True)

# --- helper: best score from retrieved nodes (if available) ---
def _best_score(nodes):
    """
    Extract the top similarity score from the retrieved nodes, if present.

    This is used only for lightweight observability (what was the best match?).
    """
    try:
        return getattr(nodes[0], "score", None) if nodes else None
    except Exception:
        return None

# --- DIY trace storage (local JSON) ---
# We keep a rolling set of simple traces in a local JSON file so anyone can
# open the "DIY Observability" tab and see timings without a full tracing stack.
TRACE_FILE = "local_traces.json"
MAX_TRACES = 200  # keep only the last 200 traces

def store_trace(span_name: str, start_s: float, end_s: float, attributes: dict | None = None):
    """
    Persist one trace event to a local JSON file.

    Parameters
    ----------
    span_name : str
        A label for what we measured (e.g., 'retrieve.topk', 'engine.chat', 'rag.e2e').
    start_s / end_s : float
        Start/end timestamps in seconds (time.perf_counter()).
    attributes : dict | None
        Any extra bits to save (like request_id, model name, 'k' used, etc.).

    Notes
    -----
    - We convert to milliseconds for readability.
    - We keep only the most recent MAX_TRACES events to avoid runaway file size.
    """
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
        if len(existing) > MAX_TRACES:
            existing = existing[-MAX_TRACES:]
        with open(TRACE_FILE, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        st.warning(f"Could not write {TRACE_FILE}: {e}")

def load_traces_df() -> pd.DataFrame:
    """
    Load all stored traces as a pandas DataFrame for display.

    Returns
    -------
    pd.DataFrame
        One row per trace event, or empty DataFrame if none yet.
    """
    if not os.path.exists(TRACE_FILE):
        return pd.DataFrame()
    try:
        return pd.DataFrame(json.load(open(TRACE_FILE)))
    except Exception:
        return pd.DataFrame()

# --- UI header ---
# Set Streamlit page options (title + centered layout), then show a headline + caption.
st.set_page_config(page_title="📖 Chat with Dostoevsky", layout="centered")
st.title("Conversations with *Crime and Punishment*")
st.caption("Ask anything about Dostoevsky's masterpiece and let LLaMA 3 (Groq) guide you.")
# (Hidden internal debug caption removed)

# --- Custom CSS tweaks (cards + tabs + expander title) ---
# These are purely cosmetic: info cards, tab styling, and expander header emphasis.
st.markdown("""
<style>
/* Info cards */
.infocard {
  border: 1px solid #444;
  border-radius: 12px;
  padding: 12px 14px;
  text-align: left;
  background-color: #262730;
  color: #ffffff;
  font-size: 14px;
  font-weight: 500;
  box-shadow: 0 2px 6px rgba(0,0,0,0.35);
  margin: 6px 0;
  word-break: break-word;
}
.infocard .label { opacity: 0.9; font-weight: 700; display: block; margin-bottom: 8px; }
.infocard .value { font-weight: 800; font-size: 16px; }

/* Tabs: bigger, bold, golden */
div[data-baseweb="tab-list"] { gap: 28px; }
button[data-baseweb="tab"] {
  font-size: 18px !important;
  font-weight: 900 !important;
  color: #FFCF33 !important;     /* golden */
  padding-bottom: 10px !important;
  letter-spacing: 0.3px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  border-bottom: 4px solid #FFC000 !important;
  color: #FFE066 !important;
}

/* Expander header: stronger and larger */
div[data-testid="stExpander"] > details > summary {
  font-size: 18px !important;
  font-weight: 900 !important;
}
</style>
""", unsafe_allow_html=True)

# Stable session id
# Each browser session gets a short, random id. Useful for correlating traces/logs.
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"s_{uuid.uuid4().hex[:8]}"

# Global tracer used for timing logical spans (build index, chat, etc.)
tracer = get_tracer()

# --- Info cards (only what you want public) ---
# We show 3 small "cards" that reveal the corpus file name, LLM model, and embedding model.
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f'<div class="infocard"><span class="label">📄 Corpus file</span>'
        f'<span class="value">{pathlib.Path(cfg.TEXT_FILE).name}</span></div>',
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f'<div class="infocard"><span class="label">🧠 LLM model</span>'
        f'<span class="value">{cfg.GROQ_MODEL}</span></div>',
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f'<div class="infocard"><span class="label">🔡 Embeddings</span>'
        f'<span class="value">{cfg.EMBEDDING_MODEL.split("/")[-1]}</span></div>',
        unsafe_allow_html=True
    )

# === Tabs: main chat & DIY ===
tab_chat, tab_diy = st.tabs(["💬 Chat", "🛠️ DIY Observability"])

# --- Cached retriever builder (unchanged core) ---
@st.cache_resource
def build_or_load_index(top_k: int):
    """
    Create or load the document index and return a retriever with the given top_k.

    What this does (simplified):
    1) Build embeddings for the book's text and store them locally (first run), OR
    2) Reuse the already-stored index (subsequent runs).
    3) Return a retriever that will fetch the 'top_k' most similar passages for any query.

    Why cache_resource:
    - Streamlit remembers the built index between reruns as long as the inputs
      to this function stay the same, avoiding slow rebuilds.
    """
    t0 = time.perf_counter()
    with tracer.start_as_current_span("index.load_or_build") as span:
        embed_model = HuggingFaceEmbedding(model_name=cfg.EMBEDDING_MODEL)

        # If no persisted index exists yet, build from the text file and save it.
        if not os.path.exists(cfg.PERSIST_DIR) or not os.listdir(cfg.PERSIST_DIR):
            docs = SimpleDirectoryReader(input_files=[cfg.TEXT_FILE]).load_data()
            index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
            index.storage_context.persist(persist_dir=cfg.PERSIST_DIR)
            span.set_attribute("built", True)
        else:
            # Otherwise load the previously built index from disk.
            storage_context = StorageContext.from_defaults(persist_dir=cfg.PERSIST_DIR)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            span.set_attribute("built", False)

        # A couple of trace attributes for debugging/observability.
        span.set_attribute("top_k_default", top_k)
        span.set_attribute("embedding_model", cfg.EMBEDDING_MODEL)

    _ = time.perf_counter() - t0
    # Return a retriever configured to return 'top_k' passages for each query.
    return index.as_retriever(similarity_top_k=top_k)

# --- Persistent LLM & memory; engine recreated on k-change but keeps memory ---
# We create the LLM client once per session (same for memory), so chat history persists.
if "llm" not in st.session_state:
    st.session_state["llm"] = Groq(model=cfg.GROQ_MODEL, api_key=GROQ_API_KEY)

if "chat_memory" not in st.session_state:
    # ChatMemoryBuffer stores the back-and-forth conversation so the model can
    # consider previous questions/answers.
    st.session_state["chat_memory"] = ChatMemoryBuffer.from_defaults()

# System-level instructions to keep answers grounded in retrieved context and brief.
prefix_messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=MessageRole.SYSTEM, content="Answer using only the provided context and chat history. Keep your response concise."),
]

def _make_engine(retriever):
    """
    Construct a ContextChatEngine that:
    - Uses the configured LLM (Groq/LLaMA 3)
    - Pulls context via the retriever (vector search over the book)
    - Remembers conversation with ChatMemoryBuffer
    - Injects system instructions from 'prefix_messages'
    """
    return ContextChatEngine(
        llm=st.session_state["llm"],
        retriever=retriever,
        memory=st.session_state["chat_memory"],
        prefix_messages=prefix_messages,
    )

with tab_chat:
    # === How it works + controls ===
    # An always-visible explainer that summarizes the pipeline in one line.
    with st.expander(" How it works", expanded=True):
        st.markdown("""
**RAG flow:** retrieve relevant chunks → build prompt with context → generate with LLaMA 3 (Groq) → show answer.

You can control how many chunks to retrieve (`top_k`).
""")

    # --- guard: stop early if corpus missing ---
    # If the book file is missing, we show a helpful error and stop the app.
    if not corpus_exists():
        st.error(
            f"Corpus file not found at:\n**{cfg.TEXT_FILE}**\n\n"
            "Fix it by either moving the book there or setting `TEXT_FILE` in `.env` "
            "to the correct path (e.g., `assets/crime_and_punishment.txt`)."
        )
        st.stop()

    # --- Slider remembers last value; default 2 ---
    # The slider lets you pick how many relevant passages to retrieve for each question.
    initial_k = st.session_state.get("current_top_k", 2)
    top_k = st.slider("🔍 Retrieve how many relevant passages?", min_value=1, max_value=10, value=initial_k, step=1)

    # --- Ensure engine exists; recreate ONLY if k changed; memory persists ---
    # We build (or reuse) the retriever and create the chat engine. If you change 'top_k',
    # we rebuild the retriever and engine but keep the same memory so conversation history isn't lost.
    if "chat_engine" not in st.session_state:
        retriever = build_or_load_index(top_k)
        st.session_state["retriever"] = retriever
        st.session_state["chat_engine"] = _make_engine(retriever)
        st.session_state["current_top_k"] = top_k
    elif top_k != st.session_state.get("current_top_k"):
        retriever = build_or_load_index(top_k)
        st.session_state["retriever"] = retriever
        # Recreate engine but reuse SAME memory → history persists
        st.session_state["chat_engine"] = _make_engine(retriever)
        st.session_state["current_top_k"] = top_k

    chat_engine = st.session_state["chat_engine"]
    retriever = st.session_state["retriever"]

    # === Chat history ===
    # We draw the existing conversation so the user can see previous Q&A.
    st.divider()
    st.subheader("🗨️ Ask away")
    for message in chat_engine.chat_history:
        with st.chat_message(message.role):
            st.markdown(message.content)

    # === Input ===
    # The main chat input. When the user submits a question:
    # - we log/trace
    # - we run retrieval and generation
    # - we show the answer
    if prompt := st.chat_input(" Ask a philosophical or narrative question..."):
        REQS.labels(route="/chat").inc()  # increment request count metric
        req_id = f"r_{uuid.uuid4().hex[:8]}"
        start = time.perf_counter()
        log_event(
            "query_received",
            request_id=req_id,
            session_id=st.session_state["session_id"],
            prompt=prompt,
            top_k=top_k,
        )

        # Echo the user's message into the transcript immediately.
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("🧠 Thinking deeply like Raskolnikov..."):
                # 1) Retrieval timing
                # We time how long it takes to fetch the top_k most relevant passages.
                t_ret0 = time.perf_counter()
                nodes = retriever.retrieve(prompt)
                t_ret1 = time.perf_counter()
                store_trace("retrieve.topk", t_ret0, t_ret1, {
                    "k": top_k, "hits": len(nodes), "best_score": _best_score(nodes), "request_id": req_id
                })

                # 2) Generation timing
                # We time the LLM call using the chat engine (includes memory and context).
                t_gen0 = time.perf_counter()
                with tracer.start_as_current_span("engine.chat") as s:
                    s.set_attribute("model", cfg.GROQ_MODEL)
                    answer = chat_engine.chat(prompt)
                    response = answer.response or ""
                t_gen1 = time.perf_counter()
                store_trace("engine.chat", t_gen0, t_gen1, {"model": cfg.GROQ_MODEL, "request_id": req_id})

                # 3) End-to-end timing
                # Full roundtrip: from receiving the question to having an answer.
                t_e2e1 = time.perf_counter()
                store_trace("rag.e2e", start, t_e2e1, {"top_k": top_k, "request_id": req_id})

            # Record the duration metric and a success log.
            LAT_E2E.labels(route="/chat").observe(time.perf_counter() - start)
            log_event("query_answered", request_id=req_id, response_len=len(response), model=cfg.GROQ_MODEL)

            # Finally, show the assistant's response in the chat transcript.
            with st.chat_message("assistant"):
                st.markdown(response)

        except Exception as e:
            # If anything fails, we log and show a friendly error.
            ERRS.labels(route="/chat", type=type(e).__name__).inc()
            log_event("error", request_id=req_id, error=str(e))
            if "Invalid API Key" in str(e) or "401" in str(e):
                st.error("Groq rejected your API key (401). Check `GROQ_API_KEY` again.")
            else:
                st.error(f"Something went wrong: {e}")

# --- DIY Observability tab ---                
with tab_diy:
    st.subheader("🔎 DIY Observability")

    PRETTY_STAGE = {
        "retrieve.topk": "Find passages",
        "engine.chat": "Write answer",
        "rag.e2e": "Total roundtrip",
        "ui.smoke": "Test ping",
    }

    # Emit a test trace
    if st.button("Emit test trace"):
        t0 = time.perf_counter()
        with get_tracer().start_as_current_span("ui.smoke") as s:
            s.set_attribute("clicked_at", int(time.time()))
        t1 = time.perf_counter()
        store_trace("ui.smoke", t0, t1, {"clicked_at": int(time.time())})
        st.success("Emitted a test trace locally (local_traces.json).")

    df = load_traces_df()
    if df.empty:
        st.info("No local traces yet. Ask a question to record retrieval, generation, and end-to-end timings.")
    else:
        # ---- prep ----
        df = df.copy()
        df["duration_s"] = pd.to_numeric(df["duration_ms"], errors="coerce") / 1000.0
        df["start_ts_s"] = pd.to_numeric(df["start_ts_ms"], errors="coerce") / 1000.0

        # Optional filter: only this browser session
        only_mine = st.checkbox("Show only my session", value=False)
        if only_mine:
            df = df[df["session_id"] == st.session_state.get("session_id")]

        if df.empty:
            st.info("No traces for the current filter yet.")
            st.stop()

        # Take the latest 20 events overall (keeps original behavior for KPIs/table)
        recent = df.sort_values("start_ts_ms", ascending=False).head(20).copy()

        # KPI tiles (same logic, raw span names)
        def _avg_s(span: str) -> float:
            vals = recent.loc[recent["span_name"] == span, "duration_s"]
            ser = pd.to_numeric(vals, errors="coerce")
            ser = pd.Series(ser).dropna()
            return float(ser.mean()) if not ser.empty else 0.0

        k1, k2, k3 = st.columns(3)
        k1.metric("Retrieval avg (s)", f"{_avg_s('retrieve.topk'):.3f}")
        k2.metric("Generation avg (s)", f"{_avg_s('engine.chat'):.3f}")
        k3.metric("End-to-end avg (s)", f"{_avg_s('rag.e2e'):.3f}")

        # ----- Friendly table: newest first, 1..N indexing -----
        recent["stage"] = recent["span_name"].map(PRETTY_STAGE).fillna(recent["span_name"])
        show = recent[["stage", "duration_s", "request_id", "session_id"]].rename(
            columns={"duration_s": "duration (s)"}
        ).reset_index(drop=True)
        show.index = show.index + 1
        st.dataframe(show, use_container_width=True)

        # ----- Request-aligned chart (makes lines visible & comparable) -----
        # Build one row per request_id, with columns for each stage's duration
        req_view = (
            df.sort_values("start_ts_ms")
              .groupby(["request_id", "span_name"], as_index=False)["duration_s"]
              .max()
              .pivot(index="request_id", columns="span_name", values="duration_s")
              .rename(columns=PRETTY_STAGE)
        )

        # Keep only the most recent ~10 requests for readability
        req_view = req_view.tail(10)

        # Make a simple "Request #n" index for the x-axis (not wall time, not perf counter)
        req_view.index = [f"Req {i}" for i in range(len(req_view) - len(req_view) + 1, len(req_view) + 1)]

        # Ensure all friendly columns exist (shows missing stages as empty)
        for col in ["Find passages", "Write answer", "Total roundtrip", "Test ping"]:
            if col not in req_view.columns:
                req_view[col] = pd.NA

        # Streamlit draws lines only where there are >=2 points; still useful when you ask a few questions
        st.line_chart(req_view[["Find passages", "Write answer", "Total roundtrip"]])
