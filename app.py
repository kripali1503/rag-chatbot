"""Universal RAG Chatbot: Streamlit UI. Core logic lives in rag.py."""
import datetime
import html
import logging
from collections import deque
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv()
logging.basicConfig(
    filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("app")

EXAMPLES = ["Summarize this document", "What are the key points?", "List any dates or deadlines mentioned"]

st.set_page_config(page_title="Universal RAG Chatbot", page_icon=":books:", layout="wide")
st.markdown(f"<style>{Path('style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def get_key(name: str):
    """Env var first, then Streamlit secrets (which raise if no secrets.toml exists)."""
    import os

    try:
        return os.environ.get(name) or st.secrets.get(name)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def embeddings():
    return rag.get_embeddings()


def discover_models(provider: str, api_key: str) -> dict:
    """Models that work for free with this key (rag caches the result on disk for 24h)."""
    with st.spinner("Finding models that work with your key…"):
        try:
            return rag.find_free_models(provider, api_key)
        except Exception as e:
            log.warning("Model discovery failed for %s: %s", provider, e)
            return {"working": [], "paid": False, "checked": 0, "statuses": {}, "note": str(e)[:120]}


def queue_example():
    st.session_state["queued"] = st.session_state["example"]
    st.session_state["example"] = None


def chat_as_text(messages) -> str:
    out = [f"Chat history - {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 50, ""]
    for m in messages:
        out += [f"[{m['role'].upper()}]", m["content"].strip(), "", "-" * 50, ""]
    return "\n".join(out)


def recent_logs(limit: int = 200):
    path = Path("app.log")
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as f:
        return list(deque((ln.strip() for ln in f if ln.strip()), maxlen=limit))


def render_sources(sources, show_text: bool):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = f" · p.{s['page']}" if s["page"] else ""
            st.markdown(f"<span class='chip'>[{s['n']}] {html.escape(s['file'])}{page}</span>", unsafe_allow_html=True)
            st.caption(s["text"] if show_text else s["snippet"])


# ---------- Session state ----------
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("db", None)
ss.setdefault("stats", [])
ss.setdefault("sig", None)
ss.setdefault("force", False)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("#### Documents")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if ss["stats"]:
        for s in ss["stats"]:
            st.markdown(f"<span class='chip'>{html.escape(s['file'])}</span>", unsafe_allow_html=True)
            st.caption(f"{s['pages']} pages · {s['chunks']} chunks")

    st.markdown("#### Model")
    provider = rag.llm_provider()
    sidebar_key = rag.llm_api_key() or get_key("GLOBAL_API_KEY") or get_key("GROQ_API_KEY")
    if rag._env("LLM_MODELS", "GROQ_MODELS"):  # explicit override skips discovery
        found = {"working": rag.llm_models(), "paid": False, "checked": 0, "note": "Using LLM_MODELS from .env"}
    elif sidebar_key:
        found = discover_models(provider, sidebar_key)
    else:
        found = {"working": [], "paid": False, "checked": 0, "note": ""}
    available = found["working"] or ["(no working models)"]
    tag = "paid" if found["paid"] else "✓ free"
    primary = st.selectbox("Model", available, label_visibility="collapsed",
                           format_func=lambda m: m if m.startswith("(") else f"{m}  ·  {tag}")
    if found["checked"]:
        st.caption(f"{provider} · {len(found['working'])} of {found['checked']} models work with this key")
    else:
        st.caption(f"Provider: {provider} · {len(found['working'])} model(s)")
    if found["note"]:
        st.caption(f":orange[{found['note']}]")
    models = ([primary] + [m for m in found["working"] if m != primary])[:3]  # primary + 2 fallbacks

    with st.expander("Advanced"):
        top_k = st.slider("Passages retrieved (Top-K)", 3, 15, 8)
        show_debug = st.toggle("Show full retrieved text", value=False)
        if st.button("Rebuild index", use_container_width=True, disabled=not uploaded):
            ss["force"], ss["sig"] = True, None
            st.rerun()
        if st.button("Re-check models", use_container_width=True):
            rag.clear_model_cache()
            st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("Clear chat", use_container_width=True, disabled=not ss["messages"]):
        ss["messages"] = []
        st.rerun()
    if ss["messages"]:
        c2.download_button(
            "Export", chat_as_text(ss["messages"]),
            file_name=f"chat_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt", use_container_width=True,
        )

# ---------- Header ----------
st.markdown(
    "<div class='hero'><h1>Universal RAG Chatbot</h1>"
    "<p>Ask questions about your PDFs. Answers come with sources.</p></div>",
    unsafe_allow_html=True,
)

# ---------- Index the uploaded PDFs (only when the set of files changes) ----------
if uploaded:
    sig = tuple((f.name, f.size) for f in uploaded)
    if sig != ss["sig"]:
        files = [(f.name, f.getvalue()) for f in uploaded]
        try:
            with st.status("Preparing your documents", expanded=True) as status:
                ss["db"], ss["stats"] = rag.prepare_corpus(
                    files, embeddings(), force=ss["force"], on_step=lambda m: st.write(m)
                )
                status.update(label="Documents ready", state="complete", expanded=False)
            log.info("Indexed %d file(s): %s", len(files), [s["file"] for s in ss["stats"]])
            ss["messages"], ss["sig"], ss["force"] = [], sig, False
            st.rerun()
        except Exception as e:
            log.error("Indexing failed: %s", e)
            st.error(f"Could not read those documents: {e}")
elif ss["sig"] is not None:  # files removed
    ss["db"], ss["stats"], ss["sig"], ss["messages"] = None, [], None, []
    st.rerun()

ready = ss["db"] is not None
api_key = rag.llm_api_key() or get_key("GLOBAL_API_KEY") or get_key("GROQ_API_KEY")

prompt = st.chat_input("Ask about your documents…" if ready else "Upload a PDF to start", disabled=not ready)
prompt = prompt or ss.pop("queued", None)

tab_chat, tab_info, tab_logs, tab_notes = st.tabs(["Chat", "Info", "Logs", "Notes"])

# ---------- Chat ----------
with tab_chat:
    if not api_key:
        st.error("No API key found. Set `GLOBAL_API_KEY` (and `LLM_PROVIDER`, `LLM_MODELS`) in `.env`, then restart.")

    if not ready:
        st.markdown(
            """<div class='steps'>
            <div class='step'><i>STEP 1</i><b>Upload</b><span>Add PDFs in the sidebar</span></div>
            <div class='step'><i>STEP 2</i><b>Ask</b><span>Type a question below</span></div>
            <div class='step'><i>STEP 3</i><b>Verify</b><span>Check the cited sources</span></div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        for m in ss["messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    render_sources(m["sources"], show_debug)
        if not ss["messages"]:
            st.caption("Try one of these")
            st.pills("Examples", EXAMPLES, key="example", on_change=queue_example, label_visibility="collapsed")

    if prompt and ready and api_key:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                retriever = ss["db"].as_retriever(search_kwargs={"k": top_k})
                llm = rag.build_llm(api_key, models, provider)
                docs, tokens = rag.answer_stream(llm, retriever, prompt, ss["messages"])
                answer = st.write_stream(tokens)
                sources = rag.sources_from(docs)
                ss["messages"] += [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer, "sources": sources},
                ]
                log.info("Answered with %d sources", len(sources))
                render_sources(sources, show_debug)
            except Exception as e:
                log.error("Answer failed: %s", e)
                st.error("The model couldn't answer. Check `GLOBAL_API_KEY`, `LLM_PROVIDER` and `LLM_MODELS` in `.env`.")
                with st.expander("Error details"):
                    st.code(str(e), language="text")

# ---------- Info ----------
with tab_info:
    st.markdown(
        f"""<div class='grid'>
        <div class='card violet'><h4>How it works</h4>
            <p>PDFs are split into overlapping chunks, embedded locally, and stored in a FAISS index.
            Each question retrieves the closest passages and your configured LLM answers from them.</p></div>
        <div class='card pink'><h4>Stack</h4>
            <p>Streamlit · LangChain · FAISS</p><p>Embeddings: all-mpnet-base-v2 (local)</p><p>LLM: {html.escape(provider)} (any LangChain provider)</p></div>
        <div class='card green'><h4>Current settings</h4>
            <p>Chunk size {rag.CHUNK_SIZE}, overlap {rag.CHUNK_OVERLAP}</p>
            <p>Primary model: {html.escape(primary)}</p><p>Fallbacks: {len(models) - 1}</p></div>
        <div class='card orange'><h4>API key</h4>
            <p>Set <code>GLOBAL_API_KEY</code>, <code>LLM_PROVIDER</code> and <code>LLM_MODELS</code> in <code>.env</code>.</p>
            <p>Optional: <code>LLM_BASE_URL</code> for any OpenAI-compatible server.</p></div>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------- Logs ----------
with tab_logs:
    if st.button("Refresh"):
        st.rerun()
    rows = []
    for line in reversed(recent_logs()):
        parts = line.split(" - ", 2)
        ts, level, msg = parts if len(parts) == 3 else ("", "INFO", line)
        cls = {"ERROR": "log-error", "WARNING": "log-warn"}.get(level, "log-info")
        rows.append(f"<div class='log-row'><span class='t'>{html.escape(ts)}</span><span class='{cls}'>{html.escape(msg)}</span></div>")
    st.html(f"<div class='log-box'>{''.join(rows) or 'No logs yet'}</div>")

# ---------- Notes ----------
with tab_notes:
    st.markdown(
        """<div class='grid'>
        <div class='card violet'><h4>Grounded answers</h4>
            <p>The assistant answers only from your documents and says so when the answer isn't there.
            Always verify important answers against the cited sources.</p></div>
        <div class='card pink'><h4>Saved indexes</h4>
            <p>Each set of PDFs gets its own saved index in <code>faiss_index_storage/</code>, so re-uploading the same files is instant.
            Use <b>Rebuild index</b> under Advanced to force a fresh one.</p></div>
        <div class='card green'><h4>Conversation memory</h4>
            <p>The last three turns are included with each question, so follow-ups work.
            History lives in your session and clears when you close the tab.</p></div>
        <div class='card blue'><h4>Model fallback</h4>
            <p>If the selected model fails, the next one in <code>LLM_MODELS</code> is tried automatically.</p></div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div class='footer'>Built by KRIPALI N · LangChain · FAISS</div>", unsafe_allow_html=True)
