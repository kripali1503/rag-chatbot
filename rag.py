"""RAG core: PDF loading, indexing, and streamed answers. No Streamlit imports."""
import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

log = logging.getLogger("rag")

INDEX_ROOT = Path("faiss_index_storage")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
HISTORY_TURNS = 3

PROMPT = ChatPromptTemplate.from_template("""You are a careful assistant answering questions about the user's documents.

Answer ONLY from the context below. Cite the passages you used with their [n] markers.
If the context does not contain the answer, say so plainly; do not guess.

Conversation so far:
{history}

Context:
{context}

Question: {question}

Answer:""")

Files = List[Tuple[str, bytes]]
Step = Callable[[str], None]


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# One row per provider. "langchain" is the init_chat_model provider used to talk to it.
PROVIDERS = {
    "groq": dict(prefix="gsk_", base_url=None, models_url="https://api.groq.com/openai/v1/models",
                 auth="bearer", langchain="groq", free_tier=True,
                 defaults=["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]),
    "openrouter": dict(prefix="sk-or-", base_url=OPENROUTER_BASE_URL, models_url=OPENROUTER_BASE_URL + "/models",
                       auth="bearer", langchain="openai", free_tier=True, defaults=[]),
    "anthropic": dict(prefix="sk-ant-", base_url=None, models_url="https://api.anthropic.com/v1/models",
                      auth="x-api-key", langchain="anthropic", free_tier=False,
                      defaults=["claude-haiku-5-5", "claude-sonnet-5-5"]),
    "gemini": dict(prefix="AIza", base_url=GEMINI_BASE_URL, models_url=GEMINI_BASE_URL + "models",
                   auth="bearer", langchain="openai", free_tier=True,
                   defaults=["gemini-2.5-flash", "gemini-2.5-flash-lite"]),
    "mistralai": dict(prefix=None, base_url=None, models_url="https://api.mistral.ai/v1/models",
                      auth="bearer", langchain="mistralai", free_tier=True, defaults=["mistral-small-latest"]),
    "openai": dict(prefix="sk-", base_url=None, models_url="https://api.openai.com/v1/models",
                   auth="bearer", langchain="openai", free_tier=False, defaults=["gpt-4o-mini"]),
}
NON_CHAT = ("whisper", "tts", "embed", "guard", "moderation", "dall-e", "audio", "image", "imagen",
            "transcribe", "orpheus", "aqa", "veo", "realtime", "search")
MAX_PROBES = 20
CACHE_FILE = Path(".model_cache.json")
CACHE_TTL = 24 * 3600


def _env(*names: str) -> Optional[str]:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def llm_api_key() -> Optional[str]:
    """One key for whichever provider is configured. Legacy GROQ_API_KEY still works."""
    return _env("GLOBAL_API_KEY", "Global_api_key", "GROQ_API_KEY")


def detect_provider(key: Optional[str]) -> Optional[str]:
    """Guess the provider from the key's prefix (longest prefix wins, so sk-or-/sk-ant- beat sk-)."""
    key = key or ""
    ranked = sorted((p for p in PROVIDERS.items() if p[1]["prefix"]), key=lambda p: -len(p[1]["prefix"]))
    return next((name for name, spec in ranked if key.startswith(spec["prefix"])), None)


def llm_provider() -> str:
    """LLM_PROVIDER if set, else 'custom' when LLM_BASE_URL is set, else detected from the key."""
    explicit = _env("LLM_PROVIDER")
    if explicit:
        return explicit.strip().lower()
    if _env("LLM_BASE_URL"):
        return "custom"
    return detect_provider(llm_api_key()) or "groq"


def provider_spec(provider: str) -> dict:
    """Row from PROVIDERS; unknown names or LLM_BASE_URL mean an OpenAI-compatible server."""
    base = _env("LLM_BASE_URL")
    spec = dict(PROVIDERS.get(provider) or dict(
        prefix=None, base_url=None, models_url=None, auth="bearer", langchain="openai", free_tier=True, defaults=[]))
    if base:
        spec.update(base_url=base, models_url=base.rstrip("/") + "/models", langchain="openai")
    return spec


def llm_base_url(provider: str) -> Optional[str]:
    return provider_spec(provider)["base_url"]


def _fetch_model_entries(provider: str, api_key: str) -> List[dict]:
    import urllib.request

    spec = provider_spec(provider)
    if not spec["models_url"]:
        return []
    headers = {"User-Agent": "rag-chatbot"}
    if spec["auth"] == "x-api-key":
        headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    with urllib.request.urlopen(urllib.request.Request(spec["models_url"], headers=headers), timeout=15) as r:
        entries = json.load(r).get("data", [])
    out = []
    for m in entries:
        if "id" not in m:
            continue
        m = {**m, "id": m["id"].removeprefix("models/")}
        if not any(s in m["id"].lower() for s in NON_CHAT):
            out.append(m)
    return out


def list_models(provider: str, api_key: str) -> List[str]:
    """Every chat model the key can see (not necessarily usable). Raises on auth/network errors."""
    return sorted(m["id"] for m in _fetch_model_entries(provider, api_key))


def _is_free_listing(entry: dict) -> bool:
    """OpenRouter publishes prices; a free model costs 0 for prompt and completion."""
    pricing = entry.get("pricing") or {}
    try:
        return float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
    except (TypeError, ValueError):
        return entry["id"].endswith(":free")


def classify_error(err: Exception) -> str:
    """Map a failed test call to: ok (rate-limited still counts) | no_access | billing | auth | error."""
    code = getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)
    if code is None:
        code = getattr(err, "code", None) if isinstance(getattr(err, "code", None), int) else None
    text = str(err).lower()
    if code == 429 or "rate limit" in text or "rate_limit" in text:
        return "ok"
    if code == 401 or "invalid api key" in text or "invalid_api_key" in text or "unauthorized" in text:
        return "auth"
    if code == 402 or any(w in text for w in ("insufficient_quota", "billing", "payment", "credit")):
        return "billing"
    if code in (403, 404) or any(w in text for w in ("not found", "does not exist", "access", "permission")):
        return "no_access"
    return "error"


def probe(provider: str, api_key: str, model: str) -> str:
    """Send a 1-token request. Returns a classify_error status ('ok' when the model answers)."""
    try:
        llm = build_llm(api_key, [model], provider, max_tokens=1, timeout=10, max_retries=0)
        llm.invoke("hi")
        return "ok"
    except Exception as e:
        status = classify_error(e)
        log.info("Model %s: %s (%s)", model, status, str(e)[:160])
        return status


def _size_score(model_id: str) -> float:
    """Rough capability rank from the name: parameter count (70b, 120b) plus a nudge for 'pro'."""
    import re

    low = model_id.lower()
    m = re.search(r"(\d+(?:\.\d+)?)b\b", low)
    score = float(m.group(1)) if m else 0.0
    return score + (50 if "pro" in low else 0) - (5 if "preview" in low or "exp" in low else 0)


def rank_models(provider: str, ids: List[str]) -> List[str]:
    """Provider's preferred defaults first, then bigger models first."""
    defaults = [d for d in provider_spec(provider)["defaults"] if d in ids]
    rest = sorted((i for i in ids if i not in defaults), key=lambda i: (-_size_score(i), i))
    return defaults + rest


def _cache_key(provider: str, api_key: str) -> str:
    salt = provider_spec(provider)["base_url"] or ""
    return provider + ":" + hashlib.sha256((api_key + salt).encode()).hexdigest()[:16]


def _read_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def clear_model_cache() -> None:
    CACHE_FILE.unlink(missing_ok=True)


def find_free_models(provider: str, api_key: str, force: bool = False) -> dict:
    """Which models actually work for free with this key. Cached 24h on disk (key stored only as a hash).

    Returns {"working": [ids best-first], "paid": bool, "checked": int, "statuses": {id: status}, "note": str}.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    key = _cache_key(provider, api_key)
    cache = _read_cache()
    hit = cache.get(key)
    if hit and not force and time.time() - hit["ts"] < CACHE_TTL:
        return hit["result"]

    spec = provider_spec(provider)
    try:
        entries = _fetch_model_entries(provider, api_key)
    except Exception as e:
        reason = ("Key rejected by the provider (check GLOBAL_API_KEY / LLM_PROVIDER)."
                  if classify_error(e) == "auth" else f"Couldn't list models: {str(e)[:120]}")
        return {"working": [], "paid": False, "checked": 0, "statuses": {}, "note": reason}  # not cached

    ids = [m["id"] for m in entries]
    if provider == "openrouter":
        free = [m["id"] for m in entries if _is_free_listing(m)]
        result = {"working": rank_models(provider, free), "paid": False, "checked": len(ids), "statuses": {}, "note": ""}
    elif not spec["free_tier"]:
        result = {"working": rank_models(provider, ids), "paid": True, "checked": 0, "statuses": {},
                  "note": f"{provider} has no free tier; usage is billed to your account."}
    else:
        candidates = rank_models(provider, ids)[:MAX_PROBES]
        with ThreadPoolExecutor(max_workers=6) as pool:
            statuses = dict(zip(candidates, pool.map(lambda m: probe(provider, api_key, m), candidates)))
        working = [m for m in candidates if statuses[m] == "ok"]
        note = ""
        if not working:
            seen = set(statuses.values())
            note = ("Key rejected by the provider." if "auth" in seen else
                    "All models need billing on this account." if seen == {"billing"} else
                    "No model answered with this key.")
        result = {"working": working, "paid": False, "checked": len(candidates), "statuses": statuses, "note": note}

    if result["working"]:
        cache[key] = {"ts": time.time(), "result": result}
        CACHE_FILE.write_text(json.dumps(cache))
    return result


def llm_models() -> List[str]:
    """Explicit LLM_MODELS / GROQ_MODELS override, else the provider's defaults."""
    raw = _env("LLM_MODELS", "GROQ_MODELS")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(provider_spec(llm_provider())["defaults"])


groq_models = llm_models  # backwards-compatible name


def corpus_key(files: Files) -> str:
    """Stable id for a set of PDFs: same files -> same key, any change -> new key."""
    h = hashlib.sha256()
    for name, data in sorted(files, key=lambda f: f[0]):
        h.update(name.encode())
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()[:16]


def _extract_pages(path: Path) -> List[Document]:
    pages = PyMuPDFLoader(str(path)).load()
    if len("".join(p.page_content for p in pages).strip()) >= 100:
        return pages
    log.info("Low text in %s, trying OCR", path.name)
    try:
        return UnstructuredPDFLoader(str(path), mode="elements", strategy="hi_res").load()
    except Exception as e:  # OCR deps are optional; keep what PyMuPDF gave us
        log.warning("OCR failed for %s: %s", path.name, e)
        return pages


def load_pdfs(files: Files) -> Tuple[List[Document], List[dict]]:
    """Return (chunks, per-file stats). Temp files are removed on exit."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks: List[Document] = []
    stats: List[dict] = []
    with tempfile.TemporaryDirectory(prefix="rag_") as tmp:
        for name, data in files:
            path = Path(tmp) / Path(name).name
            path.write_bytes(data)
            try:
                pages = _extract_pages(path)
            except Exception as e:
                log.error("Failed to load %s: %s", name, e)
                continue
            file_chunks = splitter.split_documents(pages)
            for c in file_chunks:
                c.metadata["source"] = name
            chunks.extend(file_chunks)
            stats.append({"file": name, "pages": len(pages), "chunks": len(file_chunks)})
    return chunks, stats


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def prepare_corpus(
    files: Files,
    embeddings: HuggingFaceEmbeddings,
    force: bool = False,
    on_step: Optional[Step] = None,
) -> Tuple[FAISS, List[dict]]:
    """Load the index for exactly these PDFs from disk, or build and save it."""
    step = on_step or (lambda _: None)
    index_dir = INDEX_ROOT / corpus_key(files)
    manifest = index_dir / "manifest.json"

    if force:
        shutil.rmtree(index_dir, ignore_errors=True)
    if manifest.exists():
        try:
            step("Loading saved index")
            db = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            return db, json.loads(manifest.read_text())
        except Exception as e:
            log.warning("Saved index unusable (%s); rebuilding", e)
            shutil.rmtree(index_dir, ignore_errors=True)

    step(f"Reading {len(files)} PDF(s)")
    chunks, stats = load_pdfs(files)
    if not chunks:
        raise ValueError("No readable text found in the uploaded PDFs.")
    step(f"Embedding {len(chunks)} chunks")
    db = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(index_dir))
    manifest.write_text(json.dumps(stats))
    return db, stats


def build_llm(api_key: str, models: List[str], provider: Optional[str] = None,
              base_url: Optional[str] = None, max_tokens: int = 4096, **client_kwargs):
    """Any provider in PROVIDERS (or any OpenAI-compatible server); first model is primary, rest are fallbacks."""
    from langchain.chat_models import init_chat_model

    provider = provider or llm_provider()
    spec = provider_spec(provider)
    base_url = base_url if base_url is not None else spec["base_url"]
    extra = dict(client_kwargs)
    if base_url:
        extra["base_url"] = base_url
    llms = [
        init_chat_model(m, model_provider=spec["langchain"], api_key=api_key,
                        temperature=0.0, max_tokens=max_tokens, **extra)
        for m in models
    ]
    return llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]


def _format_history(history: List[dict]) -> str:
    turns = [m for m in history if m.get("content")][-HISTORY_TURNS * 2:]
    return "\n".join(f"{m['role'].title()}: {m['content']}" for m in turns) or "(none)"


def sources_from(docs: List[Document]) -> List[dict]:
    return [
        {
            "n": i,
            "file": d.metadata.get("source", "document"),
            "page": int(d.metadata["page"]) + 1 if isinstance(d.metadata.get("page"), int) else None,
            "snippet": " ".join(d.page_content.split())[:240],
            "text": " ".join(d.page_content.split()),
        }
        for i, d in enumerate(docs, 1)
    ]


def answer_stream(llm, retriever, question: str, history: List[dict]) -> Tuple[List[Document], Iterator[str]]:
    """Retrieve, then return (docs, token iterator). Retrieval happens before streaming."""
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{i}] ({d.metadata.get('source')}) {d.page_content}" for i, d in enumerate(docs, 1))
    chain = PROMPT | llm | StrOutputParser()
    tokens = chain.stream({"context": context, "question": question, "history": _format_history(history)})
    return docs, tokens
