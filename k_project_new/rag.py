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
DEFAULT_GROQ_MODELS = "llama-3.3-70b-versatile,llama-3.1-8b-instant"

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


DEFAULT_MODELS = {
    "groq": DEFAULT_GROQ_MODELS,
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5,claude-sonnet-5-5",
    "mistralai": "mistral-small-latest",
}


def _env(*names: str) -> Optional[str]:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def llm_provider() -> str:
    """groq | openai | anthropic | mistralai. With LLM_BASE_URL set, 'openai' means any OpenAI-compatible server."""
    return (_env("LLM_PROVIDER") or ("openai" if _env("LLM_BASE_URL") else "groq")).strip().lower()


def llm_api_key() -> Optional[str]:
    """One key for whichever provider is configured. Legacy GROQ_API_KEY still works."""
    return _env("GLOBAL_API_KEY", "Global_api_key", "GROQ_API_KEY")


def llm_models() -> List[str]:
    raw = _env("LLM_MODELS", "GROQ_MODELS") or DEFAULT_MODELS.get(llm_provider(), "")
    return [m.strip() for m in raw.split(",") if m.strip()]


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


def build_llm(api_key: str, models: List[str], provider: Optional[str] = None, base_url: Optional[str] = None):
    """Any provider LangChain supports; first model is primary, the rest are fallbacks."""
    from langchain.chat_models import init_chat_model

    provider = provider or llm_provider()
    base_url = base_url if base_url is not None else _env("LLM_BASE_URL")
    extra = {"base_url": base_url} if base_url else {}
    llms = [
        init_chat_model(m, model_provider=provider, api_key=api_key, temperature=0.0, max_tokens=4096, **extra)
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
