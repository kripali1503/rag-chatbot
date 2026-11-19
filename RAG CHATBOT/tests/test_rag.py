import fitz
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.documents import Document

import rag


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    for i in range(2):
        doc.new_page().insert_text((72, 72), f"{text} page {i + 1}. " * 12)
    return doc.tobytes()


def test_load_pdfs_keeps_source_and_page():
    chunks, stats = rag.load_pdfs([("a.pdf", make_pdf("alpha"))])
    assert chunks and stats == [{"file": "a.pdf", "pages": 2, "chunks": len(chunks)}]
    assert all(c.metadata["source"] == "a.pdf" and "page" in c.metadata for c in chunks)


def test_corpus_key_stable_and_distinct():
    a, b = ("a.pdf", make_pdf("alpha")), ("b.pdf", make_pdf("beta"))
    assert rag.corpus_key([a, b]) == rag.corpus_key([b, a])
    assert rag.corpus_key([a]) != rag.corpus_key([b])


def test_answer_stream_returns_docs_and_tokens():
    class R:
        def invoke(self, q):
            return [Document(page_content="hello world", metadata={"source": "a.pdf", "page": 0})]

    docs, tokens = rag.answer_stream(FakeListChatModel(responses=["forty two"]), R(), "q?", [])
    assert "".join(tokens) == "forty two"
    assert rag.sources_from(docs)[0]["page"] == 1


def test_provider_config_from_env(monkeypatch):
    for v in ("GLOBAL_API_KEY", "Global_api_key", "GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODELS", "GROQ_MODELS", "LLM_BASE_URL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "legacy")
    assert rag.llm_provider() == "groq" and rag.llm_api_key() == "legacy"
    monkeypatch.setenv("GLOBAL_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODELS", "claude-haiku-5-5, claude-sonnet-5-5")
    assert rag.llm_api_key() == "k" and rag.llm_models() == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert type(rag.build_llm("k", ["claude-haiku-5-5"])).__name__ == "ChatAnthropic"
    llm = rag.build_llm("k", ["a", "b"], provider="openai", base_url="http://localhost:11434/v1")
    assert type(llm.runnable).__name__ == "ChatOpenAI" and len(llm.fallbacks) == 1
