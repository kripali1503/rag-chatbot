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


def test_detect_provider_from_key_prefix():
    assert rag.detect_provider("gsk_x") == "groq"
    assert rag.detect_provider("sk-ant-x") == "anthropic"
    assert rag.detect_provider("sk-proj-x") == "openai"
    assert rag.detect_provider("AIzaX") == "gemini"
    assert rag.detect_provider("other") is None
    llm = rag.build_llm("AIzaX", ["gemini-2.5-flash"], provider="gemini")
    assert type(llm).__name__ == "ChatOpenAI" and "generativelanguage" in str(llm.openai_api_base)


def test_openrouter_prefix_beats_openai():
    assert rag.detect_provider("sk-or-v1-abc") == "openrouter"
    assert rag.detect_provider("sk-ant-abc") == "anthropic"
    assert rag.detect_provider("sk-abc") == "openai"


def test_openrouter_free_filter():
    assert rag._is_free_listing({"id": "x:free", "pricing": {"prompt": "0", "completion": "0"}})
    assert not rag._is_free_listing({"id": "y", "pricing": {"prompt": "0.000001", "completion": "0"}})
    assert rag._is_free_listing({"id": "z:free", "pricing": {"prompt": "n/a"}})


def test_classify_error():
    class E(Exception):
        def __init__(self, code, msg=""):
            super().__init__(msg)
            self.status_code = code

    assert rag.classify_error(E(429)) == "ok"
    assert rag.classify_error(E(404, "model_not_found")) == "no_access"
    assert rag.classify_error(E(402)) == "billing"
    assert rag.classify_error(E(401)) == "auth"
    assert rag.classify_error(Exception("insufficient_quota")) == "billing"


def test_find_free_models_probes_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setattr(rag, "_fetch_model_entries", lambda p, k: [{"id": "good-70b"}, {"id": "locked"}])
    calls = []

    def fake_probe(provider, key, model):
        calls.append(model)
        return "ok" if model == "good-70b" else "no_access"

    monkeypatch.setattr(rag, "probe", fake_probe)
    result = rag.find_free_models("groq", "gsk_secret")
    assert result["working"] == ["good-70b"] and result["checked"] == 2
    assert "gsk_secret" not in (tmp_path / "cache.json").read_text()  # key stored only as a hash
    rag.find_free_models("groq", "gsk_secret")
    assert len(calls) == 2  # second call served from cache, no new probes


def test_paid_provider_is_not_probed(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(rag, "_fetch_model_entries", lambda p, k: [{"id": "gpt-x"}])
    monkeypatch.setattr(rag, "probe", lambda *a: (_ for _ in ()).throw(AssertionError("should not probe")))
    result = rag.find_free_models("openai", "sk-abc")
    assert result["paid"] and result["working"] == ["gpt-x"]
