# AI Knowledge Assistant – RAG Chatbot

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions in natural language. The application retrieves relevant information from uploaded documents and generates context-grounded answers with source citations.

## Features

- **Multi-PDF Upload:** Upload and process multiple PDF documents.
- **Retrieval-Augmented Generation (RAG):** Retrieves relevant document passages before generating answers.
- **Source Citations:** Displays source filenames, page numbers, and relevant snippets.
- **Streaming Responses:** Displays generated answers progressively.
- **Contextual Follow-up Questions:** Maintains recent conversation history.
- **FAISS Vector Search:** Stores and retrieves document embeddings locally.
- **Multiple LLM Providers:** Supports Groq, OpenAI, Anthropic, Mistral, and compatible endpoints.
- **Model Fallback:** Supports alternative models when the selected model fails.
- **Chat Management:** Includes chat export, index rebuilding, and retrieval controls.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM Orchestration | LangChain |
| Vector Store | FAISS |
| Embeddings | Sentence Transformers |
| LLM Integration | Groq and other supported providers |
| PDF Processing | PyMuPDF |
| Testing | Pytest |

## RAG Architecture

The application follows a Retrieval-Augmented Generation workflow:

1. **Document Ingestion:** Users upload PDF documents.
2. **Text Extraction:** Text is extracted using PyMuPDF, with an OCR fallback for scanned documents.
3. **Text Chunking:** Extracted content is divided into smaller overlapping chunks.
4. **Embedding Generation:** Sentence Transformers converts text chunks into numerical embeddings.
5. **Vector Indexing:** FAISS stores the embeddings for similarity-based retrieval.
6. **Query Processing:** User questions are converted into retrieval queries.
7. **Context Retrieval:** Relevant document chunks are retrieved using similarity search.
8. **Answer Generation:** Retrieved context and recent conversation history are passed to the configured LLM.
9. **Source Attribution:** The response includes document and page references.

## Project Structure

```text
rag-chatbot/
├── app.py
├── rag.py
├── config.py
├── style.css
├── requirements.txt
├── README.md
├── Dockerfile
├── setup.bat
├── run.bat
├── tests/
├── .streamlit/
└── .env.example
```

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- An API key for a supported LLM provider

### 1. Clone the repository

```bash
git clone https://github.com/kripali1503/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example` as the template.

Configure the required API key and model settings:

```env
GLOBAL_API_KEY=your_api_key
LLM_PROVIDER=groq
LLM_MODELS=your_supported_model
```

Never commit your actual `.env` file or API keys.

### 5. Run the application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Testing

Run the project tests using:

```bash
pip install pytest
pytest tests -q
```

## Docker

Build the Docker image:

```bash
docker build -t rag-chatbot .
```

Run the application:

```bash
docker run -p 8501:8501 --env-file .env rag-chatbot
```

## Security

- API keys and environment files are excluded from version control.
- Embeddings are generated locally.
- Retrieved passages and user questions are sent to the configured LLM provider for answer generation.

## Project Attribution

This project is an adapted implementation based on the original Universal PDF RAG Chatbot project by Ratnesh Singh.

Original repository: https://github.com/Ratnesh-181998/Universal-PDF-RAG-Chatbot

Additional modifications and project configuration are maintained in this repository.

---

**Repository:** https://github.com/kripali1503/rag-chatbot
