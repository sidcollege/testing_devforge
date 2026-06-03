# 🚀 DevForge RAG Testing Suite

Welcome to the `rag` branch of the DevForge testing repository. This branch is dedicated to the rigorous validation and quality assurance of the **Retrieval-Augmented Generation (RAG)** pipeline within the DevForge backend.

## 🎯 Objective
The primary goal of this suite is to ensure the reliability, precision, and performance of the end-to-end RAG workflow, from document ingestion to the final LLM response.

## 🛠️ Core Capabilities Tested

### 1. Data Ingestion & Processing
- **Document Reading**: Support for PDF, Markdown, and Text files.
- **Intelligent Chunking**: Validation of overlap and size constraints to maintain context.
- **Embedding Generation**: Testing multiple models (e.g., `nomic-embed-text`, `bge-m3`) for optimal vectorization.

### 2. Vector Storage & Retrieval
- **Store Initialization**: Verification of ChromaDB and Qdrant integrations.
- **Semantic Search**: Validating top-k retrieval and score-based threshold filtering.
- **Hybrid Fusion**: Ensuring the balance between keyword (BM25) and vector search.

### 3. Response Generation
- **Contextual Augmentation**: Verifying that the LLM uses retrieved documents to answer queries.
- **Fallback Handling**: Testing "insufficient information" responses when no relevant docs are found.

## 📂 Repository Structure

| Path | Description |
| :--- | :--- |
| `tests/test_rag.py` | **Main Test Suite**. Comprehensive unit and integration tests for the entire RAG lifecycle. |
| `src/` | Source code for RAG agents and tools. |
| `utils.py` | General utility functions used across the RAG suite. |
| `hello.py` | Basic connectivity/smoke test script. |
| `README.md` | Project documentation and onboarding guide. |

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Pytest installed (`pip install pytest pytest-asyncio`)
- Access to an Ollama instance (for embedding/LLM tests)

### Running the Tests
To execute the RAG test suite, run:

```bash
pytest tests/test_rag.py -v
```

To run a specific test with detailed output:

```bash
pytest tests/test_rag.py::test_end_to_end_workflow -v -s
```

## 📈 Current Status
- [x] Core RAG workflow tests implemented
- [x] Vector store mocking verified
- [ ] Performance benchmarking for reranking (Pending)
- [ ] Integration tests for pgvector (Pending)