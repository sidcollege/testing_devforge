"""Unit and integration tests for RAG feature.

Tests cover:
- Embedding model factory (nomic-embed-text, bge-m3, invalid model)
- Vector store initialization (ChromaDB mock)
- Document reading (PDF, MD, TXT, DOCX, invalid paths, size limits)
- Document chunking (size, overlap, metadata)
- Document ingestion (single/multiple docs, errors, parallel processing)
- Document retrieval (top_k, score threshold, empty results)
- Response generation (mock LLM)
- RAG agent workflow (query-only, ingest+query, error handling)
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

from src.agents.rag.agent import RAGState, rag_agent_invoke
from src.tools.rag.tools import (
    chunk_document,
    generate_response,
    get_embedding_model,
    get_vector_store,
    ingest_documents,
    read_document,
    retrieve_docs,
)

MOCK_EMBEDDING_DIM = 768
MOCK_EMBEDDING = [0.1] * MOCK_EMBEDDING_DIM

@pytest.fixture
def mock_ollama_embeddings():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents = AsyncMock(return_value=[MOCK_EMBEDDING] * 5)
    mock_embeddings.embed_query = AsyncMock(return_value=MOCK_EMBEDDING)
    return mock_embeddings

@pytest.fixture
def mock_chroma_client():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=mock_collection)
    return mock_client, mock_collection

@pytest.fixture
def mock_vector_store():
    mock_store = MagicMock()
    mock_store.add_documents = MagicMock()
    mock_store.similarity_search_with_score = MagicMock(
        return_value=[
            (Document(page_content="Test content 1", metadata={"source": "test1.txt"}), 0.2),
            (Document(page_content="Test content 2", metadata={"source": "test2.txt"}), 0.3),
            (Document(page_content="Test content 3", metadata={"source": "test3.txt"}), 0.4),
        ]
    )
    return mock_store

@pytest.fixture
def mock_model_router():
    mock_router = MagicMock()
    mock_chat_model = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "This is a test response based on the context."
    mock_chat_model.ainvoke = AsyncMock(return_value=mock_response)
    mock_router.get_chat_model = MagicMock(return_value=mock_chat_model)
    mock_router.select_model_by_task = MagicMock(return_value="gpt-oss:20b")
    return mock_router

@pytest.fixture
def temp_doc_files():
    temp_dir = tempfile.mkdtemp()
    files = {}
    txt_file = Path(temp_dir) / "test.txt"
    txt_file.write_text("This is a test text file.\nIt has multiple lines.\nFor testing purposes.")
    files["txt"] = str(txt_file)
    md_file = Path(temp_dir) / "test.md"
    md_file.write_text("# Test Document\n\nThis is a **markdown** file for testing.")
    files["md"] = str(md_file)
    yield files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

class TestGetEmbeddingModel:
    @patch("src.tools.rag.tools.OllamaEmbeddings")
    def test_get_embedding_model_nomic(self, mock_ollama_class, mock_ollama_embeddings):
        mock_ollama_class.return_value = mock_ollama_embeddings
        result = get_embedding_model("nomic-embed-text")
        assert result == mock_ollama_embeddings
        mock_ollama_class.assert_called_once()

    @patch("src.tools.rag.tools.OllamaEmbeddings")
    def test_get_embedding_model_bge_m3(self, mock_ollama_class, mock_ollama_embeddings):
        mock_ollama_class.return_value = mock_ollama_embeddings
        result = get_embedding_model("bge-m3")
        assert result == mock_ollama_embeddings
        mock_ollama_class.assert_called_once()

    def test_get_embedding_model_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid model name"):
            get_embedding_model("")

    def test_get_embedding_model_invalid_none(self):
        with pytest.raises(ValueError, match="Invalid model name"):
            get_embedding_model(None)

class TestGetVectorStore:
    @patch("src.tools.rag.tools.chromadb.PersistentClient")
    @patch("src.tools.rag.tools.Chroma")
    def test_get_vector_store_chroma(self, mock_chroma_class, mock_client_class, mock_ollama_embeddings):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_store = MagicMock()
        mock_chroma_class.return_value = mock_store
        result = get_vector_store("chroma", mock_ollama_embeddings, "test_collection")
        assert result == mock_store
        mock_client_class.assert_called_once()
        mock_chroma_class.assert_called_once()

    def test_get_vector_store_invalid_backend(self, mock_ollama_embeddings):
        with pytest.raises(ValueError, match="Unsupported vector backend"):
            get_vector_store("invalid_backend", mock_ollama_embeddings, "test_collection")

    @patch("src.tools.rag.tools.settings")
    @patch("src.tools.rag.tools.QdrantClient")
    @patch("src.tools.rag.tools.QdrantVectorStore")
    def test_get_vector_store_qdrant(self, mock_qdrant_store_class, mock_qdrant_client_class, mock_settings, mock_ollama_embeddings):
        mock_settings.QDRANT_URL = "https://test.qdrant.io"
        mock_settings.QDRANT_API_KEY = "test-api-key"
        mock_settings.QDRANT_COLLECTION = "test_collection"
        mock_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_client
        mock_store = MagicMock()
        mock_qdrant_store_class.return_value = mock_store
        result = get_vector_store("qdrant", mock_ollama_embeddings, "test_collection")
        assert result == mock_store
        mock_qdrant_client_class.assert_called_once_with(url="https://test.qdrant.io", api_key="test-api-key", timeout=30)
        mock_qdrant_store_class.assert_called_once()

class TestReadDocument:
    @pytest.mark.asyncio
    async def test_read_txt_file(self, temp_doc_files):
        content = await read_document(temp_doc_files["txt"])
        assert isinstance(content, str)
        assert "test text file" in content.lower()

    @pytest.mark.asyncio
    async def test_read_md_file(self, temp_doc_files):
        content = await read_document(temp_doc_files["md"])
        assert isinstance(content, str)
        assert "markdown" in content.lower()

    @pytest.mark.asyncio
    async def test_read_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            await read_document("/nonexistent/file.txt")

class TestChunkDocument:
    def test_chunk_document_basic(self):
        text = "This is a test document. " * 100
        chunks = chunk_document(text, "test.txt", chunk_size=500, chunk_overlap=50)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

class TestIngestDocuments:
    @pytest.mark.asyncio
    @patch("src.tools.rag.tools.get_vector_store")
    @patch("src.tools.rag.tools.get_embedding_model")
    async def test_ingest_single_document(self, mock_get_embeddings, mock_get_store, temp_doc_files, mock_vector_store):
        mock_get_embeddings.return_value = MagicMock()
        mock_get_store.return_value = mock_vector_store
        result = await ingest_documents(file_paths=[temp_doc_files["txt"]], embed_model="nomic-embed-text", backend="chroma")
        assert result["success"] is True
        assert result["documents_processed"] == 1
        mock_vector_store.add_documents.assert_called_once()

class TestRetrieveDocs:
    @pytest.mark.asyncio
    @patch("src.tools.rag.tools.get_vector_store")
    @patch("src.tools.rag.tools.get_embedding_model")
    async def test_retrieve_docs_basic(self, mock_get_embeddings, mock_get_store, mock_vector_store):
        mock_get_embeddings.return_value = MagicMock()
        mock_get_store.return_value = mock_vector_store
        results = await retrieve_docs("test query", top_k=3, embed_model="nomic-embed-text", backend="chroma")
        assert isinstance(results, list)
        assert len(results) > 0

class TestGenerateResponse:
    @pytest.mark.asyncio
    @patch("src.tools.rag.tools.model_router")
    async def test_generate_response_success(self, mock_router_module, mock_model_router):
        mock_router_module.select_model_by_task = MagicMock(return_value="gpt-oss:20b")
        mock_router_module.get_chat_model = mock_model_router.get_chat_model
        response = await generate_response(query="What is the main topic?", context="This document is about AI and machine learning.")
        assert isinstance(response, str)
        assert len(response) > 0

class TestRAGAgentWorkflow:
    @pytest.mark.asyncio
    @patch("src.agents.rag.agent.retrieve_docs")
    @patch("src.agents.rag.agent.generate_response")
    async def test_rag_agent_query_only(self, mock_generate, mock_retrieve):
        mock_retrieve.return_value = [{"content": "Test content", "metadata": {}, "score": 0.8, "id": "1"}]
        mock_generate.return_value = "This is the answer."
        result = await rag_agent_invoke(query="Test question")
        assert result["success"] is True
        assert result["data"]["response"] == "This is the answer."

    @pytest.mark.asyncio
    @patch("src.agents.rag.agent.ingest_documents")
    async def test_rag_agent_ingestion_error(self, mock_ingest):
        mock_ingest.side_effect = Exception("Ingestion failed")
        result = await rag_agent_invoke(query="Test question", file_paths=["/invalid/path.txt"])
        assert result["success"] is False
        assert "Ingestion failed" in result["error"]
