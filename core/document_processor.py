"""
core/document_processor.py — FINAL v8
Embeddings: Groq does NOT have an embedding API.
Solution: Use a pure local embedding with no PyTorch dependency.
Package: chromadb has a built-in default embedding (uses onnxruntime, not torch).
This avoids ALL API quota issues entirely.
"""
import os
import hashlib
import time
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.embeddings import Embeddings

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIR, UPLOAD_DIR


# ── 100% Local Embeddings — no API, no quota, no PyTorch ─────────────────────
class FastLocalEmbeddings(Embeddings):
    """
    Uses chromadb's built-in default embedding function (onnxruntime-based).
    - Zero API calls
    - Zero quota usage
    - No PyTorch dependency
    - Works offline
    - Model: all-MiniLM-L6-v2 via onnxruntime (not sentence-transformers)
    """
    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(x) for x in vec] for vec in self._fn(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [float(x) for x in self._fn([text])[0]]


def _get_embeddings() -> FastLocalEmbeddings:
    return FastLocalEmbeddings()


def _get_or_create_vectorstore(collection_name: str = "research_docs") -> Chroma:
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def _doc_id(source: str) -> str:
    return hashlib.md5(source.encode()).hexdigest()[:8]


def _safe_add_documents(vs: Chroma, chunks: List[Document]) -> None:
    valid = [c for c in chunks if c.page_content and c.page_content.strip()]
    if not valid:
        raise ValueError(
            "No valid text extracted from document.\n"
            "Use a text-based PDF (e.g. from arxiv.org), not a scanned image PDF."
        )
    for i in range(0, len(valid), 50):
        vs.add_documents(valid[i:i + 50])


def _load_pdf(file_path: str) -> List[Document]:
    # Attempt 1: pdfplumber
    try:
        import pdfplumber
        docs = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": i}
                    ))
        if docs:
            return docs
    except Exception as e:
        print(f"[PDF] pdfplumber failed: {e}")

    # Attempt 2: PyPDFLoader
    try:
        from langchain_community.document_loaders import PyPDFLoader
        docs = [d for d in PyPDFLoader(file_path).load() if d.page_content.strip()]
        if docs:
            return docs
    except Exception as e:
        print(f"[PDF] PyPDFLoader failed: {e}")

    raise ValueError(
        "Could not extract text from this PDF.\n"
        "Use a text-based PDF (arxiv.org papers work perfectly)."
    )


def ingest_pdf(file_path: str) -> Tuple[int, str]:
    pages    = _load_pdf(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata.update({"doc_id": _doc_id(file_path), "doc_type": "pdf", "chunk_idx": i})
    _safe_add_documents(_get_or_create_vectorstore(), chunks)
    return len(chunks), _doc_id(file_path)


def ingest_url(url: str) -> Tuple[int, str]:
    docs   = WebBaseLoader(url).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata.update({"doc_id": _doc_id(url), "doc_type": "url", "source": url, "chunk_idx": i})
    _safe_add_documents(_get_or_create_vectorstore(), chunks)
    return len(chunks), _doc_id(url)


def ingest_text_file(file_path: str) -> Tuple[int, str]:
    ext    = os.path.splitext(file_path)[1].lower()
    loader = Docx2txtLoader(file_path) if ext == ".docx" else TextLoader(file_path, encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(loader.load())
    for i, c in enumerate(chunks):
        c.metadata.update({"doc_id": _doc_id(file_path), "doc_type": ext.lstrip("."), "chunk_idx": i})
    _safe_add_documents(_get_or_create_vectorstore(), chunks)
    return len(chunks), _doc_id(file_path)


def get_retriever(top_k: int = 6):
    return _get_or_create_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )


def get_vectorstore_stats() -> dict:
    try:
        collection = _get_or_create_vectorstore()._collection

        total_chunks = collection.count()

        data = collection.get()

        unique_docs = len(
            set(
                meta.get("source", "Unknown")
                for meta in data.get("metadatas", [])
            )
        )

        return {
            "total_chunks": total_chunks,
            "total_documents": unique_docs,
            "status": "active"
        }

    except Exception as e:
        print(f"Vectorstore stats error: {e}")

        return {
            "total_chunks": 0,
            "total_documents": 0,
            "status": "empty"
        }


def clear_vectorstore():
    try:
        _get_or_create_vectorstore().delete_collection()
    except Exception:
        import shutil
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)