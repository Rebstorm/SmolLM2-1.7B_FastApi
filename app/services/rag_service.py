import hashlib
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DEFAULT_PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "cv-documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class CVRagService:
    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embedding_fn
        )

    def _stored_source_hash(self) -> Optional[str]:
        meta = self.collection.metadata or {}
        return meta.get("source_hash")

    def index_cv(self, cv_path: str, force: bool = False) -> bool:
        current_hash = file_hash(cv_path)
        if not force and self.collection.count() > 0:
            if self._stored_source_hash() == current_hash:
                print(f"CV index up to date (hash={current_hash[:8]}...), skipping.")
                return False

        print(f"Indexing CV from {cv_path}...")
        text = extract_text_from_pdf(cv_path)
        chunks = chunk_text(text)

        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"source_hash": current_hash, "source_path": cv_path},
        )
        self.collection.add(
            ids=[f"chunk-{i}" for i in range(len(chunks))],
            documents=chunks,
        )
        print(f"Indexed {len(chunks)} chunks from CV.")
        return True

    def query(self, query_text: str, top_k: int = 3) -> List[str]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        documents = results.get("documents") or [[]]
        return list(documents[0])
