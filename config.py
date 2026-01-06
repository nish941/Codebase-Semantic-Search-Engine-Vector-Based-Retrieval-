"""
Configuration for the semantic search engine
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Application configuration"""
    
    # Flask settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 5000))
    
    # Model settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    EMBEDDING_DIMENSION: int = 768
    MAX_SEQUENCE_LENGTH: int = 512
    DEVICE: str = os.getenv("DEVICE", "cpu")  # or "cuda"
    
    # FAISS settings
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "IVF")  # "Flat", "IVF", "HNSW"
    FAISS_N_PROBE: int = int(os.getenv("FAISS_N_PROBE", 10))
    
    # Elasticsearch settings
    ELASTICSEARCH_HOST: str = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    ELASTICSEARCH_INDEX: str = os.getenv("ELASTICSEARCH_INDEX", "codebase")
    
    # ChromaDB settings
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    
    # OpenAI settings (optional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Search settings
    DEFAULT_TOPK: int = int(os.getenv("DEFAULT_TOPK", 10))
    MIN_SIMILARITY_SCORE: float = float(os.getenv("MIN_SIMILARITY_SCORE", 0.5))
    
    # Code parsing settings
    SUPPORTED_EXTENSIONS: list = [
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', 
        '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
        '.html', '.css', '.scss', '.json', '.yml', '.yaml', '.xml', '.md'
    ]
    
    IGNORE_PATTERNS: list = [
        '__pycache__', '.git', '.venv', 'node_modules', 'dist', 'build',
        '*.min.js', '*.min.css', '*.log', '*.tmp'
    ]
    
    # Chunking settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    
    # Cache settings
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./data/cache")
    EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", 10000))
    
    # Performance settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 32))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 4))
    
    @property
    def elasticsearch_enabled(self) -> bool:
        return bool(self.ELASTICSEARCH_HOST and self.ELASTICSEARCH_HOST != "disabled")
    
    @property
    def openai_enabled(self) -> bool:
        return bool(self.OPENAI_API_KEY)
    
    @property
    def chromadb_enabled(self) -> bool:
        return os.getenv("CHROMADB_ENABLED", "true").lower() == "true"

config = Config()
