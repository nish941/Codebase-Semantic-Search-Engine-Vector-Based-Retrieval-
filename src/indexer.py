"""
Indexing utilities for vector search
"""
import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
from loguru import logger
import faiss
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from config import config
from src.embedding import embedding_manager
from src.parser import CodeParser, ParsedFile, CodeChunk

@dataclass
class IndexResult:
    """Result of indexing operation"""
    success: bool
    num_files: int
    num_chunks: int
    index_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

class VectorIndex:
    """Manages vector indexing and search"""
    
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.chunk_store: Dict[str, CodeChunk] = {}
        
        # Initialize ChromaDB if enabled
        if config.chromadb_enabled:
            self.chroma_client = chromadb.PersistentClient(
                path=config.CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            self.chroma_collection = None
        else:
            self.chroma_client = None
            self.chroma_collection = None
        
        # Initialize Elasticsearch if enabled
        if config.elasticsearch_enabled:
            self.es_client = Elasticsearch(config.ELASTICSEARCH_HOST)
            try:
                self.es_client.info()
                logger.info(f"Connected to Elasticsearch at {config.ELASTICSEARCH_HOST}")
            except Exception as e:
                logger.error(f"Failed to connect to Elasticsearch: {e}")
                self.es_client = None
        else:
            self.es_client = None
    
    def create_faiss_index(self, embedding_dim: int) -> faiss.Index:
        """Create FAISS index based on configuration"""
        if config.FAISS_INDEX_TYPE == "Flat":
            index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            logger.info(f"Created Flat FAISS index with dimension {embedding_dim}")
        
        elif config.FAISS_INDEX_TYPE == "IVF":
            nlist = min(100, embedding_dim * 4)  # Number of clusters
            quantizer = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Created IVF FAISS index with {nlist} clusters")
        
        elif config.FAISS_INDEX_TYPE == "HNSW":
            # HNSW parameters
            M = 16  # Number of neighbors
            ef_construction = 200  # Construction time/accuracy trade-off
            index = faiss.IndexHNSWFlat(embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            logger.info(f"Created HNSW FAISS index with M={M}")
        
        else:
            raise ValueError(f"Unsupported FAISS index type: {config.FAISS_INDEX_TYPE}")
        
        return index
    
    def index_codebase(self, repo_path: str, 
                       output_dir: str = "./data/indices") -> IndexResult:
        """
        Index an entire codebase
        
        Args:
            repo_path: Path to the codebase
            output_dir: Directory to save index files
        
        Returns:
            IndexResult object
        """
        try:
            repo_path = Path(repo_path).resolve()
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all source files
            source_files = self._find_source_files(repo_path)
            
            if not source_files:
                return IndexResult(
                    success=False,
                    num_files=0,
                    num_chunks=0,
                    error="No source files found"
                )
            
            logger.info(f"Found {len(source_files)} source files to index")
            
            # Parse files and extract chunks
            all_chunks = []
            parsed_files = []
            
            for file_path in tqdm(source_files, desc="Parsing files"):
                parsed_file = code_parser.parse_file(str(file_path))
                if parsed_file:
                    parsed_files.append(parsed_file)
                    all_chunks.extend(parsed_file.chunks)
            
            logger.info(f"Parsed {len(parsed_files)} files, extracted {len(all_chunks)} chunks")
            
            # Generate embeddings for chunks
            chunk_texts = [self._prepare_chunk_text(chunk) for chunk in all_chunks]
            
            logger.info("Generating embeddings...")
            embedding_result = embedding_manager.embed(
                chunk_texts,
                model_type="sentence-transformer"
            )
            
            embeddings = embedding_result.embeddings
            logger.info(f"Generated embeddings with shape {embeddings.shape}")
            
            # Create FAISS index
            self.index = self.create_faiss_index(embeddings.shape[1])
            
            # Add vectors to index
            if config.FAISS_INDEX_TYPE == "IVF":
                # Train IVF index
                self.index.train(embeddings)
            
            self.index.add(embeddings)
            
            # Store metadata
            self.metadata = []
            self.chunk_store = {}
            
            for i, chunk in enumerate(all_chunks):
                metadata = {
                    'id': i,
                    'chunk_id': chunk.id,
                    'file_path': chunk.file_path,
                    'language': chunk.language,
                    'chunk_type': chunk.chunk_type,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'metadata': chunk.metadata,
                    'embedding_model': embedding_result.model_name,
                }
                self.metadata.append(metadata)
                self.chunk_store[chunk.id] = chunk
            
            # Save index and metadata
            index_path = output_dir / "faiss_index.bin"
            metadata_path = output_dir / "metadata.json"
            chunks_path = output_dir / "chunks.pkl"
            
            faiss.write_index(self.index, str(index_path))
            
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunk_store, f)
            
            # Index in ChromaDB if enabled
            if config.chromadb_enabled:
                self._index_in_chromadb(all_chunks, chunk_texts, embedding_result)
            
            # Index in Elasticsearch if enabled
            if self.es_client:
                self._index_in_elasticsearch(all_chunks)
            
            logger.info(f"Indexing completed. Saved to {output_dir}")
            
            return IndexResult(
                success=True,
                num_files=len(parsed_files),
                num_chunks=len(all_chunks),
                index_path=str(index_path),
                metadata={
                    'embedding_model': embedding_result.model_name,
                    'embedding_dim': embedding_result.embeddings.shape[1],
                    'index_type': config.FAISS_INDEX_TYPE,
                    'total_size_mb': os.path.getsize(index_path) / (1024 * 1024)
                }
            )
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return IndexResult(
                success=False,
                num_files=0,
                num_chunks=0,
                error=str(e)
            )
    
    def _find_source_files(self, repo_path: Path) -> List[Path]:
        """Find all source files in repository"""
        source_files = []
        
        for ext in config.SUPPORTED_EXTENSIONS:
            pattern = f"**/*{ext}"
            for file_path in repo_path.rglob(pattern):
                if not code_parser.should_ignore(str(file_path)):
                    source_files.append(file_path)
        
        return source_files
    
    def _prepare_chunk_text(self, chunk: CodeChunk) -> str:
        """Prepare text for embedding generation"""
        # Include metadata in the text for better semantic understanding
        metadata_text = f"File: {chunk.file_path}\n"
        metadata_text += f"Type: {chunk.chunk_type}\n"
        
        if 'name' in chunk.metadata:
            metadata_text += f"Name: {chunk.metadata['name']}\n"
        
        if 'docstring' in chunk.metadata:
            metadata_text += f"Docstring: {chunk.metadata['docstring']}\n"
        
        return metadata_text + "\n" + chunk.content
    
    def _index_in_chromadb(self, chunks: List[CodeChunk], texts: List[str], 
                          embedding_result):
        """Index chunks in ChromaDB"""
        try:
            # Create or get collection
            collection_name = "codebase_chunks"
            
            if collection_name in [c.name for c in self.chroma_client.list_collections()]:
                self.chroma_client.delete_collection(collection_name)
            
            # Create collection with embedding function
            embedding_func = embedding_manager.get_embedding_function(
                model_type="sentence-transformer"
            )
            
            self.chroma_collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_func,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Add documents
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                documents.append(texts[i])
                metadatas.append({
                    'file_path': chunk.file_path,
                    'language': chunk.language,
                    'chunk_type': chunk.chunk_type,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    **chunk.metadata
                })
                ids.append(chunk.id)
            
            # Add in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                self.chroma_collection.add(
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    ids=ids[i:i+batch_size]
                )
            
            logger.info(f"Indexed {len(chunks)} chunks in ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to index in ChromaDB: {e}")
    
    def _index_in_elasticsearch(self, chunks: List[CodeChunk]):
        """Index chunks in Elasticsearch"""
        try:
            # Create index if it doesn't exist
            if not self.es_client.indices.exists(index=config.ELASTICSEARCH_INDEX):
                index_settings = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "code_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "asciifolding"]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "content": {
                                "type": "text",
                                "analyzer": "code_analyzer"
                            },
                            "file_path": {"type": "keyword"},
                            "language": {"type": "keyword"},
                            "chunk_type": {"type": "keyword"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                            "metadata": {"type": "object"}
                        }
                    }
                }
                self.es_client.indices.create(
                    index=config.ELASTICSEARCH_INDEX,
                    body=index_settings
                )
            
            # Prepare documents for bulk indexing
            actions = []
            for i, chunk in enumerate(chunks):
                action = {
                    "_index": config.ELASTICSEARCH_INDEX,
                    "_id": chunk.id,
                    "_source": {
                        "content": chunk.content,
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "chunk_type": chunk.chunk_type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "metadata": chunk.metadata
                    }
                }
                actions.append(action)
            
            # Bulk index
            success, failed = bulk(self.es_client, actions)
            logger.info(f"Indexed {success} documents in Elasticsearch")
            if failed:
                logger.warning(f"Failed to index {len(failed)} documents")
            
        except Exception as e:
            logger.error(f"Failed to index in Elasticsearch: {e}")
    
    def load_index(self, index_path: str, metadata_path: str, chunks_path: str) -> bool:
        """Load existing index from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Load chunks
            with open(chunks_path, 'rb') as f:
                self.chunk_store = pickle.load(f)
            
            logger.info(f"Loaded index with {len(self.metadata)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar code chunks
        
        Args:
            query: Natural language query
            top_k: Number of results to return
        
        Returns:
            List of search results with similarity scores
        """
        if top_k is None:
            top_k = config.DEFAULT_TOPK
        
        try:
            # Generate query embedding
            embedding_result = embedding_manager.embed(
                query,
                model_type="sentence-transformer"
            )
            query_vector = embedding_result.embeddings[0].reshape(1, -1)
            
            # Search in FAISS index
            if config.FAISS_INDEX_TYPE == "IVF":
                self.index.nprobe = config.FAISS_N_PROBE
            
            scores, indices = self.index.search(query_vector, top_k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):  # Invalid index
                    continue
                
                if score < config.MIN_SIMILARITY_SCORE:  # Low similarity
                    continue
                
                metadata = self.metadata[idx]
                chunk_id = metadata['chunk_id']
                
                if chunk_id in self.chunk_store:
                    chunk = self.chunk_store[chunk_id]
                    
                    result = {
                        'score': float(score),
                        'chunk': chunk,
                        'metadata': metadata,
                        'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                    }
                    results.append(result)
            
            # Sort by score descending
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using multiple indices
        
        Args:
            query: Natural language query
            top_k: Number of results to return
        
        Returns:
            Combined search results
        """
        if top_k is None:
            top_k = config.DEFAULT_TOPK
        
        all_results = []
        
        # FAISS search (semantic)
        faiss_results = self.search(query, top_k * 2)
        all_results.extend(faiss_results)
        
        # ChromaDB search if available
        if self.chroma_collection:
            try:
                chroma_results = self.chroma_collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                for i, (doc, meta, distance) in enumerate(zip(
                    chroma_results['documents'][0],
                    chroma_results['metadatas'][0],
                    chroma_results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    score = 1.0 - distance
                    
                    if score >= config.MIN_SIMILARITY_SCORE:
                        chunk = CodeChunk(
                            content=doc,
                            file_path=meta['file_path'],
                            language=meta['language'],
                            start_line=meta['start_line'],
                            end_line=meta['end_line'],
                            chunk_type=meta['chunk_type'],
                            metadata={k: v for k, v in meta.items() 
                                     if k not in ['file_path', 'language', 'chunk_type', 'start_line', 'end_line']}
                        )
                        
                        result = {
                            'score': score,
                            'chunk': chunk,
                            'metadata': meta,
                            'source': 'chromadb',
                            'content_preview': doc[:200] + "..." if len(doc) > 200 else doc
                        }
                        all_results.append(result)
            except Exception as e:
                logger.error(f"ChromaDB search failed: {e}")
        
        # Elasticsearch search if available
        if self.es_client:
            try:
                es_query = {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "metadata.name^2"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    "size": top_k
                }
                
                response = self.es_client.search(
                    index=config.ELASTICSEARCH_INDEX,
                    body=es_query
                )
                
                for hit in response['hits']['hits']:
                    source = hit['_source']
                    
                    chunk = CodeChunk(
                        content=source['content'],
                        file_path=source['file_path'],
                        language=source['language'],
                        start_line=source['start_line'],
                        end_line=source['end_line'],
                        chunk_type=source['chunk_type'],
                        metadata=source['metadata']
                    )
                    
                    result = {
                        'score': hit['_score'],
                        'chunk': chunk,
                        'metadata': source,
                        'source': 'elasticsearch',
                        'content_preview': source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
                    }
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Elasticsearch search failed: {e}")
        
        # Deduplicate and sort results
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            chunk_id = result['chunk'].id
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        
        return unique_results[:top_k]

# Global vector index instance
vector_index = VectorIndex()
