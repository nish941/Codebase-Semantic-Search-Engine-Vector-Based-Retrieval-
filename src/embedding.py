"""
Embedding generation utilities using various models
"""
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Union
import numpy as np
from loguru import logger
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import openai
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
import chromadb.utils.embedding_functions as embedding_functions

from config import config

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embeddings: np.ndarray
    model_name: str
    metadata: Dict[str, Any]
    cache_key: Optional[str] = None

class EmbeddingManager:
    """Manages embedding generation with caching and multiple model support"""
    
    def __init__(self):
        self.cache: Dict[str, np.ndarray] = {}
        self.models: Dict[str, Any] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models"""
        logger.info(f"Initializing embedding models on {config.DEVICE}")
        
        # Initialize Sentence Transformer
        try:
            self.models["sentence-transformer"] = SentenceTransformer(
                config.EMBEDDING_MODEL,
                device=config.DEVICE
            )
            logger.info(f"Loaded Sentence Transformer: {config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer: {e}")
            self.models["sentence-transformer"] = None
        
        # Initialize HuggingFace transformer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
            self.transformer_model = AutoModel.from_pretrained(
                config.EMBEDDING_MODEL,
                torch_dtype=torch.float32 if config.DEVICE == "cpu" else torch.float16
            ).to(config.DEVICE)
            self.models["huggingface"] = self.transformer_model
            logger.info(f"Loaded HuggingFace model: {config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            self.models["huggingface"] = None
        
        # Initialize OpenAI embeddings if API key is available
        if config.openai_enabled:
            try:
                self.models["openai"] = OpenAIEmbeddings(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    openai_api_key=config.OPENAI_API_KEY
                )
                logger.info(f"Initialized OpenAI embeddings: {config.OPENAI_EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                self.models["openai"] = None
    
    def _generate_cache_key(self, texts: List[str], model_name: str) -> str:
        """Generate cache key for embeddings"""
        text_hash = hashlib.md5("".join(texts).encode()).hexdigest()
        return f"{model_name}_{text_hash}"
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Sentence Transformer"""
        if self.models["sentence-transformer"] is None:
            raise ValueError("Sentence Transformer model not initialized")
        
        embeddings = self.models["sentence-transformer"].encode(
            texts,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def embed_with_huggingface(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using HuggingFace transformers"""
        if self.models["huggingface"] is None:
            raise ValueError("HuggingFace model not initialized")
        
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        ).to(config.DEVICE)
        
        with torch.no_grad():
            model_output = self.transformer_model(**encoded_input)
        
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = embeddings.cpu().numpy()
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def embed_with_openai(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        if not config.openai_enabled or self.models["openai"] is None:
            raise ValueError("OpenAI embeddings not available")
        
        embeddings = self.models["openai"].embed_documents(texts)
        return np.array(embeddings)
    
    def embed(self, texts: Union[str, List[str]], 
              model_type: str = "sentence-transformer") -> EmbeddingResult:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of texts
            model_type: One of "sentence-transformer", "huggingface", "openai"
        
        Returns:
            EmbeddingResult object
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        cache_key = self._generate_cache_key(texts, model_type)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for {len(texts)} texts")
            return EmbeddingResult(
                embeddings=self.cache[cache_key],
                model_name=model_type,
                metadata={"cached": True},
                cache_key=cache_key
            )
        
        logger.info(f"Generating embeddings for {len(texts)} texts using {model_type}")
        
        # Generate embeddings
        if model_type == "sentence-transformer":
            embeddings = self.embed_with_sentence_transformer(texts)
        elif model_type == "huggingface":
            embeddings = self.embed_with_huggingface(texts)
        elif model_type == "openai":
            embeddings = self.embed_with_openai(texts)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Update cache
        if len(self.cache) < config.EMBEDDING_CACHE_SIZE:
            self.cache[cache_key] = embeddings
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=model_type,
            metadata={
                "cached": False,
                "num_texts": len(texts),
                "embedding_dim": embeddings.shape[1]
            },
            cache_key=cache_key
        )
    
    def get_embedding_function(self, model_type: str = "sentence-transformer"):
        """Get embedding function for ChromaDB"""
        if model_type == "sentence-transformer":
            def embedding_function(texts):
                result = self.embed(texts, model_type)
                return result.embeddings.tolist()
            return embedding_function
        elif model_type == "openai" and config.openai_enabled:
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=config.OPENAI_API_KEY,
                model_name=config.OPENAI_EMBEDDING_MODEL
            )
        else:
            raise ValueError(f"Unsupported model type for ChromaDB: {model_type}")
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")

# Global embedding manager instance
embedding_manager = EmbeddingManager()
