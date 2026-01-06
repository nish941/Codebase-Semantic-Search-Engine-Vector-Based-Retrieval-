"""
Tests for embedding functionality
"""
import pytest
import numpy as np
from src.embedding import EmbeddingManager

@pytest.fixture
def embedding_manager():
    return EmbeddingManager()

def test_embedding_initialization(embedding_manager):
    """Test embedding manager initialization"""
    assert embedding_manager.models is not None
    assert 'sentence-transformer' in embedding_manager.models

def test_embed_single_text(embedding_manager):
    """Test embedding single text"""
    text = "This is a test function for authentication"
    result = embedding_manager.embed(text, model_type="sentence-transformer")
    
    assert result.embeddings is not None
    assert result.embeddings.shape[0] == 1  # Single embedding
    assert result.embeddings.shape[1] == 768  # MPNet dimension
    assert result.model_name == "sentence-transformer"
    assert result.metadata['num_texts'] == 1

def test_embed_multiple_texts(embedding_manager):
    """Test embedding multiple texts"""
    texts = [
        "function for user authentication",
        "class for handling database connections",
        "method to validate email addresses"
    ]
    
    result = embedding_manager.embed(texts, model_type="sentence-transformer")
    
    assert result.embeddings.shape[0] == 3  # Three embeddings
    assert result.embeddings.shape[1] == 768

def test_embedding_cache(embedding_manager):
    """Test embedding caching"""
    text = "Cache test function"
    
    # First call - should not be cached
    result1 = embedding_manager.embed(text)
    assert result1.metadata['cached'] == False
    
    # Second call - should be cached
    result2 = embedding_manager.embed(text)
    assert result2.metadata['cached'] == True
    
    # Embeddings should be identical
    np.testing.assert_array_equal(result1.embeddings, result2.embeddings)

def test_invalid_model_type(embedding_manager):
    """Test with invalid model type"""
    with pytest.raises(ValueError):
        embedding_manager.embed("test", model_type="invalid-model")

def test_embedding_normalization(embedding_manager):
    """Test that embeddings are normalized"""
    texts = ["test text 1", "test text 2"]
    result = embedding_manager.embed(texts)
    
    # Check if embeddings are normalized (unit vectors)
    for i in range(result.embeddings.shape[0]):
        norm = np.linalg.norm(result.embeddings[i])
        assert abs(norm - 1.0) < 1e-6, f"Embedding {i} not normalized: norm={norm}"
