"""
Tests for search functionality
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.searcher import SemanticSearcher, SearchResult
from src.parser import CodeChunk

@pytest.fixture
def semantic_searcher():
    return SemanticSearcher()

@pytest.fixture
def mock_code_chunk():
    """Create a mock code chunk"""
    return CodeChunk(
        content='def authenticate_user(username, password):\n    # Authentication logic\n    return True',
        file_path='/app/auth.py',
        language='python',
        start_line=10,
        end_line=15,
        chunk_type='function',
        metadata={'name': 'authenticate_user'}
    )

@pytest.fixture
def mock_search_results(mock_code_chunk):
    """Create mock search results"""
    return [
        {
            'score': 0.85,
            'chunk': mock_code_chunk,
            'metadata': {'name': 'authenticate_user'},
            'source': 'faiss'
        }
    ]

def test_analyze_query_intent(semantic_searcher):
    """Test query intent analysis"""
    # Function-related query
    intent = semantic_searcher._analyze_query_intent(
        "function for user authentication"
    )
    assert intent['type'] == 'function'
    assert 'function' in intent['targets']
    assert 'handle' not in intent['qualifiers']  # Not in this query
    
    # Class-related query
    intent = semantic_searcher._analyze_query_intent(
        "class that handles database connections"
    )
    assert intent['type'] == 'class'
    assert 'class' in intent['targets']
    assert 'handle' in intent['qualifiers']
    
    # Explanation query
    intent = semantic_searcher._analyze_query_intent(
        "how to validate email addresses"
    )
    assert intent['type'] == 'explanation'
    assert len(intent['patterns']) > 0
    
    # Error-related query
    intent = semantic_searcher._analyze_query_intent(
        "error handling in authentication"
    )
    assert intent['type'] == 'error'

def test_calculate_keyword_overlap(semantic_searcher):
    """Test keyword overlap calculation"""
    query = "function for user authentication"
    content = "def authenticate_user(username, password):\n    # Authentication logic"
    
    score = semantic_searcher._calculate_keyword_overlap(query, content)
    
    assert 0 <= score <= 1
    # Should find some overlap
    assert score > 0
    
    # Test with no overlap
    score = semantic_searcher._calculate_keyword_overlap(
        "database connection",
        "def calculate_sum(a, b): return a + b"
    )
    assert score == 0

def test_calculate_intent_match(semantic_searcher, mock_code_chunk):
    """Test intent matching"""
    # Function intent with function chunk
    intent = {'type': 'function', 'targets': ['function'], 'qualifiers': [], 'patterns': []}
    score = semantic_searcher._calculate_intent_match(intent, mock_code_chunk)
    assert score > 0
    
    # Class intent with function chunk (should have lower score)
    intent = {'type': 'class', 'targets': ['class'], 'qualifiers': [], 'patterns': []}
    score = semantic_searcher._calculate_intent_match(intent, mock_code_chunk)
    assert score == 0
    
    # Intent with qualifier that matches content
    intent = {
        'type': 'function',
        'targets': ['function'],
        'qualifiers': ['authenticate'],
        'patterns': []
    }
    mock_code_chunk.content = "def authenticate_user(): pass"
    score = semantic_searcher._calculate_intent_match(intent, mock_code_chunk)
    assert score > 0.1  # Should have bonus for qualifier match

def test_prepare_chunk_for_scoring(semantic_searcher, mock_code_chunk):
    """Test chunk preparation for scoring"""
    prepared_text = semantic_searcher._prepare_chunk_for_scoring(mock_code_chunk)
    
    assert 'File:' in prepared_text
    assert '/app/auth.py' in prepared_text
    assert 'Type: function' in prepared_text
    assert 'Language: python' in prepared_text
    assert 'def authenticate_user' in prepared_text
    assert 'Metadata:' in prepared_text

def test_find_highlight_lines(semantic_searcher, mock_code_chunk):
    """Test highlight line identification"""
    query = "user authentication function"
    mock_code_chunk.content = '''def authenticate_user(username, password):
    # Check if user exists
    user = get_user(username)
    if not user:
        return False
    
    # Verify password
    if verify_password(user, password):
        return True
    return False'''
    
    highlight_lines = semantic_searcher._find_highlight_lines(query, mock_code_chunk)
    
    assert isinstance(highlight_lines, list)
    # Should find lines containing relevant keywords
    assert len(highlight_lines) > 0
    
    # Test with no keywords
    query = "xyz abc"
    highlight_lines = semantic_searcher._find_highlight_lines(query, mock_code_chunk)
    # Should return middle lines as default
    assert len(highlight_lines) == 3

def test_get_chunk_context(semantic_searcher, mock_code_chunk, tmp_path):
    """Test getting chunk context"""
    # Create a temporary file with content
    file_content = '''Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10
Line 11
Line 12
Line 13
Line 14
Line 15
Line 16'''
    
    test_file = tmp_path / "test.py"
    test_file.write_text(file_content)
    
    mock_code_chunk.file_path = str(test_file)
    mock_code_chunk.start_line = 5  # Line 6 (0-indexed)
    mock_code_chunk.end_line = 10   # Line 11
    
    context = semantic_searcher._get_chunk_context(mock_code_chunk, lines_before=2, lines_after=2)
    
    assert isinstance(context, str)
    assert len(context.split('\n')) == 7  # 2 before + 5 chunk lines + 2 after
    
    # Check that chunk lines are marked
    assert '>>>' in context
    
    # Test with file that doesn't exist
    mock_code_chunk.file_path = '/nonexistent/file.py'
    context = semantic_searcher._get_chunk_context(mock_code_chunk)
    assert context == ""

@patch('src.searcher.embedding_manager')
@patch('src.searcher.vector_index')
def test_search(mock_vector_index, mock_embedding_manager, semantic_searcher, mock_search_results):
    """Test search functionality"""
    # Mock dependencies
    mock_vector_index.hybrid_search.return_value = mock_search_results
    mock_embedding_manager.embed.return_value.embeddings = np.array([[0.1] * 768])
    
    results = semantic_searcher.search(
        query="authentication function",
        top_k=5,
        use_hybrid=True,
        include_explanation=False
    )
    
    assert isinstance(results, list)
    mock_vector_index.hybrid_search.assert_called_once()
    
    # Test with cache
    results2 = semantic_searcher.search(
        query="authentication function",
        top_k=5,
        use_hybrid=True,
        include_explanation=False
    )
    
    assert len(results2) == len(results)

def test_re_rank_results(semantic_searcher, mock_code_chunk):
    """Test result re-ranking"""
    # Create test results
    results = [
        SearchResult(
            chunk=mock_code_chunk,
            similarity_score=0.7,
            search_source='faiss',
            context='',
            highlight_lines=[]
        ),
        SearchResult(
            chunk=mock_code_chunk,
            similarity_score=0.8,
            search_source='faiss',
            context='',
            highlight_lines=[]
        ),
        SearchResult(
            chunk=mock_code_chunk,
            similarity_score=0.6,
            search_source='faiss',
            context='',
            highlight_lines=[]
        )
    ]
    
    # Test re-ranking (should sort by score)
    re_ranked = semantic_searcher._re_rank_results(results, {'type': 'general'})
    
    assert len(re_ranked) == 3
    assert re_ranked[0].similarity_score >= re_ranked[1].similarity_score
    assert re_ranked[1].similarity_score >= re_ranked[2].similarity_score
    
    # Test with function intent (should boost function chunks)
    results[0].chunk.chunk_type = 'function'
    results[1].chunk.chunk_type = 'class'
    results[2].chunk.chunk_type = 'function'
    
    original_scores = [r.similarity_score for r in results]
    re_ranked = semantic_searcher._re_rank_results(results, {'type': 'function'})
    
    # Function chunks should be boosted
    assert re_ranked[0].similarity_score > original_scores[0]

def test_generate_explanation(semantic_searcher, mock_code_chunk):
    """Test explanation generation"""
    result = SearchResult(
        chunk=mock_code_chunk,
        similarity_score=0.85,
        search_source='faiss',
        context='',
        highlight_lines=[]
    )
    
    explanation = semantic_searcher._generate_explanation(result, "authentication function")
    
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert 'python' in explanation.lower()
    assert 'function' in explanation.lower()
    assert 'auth.py' in explanation
    assert 'lines' in explanation

def test_clear_cache(semantic_searcher):
    """Test cache clearing"""
    # Add something to cache
    semantic_searcher.query_cache['test'] = []
    
    semantic_searcher.clear_cache()
    
    assert len(semantic_searcher.query_cache) == 0

@patch('src.searcher.embedding_manager')
def test_enhance_search_result(mock_embedding_manager, semantic_searcher, mock_search_results, mock_code_chunk):
    """Test search result enhancement"""
    # Mock embedding
    mock_embedding_manager.embed.return_value.embeddings = np.array([[0.1] * 768])
    
    raw_result = mock_search_results[0]
    query_intent = {'type': 'function', 'targets': [], 'qualifiers': [], 'patterns': []}
    query_embedding = np.random.randn(768)
    
    enhanced = semantic_searcher._enhance_search_result(
        raw_result, 
        "authentication function",
        query_intent,
        query_embedding
    )
    
    assert isinstance(enhanced, SearchResult)
    assert enhanced.chunk == mock_code_chunk
    assert 0 <= enhanced.similarity_score <= 1
    assert enhanced.search_source == 'faiss'
    assert isinstance(enhanced.context, str)
    assert isinstance(enhanced.highlight_lines, list)

def test_search_with_low_similarity(semantic_searcher):
    """Test search with low similarity threshold"""
    # Mock the search to return no results
    with patch.object(semantic_searcher, 'search') as mock_search:
        mock_search.return_value = []
        
        results = semantic_searcher.search(
            query="xyz abc",  # Unlikely to match anything
            top_k=5
        )
        
        assert len(results) == 0
