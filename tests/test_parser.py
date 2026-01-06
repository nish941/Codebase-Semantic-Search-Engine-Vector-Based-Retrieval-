"""
Tests for code parsing functionality
"""
import pytest
from pathlib import Path
import tempfile
from src.parser import CodeParser, ParsedFile

@pytest.fixture
def code_parser():
    return CodeParser()

@pytest.fixture
def python_test_file():
    """Create a temporary Python test file"""
    code = '''
"""
Module docstring
"""
import os
import sys

def hello_world(name: str) -> str:
    """Greet the user"""
    return f"Hello, {name}!"

class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        self.result = 0
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers"""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

# This is a comment
def helper_function():
    pass
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()

@pytest.fixture
def javascript_test_file():
    """Create a temporary JavaScript test file"""
    code = '''
/**
 * This is a JavaScript module
 */

function calculateSum(a, b) {
    // Add two numbers
    return a + b;
}

class User {
    constructor(name, email) {
        this.name = name;
        this.email = email;
    }
    
    getUserInfo() {
        return `${this.name} (${this.email})`;
    }
}

const PI = 3.14159;

// Arrow function
const multiply = (a, b) => a * b;
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()

def test_detect_language(code_parser):
    """Test language detection"""
    assert code_parser.detect_language('/path/to/file.py') == 'python'
    assert code_parser.detect_language('/path/to/file.js') == 'javascript'
    assert code_parser.detect_language('/path/to/file.ts') == 'typescript'
    assert code_parser.detect_language('/path/to/file.java') == 'java'
    assert code_parser.detect_language('/path/to/file.cpp') == 'cpp'
    assert code_parser.detect_language('/path/to/unknown.ext') == 'unknown'

def test_should_ignore(code_parser):
    """Test ignore patterns"""
    # Should ignore
    assert code_parser.should_ignore('/path/to/__pycache__/file.py') == True
    assert code_parser.should_ignore('/path/to/.git/config') == True
    assert code_parser.should_ignore('/path/to/node_modules/package.json') == True
    assert code_parser.should_ignore('/path/to/file.min.js') == True
    assert code_parser.should_ignore('/path/to/file.log') == True
    
    # Should not ignore
    assert code_parser.should_ignore('/path/to/file.py') == False
    assert code_parser.should_ignore('/path/to/src/main.py') == False
    assert code_parser.should_ignore('/path/to/test.js') == False

def test_parse_python_file(code_parser, python_test_file):
    """Test parsing Python file"""
    parsed_file = code_parser.parse_file(python_test_file)
    
    assert parsed_file is not None
    assert parsed_file.language == 'python'
    assert parsed_file.path == python_test_file
    assert len(parsed_file.content) > 0
    
    # Check chunks
    assert len(parsed_file.chunks) > 0
    
    # Find function chunk
    function_chunks = [c for c in parsed_file.chunks if c.chunk_type == 'function']
    assert len(function_chunks) >= 2  # hello_world and helper_function
    
    # Find class chunk
    class_chunks = [c for c in parsed_file.chunks if c.chunk_type == 'class']
    assert len(class_chunks) >= 1
    
    # Check chunk metadata
    hello_chunk = next((c for c in function_chunks if 'hello_world' in c.content), None)
    assert hello_chunk is not None
    assert 'name' in hello_chunk.metadata
    assert hello_chunk.metadata['name'] == 'hello_world'
    assert 'docstring' in hello_chunk.metadata
    
    # Check file chunk exists
    file_chunks = [c for c in parsed_file.chunks if c.chunk_type == 'file']
    assert len(file_chunks) == 1

def test_parse_javascript_file(code_parser, javascript_test_file):
    """Test parsing JavaScript file"""
    parsed_file = code_parser.parse_file(javascript_test_file)
    
    assert parsed_file is not None
    assert parsed_file.language == 'javascript'
    assert parsed_file.path == javascript_test_file
    assert len(parsed_file.content) > 0
    
    # Check chunks
    assert len(parsed_file.chunks) > 0
    
    # Find function chunks
    function_chunks = [c for c in parsed_file.chunks if c.chunk_type == 'function']
    assert len(function_chunks) >= 2  # calculateSum and multiply (arrow function)
    
    # Find class chunk
    class_chunks = [c for c in parsed_file.chunks if c.chunk_type == 'class']
    assert len(class_chunks) >= 1
    
    # Check file chunk
    file_chunks = [c for c in parsed_file.chunks if c.chunk_type == 'file']
    assert len(file_chunks) == 1

def test_parse_nonexistent_file(code_parser):
    """Test parsing non-existent file"""
    parsed_file = code_parser.parse_file('/nonexistent/path/file.py')
    assert parsed_file is None

def test_parse_empty_file(code_parser):
    """Test parsing empty file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('')
        temp_path = f.name
    
    try:
        parsed_file = code_parser.parse_file(temp_path)
        assert parsed_file is not None
        assert len(parsed_file.content) == 0
        assert len(parsed_file.chunks) == 1  # Only file chunk
    finally:
        Path(temp_path).unlink()

def test_split_large_chunk(code_parser):
    """Test splitting large chunks"""
    # Create a large chunk
    large_content = '\n'.join([f"Line {i}" for i in range(100)])
    
    chunk = code_parser.CodeChunk(
        content=large_content,
        file_path='/test/file.py',
        language='python',
        start_line=0,
        end_line=99,
        chunk_type='function',
        metadata={'name': 'large_function'}
    )
    
    # Split the chunk
    split_chunks = code_parser.split_large_chunk(chunk)
    
    assert len(split_chunks) > 1
    assert all(isinstance(c, code_parser.CodeChunk) for c in split_chunks)
    
    # Check that content is preserved
    combined_content = '\n'.join([c.content for c in split_chunks])
    assert combined_content == large_content
    
    # Check metadata
    for i, sub_chunk in enumerate(split_chunks):
        assert 'part_index' in sub_chunk.metadata
        assert 'total_parts' in sub_chunk.metadata
        assert sub_chunk.metadata['total_parts'] == len(split_chunks)

def test_extract_comments(code_parser):
    """Test comment extraction"""
    python_code = '''
# Single line comment
def func():
    """Docstring comment"""
    pass

"""
Multi-line
comment
"""
'''
    
    comments = code_parser.extract_comments(
        python_code,
        '/test/file.py',
        'python'
    )
    
    assert len(comments) >= 2  # Single line and multi-line
    
    # Check comment types
    comment_types = [c.metadata['comment_type'] for c in comments]
    assert 'single' in comment_types
    assert 'multi' in comment_types

def test_chunk_id_generation(code_parser):
    """Test chunk ID generation"""
    chunk = code_parser.CodeChunk(
        content='def test(): pass',
        file_path='/test/file.py',
        language='python',
        start_line=10,
        end_line=12,
        chunk_type='function',
        metadata={'name': 'test'}
    )
    
    assert chunk.id == '/test/file.py:10:12'
    assert ':10:12' in chunk.id
