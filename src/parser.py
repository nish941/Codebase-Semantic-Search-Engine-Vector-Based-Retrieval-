"""
Code parsing and chunking utilities
"""
import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import tree_sitter
from tree_sitter_languages import get_language, get_parser
import pygments
from pygments.lexers import get_lexer_for_filename, guess_lexer
from pygments.token import Token

from config import config

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'block', 'comment'
    metadata: Dict[str, Any]
    
    @property
    def id(self) -> str:
        """Generate unique ID for chunk"""
        return f"{self.file_path}:{self.start_line}:{self.end_line}"
    
    @property
    def display_text(self) -> str:
        """Get display text with line numbers"""
        lines = self.content.split('\n')
        numbered_lines = [f"{self.start_line + i}: {line}" for i, line in enumerate(lines)]
        return '\n'.join(numbered_lines)

@dataclass
class ParsedFile:
    """Represents a parsed source file"""
    path: str
    language: str
    content: str
    chunks: List[CodeChunk]
    metadata: Dict[str, Any]
    
    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

class CodeParser:
    """Parses code files and extracts meaningful chunks"""
    
    def __init__(self):
        self.language_parsers: Dict[str, Any] = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages"""
        supported_languages = {
            'python': 'python',
            'javascript': 'javascript',
            'typescript': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'go': 'go',
            'rust': 'rust',
            'ruby': 'ruby',
        }
        
        for lang_name, lang_id in supported_languages.items():
            try:
                language = get_language(lang_id)
                parser = get_parser(lang_id)
                self.language_parsers[lang_name] = (language, parser)
                logger.debug(f"Initialized parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize parser for {lang_name}: {e}")
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'cpp',
            '.h': 'cpp',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.cs': 'csharp',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'css',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.xml': 'xml',
            '.md': 'markdown',
        }
        
        return language_map.get(ext, 'unknown')
    
    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored"""
        path_str = str(file_path)
        
        for pattern in config.IGNORE_PATTERNS:
            if '*' in pattern:
                # Simple wildcard matching
                if pattern.startswith('*'):
                    if path_str.endswith(pattern[1:]):
                        return True
            elif pattern in path_str:
                return True
        
        ext = Path(file_path).suffix.lower()
        if ext not in config.SUPPORTED_EXTENSIONS:
            return True
        
        return False
    
    def parse_file(self, file_path: str) -> Optional[ParsedFile]:
        """
        Parse a single file and extract chunks
        
        Args:
            file_path: Path to the file
        
        Returns:
            ParsedFile object or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language = self.detect_language(file_path)
            
            if language == 'unknown':
                logger.warning(f"Unknown language for file: {file_path}")
                return None
            
            # Extract chunks based on language
            if language in self.language_parsers:
                chunks = self.parse_with_tree_sitter(file_path, content, language)
            else:
                chunks = self.parse_with_regex(content, file_path, language)
            
            # Add file-level chunk
            file_chunk = CodeChunk(
                content=content,
                file_path=file_path,
                language=language,
                start_line=0,
                end_line=len(content.split('\n')) - 1,
                chunk_type='file',
                metadata={
                    'filename': os.path.basename(file_path),
                    'file_size': len(content),
                    'num_lines': len(content.split('\n')),
                }
            )
            chunks.append(file_chunk)
            
            # Split large chunks
            all_chunks = []
            for chunk in chunks:
                if len(chunk.content) > config.CHUNK_SIZE * 2:
                    all_chunks.extend(self.split_large_chunk(chunk))
                else:
                    all_chunks.append(chunk)
            
            metadata = {
                'language': language,
                'file_size': len(content),
                'num_chunks': len(all_chunks),
                'parse_method': 'tree-sitter' if language in self.language_parsers else 'regex'
            }
            
            return ParsedFile(
                path=file_path,
                language=language,
                content=content,
                chunks=all_chunks,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return None
    
    def parse_with_tree_sitter(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Parse file using tree-sitter"""
        language_obj, parser = self.language_parsers[language]
        tree = parser.parse(bytes(content, 'utf-8'))
        root_node = tree.root_node
        
        chunks = []
        
        # Language-specific extraction rules
        if language == 'python':
            chunks.extend(self._extract_python_nodes(root_node, content, file_path))
        elif language in ['javascript', 'typescript']:
            chunks.extend(self._extract_javascript_nodes(root_node, content, file_path))
        elif language == 'java':
            chunks.extend(self._extract_java_nodes(root_node, content, file_path))
        elif language == 'cpp':
            chunks.extend(self._extract_cpp_nodes(root_node, content, file_path))
        
        return chunks
    
    def _extract_python_nodes(self, root_node, content: str, file_path: str) -> List[CodeChunk]:
        """Extract chunks from Python AST"""
        chunks = []
        
        def walk(node):
            if node.type == 'function_definition':
                # Extract function
                func_name = None
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = content[child.start_byte:child.end_byte]
                        break
                
                chunk_content = content[node.start_byte:node.end_byte]
                lines = chunk_content.split('\n')
                
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    language='python',
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    chunk_type='function',
                    metadata={
                        'name': func_name or 'anonymous',
                        'node_type': node.type,
                        'docstring': self._extract_docstring(node, content)
                    }
                )
                chunks.append(chunk)
            
            elif node.type == 'class_definition':
                # Extract class
                class_name = None
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = content[child.start_byte:child.end_byte]
                        break
                
                chunk_content = content[node.start_byte:node.end_byte]
                
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    language='python',
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    chunk_type='class',
                    metadata={
                        'name': class_name or 'anonymous',
                        'node_type': node.type
                    }
                )
                chunks.append(chunk)
            
            # Walk children
            for child in node.children:
                walk(child)
        
        walk(root_node)
        return chunks
    
    def _extract_javascript_nodes(self, root_node, content: str, file_path: str) -> List[CodeChunk]:
        """Extract chunks from JavaScript/TypeScript AST"""
        chunks = []
        
        def walk(node):
            if node.type in ['function_declaration', 'function', 'arrow_function', 'method_definition']:
                # Extract function/method
                func_name = 'anonymous'
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = content[child.start_byte:child.end_byte]
                        break
                
                chunk_content = content[node.start_byte:node.end_byte]
                
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    language='javascript',
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    chunk_type='function',
                    metadata={
                        'name': func_name,
                        'node_type': node.type
                    }
                )
                chunks.append(chunk)
            
            elif node.type in ['class_declaration', 'class']:
                # Extract class
                class_name = 'anonymous'
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = content[child.start_byte:child.end_byte]
                        break
                
                chunk_content = content[node.start_byte:node.end_byte]
                
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    language='javascript',
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    chunk_type='class',
                    metadata={
                        'name': class_name,
                        'node_type': node.type
                    }
                )
                chunks.append(chunk)
            
            # Walk children
            for child in node.children:
                walk(child)
        
        walk(root_node)
        return chunks
    
    def _extract_docstring(self, node, content: str) -> Optional[str]:
        """Extract docstring from Python node"""
        for child in node.children:
            if child.type == 'expression_statement':
                expr_content = content[child.start_byte:child.end_byte].strip()
                if (expr_content.startswith('"""') or expr_content.startswith("'''") or
                    expr_content.startswith('r"""') or expr_content.startswith("r'''")):
                    return expr_content
        return None
    
    def parse_with_regex(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """Fallback parsing using regex patterns"""
        chunks = []
        
        # Language-specific regex patterns
        patterns = {
            'python': [
                (r'def\s+(\w+)\s*\([^)]*\)\s*:', 'function'),
                (r'class\s+(\w+)\s*(?:\([^)]*\))?\s*:', 'class'),
            ],
            'javascript': [
                (r'(?:function|const|let|var)\s+(\w+)\s*=\s*(?:function\s*)?\([^)]*\)\s*\{', 'function'),
                (r'class\s+(\w+)\s*\{', 'class'),
            ],
            'java': [
                (r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{', 'method'),
                (r'class\s+(\w+)\s*\{', 'class'),
            ],
            'cpp': [
                (r'(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{', 'function'),
                (r'class\s+(\w+)\s*\{', 'class'),
            ]
        }
        
        if language in patterns:
            lines = content.split('\n')
            for pattern, chunk_type in patterns[language]:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    start_line = content[:match.start()].count('\n')
                    end_line = self._find_matching_brace(content, match.end())
                    
                    if end_line > start_line:
                        chunk_content = '\n'.join(lines[start_line:end_line + 1])
                        
                        chunk = CodeChunk(
                            content=chunk_content,
                            file_path=file_path,
                            language=language,
                            start_line=start_line,
                            end_line=end_line,
                            chunk_type=chunk_type,
                            metadata={
                                'name': match.group(1) if match.groups() else 'anonymous',
                                'parsed_with': 'regex'
                            }
                        )
                        chunks.append(chunk)
        
        return chunks
    
    def _find_matching_brace(self, content: str, start_pos: int) -> int:
        """Find matching closing brace"""
        brace_stack = 0
        i = start_pos
        
        while i < len(content):
            if content[i] == '{':
                brace_stack += 1
            elif content[i] == '}':
                brace_stack -= 1
                if brace_stack == 0:
                    return content[:i].count('\n')
            i += 1
        
        return len(content.split('\n')) - 1
    
    def split_large_chunk(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split large chunk into smaller chunks"""
        if len(chunk.content) <= config.CHUNK_SIZE:
            return [chunk]
        
        chunks = []
        lines = chunk.content.split('\n')
        current_start = chunk.start_line
        
        for i in range(0, len(lines), config.CHUNK_SIZE - config.CHUNK_OVERLAP):
            end_idx = min(i + config.CHUNK_SIZE, len(lines))
            chunk_lines = lines[i:end_idx]
            
            sub_chunk = CodeChunk(
                content='\n'.join(chunk_lines),
                file_path=chunk.file_path,
                language=chunk.language,
                start_line=current_start + i,
                end_line=current_start + end_idx - 1,
                chunk_type=f"{chunk.chunk_type}_part",
                metadata={
                    **chunk.metadata,
                    'part_index': len(chunks),
                    'total_parts': -1  # Will be updated
                }
            )
            chunks.append(sub_chunk)
        
        # Update total_parts in metadata
        for i, sub_chunk in enumerate(chunks):
            sub_chunk.metadata['total_parts'] = len(chunks)
        
        return chunks
    
    def extract_comments(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """Extract comments from code"""
        chunks = []
        
        comment_patterns = {
            'python': r'#.*$|\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'',
            'javascript': r'//.*$|/\*[\s\S]*?\*/',
            'java': r'//.*$|/\*[\s\S]*?\*/',
            'cpp': r'//.*$|/\*[\s\S]*?\*/',
            'csharp': r'//.*$|/\*[\s\S]*?\*/',
        }
        
        if language in comment_patterns:
            lines = content.split('\n')
            pattern = comment_patterns[language]
            
            for match in re.finditer(pattern, content, re.MULTILINE):
                start_line = content[:match.start()].count('\n')
                end_line = content[:match.end()].count('\n')
                
                comment_chunk = CodeChunk(
                    content=match.group(0),
                    file_path=file_path,
                    language=language,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type='comment',
                    metadata={
                        'comment_type': 'single' if '//' in match.group(0) or '#' in match.group(0) else 'multi'
                    }
                )
                chunks.append(comment_chunk)
        
        return chunks

# Global parser instance
code_parser = CodeParser()
