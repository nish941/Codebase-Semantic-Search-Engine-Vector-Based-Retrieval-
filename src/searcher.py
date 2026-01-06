"""
Search functionality for the semantic search engine
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import config
from src.embedding import embedding_manager
from src.indexer import vector_index
from src.parser import CodeChunk

@dataclass
class SearchResult:
    """Enhanced search result with additional metadata"""
    chunk: CodeChunk
    similarity_score: float
    search_source: str
    context: str
    highlight_lines: List[int]
    explanation: Optional[str] = None
    
    @property
    def formatted_result(self) -> Dict[str, Any]:
        """Format result for API response"""
        return {
            'file_path': self.chunk.file_path,
            'language': self.chunk.language,
            'start_line': self.chunk.start_line,
            'end_line': self.chunk.end_line,
            'content': self.chunk.content,
            'similarity_score': self.similarity_score,
            'chunk_type': self.chunk.chunk_type,
            'metadata': self.chunk.metadata,
            'search_source': self.search_source,
            'context': self.context,
            'highlight_lines': self.highlight_lines,
            'explanation': self.explanation
        }

class SemanticSearcher:
    """Advanced semantic search with query understanding and result ranking"""
    
    def __init__(self):
        self.query_cache: Dict[str, List[SearchResult]] = {}
    
    def search(self, query: str, 
               top_k: int = None,
               use_hybrid: bool = True,
               include_explanation: bool = False) -> List[SearchResult]:
        """
        Perform semantic search
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            use_hybrid: Use hybrid search (FAISS + keyword)
            include_explanation: Include AI-generated explanation
        
        Returns:
            List of SearchResult objects
        """
        if top_k is None:
            top_k = config.DEFAULT_TOPK
        
        # Check cache
        cache_key = f"{query}_{top_k}_{use_hybrid}"
        if cache_key in self.query_cache:
            logger.debug(f"Cache hit for query: {query}")
            return self.query_cache[cache_key]
        
        logger.info(f"Processing query: '{query}'")
        
        # Parse and understand query
        query_intent = self._analyze_query_intent(query)
        logger.debug(f"Query intent: {query_intent}")
        
        # Perform search
        if use_hybrid:
            raw_results = vector_index.hybrid_search(query, top_k * 2)
        else:
            raw_results = vector_index.search(query, top_k * 2)
        
        # Process and enhance results
        enhanced_results = []
        query_embedding = embedding_manager.embed(query).embeddings[0]
        
        for raw_result in raw_results[:top_k * 3]:  # Consider more for re-ranking
            result = self._enhance_search_result(
                raw_result, query, query_intent, query_embedding
            )
            
            if result.similarity_score >= config.MIN_SIMILARITY_SCORE:
                enhanced_results.append(result)
        
        # Re-rank results
        re_ranked_results = self._re_rank_results(enhanced_results, query_intent)
        
        # Limit to top_k
        final_results = re_ranked_results[:top_k]
        
        # Generate explanations if requested
        if include_explanation:
            for result in final_results:
                result.explanation = self._generate_explanation(result, query)
        
        # Cache results
        if len(self.query_cache) < 100:  # Limit cache size
            self.query_cache[cache_key] = final_results
        
        logger.info(f"Search completed. Found {len(final_results)} results")
        
        return final_results
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and extract patterns"""
        intent = {
            'type': 'general',
            'targets': [],
            'patterns': [],
            'qualifiers': []
        }
        
        # Detect query type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'where', 'why', 'what']):
            intent['type'] = 'explanation'
        
        if 'function' in query_lower or 'method' in query_lower:
            intent['type'] = 'function'
            intent['targets'].append('function')
        
        if 'class' in query_lower:
            intent['type'] = 'class'
            intent['targets'].append('class')
        
        if 'error' in query_lower or 'exception' in query_lower:
            intent['type'] = 'error'
        
        if 'test' in query_lower:
            intent['type'] = 'test'
        
        # Extract specific patterns
        patterns = {
            r'function\s+that\s+(\w+)': 'function_with_action',
            r'class\s+that\s+(\w+)': 'class_with_purpose',
            r'where\s+is\s+(\w+)': 'location',
            r'how\s+to\s+(\w+)': 'howto'
        }
        
        for pattern, intent_name in patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                intent['patterns'].append((intent_name, match.group(1)))
        
        # Extract qualifiers
        qualifiers = ['handle', 'process', 'validate', 'create', 'update', 'delete']
        for qualifier in qualifiers:
            if qualifier in query_lower:
                intent['qualifiers'].append(qualifier)
        
        return intent
    
    def _enhance_search_result(self, raw_result: Dict[str, Any], 
                              query: str, query_intent: Dict[str, Any],
                              query_embedding: np.ndarray) -> SearchResult:
        """Enhance search result with additional context and scoring"""
        chunk = raw_result['chunk']
        
        # Calculate additional similarity metrics
        chunk_text = self._prepare_chunk_for_scoring(chunk)
        chunk_embedding = embedding_manager.embed(chunk_text).embeddings[0]
        
        # Calculate semantic similarity
        semantic_similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            chunk_embedding.reshape(1, -1)
        )[0][0]
        
        # Calculate keyword overlap score
        keyword_score = self._calculate_keyword_overlap(query, chunk.content)
        
        # Calculate intent matching score
        intent_score = self._calculate_intent_match(query_intent, chunk)
        
        # Combine scores
        final_score = (
            0.6 * semantic_similarity +
            0.2 * keyword_score +
            0.2 * intent_score
        )
        
        # Get context around the chunk
        context = self._get_chunk_context(chunk)
        
        # Find highlight lines (lines that are most relevant)
        highlight_lines = self._find_highlight_lines(query, chunk)
        
        return SearchResult(
            chunk=chunk,
            similarity_score=float(final_score),
            search_source=raw_result.get('source', 'faiss'),
            context=context,
            highlight_lines=highlight_lines
        )
    
    def _prepare_chunk_for_scoring(self, chunk: CodeChunk) -> str:
        """Prepare chunk text for scoring"""
        # Include metadata for better semantic understanding
        text = f"""
        File: {chunk.file_path}
        Type: {chunk.chunk_type}
        Language: {chunk.language}
        
        Content:
        {chunk.content}
        """
        
        # Add metadata information
        if chunk.metadata:
            text += "\nMetadata:\n"
            for key, value in chunk.metadata.items():
                if isinstance(value, str) and len(value) < 100:
                    text += f"{key}: {value}\n"
        
        return text
    
    def _calculate_keyword_overlap(self, query: str, content: str) -> float:
        """Calculate keyword overlap score"""
        # Extract keywords from query (excluding common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        query_keywords = query_words - common_words
        
        if not query_keywords:
            return 0.5  # Neutral score if no specific keywords
        
        # Extract words from content
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(query_keywords.intersection(content_words))
        union = len(query_keywords.union(content_words))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_intent_match(self, intent: Dict[str, Any], chunk: CodeChunk) -> float:
        """Calculate how well the chunk matches query intent"""
        score = 0.0
        
        # Match chunk type with intent targets
        if intent['type'] == 'function' and chunk.chunk_type in ['function', 'method']:
            score += 0.3
        
        if intent['type'] == 'class' and chunk.chunk_type == 'class':
            score += 0.3
        
        # Match qualifiers in content
        content_lower = chunk.content.lower()
        for qualifier in intent['qualifiers']:
            if qualifier in content_lower:
                score += 0.1
        
        # Match patterns
        for pattern_type, pattern_value in intent['patterns']:
            if pattern_value in content_lower:
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_chunk_context(self, chunk: CodeChunk, lines_before: int = 3, 
                          lines_after: int = 3) -> str:
        """Get context around the chunk (neighboring lines)"""
        try:
            with open(chunk.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
            
            start_line = max(0, chunk.start_line - lines_before)
            end_line = min(len(all_lines) - 1, chunk.end_line + lines_after)
            
            context_lines = []
            for i in range(start_line, end_line + 1):
                prefix = '>>> ' if chunk.start_line <= i <= chunk.end_line else '    '
                context_lines.append(f"{prefix}{i+1}: {all_lines[i].rstrip()}")
            
            return '\n'.join(context_lines)
            
        except Exception as e:
            logger.warning(f"Failed to get context for {chunk.file_path}: {e}")
            return ""
    
    def _find_highlight_lines(self, query: str, chunk: CodeChunk) -> List[int]:
        """Find lines within chunk that are most relevant to query"""
        highlight_lines = []
        lines = chunk.content.split('\n')
        
        # Extract keywords from query
        keywords = re.findall(r'\b\w+\b', query.lower())
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [k for k in keywords if k not in common_words and len(k) > 2]
        
        if not keywords:
            # Return middle lines if no specific keywords
            mid_point = len(lines) // 2
            return list(range(mid_point - 1, min(mid_point + 2, len(lines))))
        
        # Score each line
        line_scores = []
        for i, line in enumerate(lines):
            score = 0
            line_lower = line.lower()
            
            for keyword in keywords:
                if keyword in line_lower:
                    score += 1
            
            # Bonus for exact matches
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', line_lower):
                    score += 2
            
            line_scores.append((i, score))
        
        # Get top scoring lines
        line_scores.sort(key=lambda x: x[1], reverse=True)
        top_lines = [line_num for line_num, score in line_scores[:5] if score > 0]
        
        # Convert to absolute line numbers
        absolute_lines = [chunk.start_line + line_num for line_num in top_lines]
        
        return absolute_lines
    
    def _re_rank_results(self, results: List[SearchResult], 
                        query_intent: Dict[str, Any]) -> List[SearchResult]:
        """Re-rank results based on additional factors"""
        if not results:
            return results
        
        # Calculate additional ranking factors
        for result in results:
            # Boost score for exact type matches
            if query_intent['type'] == 'function' and result.chunk.chunk_type in ['function', 'method']:
                result.similarity_score *= 1.2
            
            if query_intent['type'] == 'class' and result.chunk.chunk_type == 'class':
                result.similarity_score *= 1.2
            
            # Boost for recent/modified files (if metadata available)
            if 'modified_time' in result.chunk.metadata:
                # Simple recency boost
                result.similarity_score *= 1.1
        
        # Sort by final score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    def _generate_explanation(self, result: SearchResult, query: str) -> str:
        """Generate AI-powered explanation for why result is relevant"""
        # This is a simplified version - in production, you might use an LLM
        explanation_parts = []
        
        # Explain based on chunk type
        if result.chunk.chunk_type == 'function':
            explanation_parts.append(f"This is a {result.chunk.language} function")
            if 'name' in result.chunk.metadata:
                explanation_parts.append(f"named '{result.chunk.metadata['name']}'")
        
        elif result.chunk.chunk_type == 'class':
            explanation_parts.append(f"This is a {result.chunk.language} class")
            if 'name' in result.chunk.metadata:
                explanation_parts.append(f"named '{result.chunk.metadata['name']}'")
        
        # Explain relevance
        if result.similarity_score > 0.8:
            explanation_parts.append("It's highly relevant to your query.")
        elif result.similarity_score > 0.6:
            explanation_parts.append("It's relevant to your query.")
        else:
            explanation_parts.append("It's somewhat related to your query.")
        
        # Add location info
        explanation_parts.append(f"Located in {result.chunk.file_path} (lines {result.chunk.start_line+1}-{result.chunk.end_line+1})")
        
        return ' '.join(explanation_parts)
    
    def clear_cache(self):
        """Clear search cache"""
        self.query_cache.clear()
        logger.info("Search cache cleared")

# Global searcher instance
semantic_searcher = SemanticSearcher()
