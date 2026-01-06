"""
Flask application for the semantic search engine API
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_restful import Api, Resource
from loguru import logger
import traceback

from config import config
from src.indexer import vector_index
from src.searcher import SemanticSearcher, SearchResult

# Initialize Flask app
app = Flask(__name__, 
           template_folder='frontend' if Path('frontend').exists() else None)
CORS(app)
api = Api(app)

# Initialize searcher
searcher = SemanticSearcher()

class HealthCheck(Resource):
    """Health check endpoint"""
    def get(self):
        return {
            "status": "healthy",
            "service": "codebase-semantic-search-engine",
            "version": "1.0.0",
            "index_loaded": vector_index.index is not None,
            "config": {
                "embedding_model": config.EMBEDDING_MODEL,
                "faiss_index_type": config.FAISS_INDEX_TYPE,
                "default_topk": config.DEFAULT_TOPK
            }
        }

class SearchAPI(Resource):
    """Search endpoint"""
    
    def post(self):
        try:
            data = request.get_json()
            
            if not data or 'query' not in data:
                return {
                    "error": "Missing 'query' in request body"
                }, 400
            
            query = data['query']
            top_k = data.get('top_k', config.DEFAULT_TOPK)
            use_hybrid = data.get('use_hybrid', True)
            include_explanation = data.get('include_explanation', False)
            
            logger.info(f"Search request: query='{query}', top_k={top_k}")
            
            # Perform search
            results = searcher.search(
                query=query,
                top_k=top_k,
                use_hybrid=use_hybrid,
                include_explanation=include_explanation
            )
            
            # Format response
            response = {
                "query": query,
                "total_results": len(results),
                "results": [result.formatted_result for result in results],
                "search_metadata": {
                    "top_k": top_k,
                    "use_hybrid": use_hybrid,
                    "min_similarity_score": config.MIN_SIMILARITY_SCORE
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Search error: {e}\n{traceback.format_exc()}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc() if config.DEBUG else None
            }, 500

class IndexAPI(Resource):
    """Index management endpoint"""
    
    def post(self):
        """Create new index"""
        try:
            data = request.get_json()
            
            if not data or 'repo_path' not in data:
                return {
                    "error": "Missing 'repo_path' in request body"
                }, 400
            
            repo_path = data['repo_path']
            output_dir = data.get('output_dir', './data/indices')
            
            if not Path(repo_path).exists():
                return {
                    "error": f"Repository path does not exist: {repo_path}"
                }, 400
            
            logger.info(f"Indexing request: repo_path='{repo_path}'")
            
            # Index the codebase
            result = vector_index.index_codebase(repo_path, output_dir)
            
            if result.success:
                return {
                    "status": "success",
                    "message": "Indexing completed successfully",
                    "stats": {
                        "num_files": result.num_files,
                        "num_chunks": result.num_chunks,
                        "index_path": result.index_path
                    },
                    "metadata": result.metadata
                }
            else:
                return {
                    "status": "error",
                    "message": "Indexing failed",
                    "error": result.error
                }, 500
            
        except Exception as e:
            logger.error(f"Indexing error: {e}\n{traceback.format_exc()}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc() if config.DEBUG else None
            }, 500
    
    def get(self):
        """Get index status"""
        return {
            "index_loaded": vector_index.index is not None,
            "num_chunks": len(vector_index.metadata) if vector_index.metadata else 0,
            "index_type": config.FAISS_INDEX_TYPE if vector_index.index else None,
            "embedding_model": config.EMBEDDING_MODEL
        }

class SimilarCodeAPI(Resource):
    """Find similar code snippets"""
    
    def post(self):
        try:
            data = request.get_json()
            
            if not data or 'code' not in data:
                return {
                    "error": "Missing 'code' in request body"
                }, 400
            
            code = data['code']
            top_k = data.get('top_k', 5)
            
            logger.info(f"Similar code search request: code_length={len(code)}")
            
            # Use the code as query
            results = searcher.search(
                query=code,
                top_k=top_k,
                use_hybrid=True,
                include_explanation=True
            )
            
            response = {
                "query_code": code[:500] + "..." if len(code) > 500 else code,
                "total_results": len(results),
                "results": [result.formatted_result for result in results]
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Similar code error: {e}\n{traceback.format_exc()}")
            return {
                "error": str(e)
            }, 500

class BatchSearchAPI(Resource):
    """Batch search endpoint for multiple queries"""
    
    def post(self):
        try:
            data = request.get_json()
            
            if not data or 'queries' not in data:
                return {
                    "error": "Missing 'queries' in request body"
                }, 400
            
            queries = data['queries']
            top_k = data.get('top_k', 5)
            
            if not isinstance(queries, list):
                return {
                    "error": "'queries' must be a list"
                }, 400
            
            if len(queries) > 100:
                return {
                    "error": "Maximum 100 queries allowed per batch"
                }, 400
            
            logger.info(f"Batch search request: {len(queries)} queries")
            
            results = []
            for query in queries:
                try:
                    search_results = searcher.search(query, top_k=top_k)
                    results.append({
                        "query": query,
                        "results": [r.formatted_result for r in search_results],
                        "total_results": len(search_results)
                    })
                except Exception as e:
                    results.append({
                        "query": query,
                        "error": str(e),
                        "results": []
                    })
            
            return {
                "total_queries": len(queries),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Batch search error: {e}\n{traceback.format_exc()}")
            return {
                "error": str(e)
            }, 500

class StatisticsAPI(Resource):
    """Get search statistics"""
    
    def get(self):
        stats = {
            "cache_size": len(searcher.query_cache),
            "embedding_cache_size": len(embedding_manager.cache),
            "index_info": {
                "loaded": vector_index.index is not None,
                "num_chunks": len(vector_index.metadata) if vector_index.metadata else 0,
                "chromadb_enabled": config.chromadb_enabled,
                "elasticsearch_enabled": config.elasticsearch_enabled
            },
            "config": {
                "embedding_model": config.EMBEDDING_MODEL,
                "embedding_dimension": config.EMBEDDING_DIMENSION,
                "faiss_index_type": config.FAISS_INDEX_TYPE,
                "min_similarity_score": config.MIN_SIMILARITY_SCORE
            }
        }
        
        return stats

# Register API endpoints
api.add_resource(HealthCheck, '/api/health')
api.add_resource(SearchAPI, '/api/search')
api.add_resource(IndexAPI, '/api/index')
api.add_resource(SimilarCodeAPI, '/api/similar')
api.add_resource(BatchSearchAPI, '/api/batch-search')
api.add_resource(StatisticsAPI, '/api/stats')

# Web interface
@app.route('/')
def index():
    """Serve web interface"""
    if Path('frontend/index.html').exists():
        return render_template('index.html')
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Codebase Semantic Search Engine</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { display: flex; flex-direction: column; gap: 20px; }
            .search-box { display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            .results { margin-top: 20px; }
            .result { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 5px; }
            .score { color: #28a745; font-weight: bold; }
            .file { color: #6c757d; font-size: 14px; }
            .content { background: #f8f9fa; padding: 10px; margin-top: 10px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Codebase Semantic Search Engine</h1>
            <div class="search-box">
                <input type="text" id="query" placeholder="Search for code using natural language...">
                <button onclick="search()">Search</button>
            </div>
            <div id="results" class="results"></div>
        </div>
        <script>
            async function search() {
                const query = document.getElementById('query').value;
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<p>Searching...</p>';
                
                try {
                    const response = await fetch('/api/search', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: query, top_k: 10})
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                        return;
                    }
                    
                    if (data.total_results === 0) {
                        resultsDiv.innerHTML = '<p>No results found.</p>';
                        return;
                    }
                    
                    let html = `<h3>Found ${data.total_results} results:</h3>`;
                    
                    data.results.forEach(result => {
                        html += `
                            <div class="result">
                                <div class="score">Score: ${result.similarity_score.toFixed(3)}</div>
                                <div class="file">${result.file_path} (${result.language})</div>
                                <div class="content">${result.content_preview}</div>
                            </div>
                        `;
                    });
                    
                    resultsDiv.innerHTML = html;
                } catch (error) {
                    resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
            
            // Allow Enter key to trigger search
            document.getElementById('query').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    search();
                }
            });
        </script>
    </body>
    </html>
    """

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

def main():
    """Main entry point"""
    logger.info(f"Starting Codebase Semantic Search Engine on {config.HOST}:{config.PORT}")
    logger.info(f"Embedding model: {config.EMBEDDING_MODEL}")
    logger.info(f"FAISS index type: {config.FAISS_INDEX_TYPE}")
    
    # Try to load existing index
    index_path = Path("./data/indices/faiss_index.bin")
    metadata_path = Path("./data/indices/metadata.json")
    chunks_path = Path("./data/indices/chunks.pkl")
    
    if all([p.exists() for p in [index_path, metadata_path, chunks_path]]):
        logger.info("Loading existing index...")
        success = vector_index.load_index(
            str(index_path),
            str(metadata_path),
            str(chunks_path)
        )
        if success:
            logger.info(f"Loaded index with {len(vector_index.metadata)} chunks")
        else:
            logger.warning("Failed to load existing index")
    
    # Start Flask app
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )

if __name__ == "__main__":
    main()
