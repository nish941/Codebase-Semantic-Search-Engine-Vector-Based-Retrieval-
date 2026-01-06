"""
Middleware for Flask application
"""
import time
import json
from typing import Dict, Any, Optional
from flask import request, g, jsonify
from werkzeug.exceptions import HTTPException
from loguru import logger
import traceback

from config import config

class RequestLogger:
    """Middleware to log all requests"""
    
    def __init__(self, app=None):
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        @app.before_request
        def before_request():
            g.start_time = time.time()
            
            # Log request
            if config.DEBUG:
                logger.debug(
                    f"Request: {request.method} {request.path} "
                    f"from {request.remote_addr}"
                )
                if request.method in ['POST', 'PUT']:
                    logger.debug(f"Request data: {request.get_json(silent=True)}")
        
        @app.after_request
        def after_request(response):
            # Calculate response time
            response_time = time.time() - g.start_time
            
            # Add response time header
            response.headers['X-Response-Time'] = f'{response_time:.3f}s'
            
            # Log response
            logger.info(
                f"Response: {request.method} {request.path} "
                f"=> {response.status_code} ({response_time:.3f}s)"
            )
            
            return response

class ErrorHandler:
    """Global error handler middleware"""
    
    def __init__(self, app=None):
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        @app.errorhandler(HTTPException)
        def handle_http_exception(error):
            """Handle HTTP exceptions"""
            logger.warning(
                f"HTTP Error {error.code}: {error.name} - {error.description}"
            )
            
            return jsonify({
                'error': error.name,
                'message': error.description,
                'status_code': error.code
            }), error.code
        
        @app.errorhandler(Exception)
        def handle_general_exception(error):
            """Handle all other exceptions"""
            error_id = f"ERR_{int(time.time())}_{hash(error)}"
            
            logger.error(
                f"Unhandled Exception [{error_id}]: {str(error)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            
            response = {
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'error_id': error_id,
                'status_code': 500
            }
            
            if config.DEBUG:
                response['debug_info'] = {
                    'error_type': error.__class__.__name__,
                    'error_message': str(error),
                    'traceback': traceback.format_exc().split('\n')
                }
            
            return jsonify(response), 500

class RateLimiter:
    """Simple rate limiting middleware"""
    
    def __init__(self, app=None, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        @app.before_request
        def check_rate_limit():
            if not config.DEBUG:
                client_ip = request.remote_addr
                current_time = time.time()
                
                # Initialize or clean up old entries
                if client_ip not in self.request_counts:
                    self.request_counts[client_ip] = {
                        'count': 0,
                        'window_start': current_time
                    }
                
                client_data = self.request_counts[client_ip]
                
                # Reset counter if window has passed
                if current_time - client_data['window_start'] > 60:
                    client_data['count'] = 0
                    client_data['window_start'] = current_time
                
                # Check if limit exceeded
                if client_data['count'] >= self.requests_per_minute:
                    logger.warning(f"Rate limit exceeded for {client_ip}")
                    
                    return jsonify({
                        'error': 'Rate Limit Exceeded',
                        'message': f'Maximum {self.requests_per_minute} requests per minute',
                        'retry_after': 60 - int(current_time - client_data['window_start'])
                    }), 429
                
                # Increment counter
                client_data['count'] += 1
    
    def cleanup_old_entries(self):
        """Clean up old rate limiting entries"""
        current_time = time.time()
        old_ips = [
            ip for ip, data in self.request_counts.items()
            if current_time - data['window_start'] > 120  # 2 minutes
        ]
        
        for ip in old_ips:
            del self.request_counts[ip]

class CORSHandler:
    """CORS configuration middleware"""
    
    def __init__(self, app=None):
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        @app.after_request
        def add_cors_headers(response):
            """Add CORS headers to all responses"""
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Max-Age'] = '86400'  # 24 hours
            
            return response
        
        @app.route('/api/<path:path>', methods=['OPTIONS'])
        def handle_options(path):
            """Handle preflight requests"""
            return '', 200

class AuthenticationMiddleware:
    """Simple API key authentication middleware"""
    
    def __init__(self, app=None, api_keys: Optional[list] = None):
        self.api_keys = api_keys or []
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        @app.before_request
        def check_authentication():
            # Skip authentication for health check and OPTIONS
            if request.path == '/api/health' or request.method == 'OPTIONS':
                return
            
            # Check for API key in headers
            api_key = request.headers.get('X-API-Key')
            
            if api_key and api_key in self.api_keys:
                # Valid API key
                g.api_key = api_key
                return
            
            # Check for API key in query parameters (for debugging only)
            if config.DEBUG:
                api_key = request.args.get('api_key')
                if api_key and api_key in self.api_keys:
                    g.api_key = api_key
                    return
            
            # No valid API key found
            if self.api_keys:  # Only require auth if keys are configured
                logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
                
                return jsonify({
                    'error': 'Unauthorized',
                    'message': 'Valid API key required'
                }), 401

class RequestValidator:
    """Request validation middleware"""
    
    def init_app(self, app):
        @app.before_request
        def validate_request():
            # Only validate JSON requests
            if request.method in ['POST', 'PUT'] and request.is_json:
                try:
                    data = request.get_json()
                    
                    # Validate required fields for search endpoint
                    if request.path == '/api/search':
                        self._validate_search_request(data)
                    
                    # Validate required fields for indexing endpoint
                    elif request.path == '/api/index':
                        self._validate_index_request(data)
                    
                except ValueError as e:
                    logger.warning(f"Invalid request: {str(e)}")
                    
                    return jsonify({
                        'error': 'Bad Request',
                        'message': str(e)
                    }), 400
    
    def _validate_search_request(self, data: Dict[str, Any]):
        """Validate search request"""
        if 'query' not in data:
            raise ValueError("Missing 'query' field")
        
        if not isinstance(data['query'], str):
            raise ValueError("'query' must be a string")
        
        if len(data['query']) > 1000:
            raise ValueError("'query' too long (max 1000 characters)")
        
        if 'top_k' in data and not isinstance(data['top_k'], int):
            raise ValueError("'top_k' must be an integer")
        
        if 'use_hybrid' in data and not isinstance(data['use_hybrid'], bool):
            raise ValueError("'use_hybrid' must be a boolean")
    
    def _validate_index_request(self, data: Dict[str, Any]):
        """Validate index request"""
        if 'repo_path' not in data:
            raise ValueError("Missing 'repo_path' field")
        
        if not isinstance(data['repo_path'], str):
            raise ValueError("'repo_path' must be a string")
        
        import os
        if not os.path.exists(data['repo_path']):
            raise ValueError(f"Repository path does not exist: {data['repo_path']}")

# Initialize middleware instances
request_logger = RequestLogger()
error_handler = ErrorHandler()
rate_limiter = RateLimiter()
cors_handler = CORSHandler()
auth_middleware = AuthenticationMiddleware()
request_validator = RequestValidator()

def init_middleware(app):
    """Initialize all middleware"""
    request_logger.init_app(app)
    error_handler.init_app(app)
    rate_limiter.init_app(app)
    cors_handler.init_app(app)
    auth_middleware.init_app(app)
    request_validator.init_app(app)
    
    logger.info("Middleware initialized")
