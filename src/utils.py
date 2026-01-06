"""
Utility functions for the semantic search engine
"""
import os
import sys
import json
import hashlib
import pickle
import gzip
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from datetime import datetime, timedelta
import time
from loguru import logger
import numpy as np
import psutil

from config import config

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        
        logger.debug(f"Function {func.__name__} took {elapsed:.3f} seconds")
        return result
    return wrapper

def cache_to_disk(func: Callable) -> Callable:
    """Decorator to cache function results to disk"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from function name and arguments
        cache_key = hashlib.md5(
            f"{func.__name__}{args}{tuple(sorted(kwargs.items()))}".encode()
        ).hexdigest()
        
        cache_file = Path(config.CACHE_DIR) / f"{cache_key}.pkl.gz"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if cache exists and is recent (less than 7 days old)
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < 60 * 60 * 24 * 7:  # 7 days
                try:
                    with gzip.open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
        
        # Execute function and cache result
        result = func(*args, **kwargs)
        
        try:
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Cached result for {func.__name__}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
        
        return result
    return wrapper

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_hash(file_path: Union[str, Path]) -> str:
    """Get MD5 hash of file content"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    
    return hasher.hexdigest()

def format_bytes(size: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_used_mb': memory_info.rss / (1024 * 1024),
        'memory_percent': process.memory_percent(),
        'threads': process.num_threads(),
        'open_files': len(process.open_files()),
        'system_memory': {
            'total': format_bytes(psutil.virtual_memory().total),
            'available': format_bytes(psutil.virtual_memory().available),
            'percent': psutil.virtual_memory().percent
        }
    }

class ProgressBar:
    """Custom progress bar for indexing operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.5  # seconds
        
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        current_time = time.time()
        
        # Only update if enough time has passed
        if current_time - self.last_update >= self.update_interval:
            self._display()
            self.last_update = current_time
    
    def _display(self):
        """Display progress bar"""
        elapsed = time.time() - self.start_time
        percent = (self.current / self.total) * 100
        
        # Calculate ETA
        if self.current > 0:
            items_per_second = self.current / elapsed
            eta_seconds = (self.total - self.current) / items_per_second
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "??:??:??"
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * self.current / self.total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Format output
        sys.stdout.write(
            f"\r{self.description}: [{bar}] {percent:.1f}% "
            f"({self.current}/{self.total}) | "
            f"ETA: {eta_str} | "
            f"Elapsed: {str(timedelta(seconds=int(elapsed)))}"
        )
        sys.stdout.flush()
    
    def close(self):
        """Complete the progress bar"""
        self.current = self.total
        self._display()
        sys.stdout.write('\n')
        sys.stdout.flush()

def batch_generator(items: List[Any], batch_size: int):
    """Generate batches from list"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def normalize_text(text: str) -> str:
    """Normalize text for consistent processing"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    
    return text.strip()

def compute_similarity_matrix(embeddings1: np.ndarray, 
                             embeddings2: np.ndarray = None) -> np.ndarray:
    """Compute cosine similarity matrix between embeddings"""
    if embeddings2 is None:
        embeddings2 = embeddings1
    
    # Normalize embeddings
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    embeddings1_normalized = embeddings1 / norm1
    embeddings2_normalized = embeddings2 / norm2
    
    # Compute cosine similarity
    similarity = np.dot(embeddings1_normalized, embeddings2_normalized.T)
    
    return similarity

def find_duplicates(embeddings: np.ndarray, 
                    threshold: float = 0.95) -> List[List[int]]:
    """Find duplicate embeddings based on similarity threshold"""
    similarity_matrix = compute_similarity_matrix(embeddings)
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
    
    duplicates = []
    visited = set()
    
    for i in range(len(embeddings)):
        if i in visited:
            continue
        
        # Find indices of similar embeddings
        similar_indices = np.where(similarity_matrix[i] > threshold)[0]
        
        if len(similar_indices) > 0:
            group = [i] + similar_indices.tolist()
            duplicates.append(group)
            visited.update(group)
    
    return duplicates

def save_json(data: Any, filepath: Union[str, Path], indent: int = 2):
    """Save data as JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compress_file(input_path: Union[str, Path], 
                  output_path: Union[str, Path] = None):
    """Compress file using gzip"""
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix(input_path.suffix + '.gz')
    
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return output_path

def get_git_info(repo_path: Union[str, Path]) -> Dict[str, str]:
    """Get git repository information"""
    try:
        import git
        repo = git.Repo(repo_path)
        
        return {
            'branch': repo.active_branch.name,
            'commit': repo.head.commit.hexsha,
            'commit_message': repo.head.commit.message.strip(),
            'commit_date': repo.head.commit.committed_datetime.isoformat(),
            'is_dirty': repo.is_dirty(),
            'remote_url': next(iter(repo.remotes)).url if repo.remotes else None
        }
    except ImportError:
        logger.warning("GitPython not installed, skipping git info")
        return {}
    except Exception as e:
        logger.warning(f"Failed to get git info: {e}")
        return {}

def setup_logging(log_file: str = "logs/app.log", 
                  level: str = "INFO",
                  rotation: str = "10 MB",
                  retention: str = "7 days"):
    """Configure logging with loguru"""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Add file handler
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_file_path),
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
               "{name}:{function}:{line} - {message}",
        level=level,
        compression="gz"
    )
    
    return logger
