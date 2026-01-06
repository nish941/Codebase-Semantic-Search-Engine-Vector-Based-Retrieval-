"""
Script to index a codebase for semantic search
"""
import argparse
import sys
from pathlib import Path
from loguru import logger

from src.indexer import vector_index

def setup_logging():
    """Configure logging"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/indexing.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )

def main():
    """Main entry point for indexing"""
    parser = argparse.ArgumentParser(description="Index a codebase for semantic search")
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the codebase repository"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/indices",
        help="Directory to save index files (default: ./data/indices)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if index exists"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
    
    repo_path = Path(args.repo_path)
    output_dir = Path(args.output_dir)
    
    # Validate paths
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    if not repo_path.is_dir():
        logger.error(f"Repository path is not a directory: {repo_path}")
        sys.exit(1)
    
    # Check if index already exists
    index_exists = (output_dir / "faiss_index.bin").exists()
    
    if index_exists and not args.force:
        logger.info(f"Index already exists at {output_dir}")
        response = input("Do you want to re-index? (y/N): ")
        if response.lower() != 'y':
            logger.info("Aborting...")
            sys.exit(0)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting indexing of {repo_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Index the codebase
    result = vector_index.index_codebase(str(repo_path), str(output_dir))
    
    if result.success:
        logger.info("✅ Indexing completed successfully!")
        logger.info(f"   Files indexed: {result.num_files}")
        logger.info(f"   Chunks extracted: {result.num_chunks}")
        logger.info(f"   Index saved to: {result.index_path}")
        
        if result.metadata:
            logger.info("   Metadata:")
            for key, value in result.metadata.items():
                logger.info(f"     {key}: {value}")
    else:
        logger.error(f"❌ Indexing failed: {result.error}")
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    main()
