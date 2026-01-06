"""
Setup script for semantic search engine package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="codebase-semantic-search",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Semantic search engine for codebases using vector embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/codebase-semantic-search-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
            "torch>=2.0.0",
        ],
        "full": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "cohere>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "semantic-search=app:main",
            "index-codebase=index_codebase:main",
            "search-benchmark=scripts.benchmark:main",
            "search-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md", "*.yaml", "*.yml"],
    },
    keywords=[
        "semantic-search",
        "code-search",
        "vector-search",
        "nlp",
        "machine-learning",
        "faiss",
        "embeddings",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/codebase-semantic-search-engine/issues",
        "Source": "https://github.com/yourusername/codebase-semantic-search-engine",
        "Documentation": "https://github.com/yourusername/codebase-semantic-search-engine/wiki",
    },
)
