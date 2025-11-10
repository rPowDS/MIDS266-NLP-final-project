#!/usr/bin/env python3
"""Setup script for align-rerank package."""

from setuptools import setup, find_packages

setup(
    name="align-rerank",
    version="0.1.0",
    description="Verifier-reranking for factual abstractive summarization",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/align-rerank",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.41.0",
        "datasets>=2.20.0",
        "evaluate>=0.4.2",
        "accelerate>=0.31.0",
        "torch>=2.1",
        "numpy>=1.26",
        "pandas>=2.1",
        "scipy>=1.11",
        "tqdm>=4.66",
        "nltk>=3.8.1",
        "typer>=0.12.3",
        "rich>=13.7.0",
        "bert-score>=0.3.13",
        "rouge-score>=0.1.2",
        "sentencepiece>=0.2.0",
        "sacrebleu>=2.4.0",
        "huggingface-hub>=0.24.0",
        "orjson>=3.10.0"
    ],
    entry_points={
        "console_scripts": [
            "align-rerank=align_rerank.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
