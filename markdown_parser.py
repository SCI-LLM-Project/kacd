import re
import os

def read_markdown_file(file_path):
    """
    Basic function to read a markdown file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def process_markdown_paper(file_path):
    """Process a scientific paper in markdown format"""
    # Read the file
    content = read_markdown_file(file_path)
    
    return content.strip()

def semantic_chunk(text, max_chunk_tokens=600, overlap_tokens=100):
    """
    Chunk text using LangChain's TokenTextSplitter.

    Args:
        text: The text to chunk
        max_chunk_tokens: Maximum size of each chunk in tokens (default: 600)
        overlap_tokens: Number of tokens to overlap between chunks (default: 100)

    Returns:
        List of text chunks
    """
    from langchain.text_splitter import TokenTextSplitter

    splitter = TokenTextSplitter(
        chunk_size=max_chunk_tokens,
        chunk_overlap=overlap_tokens
    )
    return splitter.split_text(text)

