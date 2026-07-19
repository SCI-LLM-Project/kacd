import re
import os

import config

def read_markdown_file(file_path):
    """
    Basic function to read a markdown file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def remove_end_sections(content):
    """
    Truncate everything from the start of the References section onward.

    Deliberately narrower than matching every possible end-of-paper heading
    (Funding, Acknowledgements, Author Contributions, Data Availability, ...):
    those short administrative headings routinely appear in front matter too
    (e.g. "Funding sources" / "Conflicts of interest" right under the author
    block, before the abstract even starts), which caused truncation to fire
    catastrophically early on some papers - one lost 98.9% of its content this
    way. "References" doesn't have that failure mode - verified against the
    real corpus that every paper where this matches early is a genuine, large
    reference list, not a false positive.
    """
    references_patterns = [
        r'#+\s*References?\s*.*?$',          # # References
        r'^\s*REFERENCES?\s*$',              # REFERENCES on its own line
        r'\*\*\s*References?\s*\*\*',        # **References**
        r'^References?\s*\n[=\-]+\s*$',      # References\n=======
    ]

    earliest_match_pos = len(content)
    for pattern in references_patterns:
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match and match.start() < earliest_match_pos:
            earliest_match_pos = match.start()

    if earliest_match_pos < len(content):
        content = content[:earliest_match_pos]

    return content

def process_markdown_paper(file_path):
    """Process a scientific paper in markdown format"""
    # Read the file
    content = read_markdown_file(file_path)

    # Remove references section onward
    content = remove_end_sections(content)

    return content.strip()

def semantic_chunk(text, model_name=config.TOKENIZER_MODEL, max_chunk_tokens=600, overlap_tokens=100):
    """
    Chunk text using LangChain's SentenceTransformersTokenTextSplitter, so chunk
    sizes are measured with the given model's own tokenizer - by default the
    same tokenizer (config.TOKENIZER_MODEL) used for context-window budgeting
    elsewhere in the pipeline, instead of TokenTextSplitter's hardcoded tiktoken
    encoding.

    Args:
        text: The text to chunk
        model_name: HuggingFace model whose tokenizer sizes the chunks (default: config.TOKENIZER_MODEL)
        max_chunk_tokens: Maximum size of each chunk in tokens (default: 600)
        overlap_tokens: Number of tokens to overlap between chunks (default: 100)

    Returns:
        List of text chunks
    """
    from langchain_text_splitters import SentenceTransformersTokenTextSplitter

    splitter = SentenceTransformersTokenTextSplitter(
        model_name=model_name,
        tokens_per_chunk=max_chunk_tokens,
        chunk_overlap=overlap_tokens,
    )
    return splitter.split_text(text)

