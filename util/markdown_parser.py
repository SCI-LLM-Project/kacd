import re
import os

def read_markdown_file(file_path):
    """
    Basic function to read a markdown file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# unused
def remove_end_sections(content):
    """
    Truncate everything from the start of the References section onward
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
    # content = remove_end_sections(content)

    return content.strip()

def chunk(text, tokenizer=None, model_name=None, max_chunk_tokens=600, overlap_tokens=100):
    """
    Chunk text on token boundaries using a plain HuggingFace tokenizer, via
    LangChain's Tokenizer/split_text_on_tokens (the same mechanism
    TokenTextSplitter/SentenceTransformersTokenTextSplitter use internally,
    just keyed to a tokenizer instead of a full model). No model weights are
    ever loaded - SentenceTransformersTokenTextSplitter loads the entire model
    purely to reach its .tokenizer attribute, which is both wasteful and, for
    a multi-billion-parameter model, a real GPU/CPU memory risk.

    Args:
        text: The text to chunk
        tokenizer: A pre-loaded tokenizer to use instead of the default. Takes
            priority over model_name.
        model_name: Load a *different* HuggingFace model's tokenizer instead
            of the shared default. Ignored if tokenizer is given.
        max_chunk_tokens: Maximum size of each chunk in tokens (default: 600)
        overlap_tokens: Number of tokens to overlap between chunks (default: 100)

    Returns:
        List of text chunks
    """
    from langchain.text_splitter import Tokenizer, split_text_on_tokens

    if tokenizer is None:
        if model_name is None:
            # same tokenizer used for context-window budgeting everywhere else
            # in the pipeline - imported lazily so plain `import
            # util.markdown_parser` doesn't force this load on callers who
            # only want e.g. remove_end_sections
            import util.helpers as helpers
            tokenizer = helpers.tokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    lc_tokenizer = Tokenizer(
        chunk_overlap=overlap_tokens,
        tokens_per_chunk=max_chunk_tokens,
        decode=tokenizer.decode,
        encode=tokenizer.encode,
    )
    return split_text_on_tokens(text=text, tokenizer=lc_tokenizer)

