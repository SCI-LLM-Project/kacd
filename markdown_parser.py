import re
import os

def read_markdown_file(file_path):
    """
    Basic function to read a markdown file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def clean_citations(content):
    """
    Remove common citation formats from scientific papers
    """
    # Remove numeric citations [1], [1,2], [1-3]
    content = re.sub(r'\[\s*\d+(?:[-–,\s]*\d+)*\s*\]', '', content)
    
    # Remove author-year citations (Smith, 2020), (Smith et al., 2020)
    content = re.sub(r'\(\s*[A-Za-z\-]+(?:\s+et\s+al\.)?(?:,\s*\d{4}[a-z]?)+\s*\)', '', content)
    
    # Remove DOI references
    content = re.sub(r'(?:doi|DOI):\s*10\.\d{4,}\/[A-Za-z0-9\.\-_]+', '', content)
    
    return content

def remove_end_sections(content):
    """Remove references and other end sections like acknowledgements, supplementary info, etc."""
    
    end_section_patterns = [
        # References patterns
        r'#+\s*References?\s*.*?$',         # # References
        r'^\s*REFERENCES?\s*$',              # REFERENCES on its own line
        r'\*\*\s*References?\s*\*\*',        # **References**
        r'^References?\s*\n[=\-]+\s*$',      # References\n=======

        # Acknowledgements patterns
        r'#+\s*Acknowledgements?\s*.*?$',    # # Acknowledgements
        r'^\s*ACKNOWLEDGEMENTS?\s*$',        # ACKNOWLEDGEMENTS on its own line
        r'\*\*\s*Acknowledgements?\s*\*\*',  # **Acknowledgements**
        
        # Supplementary info patterns
        r'#+\s*Supplementary\s+(?:Information|Material)s?\s*.*?$',  # # Supplementary Information
        r'^\s*SUPPLEMENTARY\s+(?:INFORMATION|MATERIAL)S?\s*$',      # SUPPLEMENTARY INFORMATION
        r'\*\*\s*Supplementary\s+(?:Information|Material)s?\s*\*\*', # **Supplementary Information**
        
        # Author contributions
        r'#+\s*Author\s+Contributions?\s*.*?$',
        r'\*\*\s*Author\s+Contributions?\s*\*\*',
        
        # Funding/Financial disclosure
        r'#+\s*Funding\s*.*?$',
        r'\*\*\s*Funding\s*\*\*',
        r'#+\s*Financial\s+Disclosure\s*.*?$',
        
        # Declarations/Competing interests
        r'#+\s*(?:Competing\s+Interests|Conflict\s+of\s+Interest|Declaration)s?\s*.*?$',
        r'\*\*\s*(?:Competing\s+Interests|Conflict\s+of\s+Interest|Declaration)s?\s*\*\*',
        
        # Data availability
        r'#+\s*Data\s+Availability\s*.*?$',
        r'\*\*\s*Data\s+Availability\s*\*\*',
        
        # Appendix
        r'#+\s*Appendix\s*.*?$',
        r'\*\*\s*Appendix\s*\*\*'
    ]
    
    # Find the earliest match
    earliest_match_pos = len(content)
    
    for pattern in end_section_patterns:
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if match and match.start() < earliest_match_pos:
            earliest_match_pos = match.start()
    
    # If we found any end section, truncate the content
    if earliest_match_pos < len(content):
        content = content[:earliest_match_pos]
            
    return content

def process_markdown_paper(file_path):
    """Process a scientific paper in markdown format"""
    # Read the file
    content = read_markdown_file(file_path)
    
    # Remove reference section 
    content = remove_end_sections(content)
    
    # Clean citations
    content = clean_citations(content)
    
    # Clean up formatting for plain text
    content = re.sub(r'\n+', ' ', content)  # Replace multiple newlines with space
    content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
    
    return content.strip()

def semantic_chunk(text, max_chunk_tokens=600, overlap_tokens=100):
    """
    Chunk text using LangChain's TokenTextSplitter.

    Args:
        text: The text to chunk
        max_chunk_tokens: Maximum size of each chunk in tokens (default: 8000)
        overlap_tokens: Number of tokens to overlap between chunks (default: 200)

    Returns:
        List of text chunks
    """
    from langchain.text_splitter import TokenTextSplitter

    splitter = TokenTextSplitter(
        chunk_size=max_chunk_tokens,
        chunk_overlap=overlap_tokens
    )
    return splitter.split_text(text)

