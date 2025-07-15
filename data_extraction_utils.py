# data_extraction_utils.py
"""
Utility functions for text extraction and embedding of Japanese resumes.
"""
from dotenv import load_dotenv
from openai import OpenAI
import fitz      # PyMuPDF
import re
import os
import hashlib
import shelve
import errno
from tenacity import retry, wait_exponential, stop_after_attempt
import tiktoken

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Persistent cache directory
cache_dir = os.path.join(os.getcwd(), 'Database')
os.makedirs(cache_dir, exist_ok=True)
cache_path = os.path.join(cache_dir, 'emb_cache.db')
cache = shelve.open(cache_path)

# Static in-memory tokenizer to avoid SQLite/thread issues
tok = tiktoken.get_encoding('cl100k_base')

# Embedding model name
EMBED_MODEL = 'text-embedding-ada-002'


def compute_id(text: str) -> str:
    """
    Compute a stable SHA-1 hash for caching.
    """
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def chunk_text(text: str, max_tokens: int = 2048) -> list[str]:
    """
    Split a long text into chunks of at most max_tokens tokens.
    Uses a static in-memory tokenizer to avoid SQLite.
    """
    tokens = tok.encode(text)
    return [ tok.decode(tokens[i:i+max_tokens])
             for i in range(0, len(tokens), max_tokens) ]

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def _embed_batch(chunks: list[str], model: str) -> list[list[float]]:
    """
    Internal batched embedder with retry/backoff.
    Filters out any empty chunks before calling the API.
    """
    # Clean and filter
    cleaned = [c.replace("\n"," ").strip() for c in chunks]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        raise ValueError("No valid text for embedding after filtering.")
    # Call embeddings API
    resp = client.embeddings.create(model=model, input=cleaned)
    return [item.embedding for item in resp.data]


def get_embedding(text: str, model: str = EMBED_MODEL) -> list[float]:
    """
    Compute or retrieve from cache a single embedding for the given text.
    - Uses SHA-1 id for cache key
    - Chunks text if too long
    - Batch-embeds and average-pools
    """
    # Preprocess
    text = text.replace("\n"," ").strip()
    key = compute_id(text)
    if key in cache:
        return cache[key]

    # Chunk for long input
    chunks = chunk_text(text)
    embs = _embed_batch(chunks, model)
    # Average pooling
    avg = [ sum(dims)/len(embs) for dims in zip(*embs) ]

    # Cache and return
    cache[key] = avg
    cache.sync()
    return avg


def get_embeddings_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """
    Embed a list of short texts in one batch call.
    """
    # Filter empty
    inputs = [t.replace("\n"," ").strip() for t in texts]
    inputs = [t for t in inputs if t]
    if not inputs:
        return []
    return _embed_batch(inputs, model)


def should_process(filename: str) -> bool:
    """
    Only process files with 職務経歴書 in the name.
    """
    return '職務経歴書' in filename


def extract_text_from_pdf(path: str) -> str:
    """
    Extract all text from a PDF file on disk.
    """
    doc = fitz.open(path)
    return ''.join(page.get_text() for page in doc)


def clean_text(text: str) -> str:
    """
    Normalize whitespace in extracted text.
    """
    return re.sub(r'\s+', ' ', text).strip()


if __name__ == '__main__':
    # Quick test
    sample = 'Data/_小沼様_職務経歴書.pdf'
    raw = extract_text_from_pdf(sample)
    text = clean_text(raw)
    print('Extracted:', text[:200])
    emb = get_embedding('Hello, world!')
    print('Embedding len:', len(emb))
