# Utility functions for data extraction from Resumes
from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF
import re
import os
import hashlib
import shelve
import errno
from tenacity import retry, wait_exponential, stop_after_attempt
import tiktoken

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Simple persistent cache for embeddings
cache_dir = os.path.join(os.getcwd(), 'Database')
try:
    os.makedirs(cache_dir, exist_ok=True)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

e_cache_path = os.path.join(cache_dir, 'emb_cache.db')
cache = shelve.open(e_cache_path)

def compute_id(text: str) -> str:
    """
    Compute a stable SHA-1 hash of the text for caching.
    """
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def chunk_text(text: str, max_tokens: int = 2048) -> list[str]:
    """
    Split a long text into chunks of at most `max_tokens` tokens
    according to the model's tokenizer.
    """
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    tokens = enc.encode(text)
    return [
        enc.decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def _embed(texts: list[str], model: str = 'text-embedding-3-small') -> list[list[float]]:
    """
    Internal embedder with retry, batching via OpenAI API.
    """
    cleaned = [t.replace("\n", " ") for t in texts]
    resp = client.embeddings.create(input=cleaned, model=model)
    return [item.embedding for item in resp.data]

def get_embedding(text: str, model: str = 'text-embedding-3-small') -> list[float]:
    """
    Get or compute the embedding for a single text, with chunking and caching.
    """
    text = text.replace("\n", " ")
    tid = compute_id(text)
    if tid in cache:
        return cache[tid]

    # Handle long documents by chunking
    chunks = chunk_text(text)
    embs = _embed(chunks, model)
    # Average pooling over chunk embeddings
    avg_emb = [sum(dim) / len(embs) for dim in zip(*embs)]

    # Cache and return
    cache[tid] = avg_emb
    cache.sync()
    return avg_emb


def get_embeddings_batch(texts: list[str], model: str = 'text-embedding-3-small') -> list[list[float]]:
    """
    Embed multiple texts in one batch call. Assumes each text <= token limit.
    """
    return _embed(texts, model)


def should_process(filename: str) -> bool:
    """
    Only process PDFs whose name includes 職務経歴書.
    """
    return '職務経歴書' in filename


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Concatenated text from all pages.
    """
    doc = fitz.open(pdf_path)
    text = ''.join(page.get_text() for page in doc)
    return text


def clean_text(text: str) -> str:
    """
    Cleans the extracted text by removing excessive whitespace.
    Args:
        text (str): Raw text extracted from PDF.
    Returns:
        str: Whitespace-normalized text.
    """
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

if __name__ == '__main__':
    # Example usage
    sample = 'Data/水越牧朗_職務経歴書（1）.pdf'
    sample = 'Data/水越牧郎_職務経歴書%20(1).pdf'
    raw = extract_text_from_pdf(sample)
    text = clean_text(raw)
    emb = get_embedding(text)
    print('Cleaned Text:', text[:100], '...')
    print('Embedding Length:', len(emb))
