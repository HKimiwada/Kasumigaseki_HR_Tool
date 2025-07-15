# Pipeline to Extract Data from 職務経歴書
# 一旦は職務経歴書だけをデータ抽出対象とする。（データが豊富だから）
import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from data_extraction_utils import (
    should_process,
    extract_text_from_pdf,
    clean_text,
    get_embedding,
)

# Configuration
from config import DATA_DIR, OUT_JSONL  
BATCH_SIZE = 16  # only for grouping file reads, embeddings are per-document
os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Helper to compute stable ID from file contents
def compute_file_id(path: str) -> str:
    """Compute a SHA1 hash of file bytes for idempotency."""
    sha = hashlib.sha1()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha.update(chunk)
    except Exception as e:
        logger.error(f"Cannot open file for hashing {path}: {e}")
        raise
    return sha.hexdigest()

# Load existing IDs to skip
seen_ids = set()
if os.path.exists(OUT_JSONL):
    with open(OUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen_ids.add(rec.get('id'))
            except json.JSONDecodeError:
                continue

# Find and process files
for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith('.pdf'):
        continue
    if not should_process(fname):
        continue

    path = os.path.join(DATA_DIR, fname)
    try:
        file_id = compute_file_id(path)
    except Exception:
        # already logged
        continue
    if file_id in seen_ids:
        logger.debug(f"Skipping already processed: {fname}")
        continue

    # Extract text
    try:
        raw = extract_text_from_pdf(path)
    except Exception as e:
        logger.error(f"PDF extraction failed for {fname}: {e}")
        continue

    # Clean and validate text
    text = clean_text(raw)
    if not text:
        logger.warning(f"No textual content extracted from {fname}, skipping.")
        continue

    # Embed per document (handles chunking internally)
    try:
        emb = get_embedding(text)
    except Exception as e:
        logger.error(f"Embedding failed for {fname}: {e}")
        continue

    # Write record to JSONL
    record = {
        'id': file_id,
        'filename': fname,
        'text': text,
        'embedding': emb,
        'ingested_at': datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(OUT_JSONL, 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"Failed writing record for {fname}: {e}")
        continue

    logger.info(f"Processed and wrote: {fname}")

logger.info("Pipeline completed.")
