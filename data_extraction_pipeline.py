# Pipeline to Extract Data from 職務経歴書
# 一旦は職務経歴書だけをデータ抽出対象とする。（データが豊富だから）
import os
import json
import hashlib
import logging
from datetime import datetime
from itertools import islice
from data_extraction_utils import (
    should_process,
    extract_text_from_pdf,
    clean_text,
    get_embedding,
    get_embeddings_batch,
)

# Configuration
data_dir = 'Test'
out_jsonl = 'Database/processed_job_description.jsonl'
batch_size = 16
os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

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
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()

# Load existing IDs to skip
seen_ids = set()
if os.path.exists(out_jsonl):
    with open(out_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen_ids.add(rec.get('id'))
            except json.JSONDecodeError:
                continue

# Generator for eligible files
def pdf_files_generator():
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        if not should_process(fname):
            continue
        path = os.path.join(data_dir, fname)
        yield fname, path

# Batch iterator
def batched(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

# Main pipeline
new_records = []
for batch in batched(pdf_files_generator(), batch_size):
    texts = []
    metas = []

    # Extract & clean
    for fname, path in batch:
        try:
            file_id = compute_file_id(path)
            if file_id in seen_ids:
                logger.debug(f"Skipping already processed: {fname}")
                continue

            raw = extract_text_from_pdf(path)
            text = clean_text(raw)
            texts.append(text)
            metas.append({'id': file_id, 'filename': fname, 'text': text})
        except Exception as e:
            logger.error(f"Failed extraction for {fname}: {e}")

    if not texts:
        continue

    # Embed batch
    try:
        embeddings = get_embeddings_batch(texts)
    except Exception as e:
        logger.error(f"Embedding batch failed: {e}")
        # Fallback to individual
        embeddings = [get_embedding(t) for t in texts]

    # Write to JSONL
    with open(out_jsonl, 'a', encoding='utf-8') as out_f:
        for meta, emb in zip(metas, embeddings):
            record = {
                'id': meta['id'],
                'filename': meta['filename'],
                'text': meta['text'],
                'embedding': emb,
                'ingested_at': datetime.now(datetime.UTC).isoformat(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.info(f"Processed and wrote: {meta['filename']}")

logger.info("Pipeline completed.")