# Pipeline to Extract Data from 職務経歴書
# 一旦は職務経歴書だけをデータ抽出対象とする。（データが豊富だから）
import os
import json
import uuid
from data_extraction_utils import should_process, extract_text_from_pdf, clean_text, get_embedding  

DATA_DIR = 'Test'
OUT_JSONL = 'Database/processed_job_description.jsonl'

with open(OUT_JSONL, 'w', encoding='utf-8') as out_f:
    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith('.pdf'):
            continue
        if not should_process(fname):
            continue

        path = os.path.join(DATA_DIR, fname)
        raw = extract_text_from_pdf(path)
        text = clean_text(raw)

        emb = get_embedding(text)
        record = {
            "id": str(uuid.uuid4()),
            "filename": fname,
            "text": text,
            "embedding": emb,
        }

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
