# build_faiss_index_fixed.py
import json
import os
import numpy as np
import faiss
from pathlib import Path
from config import OUT_JSONL, V2_FAISS_INDEX_PATH, V2_ID_MAP_PATH

# --- 1) Prepare paths and directories ---
index_path = Path(V2_FAISS_INDEX_PATH)
idmap_path = Path(V2_ID_MAP_PATH)
db_dir = index_path.parent

# Ensure Database/ exists
db_dir.mkdir(parents=True, exist_ok=True)

# If anything slipped in at the exact file paths, remove it
for p in (index_path, idmap_path):
    if p.exists():
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
        else:
            p.unlink()

# --- 2) Load your embeddings & metadata from JSONL ---
ids, filenames, embs = [], [], []
with open(OUT_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        ids.append(rec['id'])
        filenames.append(rec['filename'])
        emb = rec.get('embedding')
        if emb is None:
            raise ValueError(f"Record {rec['id']} has no embedding")
        embs.append(emb)

emb_matrix = np.asarray(embs, dtype='float32')
print(f">>> Loaded {len(embs)} embeddings, dim = {emb_matrix.shape[1]}")

# --- 3) FIXED: Use Inner Product for cosine similarity ---
D = emb_matrix.shape[1]

# Normalize embeddings for cosine similarity
print(">>> Normalizing embeddings for cosine similarity...")
faiss.normalize_L2(emb_matrix)

# Use Inner Product index (equivalent to cosine similarity with normalized vectors)
base_index = faiss.IndexFlatIP(D)  # Inner Product instead of L2
index = faiss.IndexIDMap(base_index)

int_ids = np.arange(len(ids), dtype='int64')
index.add_with_ids(emb_matrix, int_ids)
print(f">>> FAISS index built with {index.ntotal} vectors using cosine similarity")

# --- 4) Write the .index file via chdir hack ---
print(f">>> Writing FAISS index to {index_path}")
orig_cwd = Path.cwd()
os.chdir(db_dir)
faiss.write_index(index, index_path.name)  
os.chdir(orig_cwd)
print("✔ FAISS index written successfully")

# --- 5) Write the ID→UUID & filename map ---
mapping = {
    str(i): {"uuid": ids[i], "filename": filenames[i]}
    for i in range(len(ids))
}

print(f">>> Writing ID map to {idmap_path}")
with open(idmap_path, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print("✔ ID map written successfully")