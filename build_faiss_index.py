# Creates Database Index for FAISS
import json, numpy as np, faiss
from config import OUT_JSONL, FAISS_INDEX_PATH, ID_MAP_PATH

# 1. Load records
ids, filenames, embs = [], [], []
with open(OUT_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        ids.append(rec['id'])
        filenames.append(rec['filename'])
        embs.append(rec['embedding'])
emb_matrix = np.array(embs, dtype='float32')

# 2. Build index
D = emb_matrix.shape[1]
index = faiss.IndexIDMap(faiss.IndexFlatL2(D))
int_ids = np.arange(len(ids), dtype='int64')
index.add_with_ids(emb_matrix, int_ids)

# 3. Persist
faiss.write_index(index, FAISS_INDEX_PATH)

# 4. Write ID map
mapping = {str(i): {'uuid': ids[i], 'filename': filenames[i]} for i in range(len(ids))}
import json
with open(ID_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print(f"Index built and saved to {FAISS_INDEX_PATH}")
print(f"ID mapping saved to {ID_MAP_PATH}")