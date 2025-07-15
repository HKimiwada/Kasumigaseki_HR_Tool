import numpy as np
import faiss, json
from data_extraction_utils import get_embedding, clean_text, extract_text_from_pdf

# load index & map
index = faiss.read_index('Database/resumes_faiss.index')
with open('Database/resumes_id_map.json','r',encoding='utf-8') as f:
    id_map = json.load(f)

def find_similar_text(text, k=5):
    emb = np.array([get_embedding(clean_text(text))], dtype='float32')
    D, I = index.search(emb, k)
    return [{
        "filename": id_map[str(int(i))]["filename"],
        "score": float(1.0/(1.0 + d))
    } for d, i in zip(D[0], I[0])]

# quick test
query = "不動産投資のプロジェクトマネジメント経験"
print(find_similar_text(query))
