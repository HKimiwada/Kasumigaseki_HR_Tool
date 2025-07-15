import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'Data')
DB_DIR   = os.path.join(BASE_DIR, 'Database')
OUT_JSONL = os.path.join(DB_DIR, 'processed_job_description.jsonl')
FAISS_INDEX_PATH = os.path.join(DB_DIR, 'resumes_faiss.index')
ID_MAP_PATH      = os.path.join(DB_DIR, 'resumes_id_map.json')