# streamlit run app.py
# Front-end UI for the HR tool (light-weight dashboard for testing)
import streamlit as st
import numpy as np
import faiss
import json
import os
from pathlib import Path
from io import BytesIO
from data_extraction_utils import clean_text, get_embedding
import fitz  # PyMuPDF for in-memory PDF
from config import DB_DIR, FAISS_INDEX_PATH, ID_MAP_PATH

@st.cache(allow_output_mutation=True)
def load_faiss_index(index_path: str):
    """
    Load and return a FAISS index, using a working-directory hack to support Unicode paths on Windows.
    """
    idx_path = Path(index_path)
    parent = idx_path.parent
    name = idx_path.name
    # Temporarily chdir into parent to avoid Windows Unicode fopen issues
    cwd = os.getcwd()
    os.chdir(str(parent))
    index = faiss.read_index(name)
    os.chdir(cwd)
    return index

@st.cache(allow_output_mutation=True)
def load_id_map(id_map_path: str):
    """
    Load and return the integer→metadata JSON map.
    """
    with open(id_map_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load resources once
index = load_faiss_index(str(FAISS_INDEX_PATH))
id_map = load_id_map(str(ID_MAP_PATH))

# Helper: extract text from PDF bytes, preserving Japanese

def extract_text_from_bytes(raw_bytes: bytes) -> str:
    """
    Extract text from PDF content in bytes using PyMuPDF.
    """
    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    return full_text

# Similarity search

def find_similar(text: str, top_k: int):
    emb = np.array([get_embedding(text)], dtype='float32')
    distances, indices = index.search(emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        meta = id_map.get(str(int(idx)), {})
        results.append({
            'filename': meta.get('filename', 'unknown'),
            'score': float(1.0 / (1.0 + dist)),
        })
    return results

# Streamlit UI
st.title("📄 Resume Similarity Search (日本語対応)")
st.markdown("日本語の職務経歴書PDFをアップロードして、類似履歴書を検索します。ファイル名の日本語表示にも対応しています。")

uploaded_file = st.file_uploader("職務経歴書PDFを選択", type=["pdf"])
top_k = st.slider("表示する類似履歴書の数", min_value=1, max_value=20, value=5)

if uploaded_file is not None:
    orig_name = uploaded_file.name
    st.write(f"### アップロードされたファイル: {orig_name}")

    raw_bytes = uploaded_file.read()
    try:
        raw_text = extract_text_from_bytes(raw_bytes)
        text = clean_text(raw_text)
        if not text:
            st.error("⚠️ PDFからテキストが抽出できませんでした。スキャン画像の可能性があります。")
        else:
            st.subheader("抽出テキストプレビュー")
            st.text_area("", text[:1000], height=200)

            with st.spinner("埋め込みと検索中…"):
                results = find_similar(text, top_k)

            if results:
                st.success(f"トップ{top_k}の類似履歴書:")
                for i, r in enumerate(results, start=1):
                    st.write(f"**{i}. {r['filename']}** — スコア: {r['score']:.3f}")
            else:
                st.info("類似する履歴書が見つかりませんでした。")
    except Exception as e:
        st.error(f"ファイル処理中にエラーが発生しました: {e}")
else:
    st.info("まずは職務経歴書PDFをアップロードしてください。")
