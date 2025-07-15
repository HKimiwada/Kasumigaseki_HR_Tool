# streamlit run app.py
# Front-end UI for the HR tool (light-weight dashboard for testing)
import os
import json
from pathlib import Path

import streamlit as st
import numpy as np
import faiss
import fitz  # PyMuPDF for in-memory PDF

# Import utility functions
from data_extraction_utils import clean_text, get_embedding
from config import FAISS_INDEX_PATH, ID_MAP_PATH

# --- 1. Load FAISS index & ID map with caching and Windows Unicode hack ---
def load_faiss_index(index_path: str) -> faiss.Index:
    """
    Load a FAISS index from disk. On Windows, temporarily cd to its folder
    so that faiss.read_index sees only the ASCII filename.
    """
    idx_path = Path(index_path)
    parent_dir = idx_path.parent
    name = idx_path.name

    old_cwd = os.getcwd()
    os.chdir(str(parent_dir))
    index = faiss.read_index(name)
    os.chdir(old_cwd)
    return index

def load_id_map(id_map_path: str) -> dict:
    """
    Load the JSON mapping of internal integer IDs to metadata.
    """
    with open(id_map_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Initialize once at startup
index = load_faiss_index(FAISS_INDEX_PATH)
id_map = load_id_map(ID_MAP_PATH)

# --- 2. Helper to extract text from PDF bytes ---
def extract_text_from_bytes(raw_bytes: bytes) -> str:
    """
    Extract all text from a PDF in-memory, preserving Japanese characters.
    """
    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    text = ''.join(page.get_text() for page in doc)
    return text

# --- 3. Similarity search function ---
def find_similar(text: str, top_k: int) -> list[dict]:
    """
    Embed the input text and query the FAISS index for top_k matches.
    Returns a list of {'filename', 'score'} dicts.
    """
    # Get embedding (handles caching and chunking internally)
    emb = np.array([get_embedding(text)], dtype='float32')
    distances, indices = index.search(emb, top_k)

    results = []
    for d, idx in zip(distances[0], indices[0]):
        meta = id_map.get(str(int(idx)), {})
        results.append({
            'filename': meta.get('filename', 'unknown'),
            'score': float(1.0 / (1.0 + d)),
        })
    return results

# --- 4. Streamlit UI layout ---
st.set_page_config(page_title="Resume Similarity Search", layout="centered")
st.title("📄 Resume Similarity Search (日本語対応)")
st.markdown(
    "日本語の職務経歴書PDFをアップロードして、類似履歴書を検索します。"
)

# File uploader (supports Japanese filenames)
uploaded_file = st.file_uploader("職務経歴書PDFを選択", type=["pdf"])
# Slider for number of results
top_k = st.slider("表示する類似履歴書の数", min_value=1, max_value=20, value=5)

if uploaded_file is not None:
    # Show original filename
    st.write(f"**アップロードされたファイル:** {uploaded_file.name}")

    raw_bytes = uploaded_file.read()
    try:
        # Extract and clean text
        raw_text = extract_text_from_bytes(raw_bytes)
        text = clean_text(raw_text)

        if not text:
            st.error("⚠️ PDFからテキストを抽出できませんでした。スキャン画像の可能性があります。")
        else:
            # Show text preview
            st.subheader("抽出テキストプレビュー")
            st.text_area("", text[:1000], height=200)

            # Perform similarity search
            with st.spinner("埋め込みと検索中…"):
                results = find_similar(text, top_k)

            # Display results
            if results:
                st.success(f"トップ{top_k}の類似履歴書:")
                for i, r in enumerate(results, start=1):
                    st.write(f"{i}. **{r['filename']}** — スコア {r['score']:.3f}")
            else:
                st.info("類似する履歴書が見つかりませんでした。")
    except Exception as e:
        st.error(f"ファイル処理中にエラーが発生しました: {e}")
else:
    st.info("まずは職務経歴書PDFをアップロードしてください。")
