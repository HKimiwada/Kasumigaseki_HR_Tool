# streamlit run app.py
import os
import json
from pathlib import Path

import streamlit as st
import numpy as np
import faiss
import fitz         # PyMuPDF
from openai import OpenAI

# ————————————————
# Configuration
# ————————————————
from config import FAISS_INDEX_PATH, ID_MAP_PATH
from testing_utils import visualize_faiss_output_st

# Load OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ————————————————
# 1) Load FAISS index & ID map (once)
# ————————————————
@st.cache_resource
def load_faiss_index(index_path: str) -> faiss.Index:
    # Windows Unicode workaround: cd into the folder
    idx = Path(index_path)
    cwd = os.getcwd()
    os.chdir(str(idx.parent))
    index = faiss.read_index(idx.name)
    os.chdir(cwd)
    return index

@st.cache_resource
def load_id_map(map_path: str) -> dict:
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)

index  = load_faiss_index(str(FAISS_INDEX_PATH))
id_map = load_id_map(str(ID_MAP_PATH))

# ————————————————
# 2) Helpers
# ————————————————
def extract_text_from_bytes(raw: bytes) -> str:
    """Extract text from an in-memory PDF (preserves Japanese)."""
    doc = fitz.open(stream=raw, filetype="pdf")
    return "".join(p.get_text() for p in doc)

def clean_text(txt: str) -> str:
    """Normalize whitespace."""
    return " ".join(txt.split())

def get_embedding(text: str, model: str="text-embedding-ada-002") -> list[float]:
    """
    Direct single-call embedding (no caching, no chunking).
    Relies on OpenAI to error if over token limit (~8191 tokens).
    """
    # Ensure it's a single string
    resp = client.embeddings.create(input=text.replace("\n"," "), model=model)
    return resp.data[0].embedding  # type: ignore

def find_similar(text: str, top_k: int):
    emb = np.array([get_embedding(text)], dtype="float32")
    D, I = index.search(emb, top_k)
    out = []
    for dist, idx in zip(D[0], I[0]):
        meta = id_map.get(str(int(idx)), {})
        out.append({
            "filename": meta.get("filename", "unknown"),
            "score":    float(1.0/(1.0+dist)),
        })
    return out

# ————————————————
# 3) Streamlit UI
# ————————————————
st.set_page_config(page_title="Resume Similarity Search", layout="centered")
st.title("📄 Resume Similarity Search (日本語対応)")
st.markdown("日本語の職務経歴書PDFをアップロードして、類似履歴書を検索します。")

uploaded = st.file_uploader("職務経歴書PDFを選択", type=["pdf"])
top_k    = st.slider("表示する類似履歴書の数", 1, 20, 5)

if uploaded:
    st.write(f"**アップロードされたファイル:** {uploaded.name}")
    raw = uploaded.read()
    try:
        text = clean_text(extract_text_from_bytes(raw))
        if not text:
            st.error("⚠️ PDFからテキストを抽出できませんでした。")
        else:
            st.subheader("抽出テキストプレビュー")
            st.text_area("", text[:1000], height=200)

            with st.spinner("埋め込みと検索中…"):
                results = find_similar(text, top_k)

            if results:
                st.success(f"トップ {top_k} の類似履歴書:")
                for i, r in enumerate(results, start=1):
                    st.write(f"{i}. **{r['filename']}** — スコア {r['score']:.3f}")
                ##### Updated Section
                st.markdown("### Debug用:結果の可視化")
                fig = visualize_faiss_output_st(
                    'Database/processed_job_description.jsonl',
                    red_filename=uploaded.name,
                    blue_filenames=[r['filename'] for r in results]
                )
                st.pyplot(fig)
                ##### End of updated section
            else:
                st.info("類似する履歴書が見つかりませんでした。")
    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
else:
    st.info("まずは職務経歴書PDFをアップロードしてください。")
