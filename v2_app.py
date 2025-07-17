# streamlit run v2_app.py
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
from config import V2_FAISS_INDEX_PATH, V2_ID_MAP_PATH
from testing_utils import visualize_faiss_output_tsne

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

index  = load_faiss_index(str(V2_FAISS_INDEX_PATH))
id_map = load_id_map(str(V2_ID_MAP_PATH))

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

def detect_index_type():
    """Detect if we're using L2 or Inner Product index"""
    # Try to determine index type
    if hasattr(index, 'index') and hasattr(index.index, 'metric_type'):
        return index.index.metric_type
    return "unknown"

def find_similar_adaptive(text: str, top_k: int):
    """
    Adaptive search that works with both L2 and IP indexes
    """
    # Get query embedding
    query_emb = np.array([get_embedding(text)], dtype="float32")
    
    # Detect index type and normalize if needed
    is_cosine_index = "IP" in str(type(index.index)) or "Inner" in str(type(index.index))
    
    if is_cosine_index:
        # Normalize for cosine similarity
        faiss.normalize_L2(query_emb)
        st.info("🔍 Using cosine similarity search")
    else:
        st.info("🔍 Using L2 distance search (consider rebuilding with cosine similarity)")
    
    # Search with more candidates to filter out generic matches
    search_k = min(top_k * 2, index.ntotal)
    D, I = index.search(query_emb, search_k)
    
    # Debug info
    st.text(f"Debug - Query embedding norm: {np.linalg.norm(query_emb):.4f}")
    st.text(f"Debug - Top distances/similarities: {D[0][:3]}")
    
    # Calculate scores based on index type
    results = []
    for dist_or_sim, idx in zip(D[0], I[0]):
        meta = id_map.get(str(int(idx)), {})
        filename = meta.get("filename", "unknown")
        
        if is_cosine_index:
            # For Inner Product with normalized vectors
            raw_score = float(dist_or_sim)
            display_score = float((dist_or_sim + 1) / 2)  # Convert [-1,1] to [0,1]
        else:
            # For L2 distance
            raw_score = float(dist_or_sim)
            display_score = float(1.0 / (1.0 + dist_or_sim))
        
        results.append({
            "filename": filename,
            "raw_score": raw_score,
            "score": display_score,
        })
    
    # Sort appropriately
    if is_cosine_index:
        results.sort(key=lambda x: x["raw_score"], reverse=True)  # Higher is better
    else:
        results.sort(key=lambda x: x["raw_score"])  # Lower distance is better
    
    # Check for suspiciously similar scores
    if len(results) >= 3:
        top_scores = [r["raw_score"] for r in results[:3]]
        score_std = np.std(top_scores)
        if score_std < 0.01:
            st.warning(f"⚠️ Top results have very similar scores (std: {score_std:.4f}). This suggests generic matching.")
    
    return results[:top_k]

def find_similar_with_diversity(text: str, top_k: int, diversity_threshold: float = 0.9):
    """
    Find similar resumes but filter out near-duplicates for diversity
    """
    # Get more candidates than needed
    candidates = find_similar_adaptive(text, top_k * 3)
    
    if not candidates:
        return []
    
    # Keep the top result
    selected = [candidates[0]]
    
    # Add diverse results
    for candidate in candidates[1:]:
        # Simple diversity check based on filename similarity
        is_diverse = True
        for selected_item in selected:
            # Basic filename-based diversity (you could make this more sophisticated)
            if candidate["filename"] == selected_item["filename"]:
                is_diverse = False
                break
        
        if is_diverse:
            selected.append(candidate)
            if len(selected) >= top_k:
                break
    
    return selected

# ————————————————
# 3) Streamlit UI
# ————————————————
st.set_page_config(page_title="Resume Similarity Search", layout="centered")
st.title("📄 Resume Similarity Search (日本語対応)")
st.markdown("日本語の職務経歴書PDFをアップロードして、類似履歴書を検索します。")

# Add search method selection
search_method = st.selectbox(
    "検索方法を選択:",
    ["Standard Search", "Diversity Search"],
    help="Diversity Search filters out very similar results"
)

uploaded = st.file_uploader("職務経歴書PDFを選択", type=["pdf"])
top_k = st.slider("表示する類似履歴書の数", 1, 20, 5)

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
                # FIXED: Use the improved search functions
                if search_method == "Diversity Search":
                    results = find_similar_with_diversity(text, top_k)
                else:
                    results = find_similar_adaptive(text, top_k)

            if results:
                st.success(f"トップ {top_k} の類似履歴書:")
                for i, r in enumerate(results, start=1):
                    st.write(f"{i}. **{r['filename']}** — スコア {r['score']:.3f}")
                
                # Visualization
                st.markdown("### Debug用:結果の可視化")
                try:
                    fig = visualize_faiss_output_tsne(
                        'Database/processed_job_description.jsonl',
                        red_filename=uploaded.name,
                        blue_filenames=[r['filename'] for r in results]
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"可視化エラー: {e}")
                    
            else:
                st.info("類似する履歴書が見つかりませんでした。")
    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
        st.text(f"Error details: {str(e)}")
else:
    st.info("まずは職務経歴書PDFをアップロードしてください。")
