# streamlit run app.py
# Front-end UI for the HR tool (light-weight dashboard for testing)
import streamlit as st
import numpy as np
import faiss
import json
from pathlib import Path
from data_extraction_utils import extract_text_from_pdf, clean_text, get_embedding
from config import DB_DIR, FAISS_INDEX_PATH, ID_MAP_PATH

@st.cache(allow_output_mutation=True)
def load_faiss_index(index_path):
    """
    Load and return a FAISS index from disk.
    Cached to avoid reloading on each interaction.
    """
    return faiss.read_index(str(index_path))

@st.cache(allow_output_mutation=True)
def load_id_map(id_map_path):
    """
    Load and return the integer→metadata JSON map.
    """
    with open(id_map_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load once
index = load_faiss_index(FAISS_INDEX_PATH)
id_map = load_id_map(ID_MAP_PATH)

def find_similar(text: str, top_k: int):
    """
    Clean, embed, and query the FAISS index, returning top_k matches.
    """
    # extract & clean text already done upstream
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
st.title("📄 Resume Similarity Search")
st.markdown("Upload a 職務経歴書 PDF to find the most similar resumes in your database.")

uploaded_file = st.file_uploader("Select a resume PDF", type=["pdf"] )
top_k = st.slider("Number of similar resumes to show", min_value=1, max_value=20, value=5)

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    try:
        # Extract & clean text
        text = clean_text(extract_text_from_pdf(raw_bytes))
        if not text:
            st.error("⚠️ No text could be extracted from the PDF. Is it a scanned image?")
        else:
            with st.spinner("Embedding and searching…"):
                results = find_similar(text, top_k)
            if results:
                st.success(f"Top {top_k} similar resumes:")
                for i, r in enumerate(results, start=1):
                    st.write(f"**{i}. {r['filename']}** — Score: {r['score']:.3f}")
            else:
                st.info("No similar resumes found.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a 職務経歴書 PDF to begin.")
