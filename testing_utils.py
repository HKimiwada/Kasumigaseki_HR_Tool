# Utility Functions to test the individual components of the HR Analysis System
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings(jsonl_path):
    """
    Load embeddings from a JSONL file, perform PCA to reduce to 2D,
    and plot a scatter plot.
    """
    # Load embeddings and filenames
    embeddings = []
    filenames = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            embeddings.append(rec['embedding'])
            filenames.append(rec['filename'])
    
    X = np.array(embeddings, dtype=np.float32)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    plt.title('PCA of Resume Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()

# Highlight Input Embedding as Red, and FAISS Matched Embeddings as Blue, Other Embeddings as Grey
def visualize_faiss_output(jsonl_path, red_filename, blue_filenames):
    """
    Load embeddings from a JSONL file, perform PCA to reduce to 2D,
    and plot a scatter plot highlighting one red point and many blue points.
    
    red_filename:   the filename of the point to color red
    blue_filenames: a list of filenames to color blue
    """
    # Load embeddings and filenames
    embeddings = []
    filenames  = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            embeddings.append(rec['embedding'])
            filenames.append(rec['filename'])
    
    X = np.array(embeddings, dtype=np.float32)
    
    # PCA to 2D
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Find index of red point
    try:
        idx_red = filenames.index(red_filename)
    except ValueError:
        raise ValueError(f"Red filename '{red_filename}' not found in JSONL.")
    
    # Find indices of blue points (ignore any missing)
    idx_blue = []
    for name in blue_filenames:
        if name in filenames:
            idx_blue.append(filenames.index(name))
        else:
            print(f"Warning: blue filename '{name}' not found, skipping.")
    
    # Build color list: default grey, then red and blue
    colors = ['grey'] * len(filenames)
    colors[idx_red] = 'red'
    for i in idx_blue:
        colors[i] = 'blue'
    
    # Plot everything
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1],
                c=colors,
                alpha=0.7,
                edgecolors='w',
                linewidths=0.5)
    plt.title('PCA of Resume Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()

# Returns Matplotlib Figure Object instead of showing it directly for displaying in Streamlit.
def visualize_faiss_output_st(
    jsonl_path: str,
    red_filename: str,
    blue_filenames: list[str]
) -> plt.Figure:
    """
    Load embeddings, perform PCA, and return a Matplotlib Figure
    with one red point, many blue points, and grey background points.
    """
    # 1. Load embeddings & filenames
    embeddings = []
    filenames  = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            embeddings.append(rec['embedding'])
            filenames.append(rec['filename'])

    X = np.array(embeddings, dtype=np.float32)

    # 2. PCA to 2D
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 3. Map filenames → indices
    idx_red  = filenames.index(red_filename)
    idx_blue = [filenames.index(fn) for fn in blue_filenames if fn in filenames]

    # 4. Create figure & axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # 5. Draw background grey points
    ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        color='lightgrey', alpha=0.7,
    )

    # 6. Overlay blue points
    ax.scatter(
        X_pca[idx_blue, 0], X_pca[idx_blue, 1],
        color='blue', s=80, label='Similar Resumes'
    )

    # 7. Overlay red point
    ax.scatter(
        X_pca[idx_red, 0], X_pca[idx_red, 1],
        color='red', s=100, label='Uploaded Resume'
    )

    # 8. Labels & legend
    ax.set_title('PCA of Resume Embeddings')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    fig.tight_layout()

    return fig

def visualize_faiss_output_tsne(
    jsonl_path: str,
    red_filename: str,
    blue_filenames: list[str],
    perplexity: int = 30,
    random_state: int = 42
) -> plt.Figure:
    """
    Load embeddings from a JSONL file, run t-SNE to reduce to 2D,
    and plot a scatter:
      - all points in light gray
      - FAISS-suggested neighbors in blue
      - the query (uploaded) resume in red

    Returns a Matplotlib Figure.
    """
    # 1) Load embeddings & filenames
    embeddings = []
    filenames = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            filenames.append(rec['filename'])
            embeddings.append(rec['embedding'])
    X = np.array(embeddings, dtype=np.float32)

    # 2) Compute 2D via t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        init='random',
        random_state=random_state,
        n_jobs=-1
    )
    X2 = tsne.fit_transform(X)  # shape (N, 2)

    # 3) Identify indices for highlights
    idx_map = {fn: i for i, fn in enumerate(filenames)}
    red_idx = idx_map.get(red_filename, None)
    blue_idxs = [idx_map[bf] for bf in blue_filenames if bf in idx_map]

    # 4) Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # all points
    ax.scatter(X2[:, 0], X2[:, 1],
               color='lightgray', alpha=0.5, s=20, label='Database')
    # neighbors
    handled = False
    for i in blue_idxs:
        ax.scatter(X2[i, 0], X2[i, 1],
                   color='blue', edgecolor='k', s=80,
                   label='FAISS neighbor' if not handled else "")
        handled = True
    # query
    if red_idx is not None:
        ax.scatter(X2[red_idx, 0], X2[red_idx, 1],
                   color='red', marker='X', s=150,
                   label='Query resume')
    # Legend and labels
    ax.legend()
    ax.set_title("t-SNE of Resume Embeddings\n(red=query, blue=FAISS neighbors)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.tight_layout()
    return fig

# Code to Diagnose Embeddings
def diagnose_embeddings():
    # Load all embeddings
    embeddings = []
    filenames = []
    with open('Database/processed_job_description.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            embeddings.append(rec['embedding'])
            filenames.append(rec['filename'])
    
    X = np.array(embeddings)
    
    # Check embedding statistics
    print(f"Embedding matrix shape: {X.shape}")
    print(f"Mean embedding norm: {np.linalg.norm(X, axis=1).mean():.4f}")
    print(f"Std embedding norm: {np.linalg.norm(X, axis=1).std():.4f}")
    
    # Check pairwise distances
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(X)
    print(f"Mean cosine similarity: {sim_matrix.mean():.4f}")
    print(f"Min cosine similarity: {sim_matrix.min():.4f}")
    print(f"Max cosine similarity: {sim_matrix.max():.4f}")

# Check what text is actually being embedded
def check_text_diversity():
    texts = []
    with open('Database/processed_job_description.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec['text'])
    
    # Check text lengths and uniqueness
    lengths = [len(t.split()) for t in texts]
    print(f"Text lengths - mean: {np.mean(lengths):.1f}, std: {np.std(lengths):.1f}")
    
    # Check for duplicate or very similar texts
    unique_texts = set(texts)
    print(f"Unique texts: {len(unique_texts)} out of {len(texts)}")

if __name__ == "__main__":
    # visualize_embeddings('Database/processed_job_description.jsonl')
    visualize_faiss_output('Database/processed_job_description.jsonl',
                           red_filename='職務経歴書_室井友伸.pdf',
                           blue_filenames=['職務経歴書（SMBC野村）.pdf', '小林学様【職務経歴書書】.pdf', '履歴書・職務経歴書.pdf', '職務経歴書（小澤金臣）.pdf'])
