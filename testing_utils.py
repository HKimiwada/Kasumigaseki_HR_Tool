# Utility Functions to test the individual components of the HR Analysis System
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

if __name__ == "__main__":
    # visualize_embeddings('Database/processed_job_description.jsonl')
    visualize_faiss_output('Database/processed_job_description.jsonl',
                           red_filename='職務経歴書_室井友伸.pdf',
                           blue_filenames=['職務経歴書（SMBC野村）.pdf', '小林学様【職務経歴書書】.pdf', '履歴書・職務経歴書.pdf', '職務経歴書（小澤金臣）.pdf'])
