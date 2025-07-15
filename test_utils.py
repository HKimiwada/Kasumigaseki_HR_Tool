# Code to test the individual components of the HR Analysis System
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

if __name__ == "__main__":
    visualize_embeddings('Database/processed_job_description.jsonl')
