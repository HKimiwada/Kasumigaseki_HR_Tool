# analyze_universal_resumes.py
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def find_universal_resumes(jsonl_path: str, top_n: int = 10):
    """
    Find resumes that are most similar to all other resumes (universal matches)
    """
    # Load embeddings
    embeddings = []
    filenames = []
    texts = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            embeddings.append(rec['embedding'])
            filenames.append(rec['filename'])
            texts.append(rec['text'])
    
    X = np.array(embeddings)
    
    # Calculate similarity matrix
    sim_matrix = cosine_similarity(X)
    
    # For each resume, calculate its average similarity to all others
    avg_similarities = []
    for i in range(len(filenames)):
        # Exclude self-similarity (diagonal = 1.0)
        others_sim = np.concatenate([sim_matrix[i][:i], sim_matrix[i][i+1:]])
        avg_sim = others_sim.mean()
        avg_similarities.append((avg_sim, filenames[i], texts[i]))
    
    # Sort by average similarity (descending)
    avg_similarities.sort(reverse=True)
    
    print("=== MOST UNIVERSAL RESUMES (most similar to everything) ===")
    for i, (avg_sim, filename, text) in enumerate(avg_similarities[:top_n]):
        print(f"\n{i+1}. {filename}")
        print(f"   Average similarity to others: {avg_sim:.4f}")
        print(f"   Text preview: {text[:200]}...")
        print("   " + "-" * 50)
    
    return avg_similarities

def analyze_common_words(jsonl_path: str, universal_count: int = 5):
    """
    Analyze what words appear most in universal resumes
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    
    # Get embeddings and find universal resumes
    embeddings = np.array([rec['embedding'] for rec in lines])
    sim_matrix = cosine_similarity(embeddings)
    avg_sims = [np.concatenate([sim_matrix[i][:i], sim_matrix[i][i+1:]]).mean() 
                for i in range(len(lines))]
    
    # Get top universal resumes
    universal_indices = np.argsort(avg_sims)[-universal_count:]
    universal_texts = [lines[i]['text'] for i in universal_indices]
    
    # Count words in universal resumes
    all_words = []
    for text in universal_texts:
        # Simple word splitting (you might want to use a proper Japanese tokenizer)
        words = text.split()
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    
    print(f"\n=== MOST COMMON WORDS IN TOP {universal_count} UNIVERSAL RESUMES ===")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")
    
    return word_counts

if __name__ == "__main__":
    # Run the analysis
    universal_resumes = find_universal_resumes('Database/processed_job_description.jsonl')
    common_words = analyze_common_words('Database/processed_job_description.jsonl')