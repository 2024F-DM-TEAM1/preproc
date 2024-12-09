import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Dict
from collections import defaultdict


class Document:
    def __init__(self, idx: int, title: str, company: str):
        self.idx = idx
        self.title = title
        self.company = company
        self._title_shingles = None
        self._comp_shingles = None

    @property
    def title_shingles(self) -> Set[str]:
        if self._title_shingles is None:
            self._title_shingles = set(self.title[i:i + 2] for i in range(len(self.title) - 1))
        return self._title_shingles

    @property
    def comp_shingles(self) -> Set[str]:
        if self._comp_shingles is None:
            self._comp_shingles = set(self.company[i:i + 2] for i in range(len(self.company) - 1))
        return self._comp_shingles


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def naive_deduplication(documents: List[Document],
                        title_threshold: float,
                        comp_threshold: float) -> Tuple[Set[Tuple[int, int]], float]:
    """Naive deduplication using pairwise Jaccard similarity"""
    start_time = time.time()
    duplicate_pairs = set()
    n = len(documents)

    for i in range(n - 1):
        for j in range(i + 1, n):
            doc1, doc2 = documents[i], documents[j]
            title_sim = jaccard_similarity(doc1.title_shingles, doc2.title_shingles)
            if title_sim > title_threshold:
                comp_sim = jaccard_similarity(doc1.comp_shingles, doc2.comp_shingles)
                if comp_sim > comp_threshold:
                    duplicate_pairs.add((i, j))

    execution_time = time.time() - start_time
    return duplicate_pairs, execution_time


class OptimizedLSH:
    def __init__(self, num_bands=16, rows_per_band=8):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.num_hashes = num_bands * rows_per_band
        self.hash_ranges = [(i * rows_per_band, (i + 1) * rows_per_band)
                            for i in range(num_bands)]
        self.hash_params = np.random.randint(1, 2 ** 32 - 1, (2, self.num_hashes))
        self.prime = 4294967311

    def minhash_signature(self, shingle_set: Set[str]) -> np.ndarray:
        signature = np.full(self.num_hashes, np.inf)
        for shingle in shingle_set:
            hash_val = hash(shingle) & 0xffffffff
            for i in range(self.num_hashes):
                h = (self.hash_params[0, i] * hash_val + self.hash_params[1, i]) % self.prime
                signature[i] = min(signature[i], h)
        return signature

    def find_duplicates(self, documents: List[Document],
                        title_threshold: float,
                        comp_threshold: float) -> Tuple[Set[Tuple[int, int]], float]:
        start_time = time.time()
        hash_tables = [defaultdict(list) for _ in range(self.num_bands)]
        duplicate_pairs = set()

        signatures = [self.minhash_signature(doc.title_shingles) for doc in documents]

        for idx, sig in enumerate(signatures):
            for band_idx, (start, end) in enumerate(self.hash_ranges):
                band_hash = hash(tuple(sig[start:end]))
                hash_tables[band_idx][band_hash].append(idx)

        for band_idx in range(self.num_bands):
            for bucket in hash_tables[band_idx].values():
                if len(bucket) > 1:
                    for i in range(len(bucket) - 1):
                        for j in range(i + 1, len(bucket)):
                            idx1, idx2 = bucket[i], bucket[j]
                            if idx1 >= idx2:
                                continue

                            doc1, doc2 = documents[idx1], documents[idx2]
                            title_sim = jaccard_similarity(doc1.title_shingles, doc2.title_shingles)
                            if title_sim > title_threshold:
                                comp_sim = jaccard_similarity(doc1.comp_shingles, doc2.comp_shingles)
                                if comp_sim > comp_threshold:
                                    duplicate_pairs.add((idx1, idx2))

        execution_time = time.time() - start_time
        return duplicate_pairs, execution_time


def calculate_dedup_ratio(duplicate_pairs: Set[Tuple[int, int]], total_docs: int) -> float:
    """Calculate deduplication ratio"""
    unique_dupes = set()
    for idx1, idx2 in duplicate_pairs:
        unique_dupes.add(idx1)
        unique_dupes.add(idx2)

    return len(unique_dupes) / total_docs


def run_threshold_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Compare methods across different thresholds and store duplicate pairs"""
    documents = [Document(idx, row['title'], row['comp_name'])
                 for idx, row in df.iterrows()]

    thresholds = np.arange(0.01, 0.51, 0.1)
    results = []
    duplicate_pairs_by_threshold = {}
    total_docs = len(documents)

    for title_th in thresholds:
        comp_th = title_th
        print(f"\nTesting threshold: {title_th:.2f}")

        print("Running naive approach...")
        naive_pairs, naive_time = naive_deduplication(documents, title_th, comp_th)
        naive_ratio = calculate_dedup_ratio(naive_pairs, total_docs)

        print("Running LSH approach...")
        lsh = OptimizedLSH(num_bands=16, rows_per_band=8)
        lsh_pairs, lsh_time = lsh.find_duplicates(documents, title_th, comp_th)
        lsh_ratio = calculate_dedup_ratio(lsh_pairs, total_docs)

        duplicate_pairs_by_threshold[title_th] = {
            'naive': naive_pairs,
            'lsh': lsh_pairs
        }

        results.append({
            'threshold': title_th,
            'naive_time': naive_time,
            'lsh_time': lsh_time,
            'naive_dedup_ratio': naive_ratio,
            'lsh_dedup_ratio': lsh_ratio,
            'speedup': naive_time / lsh_time if lsh_time > 0 else float('inf'),
            'accuracy': lsh_ratio / naive_ratio if naive_ratio > 0 else 1.0
        })

    return pd.DataFrame(results), duplicate_pairs_by_threshold


def plot_threshold_results_with_optimal(results: pd.DataFrame):
    """Plot results with highlighting specific threshold"""
    plt.rcParams.update({'font.size': 14})

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    naive_color = '#FF6B6B'
    lsh_color = '#4ECDC4'
    highlight_color = '#45B7D1'
    highlight_threshold = 0.33

    time_x = results['threshold'].values
    naive_time = results['naive_time'].values
    lsh_time = results['lsh_time'].values

    highlight_time_y = lsh_time[np.abs(time_x - highlight_threshold).argmin()]

    ax1.plot(time_x, naive_time, '-', color=naive_color, label='Naive', linewidth=3)
    ax1.plot(time_x, lsh_time, '-', color=lsh_color, label='LSH', linewidth=3)
    ax1.axvline(x=highlight_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.plot(highlight_threshold, highlight_time_y, 'o', color=highlight_color,
             label=f'Threshold = {highlight_threshold}', markersize=12)
    ax1.set_xlabel('Threshold', fontsize=16)
    ax1.set_ylabel('Time (seconds)', fontsize=16)
    ax1.set_title('Execution Time vs Threshold', fontsize=18)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)

    ratio_x = results['threshold'].values
    naive_ratio = results['naive_dedup_ratio'].values
    lsh_ratio = results['lsh_dedup_ratio'].values
    highlight_ratio_y = lsh_ratio[np.abs(ratio_x - highlight_threshold).argmin()]

    ax2.plot(ratio_x, naive_ratio, '-', color=naive_color, label='Naive', linewidth=3)
    ax2.plot(ratio_x, lsh_ratio, '-', color=lsh_color, label='LSH', linewidth=3)
    ax2.axvline(x=highlight_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax2.plot(highlight_threshold, highlight_ratio_y, 'o', color=highlight_color,
             label=f'Threshold = {highlight_threshold}', markersize=12)
    ax2.set_xlabel('Threshold', fontsize=16)
    ax2.set_ylabel('Deduplication Ratio', fontsize=16)
    ax2.set_title('Deduplication Ratio vs Threshold', fontsize=18)
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)

    speedup_x = results['threshold'].values
    speedup_y = results['speedup'].values
    highlight_speedup_y = speedup_y[np.abs(speedup_x - highlight_threshold).argmin()]

    ax3.plot(speedup_x, speedup_y, '-', color=lsh_color, linewidth=3)
    ax3.axvline(x=highlight_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax3.plot(highlight_threshold, highlight_speedup_y, 'o', color=highlight_color,
             label=f'Threshold = {highlight_threshold}', markersize=12)
    ax3.set_xlabel('Threshold', fontsize=16)
    ax3.set_ylabel('Speedup (Naive time / LSH time)', fontsize=16)
    ax3.set_title('LSH Speedup vs Threshold', fontsize=18)
    ax3.legend(fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=14)

    acc_x = results['threshold'].values
    acc_y = results['accuracy'].values
    highlight_acc_y = acc_y[np.abs(acc_x - highlight_threshold).argmin()]

    ax4.plot(acc_x, acc_y, '-', color=lsh_color, linewidth=3)
    ax4.axvline(x=highlight_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax4.plot(highlight_threshold, highlight_acc_y, 'o', color=highlight_color,
             label=f'Threshold = {highlight_threshold}', markersize=12)
    ax4.set_xlabel('Threshold', fontsize=16)
    ax4.set_ylabel('Accuracy (LSH ratio / Naive ratio)', fontsize=16)
    ax4.set_title('LSH Accuracy vs Threshold', fontsize=18)
    ax4.legend(fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=14)

    plt.tight_layout()
    fig.suptitle(f'Deduplication Performance Analysis (Highlighted Threshold: {highlight_threshold})',
                 fontsize=20, y=1.02)

    metrics_at_highlight = {
        'time': highlight_time_y,
        'ratio': highlight_ratio_y,
        'speedup': highlight_speedup_y,
        'accuracy': highlight_acc_y
    }

    return fig, metrics_at_highlight


def analyze_duplicate_pairs(df: pd.DataFrame, pairs_dict: Dict, threshold: float):
    """Analyze duplicate pairs for given threshold"""
    naive_pairs = pairs_dict[threshold]['naive']
    lsh_pairs = pairs_dict[threshold]['lsh']
    total_docs = len(df)

    naive_ratio = calculate_dedup_ratio(naive_pairs, total_docs)
    lsh_ratio = calculate_dedup_ratio(lsh_pairs, total_docs)

    print(f"\nAnalysis for threshold {threshold:.2f}:")
    print(f"Naive deduplication ratio: {naive_ratio:.3f}")
    print(f"LSH deduplication ratio: {lsh_ratio:.3f}")

    print("\nSample duplicate pairs from naive approach:")
    for idx1, idx2 in list(naive_pairs)[:3]:
        print(f"\nPair {idx1} - {idx2}:")
        print(f"Title 1: {df['title'].iloc[idx1]}")
        print(f"Title 2: {df['title'].iloc[idx2]}")
        print(f"Company 1: {df['comp_name'].iloc[idx1]}")
        print(f"Company 2: {df['comp_name'].iloc[idx2]}")

    missed_pairs = naive_pairs - lsh_pairs
    missed_ratio = calculate_dedup_ratio(missed_pairs, total_docs)
    print(f"\nRatio of documents missed by LSH: {missed_ratio:.3f}")

    if missed_pairs:
        print("\nSample missed pairs:")
        for idx1, idx2 in list(missed_pairs)[:3]:
            print(f"\nMissed Pair {idx1} - {idx2}:")
            print(f"Title 1: {df['title'].iloc[idx1]}")
            print(f"Title 2: {df['title'].iloc[idx2]}")
            print(f"Company 1: {df['comp_name'].iloc[idx1]}")
            print(f"Company 2: {df['comp_name'].iloc[idx2]}")


def main():
    print("Loading data...")
    merged = pd.read_csv('./data/_merged.csv')

    print("Starting threshold comparison...")
    results_df, duplicate_pairs_dict = run_threshold_comparison(merged)

    print("\nDetailed Results:")
    print(results_df.to_string(index=False))

    plt.rcParams.update({
        'font.size': 14,
        'figure.titlesize': 20,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    fig, optimal_thresholds = plot_threshold_results_with_optimal(results_df)

    fixed_threshold = 0.33
    closest_threshold = min(duplicate_pairs_dict.keys(),
                            key=lambda x: abs(x - fixed_threshold))

    print(f"\nAnalyzing results for threshold: {fixed_threshold:.2f}")
    analyze_duplicate_pairs(merged, duplicate_pairs_dict, closest_threshold)

    plt.show()


if __name__ == "__main__":
    main()