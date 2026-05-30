import os
import numpy as np
import polars as pl
import time
import argparse
import json
import torch
import math
from k_means_constrained import KMeansConstrained
import warnings

warnings.filterwarnings('ignore')

def balanced_kmeans_level_constrained(X, K, max_iter=100, tol=1e-7, random_state=None, verbose=False):
    """Balanced K-means implemented with k-means-constrained"""
    start_time = time.time()
    n, d = X.shape
    X = X.astype(np.float32, copy=False)
    
    # Calculate min and max cluster size
    min_size = max(1, n // K - 1)  # allow some imbalance
    max_size = n // K + 1
    
    if verbose:
        print(f"    Starting constrained K-means with K={K}, n={n}, d={d}")
        print(f"    Cluster size constraints: [{min_size}, {max_size}]")
    
    # Use k-means-constrained
    kmeans = KMeansConstrained(
        n_clusters=K,
        size_min=min_size,
        size_max=max_size,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=3,
        verbose=verbose,
        n_jobs=-1
    )
    
    # Train and get labels
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    print(f"[Time] balanced_kmeans_level_constrained (K={K}): {time.time() - start_time:.2f}s")
    
    if verbose:
        # Check cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"    Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    return labels, centroids

def residual_kmeans_constrained(X, K, L, max_iter=300, tol=1e-4, random_state=None, verbose=False):
    """Residual K-means implemented with k-means-constrained"""
    total_start = time.time()
    n, d = X.shape
    Ks = ([K] * L) if isinstance(K, int) else list(K)
    assert len(Ks) == L

    X = X.astype(np.float32, copy=False)
    R = X.copy()
    codes_all = np.empty((L, n), dtype=np.int32)
    codebooks = []

    for l in range(L):
        level_start = time.time()
        k_l = Ks[l]
        if verbose:
            mse_before = np.mean(R ** 2)
            print(f"\n=== Level {l+1}/{L} | K={k_l} ===")
            print(f"  Residual MSE before clustering: {mse_before:.6f}")

        # Generate random seed for sub-level
        seed_l = None if random_state is None else int(np.random.RandomState(random_state + l).randint(0, 2**31 - 1))
        
        codes_l, C_l = balanced_kmeans_level_constrained(
            R, k_l, max_iter=max_iter, tol=tol, random_state=seed_l, verbose=verbose
        )

        codes_all[l] = codes_l
        codebooks.append(C_l)
        
        # Subtract reconstructed part from residual
        R -= C_l[codes_l]

        print(f"[Time] Level {l+1}: {time.time() - level_start:.2f}s")
        if verbose:
            mse_after = np.mean(R ** 2)
            print(f"  Residual MSE after Level {l+1}: {mse_after:.6f}")

    recon = X - R
    print(f"[Time] residual_kmeans_constrained total: {time.time() - total_start:.2f}s")
    
    if verbose:
        total_mse = np.mean((X - recon) ** 2)
        print(f"\nFinal reconstruction MSE: {total_mse:.6f}")
    
    return codes_all, codebooks, recon

def check_constrained_availability():
    """Check k-means-constrained availability"""
    try:
        from k_means_constrained import KMeansConstrained
        return True
    except ImportError as e:
        print(f"k-means-constrained not available: {e}")
        return False
    
def deal_with_dedupilcate(df):
    """
    Handle duplicates: append a rank (p_1, p_2...) to EVERY ID sequence
    to ensure absolute uniqueness and consistent formatting.
    """
    try:
        df_with_index = df.with_row_index()
    except AttributeError:
        df_with_index = df.with_row_count()

    result_df = df_with_index.with_columns(
        pl.col("codes").list.concat(
            pl.col("index").rank(method="ordinal").over("codes").cast(pl.Int64)
        ).alias("codes")
    ).drop("index")

    return result_df

def find_nearest_head_items_torch(tail_emb, head_emb, batch_size=20000):
    """Find nearest head item index using Torch"""
    print(f"Mapping Tail items to nearest Head items using Torch... (Batch Size: {batch_size})")
    device = torch.device('cpu')
    
    head_tensor = torch.from_numpy(head_emb).to(device)
    tail_tensor = torch.from_numpy(tail_emb)
    
    head_tensor = torch.nn.functional.normalize(head_tensor, p=2, dim=1)
    
    num_tail = len(tail_emb)
    nearest_indices = np.zeros(num_tail, dtype=int)
    
    for i in range(0, num_tail, batch_size):
        end = min(i + batch_size, num_tail)
        batch_tail = tail_tensor[i:end].to(device)
        batch_tail = torch.nn.functional.normalize(batch_tail, p=2, dim=1)
        
        dists = torch.cdist(batch_tail, head_tensor)
        _, min_indices = torch.min(dists, dim=1)
        nearest_indices[i:end] = min_indices.cpu().numpy()
        
    return nearest_indices

def parse_args():
    parser = argparse.ArgumentParser(description="Residual RK-Means (Head/Tail Mixed)")
    parser.add_argument('--head_file', type=str, required=True)
    parser.add_argument('--tail_file', type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--head_depth", type=int, default=4)
    parser.add_argument("--tail_depth", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=100)
    return parser.parse_args()

def load_split_data(args):
    t0 = time.time()
    embeddings = np.load(args.data_path).astype(np.float32)
    print(f"Loaded total embeddings with shape {embeddings.shape}. [Time: {time.time()-t0:.2f}s]")
    
    with open(args.head_file, 'r') as f:
        head_ids = json.load(f)
    with open(args.tail_file, 'r') as f:
        tail_ids = json.load(f)
        
    head_ids = [int(i) for i in head_ids]
    tail_ids = [int(i) for i in tail_ids]
    
    head_emb = embeddings[head_ids]
    tail_emb = embeddings[tail_ids]
    
    print(f"Split Data: Head={len(head_ids)}, Tail={len(tail_ids)}")
    return head_ids, head_emb, tail_ids, tail_emb

if __name__ == "__main__":
    args = parse_args()
    if not check_constrained_availability():
        exit(1)
    
    head_ids, head_emb, tail_ids, tail_emb = load_split_data(args)

    # Phase 1: Head items
    print(f"\n>>> Phase 1: Clustering Head Items (Depth={args.head_depth})")
    head_k_values = [args.k] * args.head_depth
    head_codes_raw, _, head_recon = residual_kmeans_constrained(
        head_emb, K=head_k_values, L=args.head_depth, 
        random_state=42, verbose=True, max_iter=args.max_iter
    )

    # Phase 2: Mapping
    print(f"\n>>> Phase 2: Mapping Tail Items to Head Items")
    map_indices = find_nearest_head_items_torch(tail_emb, head_emb)
    tail_prefix_codes = head_codes_raw.T[map_indices] 
    tail_base_recon = head_recon[map_indices]
    
    # Phase 3: Tail suffix
    print(f"\n>>> Phase 3: Clustering Tail Item Suffixes (Depth={args.tail_depth})")
    tail_residuals = tail_emb - tail_base_recon
    tail_k_values = [args.k] * args.tail_depth
    tail_suffix_codes_raw, _, _ = residual_kmeans_constrained(
        tail_residuals, K=tail_k_values, L=args.tail_depth,
        random_state=42, verbose=True, max_iter=args.max_iter
    )
    
    # Combining
    print(f"\n>>> Combining and Saving")
    head_final_codes = head_codes_raw.T
    tail_suffix_codes = tail_suffix_codes_raw.T
    tail_final_codes = np.concatenate([tail_prefix_codes, tail_suffix_codes], axis=1)
    
    all_data = []
    for i, real_id in enumerate(head_ids):
        all_data.append({'id': real_id, 'codes': list(head_final_codes[i])})
    for i, real_id in enumerate(tail_ids):
        all_data.append({'id': real_id, 'codes': list(tail_final_codes[i])})
        
    df = pl.DataFrame(all_data)
    
    t_dedup = time.time()
    df_dedup = deal_with_dedupilcate(df)
    print(f"Global Deduplication finished. [Time: {time.time()-t_dedup:.2f}s]")

    codes_str = df.with_columns(
        pl.col("codes").map_elements(lambda x: ','.join(map(str, x)), return_dtype=pl.Utf8).alias("codes_str")
    )
    duplicates = (codes_str.group_by("codes_str").count().filter(pl.col("count") > 1))
    if len(duplicates) > 0:
        print(f"  - Total duplicate groups: {len(duplicates)}")

    output_dict = {}
    for row in df_dedup.iter_rows(named=True):
        real_id = row['id']
        code_list = row['codes']
        
        formatted_codes = []
        for i, code in enumerate(code_list):
            if i == len(code_list) - 1:
                token = f"<p_{code}>"
            else:
                token = f"<{chr(97+i)}_{code}>"
            formatted_codes.append(token)
            
        output_dict[str(real_id)] = formatted_codes
    
    sorted_keys = sorted(output_dict.keys(), key=lambda x: int(x))
    final_dict = {k: output_dict[k] for k in sorted_keys}
        
    with open(args.output_path, 'w') as f:
        json.dump(final_dict, f)
        
    print(f"JSON index saved to {args.output_path}.")