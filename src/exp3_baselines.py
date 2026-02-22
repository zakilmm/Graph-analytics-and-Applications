import os
import json
import random
import numpy as np
import networkx as nx

# ===== PATCH PYTORCH / OGB COMPATIBILITY =====
import torch
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
# ============================================

from ogb.nodeproppred import NodePropPredDataset
from community import community_louvain

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


def ensure_dirs():
    os.makedirs("outputs/tables", exist_ok=True)


def load_graph(max_nodes=30000, seed=42):
    dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph, _ = dataset[0]

    edge_index = graph["edge_index"]
    n = graph["num_nodes"]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Largest connected component
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()

    # Subsample nodes for scalability
    if G.number_of_nodes() > max_nodes:
        random.seed(seed)
        nodes = random.sample(list(G.nodes()), max_nodes)
        G = G.subgraph(nodes).copy()

    return G


def run_louvain(G, gamma=1.0, seed=42):
    return community_louvain.best_partition(G, resolution=gamma, random_state=seed)


def run_label_propagation(G):
    communities = nx.algorithms.community.label_propagation_communities(G)
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    return partition


def run_spectral_sparse(G, k=50, seed=42):
    """
    Sparse spectral clustering:
    - Build sparse adjacency A
    - Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    - Take k smallest eigenvectors (excluding trivial one if needed)
    - Run k-means on embedding
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build sparse adjacency
    rows, cols = [], []
    for u, v in G.edges():
        rows.append(idx[u]); cols.append(idx[v])
        rows.append(idx[v]); cols.append(idx[u])

    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    # Degree and D^{-1/2}
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = diags(d_inv_sqrt.astype(np.float32))

    # Normalized adjacency
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # Normalized Laplacian: L = I - A_norm
    # We want smallest eigenvalues => use eigsh with which='SM'
    I = diags(np.ones(n, dtype=np.float32))
    L = I - A_norm

    # Compute k eigenvectors (smallest)
    # Note: for clustering into k communities, we typically take k eigenvectors.
    vals, vecs = eigsh(L, k=k, which="SM")

    # Row-normalize embedding (common trick)
    embedding = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(embedding)

    return {nodes[i]: int(labels[i]) for i in range(n)}


def main():
    ensure_dirs()

    # You can reduce max_nodes if your machine is limited (e.g., 15000)
    G = load_graph(max_nodes=30000, seed=42)
    print(f"Working graph: n={G.number_of_nodes()} m={G.number_of_edges()}")

    results = {}

    # Louvain baseline
    results["louvain_gamma_1.0"] = run_louvain(G, gamma=1.0, seed=42)

    # Spectral baseline (sparse)
    # If still heavy, try k=30 or max_nodes=15000
    results["spectral_k_50"] = run_spectral_sparse(G, k=50, seed=42)

    # Label propagation baseline
    results["label_prop"] = run_label_propagation(G)

    # Save partitions
    out = {name: {str(n): int(c) for n, c in part.items()}
           for name, part in results.items()}
    with open("outputs/tables/partitions_baselines.json", "w") as f:
        json.dump(out, f)

    print("Saved: outputs/tables/partitions_baselines.json")
    print("Partitions:", list(results.keys()))


if __name__ == "__main__":
    main()
