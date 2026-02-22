import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def ensure_dirs():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

def build_ogbn_arxiv_graph():
    """
    Builds an undirected graph from ogbn-arxiv citation edges.
    Node features are provided by OGB (128-d by default).
    We also create a pseudo 'keyword count' proxy: number of non-zero feature dimensions.
    """
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph, labels = dataset[0]

    edge_index = graph["edge_index"]  # shape (2, m)
    x = graph["node_feat"]            # shape (n, d)

    n = x.shape[0]
    # Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges (make undirected)
    src = edge_index[0]
    dst = edge_index[1]
    edges = list(zip(src.tolist(), dst.tolist()))
    G.add_edges_from(edges)

    # Remove self-loops (optional but common)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Keep largest connected component (optional)
    # (good if graph has isolates; ogbn-arxiv is mostly connected)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # Create "keyword count" proxy
    # In your report, SC uses keyword sets. Here we emulate keyword count using feature sparsity.
    # If features are dense, this will be ~d; still usable for baseline plots.
    keyword_counts = (x != 0).sum(axis=1).astype(int)

    # Compute stats
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = np.array([d for _, d in G.degree()])

    avg_degree = degrees.mean()
    density = nx.density(G)

    # clustering coefficient can be expensive; sample if needed
    # ogbn-arxiv is large, so approximate by sampling nodes
    sample_size = min(10000, num_nodes)
    sampled_nodes = np.random.choice(list(G.nodes()), size=sample_size, replace=False)
    clustering_vals = []
    for v in sampled_nodes:
        clustering_vals.append(nx.clustering(G, v))
    avg_clustering_approx = float(np.mean(clustering_vals))

    stats = {
        "dataset": "ogbn-arxiv",
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "avg_degree": float(avg_degree),
        "density": float(density),
        "avg_clustering_approx": float(avg_clustering_approx),
        "feature_dim": int(x.shape[1]),
        "keyword_count_mean_proxy": float(np.mean(keyword_counts)),
        "keyword_count_median_proxy": float(np.median(keyword_counts)),
    }

    return G, degrees, keyword_counts, stats

def save_stats_table(stats: dict):
    df = pd.DataFrame([stats])
    df.to_csv("outputs/tables/,s.csv", index=False)
    return df

def plot_degree_distribution(degrees: np.ndarray):
    plt.figure()
    plt.hist(degrees, bins=50)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Degree distribution")
    plt.tight_layout()
    plt.savefig("outputs/figures/degree_distribution.png", dpi=200)
    plt.close()

def plot_keyword_count_distribution(keyword_counts: np.ndarray):
    plt.figure()
    plt.hist(keyword_counts, bins=50)
    plt.xlabel("Keyword count proxy (#non-zero features)")
    plt.ylabel("Count")
    plt.title("Keyword-count distribution (proxy)")
    plt.tight_layout()
    plt.savefig("outputs/figures/keyword_count_distribution.png", dpi=200)
    plt.close()

def main():
    ensure_dirs()
    G, degrees, keyword_counts, stats = build_ogbn_arxiv_graph()

    df = save_stats_table(stats)
    plot_degree_distribution(degrees)
    plot_keyword_count_distribution(keyword_counts)

    # Also save a lightweight JSON for your LaTeX
    with open("outputs/tables/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Saved:")
    print("- outputs/tables/dataset_stats.csv")
    print("- outputs/tables/dataset_stats.json")
    print("- outputs/figures/degree_distribution.png")
    print("- outputs/figures/keyword_count_distribution.png")
    print("\nDataset stats:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
