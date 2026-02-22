import os
import json
import numpy as np
import matplotlib.pyplot as plt

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


def ensure_dirs():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)


def build_keyword_sets(k=10):
    """
    For each node, keep the indices of the top-k largest feature values
    as its keyword set.
    """
    dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph, labels = dataset[0]
    X = graph["node_feat"]  # shape (n, d)

    keyword_sets = {}
    keyword_sizes = []

    for i in range(X.shape[0]):
        vec = X[i]
        topk_idx = np.argsort(vec)[-k:]
        keyword_sets[i] = set(int(x) for x in topk_idx)
        keyword_sizes.append(len(topk_idx))

    return keyword_sets, np.array(keyword_sizes)


def plot_keyword_size_distribution(keyword_sizes):
    plt.figure()
    plt.hist(keyword_sizes, bins=20)
    plt.xlabel("Keyword set size")
    plt.ylabel("Count")
    plt.title("Keyword set size distribution")
    plt.tight_layout()
    plt.savefig("outputs/figures/keyword_set_size_distribution.png", dpi=200)
    plt.close()


def main():
    ensure_dirs()

    k = 10  # hyperparameter (we will vary later)
    keyword_sets, keyword_sizes = build_keyword_sets(k=k)

    # Save keyword sets
    with open("outputs/tables/keyword_sets.json", "w") as f:
        json.dump({str(i): list(v) for i, v in keyword_sets.items()}, f)

    plot_keyword_size_distribution(keyword_sizes)

    stats = {
        "k": k,
        "num_nodes": len(keyword_sets),
        "keyword_size_min": int(keyword_sizes.min()),
        "keyword_size_mean": float(keyword_sizes.mean()),
        "keyword_size_max": int(keyword_sizes.max())
    }

    with open("outputs/tables/keyword_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Keyword stats:")
    print(stats)
    print("Saved:")
    print("- outputs/tables/keyword_sets.json")
    print("- outputs/tables/keyword_stats.json")
    print("- outputs/figures/keyword_set_size_distribution.png")


if __name__ == "__main__":
    main()
