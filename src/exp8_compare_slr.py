import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

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
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)


def load_keyword_sets(path="outputs/tables/keyword_sets.json"):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(n): set(map(int, kws)) for n, kws in raw.items()}


def load_slr_partition(path="outputs/tables/partition_slr.json"):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(n): int(c) for n, c in raw.items()}


def partition_to_communities(partition_dict):
    comms = defaultdict(list)
    for node, cid in partition_dict.items():
        comms[cid].append(node)
    return comms


def nC2(n):
    return n * (n - 1) // 2


def compute_sc_global(partition_dict, keyword_sets, k=10):
    """
    SC global = weighted avg over communities of average overlap coefficient.
    Efficient computation via keyword counts:
      sum_{u<v} |Su∩Sv| = sum_kw C(c_kw,2)
    """
    comms = partition_to_communities(partition_dict)

    total_pairs = 0
    total_sc_pairsum = 0.0

    for nodes in comms.values():
        s = len(nodes)
        if s < 2:
            continue

        kw_count = defaultdict(int)
        for v in nodes:
            Sv = keyword_sets.get(v, None)
            if Sv is None:
                continue
            for kw in Sv:
                kw_count[kw] += 1

        inter_sum = sum(nC2(c) for c in kw_count.values() if c >= 2)
        pairs = nC2(s)

        avg_overlap_coeff = (inter_sum / pairs) / k
        total_pairs += pairs
        total_sc_pairsum += avg_overlap_coeff * pairs

    return total_sc_pairsum / total_pairs if total_pairs > 0 else 0.0


def load_full_graph_edges():
    dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph, _ = dataset[0]
    ei = graph["edge_index"]
    return list(zip(ei[0].tolist(), ei[1].tolist()))


def build_induced_graph(nodes, edges):
    node_set = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v in edges:
        if u in node_set and v in node_set and u != v:
            G.add_edge(u, v)
    return G


def minmax_norm(series: pd.Series):
    return (series - series.min()) / (series.max() - series.min() + 1e-12)


def main():
    ensure_dirs()

    # Load baseline results (already computed)
    base_df = pd.read_csv("outputs/tables/results_q_sc.csv")

    # Load SLR partition and keywords
    slr_part = load_slr_partition("outputs/tables/partition_slr.json")
    keyword_sets = load_keyword_sets("outputs/tables/keyword_sets.json")

    # Rebuild the SAME induced graph used in baselines (from node set)
    nodes = sorted(slr_part.keys())
    edges = load_full_graph_edges()
    G = build_induced_graph(nodes, edges)

    print(f"Induced graph for comparison: n={G.number_of_nodes()} m={G.number_of_edges()}")

    # Compute Q and SC for SLR
    comms = partition_to_communities(slr_part)
    communities = [set(v) for v in comms.values()]
    Q_slr = nx.algorithms.community.modularity(G, communities)
    SC_slr = compute_sc_global(slr_part, keyword_sets, k=10)

    slr_row = {
        "method": "slr_alpha_0.5",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_communities": len(comms),
        "community_size_min": int(np.min([len(v) for v in comms.values()])),
        "community_size_mean": float(np.mean([len(v) for v in comms.values()])),
        "community_size_max": int(np.max([len(v) for v in comms.values()])),
        "modularity_Q": float(Q_slr),
        "semantic_coherence_SC": float(SC_slr),
    }

    print(f"SLR metrics: Q={Q_slr:.6f}, SC={SC_slr:.6f}, #comms={len(comms)}")

    # Combine
    combined_df = pd.concat([base_df, pd.DataFrame([slr_row])], ignore_index=True)
    combined_df.to_csv("outputs/tables/results_q_sc_with_slr.csv", index=False)
    print("Saved: outputs/tables/results_q_sc_with_slr.csv")

    # ---- Plot Q vs SC with SLR
    plt.figure()
    plt.scatter(combined_df["modularity_Q"], combined_df["semantic_coherence_SC"])
    for _, r in combined_df.iterrows():
        plt.annotate(r["method"], (r["modularity_Q"], r["semantic_coherence_SC"]))
    plt.xlabel("Modularity Q")
    plt.ylabel("Semantic Coherence SC")
    plt.title("Structure vs Semantic Trade-off (with SLR)")
    plt.tight_layout()
    plt.savefig("outputs/figures/q_vs_sc_tradeoff_with_slr.png", dpi=200)
    plt.close()
    print("Saved: outputs/figures/q_vs_sc_tradeoff_with_slr.png")

    # ---- Compute SAM vs alpha with SLR included (min-max over ALL methods incl SLR)
    combined_df["Q_norm"] = minmax_norm(combined_df["modularity_Q"])
    combined_df["SC_norm"] = minmax_norm(combined_df["semantic_coherence_SC"])

    alphas = np.linspace(0, 1, 11)
    sam_rows = []
    for a in alphas:
        sam = a * combined_df["Q_norm"] + (1 - a) * combined_df["SC_norm"]
        for i, r in combined_df.iterrows():
            sam_rows.append({"method": r["method"], "alpha": float(a), "SAM": float(sam.iloc[i])})

    sam_df = pd.DataFrame(sam_rows)
    sam_df.to_csv("outputs/tables/sam_scores_with_slr.csv", index=False)
    print("Saved: outputs/tables/sam_scores_with_slr.csv")

    # ---- Plot SAM vs alpha with SLR
    plt.figure()
    for method in sam_df["method"].unique():
        sub = sam_df[sam_df["method"] == method]
        plt.plot(sub["alpha"], sub["SAM"], marker="o", label=method)
    plt.xlabel("alpha")
    plt.ylabel("SAM score")
    plt.title("SAM sensitivity to alpha (with SLR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/sam_vs_alpha_with_slr.png", dpi=200)
    plt.close()
    print("Saved: outputs/figures/sam_vs_alpha_with_slr.png")


if __name__ == "__main__":
    main()
