import os
import json
import math
import numpy as np
import pandas as pd
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


def load_full_graph_edges():
    dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph, _ = dataset[0]
    edge_index = graph["edge_index"]
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    return edges


def build_induced_graph(node_list, edges):
    node_set = set(node_list)
    G = nx.Graph()
    G.add_nodes_from(node_list)
    for u, v in edges:
        if u in node_set and v in node_set and u != v:
            G.add_edge(u, v)
    return G


def load_keyword_sets(path="outputs/tables/keyword_sets.json"):
    with open(path, "r") as f:
        raw = json.load(f)
    # raw keys are strings of node ids
    # values are list of keyword indices
    keyword_sets = {int(n): set(map(int, kws)) for n, kws in raw.items()}
    return keyword_sets


def load_partitions(path="outputs/tables/partitions_baselines.json"):
    with open(path, "r") as f:
        raw = json.load(f)
    partitions = {}
    for name, mapping in raw.items():
        partitions[name] = {int(n): int(c) for n, c in mapping.items()}
    return partitions


def partition_to_communities(partition_dict):
    comms = defaultdict(list)
    for node, cid in partition_dict.items():
        comms[cid].append(node)
    return comms


def compute_modularity(G, partition_dict):
    # networkx expects list of sets communities
    comms = partition_to_communities(partition_dict)
    communities = [set(nodes) for nodes in comms.values()]
    return nx.algorithms.community.modularity(G, communities)


def nC2(n):
    return n * (n - 1) // 2


def compute_sc_global(partition_dict, keyword_sets, k=10):
    """
    SC global = weighted avg of community SC weighted by number of node pairs.
    For a community C: SC(C) = avg_{u<v in C} (|Su ∩ Sv| / k)
    Efficient computation via keyword counts.
    """
    comms = partition_to_communities(partition_dict)

    total_pairs = 0
    total_sc_pairsum = 0.0  # sum over pairs of overlap coefficient

    for cid, nodes in comms.items():
        s = len(nodes)
        if s < 2:
            continue

        # count keyword occurrences in this community
        kw_count = defaultdict(int)
        for v in nodes:
            Sv = keyword_sets.get(v, None)
            if Sv is None:
                continue
            for kw in Sv:
                kw_count[kw] += 1

        # sum intersections over all unordered pairs
        inter_sum = 0
        for kw, c in kw_count.items():
            if c >= 2:
                inter_sum += nC2(c)

        pairs = nC2(s)
        # average overlap coefficient in this community
        avg_overlap_coeff = (inter_sum / pairs) / k

        total_pairs += pairs
        total_sc_pairsum += avg_overlap_coeff * pairs

    if total_pairs == 0:
        return 0.0

    return total_sc_pairsum / total_pairs


def main():
    ensure_dirs()

    partitions = load_partitions("outputs/tables/partitions_baselines.json")
    keyword_sets = load_keyword_sets("outputs/tables/keyword_sets.json")

    # Use nodes from one partition (they all share the same node set)
    any_name = next(iter(partitions.keys()))
    nodes = sorted(partitions[any_name].keys())

    edges = load_full_graph_edges()
    G = build_induced_graph(nodes, edges)

    print(f"Induced graph rebuilt: n={G.number_of_nodes()} m={G.number_of_edges()}")

    rows = []
    for name, part in partitions.items():
        comms = partition_to_communities(part)
        sizes = [len(v) for v in comms.values()]
        Q = compute_modularity(G, part)
        SC = compute_sc_global(part, keyword_sets, k=10)

        row = {
            "method": name,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_communities": len(comms),
            "community_size_min": int(np.min(sizes)),
            "community_size_mean": float(np.mean(sizes)),
            "community_size_max": int(np.max(sizes)),
            "modularity_Q": float(Q),
            "semantic_coherence_SC": float(SC),
        }
        rows.append(row)
        print(name, "Q=", Q, "SC=", SC, "num_comms=", len(comms))

    df = pd.DataFrame(rows).sort_values("method")
    df.to_csv("outputs/tables/results_q_sc.csv", index=False)

    with open("outputs/tables/results_q_sc.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("Saved:")
    print("- outputs/tables/results_q_sc.csv")
    print("- outputs/tables/results_q_sc.json")


if __name__ == "__main__":
    main()
