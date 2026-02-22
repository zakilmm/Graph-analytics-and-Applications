import json
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm  # pip install tqdm
from networkx.algorithms.community import modularity

def load_partitions():
    print("📂 Chargement des partitions...")
    with open("outputs/tables/partitions_baselines.json") as f:
        raw = json.load(f)
    return {k: {int(n): int(c) for n, c in v.items()} for k, v in raw.items()}

def load_keywords():
    print("📂 Chargement des keywords...")
    with open("outputs/tables/keyword_sets.json") as f:
        raw = json.load(f)
    return {int(n): set(v) for n, v in raw.items()}

def build_graph():
    print("🔨 Construction du graphe...")
    import torch
    from ogb.nodeproppred import NodePropPredDataset

    torch.load = lambda *a, **kw: torch.serialization.load(*a, weights_only=False)
    dataset = NodePropPredDataset(name="ogbn-arxiv")
    graph, _ = dataset[0]
    edge_index = graph["edge_index"]

    nodes = set(load_partitions()["spectral_k_50"].keys())
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    print(f"   Ajout des arêtes ({len(edge_index[0])} arêtes à traiter)...")
    for u, v in tqdm(zip(edge_index[0], edge_index[1]), 
                     total=len(edge_index[0]), 
                     desc="   Arêtes"):
        u, v = int(u), int(v)
        if u in nodes and v in nodes and u != v:
            G.add_edge(u, v)
    
    print(f"   ✅ Graphe: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    return G

def partition_to_comms(part):
    comms = defaultdict(set)
    for n, c in part.items():
        comms[c].add(n)
    return comms

def compute_SC(part, keywords, k=10):
    comms = partition_to_comms(part)
    total_pairs, total = 0, 0.0
    for nodes in comms.values():
        s = len(nodes)
        if s < 2: continue
        kw_count = defaultdict(int)
        for v in nodes:
            for kw in keywords[v]:
                kw_count[kw] += 1
        inter = sum(c*(c-1)//2 for c in kw_count.values())
        pairs = s*(s-1)//2
        total += (inter / pairs) / k * pairs
        total_pairs += pairs
    return total / total_pairs if total_pairs > 0 else 0.0

def SAM(G, part, keywords, alpha):
    comms = list(partition_to_comms(part).values())
    Q = modularity(G, comms)
    SC = compute_SC(part, keywords)
    return Q, SC, alpha*Q + (1-alpha)*SC

def run_slr(alpha=0.5, max_iter=5):
    parts = load_partitions()
    part = parts["spectral_k_50"].copy()
    keywords = load_keywords()
    G = build_graph()

    print(f"\n🚀 Démarrage SLR (alpha={alpha}, max_iter={max_iter})")
    _, _, sam_cur = SAM(G, part, keywords, alpha)
    print(f"   SAM initial: {sam_cur:.6f}")

    for iteration in range(max_iter):
        print(f"\n📍 Itération {iteration + 1}/{max_iter}")
        moved = False
        nodes = list(part.keys())
        random.shuffle(nodes)

        # Barre de progression pour les nœuds
        for v in tqdm(nodes, desc=f"   Nœuds (iter {iteration+1})"):
            cur_c = part[v]
            neighbor_comms = {part[u] for u in G.neighbors(v)}

            for c in neighbor_comms:
                if c == cur_c:
                    continue
                part[v] = c
                _, _, sam_new = SAM(G, part, keywords, alpha)
                if sam_new > sam_cur:
                    sam_cur = sam_new
                    moved = True
                    break
                else:
                    part[v] = cur_c

        print(f"   SAM actuel: {sam_cur:.6f}, Déplacements: {'Oui' if moved else 'Non'}")
        
        if not moved:
            print("   ✅ Convergence atteinte")
            break

    return part, sam_cur

def main():
    alpha = 0.5
    part_slr, sam_slr = run_slr(alpha=alpha)

    print(f"\n💾 Sauvegarde de la partition...")
    with open("outputs/tables/partition_slr.json", "w") as f:
        json.dump({str(k): int(v) for k, v in part_slr.items()}, f)

    print(f"\n✨ SLR terminé ! SAM final = {sam_slr:.6f}")

if __name__ == "__main__":
    main()