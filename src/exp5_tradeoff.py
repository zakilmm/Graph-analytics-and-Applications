import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def main():
    df = pd.read_csv("outputs/tables/results_q_sc.csv")

    Q = df["modularity_Q"].values
    SC = df["semantic_coherence_SC"].values

    # Correlations
    pearson_r, pearson_p = pearsonr(Q, SC)
    spearman_r, spearman_p = spearmanr(Q, SC)

    print("Correlation results:")
    print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    print(f"Spearman r = {spearman_r:.4f}, p = {spearman_p:.4e}")

    # Scatter plot
    plt.figure()
    plt.scatter(Q, SC)

    for i, name in enumerate(df["method"]):
        plt.annotate(name, (Q[i], SC[i]))

    plt.xlabel("Modularity Q")
    plt.ylabel("Semantic Coherence SC")
    plt.title("Structure vs Semantic Trade-off")
    plt.tight_layout()
    plt.savefig("outputs/figures/q_vs_sc_tradeoff.png", dpi=200)
    plt.close()

    # Save stats
    stats = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p
    }

    pd.DataFrame([stats]).to_csv(
        "outputs/tables/q_sc_correlations.csv", index=False
    )

    print("Saved:")
    print("- outputs/figures/q_vs_sc_tradeoff.png")
    print("- outputs/tables/q_sc_correlations.csv")

if __name__ == "__main__":
    main()
