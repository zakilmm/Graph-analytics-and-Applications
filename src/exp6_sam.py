import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def main():
    df = pd.read_csv("outputs/tables/results_q_sc.csv")

    # Normalize
    df["Q_norm"] = minmax_norm(df["modularity_Q"])
    df["SC_norm"] = minmax_norm(df["semantic_coherence_SC"])

    alphas = np.linspace(0, 1, 11)
    sam_rows = []

    for alpha in alphas:
        sam = alpha * df["Q_norm"] + (1 - alpha) * df["SC_norm"]
        for i, row in df.iterrows():
            sam_rows.append({
                "method": row["method"],
                "alpha": alpha,
                "SAM": sam.iloc[i]
            })

    sam_df = pd.DataFrame(sam_rows)
    sam_df.to_csv("outputs/tables/sam_scores.csv", index=False)

    # Plot SAM vs alpha
    plt.figure()
    for method in sam_df["method"].unique():
        sub = sam_df[sam_df["method"] == method]
        plt.plot(sub["alpha"], sub["SAM"], marker="o", label=method)

    plt.xlabel("alpha")
    plt.ylabel("SAM score")
    plt.title("SAM sensitivity to alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/sam_vs_alpha.png", dpi=200)
    plt.close()

    print("Saved:")
    print("- outputs/tables/sam_scores.csv")
    print("- outputs/figures/sam_vs_alpha.png")

if __name__ == "__main__":
    main()
