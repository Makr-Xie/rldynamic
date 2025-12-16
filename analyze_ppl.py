import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")


# ----------------------------
# Prefix metric computation
# ----------------------------
def compute_prefix_stats(token_log_probs: List[float], k: int) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """
    Compute prefix mean logprob, NLL, and perplexity using K_eff=min(K,T).

    Returns:
      mean_logp (higher is better), nll (>=0), ppl, K_eff
      If no usable tokens (all None), returns (None, None, None, K_eff).
    """
    if not token_log_probs or k <= 0:
        return None, None, None, 0

    T = len(token_log_probs)
    k_eff = min(k, T)

    prefix = token_log_probs[:k_eff]
    clean = [lp for lp in prefix if lp is not None]

    if not clean:
        return None, None, None, k_eff

    mean_logp = float(np.mean(clean))         
    nll = float(-mean_logp)                   
    ppl = float(math.exp(nll))                
    return mean_logp, nll, ppl, k_eff


def load_dataset_with_k_values(data_dir: str, k_values: List[int]) -> pd.DataFrame:
    """Load all JSON files and compute prefix mean_logp/nll/ppl for specified K values."""
    json_files = sorted(glob.glob(os.path.join(data_dir, "validation_step_*.json")))
    rows = []

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        for record in data:
            token_log_probs = record.get("token_log_probs", []) or []
            prompt = record.get("prompt", "")

            row = {
                "uid": record.get("uid"),
                "prompt": prompt,  # GROUP KEY
                "step": int(record.get("step")),
                "rollout_idx": record.get("rollout_idx"),
                "reward": float(record.get("reward")),
                "num_tokens": len(token_log_probs),
            }

            # Per-K prefix stats
            for k in k_values:
                mean_logp, nll, ppl, k_eff = compute_prefix_stats(token_log_probs, k)
                row[f"mean_logp_k{k}"] = mean_logp
                row[f"nll_k{k}"] = nll
                row[f"ppl_k{k}"] = ppl
                row[f"k_eff_k{k}"] = k_eff

            # Full stats
            if token_log_probs:
                mean_logp_full, nll_full, ppl_full, _ = compute_prefix_stats(token_log_probs, len(token_log_probs))
            else:
                mean_logp_full, nll_full, ppl_full = None, None, None

            row["mean_logp_full"] = mean_logp_full
            row["nll_full"] = nll_full
            row["ppl_full"] = ppl_full

            rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# Prompt-averaged AUROC + CI
# ----------------------------
def _prompt_aurocs(
    df: pd.DataFrame,
    score_col: str,
    step: Optional[int] = None,
    prompt_col: str = "prompt",
) -> np.ndarray:
    """Return per-prompt AUROCs (skip prompts with all-0 or all-1 rewards or missing score)."""
    if step is not None:
        df = df[df["step"] == step]

    keep = df[[prompt_col, "reward", score_col]].dropna()
    if keep.empty:
        return np.array([])

    aurocs = []
    for _, g in keep.groupby(prompt_col):
        y = g["reward"].values
        s = g[score_col].values
        if len(np.unique(y)) < 2:
            continue
        try:
            au = roc_auc_score(y, s)
            aurocs.append(au)
        except Exception:
            continue

    return np.asarray(aurocs, dtype=float)


def prompt_avg_auroc_with_ci(
    df: pd.DataFrame,
    score_col: str,
    step: Optional[int] = None,
    n_boot: int = 500,
    seed: int = 0,
) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """
    Compute mean AUROC across prompts + bootstrap CI by resampling prompts.

    Returns (mean, ci_low, ci_high, n_prompts_used).
    """
    rng = np.random.default_rng(seed)
    aurocs = _prompt_aurocs(df, score_col=score_col, step=step)

    n = len(aurocs)
    if n == 0:
        return None, None, None, 0

    mean = float(np.mean(aurocs))

    if n == 1:
        return mean, mean, mean, 1

    boots = []
    for _ in range(n_boot):
        sample = rng.choice(aurocs, size=n, replace=True)
        boots.append(float(np.mean(sample)))

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi), n


def compute_auroc_vs_k_prompt_avg(
    df: pd.DataFrame,
    k_values: List[int],
    step: Optional[int],
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """
    AUROC vs K using prompt-averaged AUROC.
    Score used: mean_logp_kK (higher => more correct).
    """
    results = []
    for k in k_values:
        score_col = f"mean_logp_k{k}"
        if score_col not in df.columns:
            continue

        mean, lo, hi, n_prompts = prompt_avg_auroc_with_ci(
            df, score_col=score_col, step=step, n_boot=n_boot, seed=seed
        )
        if mean is None:
            continue

        # Length baseline (prompt-avg AUROC using num_tokens)
        mean_len, lo_len, hi_len, n_prompts_len = prompt_avg_auroc_with_ci(
            df, score_col="num_tokens", step=step, n_boot=n_boot, seed=seed + 1
        )

        results.append({
            "k": k,
            "auroc": mean,
            "auroc_ci_low": lo,
            "auroc_ci_high": hi,
            "n_prompts": n_prompts,
            "length_baseline": mean_len,
            "length_ci_low": lo_len,
            "length_ci_high": hi_len,
            "random_baseline": 0.5
        })

    return pd.DataFrame(results)


def compute_auroc_vs_step_prompt_avg(
    df: pd.DataFrame,
    k_values: List[int],
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """
    AUROC across training steps for fixed K values using prompt-avg AUROC.
    """
    results = []
    for step in sorted(df["step"].unique()):
        for k in k_values:
            score_col = f"mean_logp_k{k}"
            if score_col not in df.columns:
                continue
            mean, lo, hi, n_prompts = prompt_avg_auroc_with_ci(
                df, score_col=score_col, step=step, n_boot=n_boot, seed=seed + step + k
            )
            if mean is None:
                continue
            results.append({
                "step": step,
                "k": k,
                "auroc": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_prompts": n_prompts
            })
    return pd.DataFrame(results)


# ----------------------------
# Correlation (pooled)
# ----------------------------
def compute_correlation_vs_k(df: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    """
    Pearson correlation between prefix score and binary reward (pooled).
    Uses mean_logp (log-space) to avoid PPL outliers.
    """
    results = []
    for k in k_values:
        col = f"mean_logp_k{k}"
        if col not in df.columns:
            continue
        valid = df[[col, "reward"]].dropna()
        if len(valid) < 10:
            continue
        corr, pval = pearsonr(valid[col].values, valid["reward"].values)
        results.append({"k": k, "correlation": float(corr), "p_value": float(pval)})
    return pd.DataFrame(results)


# ----------------------------
# Figures
# ----------------------------
def generate_all_figures(datasets: Dict[str, pd.DataFrame], k_config: Dict, output_dir: str):
    k_values_full = k_config["k_values_full"]
    k_values_step = k_config["k_values_step"]
    k_fixed = k_config["k_fixed"]
    k_name = k_config["name"]

    print(f"\n{'='*70}")
    print(f"Generating {k_name} figures -> {output_dir}")
    print(f"K full: {k_values_full} | K step: {k_values_step} | K fixed: {k_fixed}")
    print(f"{'='*70}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Use last step for vs-K and pruning/distribution plots (consistent with your original intent)
    last_steps = {m: int(df["step"].max()) for m, df in datasets.items()}

    # ---- Figure 1: AUROC vs K (prompt-avg + CI) at last step
    plt.figure(figsize=(8, 5))
    for model_name, df in datasets.items():
        step = last_steps[model_name]
        auroc_df = compute_auroc_vs_k_prompt_avg(df, k_values_full, step=step)
        if auroc_df.empty:
            continue
        plt.plot(auroc_df["k"], auroc_df["auroc"], marker="o", label=f"{model_name}", linewidth=2)
        plt.fill_between(auroc_df["k"], auroc_df["auroc_ci_low"], auroc_df["auroc_ci_high"], alpha=0.15)

    # length baseline from first model (at its last step)
    first_model = list(datasets.keys())[0]
    step = last_steps[first_model]
    base_df = compute_auroc_vs_k_prompt_avg(datasets[first_model], k_values_full, step=step)
    if not base_df.empty and base_df["length_baseline"].notna().any():
        plt.plot(base_df["k"], base_df["length_baseline"], color="gray", linestyle=":", label="Length baseline", linewidth=2)
        if base_df["length_ci_low"].notna().any():
            plt.fill_between(base_df["k"], base_df["length_ci_low"], base_df["length_ci_high"], color="gray", alpha=0.10)

    plt.axhline(y=0.5, color="gray", linestyle="--", label="Random baseline", alpha=0.7)
    plt.xlabel("Prefix Length (K tokens)")
    plt.ylabel("Prompt-Avg AUROC")
    plt.title(f"AUROC vs Prefix Length ({k_name}, last step)", fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure1_auroc_vs_k.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure1_auroc_vs_k.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Saved Figure 1")

    # ---- Figure 2: AUROC vs Step (prompt-avg + CI)
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    axes = axes[0]
    for ax, (model_name, df) in zip(axes, datasets.items()):
        auroc_df = compute_auroc_vs_step_prompt_avg(df, k_values_step)
        for k in k_values_step:
            data = auroc_df[auroc_df["k"] == k]
            if data.empty:
                continue
            ax.plot(data["step"], data["auroc"], marker="o", label=f"K={k}", linewidth=2)
            ax.fill_between(data["step"], data["ci_low"], data["ci_high"], alpha=0.12)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Prompt-Avg AUROC")
        ax.set_title(model_name, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"AUROC Across Training Steps ({k_name})", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure2_auroc_vs_step.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure2_auroc_vs_step.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Saved Figure 2")

    # ---- Figure 3: Distributions at last step using NLL (more stable than PPL)
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    axes = axes[0]
    for ax, (model_name, df) in zip(axes, datasets.items()):
        last_step = last_steps[model_name]
        df_step = df[df["step"] == last_step].copy()

        col = f"nll_k{k_fixed}"
        valid = df_step[[col, "reward"]].dropna()
        correct = valid[valid["reward"] == 1.0][col]
        incorrect = valid[valid["reward"] == 0.0][col]

        ax.hist(correct, bins=30, alpha=0.6, label="Correct (1)", density=True)
        ax.hist(incorrect, bins=30, alpha=0.6, label="Incorrect (0)", density=True)

        ax.set_xlabel(f"Prefix NLL (K={k_fixed})  (lower is better)")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name} (Step {last_step})", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        if len(correct) > 0:
            ax.axvline(correct.mean(), linestyle="--", linewidth=2, alpha=0.8)
        if len(incorrect) > 0:
            ax.axvline(incorrect.mean(), linestyle="--", linewidth=2, alpha=0.8)

    plt.suptitle(f"NLL Distributions ({k_name}, K={k_fixed})", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure3_distributions.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure3_distributions.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Saved Figure 3")

    # ---- Figure 4: Mean NLL vs Step (correct vs incorrect)
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    axes = axes[0]
    for ax, (model_name, df) in zip(axes, datasets.items()):
        col = f"nll_k{k_fixed}"
        results = []
        for step in sorted(df["step"].unique()):
            df_step = df[df["step"] == step]
            valid = df_step[[col, "reward"]].dropna()
            if valid.empty:
                continue
            correct = valid[valid["reward"] == 1.0][col]
            incorrect = valid[valid["reward"] == 0.0][col]
            if len(correct) == 0 or len(incorrect) == 0:
                continue
            results.append({
                "step": step,
                "correct_mean": float(correct.mean()),
                "incorrect_mean": float(incorrect.mean()),
                "all_mean": float(valid[col].mean())
            })
        df_results = pd.DataFrame(results)
        if df_results.empty:
            continue

        ax.plot(df_results["step"], df_results["correct_mean"], marker="o", label="Correct", linewidth=2)
        ax.plot(df_results["step"], df_results["incorrect_mean"], marker="s", label="Incorrect", linewidth=2)
        ax.plot(df_results["step"], df_results["all_mean"], marker="^", label="All", linewidth=2, alpha=0.6)

        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"Mean Prefix NLL (K={k_fixed})")
        ax.set_title(model_name, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Mean NLL Over Training ({k_name}, K={k_fixed})", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure4_mean_nll_vs_step.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure4_mean_nll_vs_step.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Saved Figure 4")

    # ---- Figure 5: Pruning tradeoff (correct metrics + true token savings)
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    axes = axes[0]

    for ax, (model_name, df) in zip(axes, datasets.items()):
        last_step = last_steps[model_name]
        df_step = df[df["step"] == last_step].copy()

        score_col = f"mean_logp_k{k_fixed}"  # higher is better
        use = df_step[[score_col, "reward", "num_tokens"]].dropna()
        if use.empty:
            continue

        # sort so worst scores are first (to prune)
        use = use.sort_values(score_col, ascending=True).reset_index(drop=True)

        total_tokens = float(use["num_tokens"].sum())
        total_correct = int((use["reward"] == 1.0).sum())

        fracs = np.linspace(0, 0.9, 50)
        out = []
        for frac in fracs:
            n_prune = int(len(use) * frac)
            pruned = use.iloc[:n_prune]
            retained = use.iloc[n_prune:]

            if len(retained) == 0:
                continue

            # accuracy among retained (precision of keeping)
            acc_retained = float(retained["reward"].mean()) * 100.0

            # recall of correct retained
            if total_correct > 0:
                correct_retained = int((retained["reward"] == 1.0).sum())
                recall_correct = 100.0 * correct_retained / total_correct
            else:
                recall_correct = np.nan

            # true token savings if pruned rollouts stop at K_fixed
            if total_tokens > 0 and len(pruned) > 0:
                saved = float(np.maximum(0, pruned["num_tokens"].values - k_fixed).sum())
                token_savings = 100.0 * saved / total_tokens
            else:
                token_savings = 0.0

            out.append({
                "rollouts_pruned_pct": frac * 100.0,
                "acc_retained_pct": acc_retained,
                "correct_recall_pct": recall_correct,
                "token_savings_pct": token_savings
            })

        df_out = pd.DataFrame(out)
        if df_out.empty:
            continue

        ax.plot(df_out["rollouts_pruned_pct"], df_out["acc_retained_pct"], marker="o", markersize=4, linewidth=2, label="Accuracy among retained")
        ax.plot(df_out["rollouts_pruned_pct"], df_out["correct_recall_pct"], marker="s", markersize=4, linewidth=2, label="Correct retained (recall)")

        ax2 = ax.twinx()
        ax2.plot(df_out["rollouts_pruned_pct"], df_out["token_savings_pct"], linestyle="--", linewidth=2, alpha=0.9, label="Token savings (%)")
        ax2.set_ylabel("Token savings (%)")

        ax.set_xlabel("Rollouts pruned (%)")
        ax.set_ylabel("Percent (%)")
        ax.set_title(f"{model_name} (Step {last_step}, K={k_fixed})", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 90])
        ax.set_ylim([0, 105])

        # combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")

    plt.suptitle(f"Pruning Tradeoff ({k_name}, K={k_fixed})", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure5_pruning_tradeoff.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure5_pruning_tradeoff.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Saved Figure 5")

    # ---- Figure 6: Correlation vs K (pooled, using mean_logp)
    plt.figure(figsize=(8, 5))
    for model_name, df in datasets.items():
        corr_df = compute_correlation_vs_k(df, k_values_full)
        if corr_df.empty:
            continue
        plt.plot(corr_df["k"], corr_df["correlation"], marker="o", label=model_name, linewidth=2)

    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Prefix Length (K tokens)")
    plt.ylabel("Pearson corr(mean_logp_K, reward)")
    plt.title(f"Correlation: Prefix Score vs Reward ({k_name})", fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure6_correlation.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure6_correlation.pdf"), bbox_inches="tight")
    plt.close()
    print("✓ Saved Figure 6")

    # ---- Print quick summary table (last step)
    print("\nSummary (last step, prompt-avg AUROC):")
    for model_name, df in datasets.items():
        step = last_steps[model_name]
        auroc_df = compute_auroc_vs_k_prompt_avg(df, k_values_full, step=step)
        if auroc_df.empty:
            continue
        best_row = auroc_df.iloc[auroc_df["auroc"].astype(float).idxmax()]
        k_best = int(best_row["k"])
        au_best = float(best_row["auroc"])
        kfix_row = auroc_df[auroc_df["k"] == k_fixed]
        au_kfix = float(kfix_row["auroc"].values[0]) if not kfix_row.empty else float("nan")
        print(f"  {model_name}: step={step} | best AUROC={au_best:.3f} @K={k_best} | AUROC@K_fixed={au_kfix:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures with multiple K ranges for comparison")
    parser.add_argument(
        "--qwen1_7b",
        dest="qwen1_7b",
        type=str,
        default="ppl_tracking_qwen3_1.7b",
        help="Path to Qwen3-1.7B dataset directory",
    )
    parser.add_argument(
        "--qwen4b",
        type=str,
        default="ppl_tracking_qwen3_4b",
        help="Path to Qwen3-4B dataset directory",
    )
    parser.add_argument("--n_boot", type=int, default=500, help="Bootstrap samples for AUROC CIs (over prompts)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrapping")

    args = parser.parse_args()

    k_configs = [
        {"name": "Early", "k_values_full": [5, 10, 20, 40, 80], "k_values_step": [10, 20, 40], "k_fixed": 20, "output_dir": "figures_early"},
        {"name": "Mid", "k_values_full": [10, 20, 40, 80, 160, 320], "k_values_step": [20, 40, 80], "k_fixed": 40, "output_dir": "figures_mid"},
        {"name": "Late", "k_values_full": [50, 100, 200, 400, 800], "k_values_step": [100, 200, 400], "k_fixed": 200, "output_dir": "figures_late"},
    ]

    for config in k_configs:
        print(f"\n{'='*70}")
        print(f"Loading datasets for {config['name']} configuration")
        print(f"{'='*70}")

        datasets: Dict[str, pd.DataFrame] = {}
        all_k_values = sorted(set(config["k_values_full"] + config["k_values_step"] + [config["k_fixed"]]))

        if os.path.exists(args.qwen1_7b):
            print(f"  Loading {args.qwen1_7b} ...")
            datasets["Qwen3-1.7B"] = load_dataset_with_k_values(args.qwen1_7b, all_k_values)
            print(f"    Loaded {len(datasets['Qwen3-1.7B'])} records")

        if os.path.exists(args.qwen4b):
            print(f"  Loading {args.qwen4b} ...")
            datasets["Qwen3-4B"] = load_dataset_with_k_values(args.qwen4b, all_k_values)
            print(f"    Loaded {len(datasets['Qwen3-4B'])} records")

        if not datasets:
            print(f"Error: No datasets found for {config['name']}!")
            continue

        generate_all_figures(datasets, config, config["output_dir"])

    print(f"\n{'='*70}")
    print("✓ All figure sets generated successfully!")
    print(f"{'='*70}")
    print("\nOutput directories:")
    print("  - figures_early/  (K=[5, 10, 20, 40, 80])")
    print("  - figures_mid/    (K=[10, 20, 40, 80, 160, 320])")
    print("  - figures_late/   (K=[50, 100, 200, 400, 800])")


if __name__ == "__main__":
    main()
