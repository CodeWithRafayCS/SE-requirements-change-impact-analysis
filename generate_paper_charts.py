# -*- coding: utf-8 -*-
"""
Generates high-resolution, publication-ready charts and graphs for the research paper.
Covers:
1. performance_comparison_chart.png: Grouped comparison of Existing Methods vs. Proposed Methods across Accuracy, Precision, Recall, F1, F2, Cost.
2. matched_cost_topk_chart.png: Recall across fixed inspection budgets (Top 1, 2, 3, 4, 8).
3. hypothesis_forest_plot.png: Forest plot of statistical hypothesis testing (point estimates & 95% bootstrap CIs).
4. signal_quality_chart.png: Ranking signal discriminability (AUROC and AUPRC) across modalities.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EAEAEA'
plt.rcParams['grid.linestyle'] = '--'

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Figure 3: Performance Measures Comparison (Existing vs. Proposed Methods)
# -------------------------------------------------------------------------
def generate_performance_comparison_chart():
    methods = [
        "B1: TF-IDF\n(Lexical)",
        "B2: MiniLM\n(Dense)",
        "B3: Unpruned\n(High-Recall)",
        "ProReFiCIA\n(Cloud LLM)",
        "C1: Emb+NLI\n(Ablation)",
        "C2: Emb+Lex\n(Ablation)",
        "C3: Simplex\n(Manual)",
        "C4: Learned\n(Proposed)"
    ]
    
    # Categories: Existing vs Proposed
    colors = [
        "#5C6B73", "#1B365D", "#7F96A0", "#9C27B0", # Existing & Benchmarks
        "#E59866", "#E67E22", "#D35400", "#2E7D32"   # Proposed & Hybrids (C4 is rich green)
    ]
    
    recall = [47.6, 47.6, 85.7, 85.7, 47.6, 76.2, 19.0, 66.7]
    precision = [57.1, 57.1, 2.5, 31.6, 35.7, 11.2, 21.4, 33.3]
    f1 = [50.0, 50.0, 4.8, 46.2, 35.7, 17.6, 17.1, 39.5]
    f2 = [48.4, 48.4, 11.2, 64.2, 41.2, 30.6, 18.0, 50.2]
    cost = [1.52, 1.33, 55.62, 3.00, 2.48, 10.86, 1.52, 4.38]
    accuracy = [98.28, 98.47, 45.51, 97.20, 97.32, 89.67, 97.51, 96.18]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [2.2, 1]})
    
    # Subplot 1: Core Performance Measures (Recall, Precision, F1, F2)
    ax1 = axes[0]
    x = np.arange(len(methods))
    width = 0.20

    r1 = ax1.bar(x - 1.5*width, recall, width, label="Recall (%)", color="#1B365D", alpha=0.9)
    r2 = ax1.bar(x - 0.5*width, precision, width, label="Precision (%)", color="#2E5B88", alpha=0.85)
    r3 = ax1.bar(x + 0.5*width, f1, width, label="F1-Score (%)", color="#E67E22", alpha=0.85)
    r4 = ax1.bar(x + 1.5*width, f2, width, label="F2-Score (%)", color="#2E7D32", alpha=0.9)

    ax1.set_ylabel("Score (%)", fontsize=11, fontweight='bold', color="#1B365D")
    ax1.set_title("(a) Core Classification Measures (Recall, Precision, F1, F2)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9.5)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDD", fontsize=9.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Highlight Proposed C4 method
    ax1.axvspan(6.5, 7.5, color='#E8F5E9', alpha=0.5, zorder=0)
    ax1.text(7.0, 100, "PROPOSED\n(C4)", ha='center', va='top', fontsize=8.5, fontweight='bold', color='#1B5E20')

    # Subplot 2: Operational Inspection Cost & Accuracy
    ax2 = axes[1]
    b_cost = ax2.bar(x, cost, color=colors, width=0.55, edgecolor="#333", linewidth=0.6)
    
    # Annotate cost values above bars
    for rect in b_cost:
        height = rect.get_height()
        ax2.annotate(f"{height:.1f}%",
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_ylabel("Inspection Cost (% of Requirements)", fontsize=11, fontweight='bold', color="#1B365D")
    ax2.set_title("(b) Operational Inspection Cost", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9.5, rotation=45, ha='right')
    ax2.set_ylim(0, 65)
    ax2.axhline(5.0, color="#D32F2F", linestyle=":", linewidth=1.2, label="5% Cost Ceiling (H2)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDD", fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "performance_comparison_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Generated: {chart_path}")

# -------------------------------------------------------------------------
# Figure 4: Statistical Hypothesis Forest Plot (95% Bootstrap CIs)
# -------------------------------------------------------------------------
def generate_hypothesis_forest_plot():
    fig, ax = plt.subplots(figsize=(10, 4.8))

    hypotheses = [
        "H1: C3 vs. B2 (Top-1 Delta)",
        "H2: C4 Recall vs. Floor (66.7% vs 68.6%)",
        "H3a: C3 vs. C1 (Top-1 Delta)",
        "H3b: C4 vs. C3 (Top-1 Delta)",
        "H4: NLI vs. Cosine AUROC Delta"
    ]
    
    means = [-42.86, -1.90, 0.00, 42.86, -48.99]
    ci_lows = [-85.71, -35.24, 0.00, 14.29, -67.32]   # H2 low is 33.33 - 68.56 = -35.24
    ci_highs = [-14.29, 26.68, 0.00, 85.71, -21.69]  # H2 high is 95.24 - 68.56 = +26.68
    
    outcomes = [
        "Not Supported (p=1.000)",
        "Inconclusive (Margin -1.9 pp)",
        "Not Supported (p=1.000)",
        "Not Confirmed (p=0.375)",
        "Refuted (p=0.0005)"
    ]
    
    colors = ["#C62828", "#E65100", "#546E7A", "#1565C0", "#B71C1C"]

    y_pos = np.arange(len(hypotheses))

    # Reference zero line
    ax.axvline(0, color="#333", linestyle="--", linewidth=1.2, alpha=0.7, label="Null Difference (Delta = 0)")

    # Plot confidence interval error bars
    for i in range(len(hypotheses)):
        err_low = means[i] - ci_lows[i]
        err_high = ci_highs[i] - means[i]
        ax.errorbar(means[i], y_pos[i], xerr=[[err_low], [err_high]], 
                    fmt='o', color=colors[i], ecolor=colors[i], elinewidth=2.2, 
                    capsize=6, capthick=1.8, markersize=8, zorder=3)
        # Annotate text
        txt = f" {means[i]:+.1f} pp [95% CI: {ci_lows[i]:+.1f}, {ci_highs[i]:+.1f}] | {outcomes[i]}"
        ax.text(max(ci_highs[i] + 3, means[i] + 3), y_pos[i], txt, 
                va='center', fontsize=9, fontweight='semibold', color=colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(hypotheses, fontsize=10, fontweight='bold', color="#1B365D")
    ax.set_xlabel("Mean Difference (Percentage Points / Metric Points) with 95% Bootstrap Confidence Interval", 
                  fontsize=10.5, fontweight='bold', color="#1B365D", labelpad=8)
    ax.set_title("Formal Statistical Hypothesis Testing Outcomes (10,000 Bootstrap Resamples)", 
                 fontsize=12, fontweight='bold', pad=12, color="#1B365D")
    ax.set_xlim(-100, 110)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.invert_yaxis()  # top-down order

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "hypothesis_forest_plot.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Generated: {chart_path}")

# -------------------------------------------------------------------------
# Figure 5: Signal Quality and Discriminability Chart (AUROC & AUPRC)
# -------------------------------------------------------------------------
def generate_signal_quality_chart():
    features = [
        "Cosine Similarity\n(MiniLM Dense)",
        "Lexical/Heuristic\n(Jaccard + Dice + IDF)",
        "Zero-Shot NLI\n(DeBERTa-v3 Entailment)",
        "C4 Combiner\n(Learned Composite)"
    ]
    
    auroc = [0.952, 0.823, 0.463, 0.947]
    auprc = [0.403, 0.158, 0.038, 0.351]

    x = np.arange(len(features))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 4.8))
    rects1 = ax.bar(x - width/2, auroc, width, label="AUROC (Area Under ROC Curve)", color="#1B365D", alpha=0.9, edgecolor="#333", linewidth=0.5)
    rects2 = ax.bar(x + width/2, auprc, width, label="AUPRC (Area Under PR Curve)", color="#E67E22", alpha=0.9, edgecolor="#333", linewidth=0.5)

    # Add reference random chance line for AUROC
    ax.axhline(0.5, color="#D32F2F", linestyle=":", linewidth=1.2, label="Random Chance Baseline (AUROC = 0.500)")

    # Data labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_ylabel("Metric Score", fontsize=11, fontweight='bold', color="#1B365D")
    ax.set_title("Ranking Signal Discriminability on Stage-1 Candidate Pairs", fontsize=12, fontweight='bold', pad=12, color="#1B365D")
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=9.5)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDD", fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "signal_quality_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Generated: {chart_path}")

# -------------------------------------------------------------------------
# Figure 6: Matched Inspection Cost (Top-k) Comparison Chart
# -------------------------------------------------------------------------
def generate_matched_cost_chart():
    budgets = ["Top 1\n(1.33%)", "Top 2\n(2.67%)", "Top 3\n(4.00%)", "Top 4\n(5.33%)", "Top 8\n(10.00%)"]
    
    b2_rec = [47.6, 52.4, 57.1, 57.1, 85.7]
    c1_rec = [4.8, 47.6, 52.4, 57.1, 85.7]
    c2_rec = [47.6, 61.9, 61.9, 61.9, 76.2]
    c3_rec = [4.8, 47.6, 61.9, 66.7, 85.7]
    c4_rec = [47.6, 52.4, 66.7, 66.7, 85.7]

    x = np.arange(len(budgets))
    width = 0.16

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 2*width, b2_rec, width, label="B2: Dense Baseline (MiniLM)", color="#1B365D", alpha=0.9)
    ax.bar(x - 1*width, c1_rec, width, label="C1: Emb + NLI Cross-Encoder", color="#7F96A0", alpha=0.85)
    ax.bar(x + 0*width, c2_rec, width, label="C2: Emb + Lexical/Entity", color="#E67E22", alpha=0.85)
    ax.bar(x + 1*width, c3_rec, width, label="C3: Simplex Grid Combiner", color="#D35400", alpha=0.85)
    ax.bar(x + 2*width, c4_rec, width, label="C4: Learned Combiner (Proposed)", color="#2E7D32", alpha=0.95)

    ax.axhline(85.7, color="#C62828", linestyle="--", linewidth=1.2, label="Stage-1 Candidate Ceiling (85.7%)")

    ax.set_ylabel("Test Set Macro Recall (%)", fontsize=11, fontweight='bold', color="#1B365D")
    ax.set_xlabel("Fixed Inspection Budget (% of 75-Requirement Specification)", fontsize=10.5, fontweight='bold', color="#1B365D", labelpad=8)
    ax.set_title("Matched-Cost Recall Comparison Across Fixed Inspection Budgets", fontsize=12, fontweight='bold', pad=12, color="#1B365D")
    ax.set_xticks(x)
    ax.set_xticklabels(budgets, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DDD", fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "matched_cost_topk_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Generated: {chart_path}")

if __name__ == "__main__":
    generate_performance_comparison_chart()
    generate_hypothesis_forest_plot()
    generate_signal_quality_chart()
    generate_matched_cost_chart()
    print("All paper charts generated successfully.")
