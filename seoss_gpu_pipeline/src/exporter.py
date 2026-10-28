"""
Metrics Exporter and Visualization Engine for SEOSS Pipeline.
Saves CSVs, JSON summaries, and publication-ready plots.
Strictly separates evaluation metrics from operational/metadata fields during aggregation.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend safe for headless/Colab/server
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class MetricsExporter:
    """
    Exports benchmark tables and plots for SEOSS CIA experiments.
    """

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def save_project_metrics(self, project_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Saves per-project metrics to CSV and returns DataFrame.
        """
        df = pd.DataFrame(project_records)
        csv_path = os.path.join(self.output_dir, "per_project_metrics.csv")
        df.to_csv(csv_path, index=False)
        return df

    def save_overall_summary(
        self,
        df_metrics: pd.DataFrame,
        failed_projects: Optional[List[Dict[str, str]]] = None,
        total_attempted: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes macro-average and weighted micro-average across projects and writes summary files.
        Distinguishes evaluation metrics from operational and descriptive columns.
        """
        if df_metrics.empty:
            empty_summary = {
                "Run_Status": "FAILED",
                "Total_Projects_Attempted": total_attempted or 0,
                "Total_Projects_Evaluated": 0,
                "Total_Projects_Failed": len(failed_projects or []),
                "Failed_Projects": failed_projects or [],
                "Macro_Average": {},
                "Micro_Average": {}
            }
            json_path = os.path.join(self.output_dir, "overall_summary.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(empty_summary, f, indent=2)
            return empty_summary

        # Explicitly define non-metric metadata and operational columns to EXCLUDE from metric averages
        excluded_from_metrics = {
            "project",
            "total_issues",
            "total_ground_truth_links",
            "evaluated_queries",
            "candidate_pool_size",
            "elapsed_sec",
            "classification_threshold",
            "TP",
            "FP",
            "FN",
            "TN"
        }

        # Filter strictly evaluation metrics (e.g., Recall@K, Precision@K, MAP, MRR, Accuracy, F1, F2)
        metric_cols = [
            col for col in df_metrics.select_dtypes(include=[np.number]).columns
            if col not in excluded_from_metrics
        ]

        total_queries = int(df_metrics["evaluated_queries"].sum()) if "evaluated_queries" in df_metrics else 0
        total_eval = len(df_metrics)
        attempted = total_attempted or (total_eval + len(failed_projects or []))
        run_status = "COMPLETE" if (not failed_projects or len(failed_projects) == 0) else "PARTIAL"

        summary_data = {
            "Run_Status": run_status,
            "Total_Projects_Attempted": attempted,
            "Total_Projects_Evaluated": total_eval,
            "Total_Projects_Failed": len(failed_projects or []),
            "Total_Queries_Evaluated": total_queries,
            "Classification_Threshold": float(df_metrics["classification_threshold"].iloc[0]) if "classification_threshold" in df_metrics else 0.5,
            "Failed_Projects": failed_projects or [],
            "Macro_Average": {},
            "Micro_Average": {}
        }

        # Macro average: unweighted arithmetic mean across projects (equal weight per project)
        for col in metric_cols:
            summary_data["Macro_Average"][col] = round(float(df_metrics[col].mean()), 4)

        # Micro average: weighted mean by number of evaluated queries per project
        if total_queries > 0 and "evaluated_queries" in df_metrics:
            weights = df_metrics["evaluated_queries"]
            for col in metric_cols:
                weighted_mean = (df_metrics[col] * weights).sum() / total_queries
                summary_data["Micro_Average"][col] = round(float(weighted_mean), 4)

        # Save JSON summary
        json_path = os.path.join(self.output_dir, "overall_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Save CSV summary containing only valid aggregated metric rows
        summary_rows = [
            {"Aggregation_Type": "Macro_Average (equal project weight)", **summary_data["Macro_Average"]},
            {"Aggregation_Type": "Micro_Average (query-weighted)", **summary_data["Micro_Average"]}
        ]
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(self.output_dir, "overall_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        # If any projects failed or were skipped, save a dedicated failed_projects.csv
        if failed_projects:
            failed_df = pd.DataFrame(failed_projects)
            failed_csv_path = os.path.join(self.output_dir, "failed_projects.csv")
            failed_df.to_csv(failed_csv_path, index=False)

        return summary_data

    def generate_plots(self, df_metrics: pd.DataFrame):
        """
        Generates publication-quality charts.
        """
        if plt is None or df_metrics.empty:
            return

        # 1. Recall@K curve
        k_cols = [c for c in ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "Recall@20"] if c in df_metrics.columns]
        if k_cols:
            plt.figure(figsize=(8, 5))
            k_values = [int(c.split("@")[1]) for c in k_cols]
            mean_recalls = [df_metrics[c].mean() for c in k_cols]

            plt.plot(k_values, mean_recalls, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Macro-Avg Recall@K')
            plt.title("SEOSS Change Impact Retrieval: Recall@K Curve", fontsize=14, fontweight='bold', pad=12)
            plt.xlabel("Top-K Candidates Inspected", fontsize=12)
            plt.ylabel("Recall (Proportion of True Impact Links Found)", fontsize=12)
            plt.xticks(k_values)
            plt.ylim(0, min(1.0, max(mean_recalls) * 1.25 if mean_recalls else 1.0))
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "recall_at_k_curve.png"), dpi=300)
            plt.close()

        # 2. Project performance bar chart (F1, Recall@10, MAP)
        needed_cols = [c for c in ["Recall@10", "MAP", "F1"] if c in df_metrics.columns]
        if needed_cols and "project" in df_metrics.columns:
            plot_df = df_metrics.sort_values(by="project").head(25)  # up to 25 projects for readability
            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(plot_df))
            width = 0.25

            for i, col in enumerate(needed_cols):
                ax.bar(x + (i - 1) * width, plot_df[col], width, label=col)

            ax.set_ylabel("Score", fontsize=12)
            ax.set_title("Performance Metrics Across Evaluated SEOSS Projects", fontsize=14, fontweight='bold', pad=12)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df["project"], rotation=45, ha="right", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "per_project_comparison.png"), dpi=300)
            plt.close()

        # 3. Precision vs Recall Trade-off
        pr_pairs = [(1, "Recall@1", "Precision@1"), (3, "Recall@3", "Precision@3"),
                    (5, "Recall@5", "Precision@5"), (10, "Recall@10", "Precision@10"),
                    (20, "Recall@20", "Precision@20")]
        valid_pairs = [(k, r, p) for k, r, p in pr_pairs if r in df_metrics.columns and p in df_metrics.columns]

        if valid_pairs:
            plt.figure(figsize=(8, 5))
            r_vals = [df_metrics[r].mean() for _, r, _ in valid_pairs]
            p_vals = [df_metrics[p].mean() for _, _, p in valid_pairs]
            ks = [k for k, _, _ in valid_pairs]

            plt.plot(r_vals, p_vals, marker='s', color='#2ca02c', linewidth=2.5, markersize=8)
            for k, r, p in zip(ks, r_vals, p_vals):
                plt.annotate(f"K={k}", (r, p), textcoords="offset points", xytext=(5, 5), fontsize=10)

            plt.title("Precision-Recall Operating Curve Across Candidate Budgets", fontsize=14, fontweight='bold', pad=12)
            plt.xlabel("Macro Recall", fontsize=12)
            plt.ylabel("Macro Precision", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "precision_recall_operating_curve.png"), dpi=300)
            plt.close()
