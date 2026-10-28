"""
Comprehensive Empirical Evaluation Metrics Engine for SE Traceability & CIA.
Calculates Accuracy, Precision, Recall, F1, F2, Recall@K, MAP, MRR, and Inspection Cost.
Strictly empirical metrics — zero hypothesis testing or statistical hand-waving.

Note on Evaluation Methodology:
- Ranking metrics (Recall@K, Precision@K, MAP, MRR) are purely threshold-independent
  and evaluate candidate ranking quality directly.
- Classification metrics (Accuracy, Precision, Recall, F1, F2) evaluate binary link
  predictions at an explicitly specified, fixed decision threshold (default: 0.5)
  under a zero-shot regime without validation tuning to prevent test-set leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any, Optional


class SEOSSEvaluator:
    """
    Standard empirical metrics evaluator for software engineering traceability
    and change impact analysis benchmarks.
    """

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10, 20]

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculates standard binary classification metrics at a given decision threshold:
        Accuracy, Precision, Recall, Specificity, F1-Score, F2-Score.
        """
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        total = tp + fp + fn + tn

        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            # F2-score weights recall higher (beta = 2), critical for safety in CIA
            f2 = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)
        else:
            f1 = 0.0
            f2 = 0.0

        balanced_acc = 0.5 * (recall + specificity)

        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Threshold": round(float(threshold), 4),
            "Accuracy": round(float(accuracy), 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "Specificity": round(float(specificity), 4),
            "F1": round(float(f1), 4),
            "F2": round(float(f2), 4),
            "Balanced_Accuracy": round(float(balanced_acc), 4)
        }

    def compute_ranking_metrics(
        self,
        retrieval_df: pd.DataFrame,
        ground_truth_map: Dict[str, Set[str]],
        score_column: str = "similarity",
        rank_column: str = "rank",
        total_candidates_pool: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Computes ranking metrics for each query and aggregates:
        Recall@K, Precision@K, MAP, MRR, and Inspection Cost.
        
        Args:
            retrieval_df: DataFrame with ['source_id', 'target_id', score_column, rank_column, 'is_linked']
            ground_truth_map: Dict of query_id -> set of all true ground-truth linked target_ids
            score_column: Column used for scoring
            rank_column: Rank column
            total_candidates_pool: Total candidate pool size (used to compute % inspection cost)
        """
        if retrieval_df.empty:
            empty_res = {f"Recall@{k}": 0.0 for k in self.k_values}
            empty_res.update({f"Precision@{k}": 0.0 for k in self.k_values})
            empty_res.update({"MAP": 0.0, "MRR": 0.0, "Evaluated_Queries": 0})
            return empty_res

        grouped = retrieval_df.groupby("source_id")

        recalls_at_k = {k: [] for k in self.k_values}
        precisions_at_k = {k: [] for k in self.k_values}
        mrr_list = []
        map_list = []

        for qid, group in grouped:
            true_links = ground_truth_map.get(qid, set())
            if not true_links:
                continue

            sorted_group = group.sort_values(by=rank_column, ascending=True)
            retrieved_targets = sorted_group["target_id"].tolist()
            num_true_total = len(true_links)

            # Compute Recall@K and Precision@K
            for k in self.k_values:
                top_k_targets = set(retrieved_targets[:k])
                num_hits = len(top_k_targets & true_links)

                r_at_k = num_hits / num_true_total if num_true_total > 0 else 0.0
                p_at_k = num_hits / k if k > 0 else 0.0

                recalls_at_k[k].append(r_at_k)
                precisions_at_k[k].append(p_at_k)

            # Compute Reciprocal Rank (RR)
            rr = 0.0
            for rank_idx, tid in enumerate(retrieved_targets, start=1):
                if tid in true_links:
                    rr = 1.0 / rank_idx
                    break
            mrr_list.append(rr)

            # Compute Average Precision (AP)
            hits = 0
            sum_precisions = 0.0
            for rank_idx, tid in enumerate(retrieved_targets, start=1):
                if tid in true_links:
                    hits += 1
                    sum_precisions += hits / rank_idx
            ap = (sum_precisions / min(num_true_total, len(retrieved_targets))) if hits > 0 else 0.0
            map_list.append(ap)

        metrics = {}
        for k in self.k_values:
            metrics[f"Recall@{k}"] = round(float(np.mean(recalls_at_k[k])) if recalls_at_k[k] else 0.0, 4)
            metrics[f"Precision@{k}"] = round(float(np.mean(precisions_at_k[k])) if precisions_at_k[k] else 0.0, 4)

        metrics["MRR"] = round(float(np.mean(mrr_list)) if mrr_list else 0.0, 4)
        metrics["MAP"] = round(float(np.mean(map_list)) if map_list else 0.0, 4)
        metrics["Evaluated_Queries"] = len(mrr_list)

        if total_candidates_pool and total_candidates_pool > 0:
            for k in self.k_values:
                metrics[f"Inspection_Cost@{k}_pct"] = round(float(k / total_candidates_pool * 100), 2)

        return metrics

    def evaluate_pipeline(
        self,
        candidate_df: pd.DataFrame,
        ground_truth_map: Dict[str, Set[str]],
        threshold: float = 0.5,
        score_column: str = "similarity",
        rank_column: str = "rank",
        total_candidates_pool: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs both classification and ranking evaluations on the candidate set.
        
        Args:
            candidate_df: Candidate pair predictions
            ground_truth_map: Query to target true link mapping
            threshold: Fixed uncalibrated decision cutoff for classification metrics
            score_column: Column name containing ranking score
            rank_column: Column name containing rank
            total_candidates_pool: Total repository size for inspection cost calculation
        """
        ranking_metrics = self.compute_ranking_metrics(
            candidate_df,
            ground_truth_map=ground_truth_map,
            score_column=score_column,
            rank_column=rank_column,
            total_candidates_pool=total_candidates_pool
        )

        # Classification metrics at fixed threshold
        if score_column in candidate_df.columns and "is_linked" in candidate_df.columns:
            y_true = candidate_df["is_linked"].to_numpy()
            y_pred = (candidate_df[score_column].to_numpy() >= threshold).astype(int)
            class_metrics = self.compute_classification_metrics(y_true, y_pred, threshold=threshold)
        else:
            class_metrics = {}

        return {
            "ranking": ranking_metrics,
            "classification": class_metrics
        }
