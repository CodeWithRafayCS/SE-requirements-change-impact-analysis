"""
Cross-Encoder Re-Ranker for Stage-2 Traceability & Change Impact Analysis.
Scores (query, candidate) pairs using full joint cross-attention on GPU.
"""

import torch
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class GPUCrossEncoder:
    """
    Stage-2 Joint Cross-Attention Re-ranker.
    Only evaluates the candidate pool retrieved by Stage-1, making it fast and memory-efficient.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 256
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        if self._model is None:
            if CrossEncoder is None:
                raise ImportError(
                    "sentence-transformers is not installed. Run: pip install sentence-transformers"
                )
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )

    def rerank_candidates(
        self,
        candidate_df: pd.DataFrame,
        issue_text_map: Dict[str, str],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Takes candidate pairs DataFrame from Stage-1 retrieval,
        computes cross-encoder logits, and re-ranks within each source_id query.
        
        Args:
            candidate_df: DataFrame with ['source_id', 'target_id', 'similarity', 'rank', 'is_linked']
            issue_text_map: Dict mapping issue_id -> text representation
            show_progress: Whether to show progress bar
            
        Returns:
            Updated DataFrame with 'cross_score' and 'reranked_rank' columns.
        """
        if candidate_df.empty:
            return candidate_df

        self._load_model()

        # Build pair text tuples
        pairs = []
        for _, row in candidate_df.iterrows():
            q_text = issue_text_map.get(row["source_id"], "")
            c_text = issue_text_map.get(row["target_id"], "")
            pairs.append((q_text, c_text))

        # Predict in GPU batches
        with torch.no_grad():
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=show_progress
            )

        updated_df = candidate_df.copy()
        updated_df["cross_score"] = scores

        # Re-sort and re-rank within each query
        reranked_rows = []
        for src_id, group in updated_df.groupby("source_id", sort=False):
            sorted_group = group.sort_values(by="cross_score", ascending=False).copy()
            sorted_group["reranked_rank"] = range(1, len(sorted_group) + 1)
            reranked_rows.append(sorted_group)

        result_df = pd.concat(reranked_rows, ignore_index=True)
        return result_df
