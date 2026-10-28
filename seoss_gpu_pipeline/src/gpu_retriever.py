"""
GPU-Accelerated Candidate Retrieval for SEOSS Change Impact Analysis.
Performs chunked GPU matrix multiplication and top-k retrieval without OOM.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class GPURetriever:
    """
    High-performance Bi-Encoder retriever using chunked PyTorch matrix multiplication.
    Evaluates query embeddings against the candidate pool in chunks (default: 500)
    to avoid allocating large dense similarity matrices in GPU VRAM.
    """

    def __init__(
        self,
        device: str = "auto",
        chunk_size: int = 500,
        top_k: int = 20
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.chunk_size = chunk_size
        self.top_k = top_k

    def retrieve_top_k(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        query_ids: List[str],
        candidate_ids: List[str],
        filter_self: bool = True,
        top_k: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Retrieves top-k most similar candidates for each query.
        
        Args:
            query_embeddings: Tensor of shape (num_queries, dim)
            candidate_embeddings: Tensor of shape (num_candidates, dim)
            query_ids: List of issue IDs corresponding to query_embeddings
            candidate_ids: List of issue IDs corresponding to candidate_embeddings
            filter_self: Whether to eliminate self-matches (query_id == candidate_id)
            top_k: Number of candidates to retrieve per query (defaults to self.top_k)
            show_progress: Show tqdm progress bar
            
        Returns:
            List of dicts: [
                {
                    "query_id": str,
                    "top_candidates": List[str],
                    "top_scores": List[float],
                    "top_ranks": List[int]
                },
                ...
            ]
        """
        k = top_k or self.top_k
        k = min(k, len(candidate_ids) - (1 if filter_self else 0))
        if k <= 0:
            return []

        num_queries = query_embeddings.shape[0]
        candidate_id_arr = np.array(candidate_ids)

        # Move candidate embeddings to the compute device
        cands_dev = candidate_embeddings.to(self.device).float()
        # Ensure candidate embeddings are normalized for cosine similarity
        cands_dev = torch.nn.functional.normalize(cands_dev, p=2, dim=1)
        cands_t = cands_dev.T  # (dim, num_candidates)

        candidate_id_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids)}

        results = []
        iterator = range(0, num_queries, self.chunk_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Retrieving candidates (GPU chunked)", leave=False)

        for start_idx in iterator:
            end_idx = min(start_idx + self.chunk_size, num_queries)
            q_chunk = query_embeddings[start_idx:end_idx].to(self.device).float()
            q_chunk = torch.nn.functional.normalize(q_chunk, p=2, dim=1)

            # Cosine similarity matrix: (chunk_size, num_candidates)
            sim_matrix = torch.matmul(q_chunk, cands_t)

            # Mask out self-references if query is in candidate pool
            if filter_self:
                for row_offset, q_global_idx in enumerate(range(start_idx, end_idx)):
                    qid = query_ids[q_global_idx]
                    if qid in candidate_id_to_idx:
                        c_idx = candidate_id_to_idx[qid]
                        sim_matrix[row_offset, c_idx] = -1e9

            # Retrieve top-k on GPU
            chunk_k = min(k, sim_matrix.shape[1])
            top_scores, top_indices = torch.topk(sim_matrix, k=chunk_k, dim=1, largest=True)

            top_scores_cpu = top_scores.cpu().numpy()
            top_indices_cpu = top_indices.cpu().numpy()

            for i, q_global_idx in enumerate(range(start_idx, end_idx)):
                qid = query_ids[q_global_idx]
                retrieved_cids = candidate_id_arr[top_indices_cpu[i]].tolist()
                retrieved_scores = top_scores_cpu[i].tolist()
                ranks = list(range(1, len(retrieved_cids) + 1))

                results.append({
                    "query_id": qid,
                    "top_candidates": retrieved_cids,
                    "top_scores": retrieved_scores,
                    "top_ranks": ranks
                })

            # Free chunk memory
            del q_chunk, sim_matrix, top_scores, top_indices

        # Clean GPU memory
        del cands_dev, cands_t
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return results

    @staticmethod
    def flatten_to_dataframe(
        retrieval_results: List[Dict],
        ground_truth_map: Dict[str, set]
    ) -> pd.DataFrame:
        """
        Flattens top-k retrieval results into a tabular DataFrame for evaluation/training.
        Adds ground-truth binary label (is_linked: 0 or 1).
        """
        rows = []
        for item in retrieval_results:
            qid = item["query_id"]
            true_targets = ground_truth_map.get(qid, set())

            for cid, score, rank in zip(item["top_candidates"], item["top_scores"], item["top_ranks"]):
                is_linked = 1 if (cid in true_targets) else 0
                rows.append({
                    "source_id": qid,
                    "target_id": cid,
                    "similarity": float(score),
                    "rank": int(rank),
                    "is_linked": is_linked
                })

        return pd.DataFrame(rows)
