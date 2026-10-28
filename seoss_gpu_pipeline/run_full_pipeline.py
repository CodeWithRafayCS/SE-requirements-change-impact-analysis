#!/usr/bin/env python3
"""
Master GPU Execution Pipeline for SEOSS Traceability & Change Impact Analysis.
Executes across any or all 33 SQLite databases in the SEOSS dataset.
Supports local 8GB GPUs and Google Colab (T4/V100/A100).
"""

import os
import sys
import time
import argparse
import logging
import torch
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

# Add current folder to sys.path so src imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import SEOSSDataLoader, ProjectDataset
from src.gpu_embedder import GPUEmbedder
from src.gpu_retriever import GPURetriever
from src.cross_encoder import GPUCrossEncoder
from src.evaluator import SEOSSEvaluator
from src.exporter import MetricsExporter


def load_config_defaults(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """
    Safely loads default parameters from config.yaml if present.
    Returns flattened dictionary of baseline defaults.
    """
    defaults = {
        "data_dir": "./dataverse_files",
        "output_dir": "./results",
        "cache_dir": "./cache/embeddings",
        "device": "auto",
        "batch_size": 256,
        "chunk_size": 500,
        "top_k": 20,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "cross_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross_batch_size": 64,
        "threshold": 0.5
    }

    if not config_path or not os.path.exists(config_path) or yaml is None:
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Extract nested keys
        paths = cfg.get("paths", {})
        hw = cfg.get("hardware", {})
        models = cfg.get("models", {})
        retrieval = cfg.get("retrieval", {})

        if "data_dir" in paths: defaults["data_dir"] = paths["data_dir"]
        if "output_dir" in paths: defaults["output_dir"] = paths["output_dir"]
        if "cache_dir" in paths: defaults["cache_dir"] = paths["cache_dir"]
        if "device" in hw: defaults["device"] = hw["device"]
        if "batch_size" in hw: defaults["batch_size"] = hw["batch_size"]
        if "chunk_size" in hw: defaults["chunk_size"] = hw["chunk_size"]
        if "cross_batch_size" in hw: defaults["cross_batch_size"] = hw["cross_batch_size"]
        if "bi_encoder" in models: defaults["model_name"] = models["bi_encoder"]
        if "cross_encoder" in models: defaults["cross_model_name"] = models["cross_encoder"]
        if "top_k" in retrieval: defaults["top_k"] = retrieval["top_k"]
        if "threshold" in retrieval: defaults["threshold"] = retrieval["threshold"]
    except Exception as e:
        print(f"Note: Could not parse config file '{config_path}': {e}. Using standard defaults.")

    return defaults


def parse_args():
    # Pre-parse --config argument if supplied
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="./config.yaml", help="Path to config.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    # Load baseline defaults from config.yaml
    cfg_defaults = load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="High-Throughput GPU Pipeline for the Full 3GB SEOSS Dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help="Path to YAML configuration file (CLI flags take precedence)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=cfg_defaults["data_dir"],
        help=f"Path to folder containing SEOSS .sqlite3 databases (default: {cfg_defaults['data_dir']})."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=cfg_defaults["output_dir"],
        help=f"Directory to save metric CSVs, JSON summaries, and plots (default: {cfg_defaults['output_dir']})."
    )
    parser.add_argument(
        "--projects",
        type=str,
        default="all",
        help="Comma-separated project names (e.g. 'derby,archiva,camel') or 'all' to run all 33 databases."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg_defaults["device"],
        choices=["auto", "cuda", "cpu"],
        help=f"Compute device: 'auto' (prefers CUDA), 'cuda', or 'cpu' (default: {cfg_defaults['device']})."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=cfg_defaults["batch_size"],
        help=f"Batch size for embedding generation (default: {cfg_defaults['batch_size']})."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=cfg_defaults["chunk_size"],
        help=f"Query chunk size for GPU matrix multiplication (default: {cfg_defaults['chunk_size']})."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=cfg_defaults["top_k"],
        help=f"Number of candidate issues to retrieve per query (default: {cfg_defaults['top_k']})."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=cfg_defaults["model_name"],
        help=f"Pretrained transformer model for dense bi-encoder retrieval (default: {cfg_defaults['model_name']})."
    )
    parser.add_argument(
        "--use_cross_encoder",
        action="store_true",
        help="Enable Stage-2 Cross-Encoder neural re-ranking on top-k candidates."
    )
    parser.add_argument(
        "--cross_model_name",
        type=str,
        default=cfg_defaults["cross_model_name"],
        help=f"Pretrained cross-encoder model for Stage-2 re-ranking (default: {cfg_defaults['cross_model_name']})."
    )
    parser.add_argument(
        "--cross_batch_size",
        type=int,
        default=cfg_defaults["cross_batch_size"],
        help=f"Batch size for Cross-Encoder scoring (default: {cfg_defaults['cross_batch_size']})."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=cfg_defaults["threshold"],
        help=f"Fixed decision threshold for binary classification metrics (uncalibrated zero-shot, default: {cfg_defaults['threshold']})."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=cfg_defaults["cache_dir"],
        help=f"Directory to cache computed embeddings for instant re-runs (default: {cfg_defaults['cache_dir']})."
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable disk caching of embeddings."
    )
    return parser.parse_args()


def setup_device(requested_device: str) -> str:
    if requested_device == "auto":
        target = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target = requested_device

    print("=" * 70)
    print("SEOSS CHANGE IMPACT ANALYSIS GPU PIPELINE")
    print("=" * 70)
    print(f"Target compute device: {target.upper()}")
    if target == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Detected: {gpu_name} ({total_vram:.2f} GB VRAM)")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print("Mixed precision: FP16 Autocast Enabled")
    else:
        print("Notice: Running on CPU. Execution will be slower.")
    print("=" * 70)
    return target


def main():
    args = parse_args()
    device = setup_device(args.device)

    # Initialize modules
    cache_path = None if args.no_cache else args.cache_dir
    embedder = GPUEmbedder(
        model_name=args.model_name,
        device=device,
        use_fp16=True,
        batch_size=args.batch_size,
        cache_dir=cache_path
    )

    retriever = GPURetriever(
        device=device,
        chunk_size=args.chunk_size,
        top_k=args.top_k
    )

    cross_encoder = None
    if args.use_cross_encoder:
        print(f"Stage-2 Cross-Encoder Enabled: {args.cross_model_name}")
        cross_encoder = GPUCrossEncoder(
            model_name=args.cross_model_name,
            device=device,
            batch_size=args.cross_batch_size
        )

    evaluator = SEOSSEvaluator(k_values=[1, 3, 5, 10, 20])
    exporter = MetricsExporter(output_dir=args.output_dir)

    # Resolve Data Path
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        # Check fallback locations (e.g. ../dataverse_files)
        parent_candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataverse_files"))
        if os.path.exists(parent_candidate):
            data_dir = parent_candidate
        else:
            raise FileNotFoundError(f"Could not locate dataset folder at: {args.data_dir}")

    print(f"Dataset root directory: {data_dir}")
    print(f"Classification threshold: {args.threshold} (Fixed, zero-shot uncalibrated)")

    loader = SEOSSDataLoader(data_dir=data_dir)
    all_available = loader.list_available_projects()
    print(f"Found {len(all_available)} project databases in dataset.")

    # Select projects to run
    if args.projects.strip().lower() == "all":
        selected_projects = all_available
    else:
        requested = [p.strip() for p in args.projects.split(",") if p.strip()]
        selected_projects = [p for p in all_available if p in requested]
        if not selected_projects:
            raise ValueError(f"None of the requested projects {requested} were found in {all_available}")

    print(f"Projects queued for execution ({len(selected_projects)}): {', '.join(selected_projects)}")
    print("-" * 70)

    project_results = []
    failed_projects = []
    overall_start_time = time.time()

    for idx, project_name in enumerate(selected_projects, start=1):
        print(f"\n[{idx}/{len(selected_projects)}] Processing project: '{project_name}'")
        p_start = time.time()

        try:
            # 1. Load project data
            dataset: Optional[ProjectDataset] = loader.load_project(project_name)
            
            if dataset is None:
                err_msg = "Database extraction returned None (empty issue table or zero valid issues)"
                print(f"  ✗ Skipping {project_name}: {err_msg}")
                failed_projects.append({"project": project_name, "reason": err_msg})
                continue

            total_issues = len(dataset.issues)
            total_crs = dataset.num_change_requests
            total_links = dataset.num_total_links

            print(f"  - Issues: {total_issues:,} | Change Requests (CRs): {total_crs:,} | Ground-Truth Links: {total_links:,}")

            if total_links == 0:
                err_msg = "No ground-truth issue links available in database"
                print(f"  ✗ Skipping {project_name}: {err_msg}")
                failed_projects.append({"project": project_name, "reason": err_msg})
                continue

            # Ground truth map: source_id -> set(target_ids)
            gt_map = dataset.ground_truth_map

            # Split evaluation: Use test issues that have verified ground-truth links as queries
            # All project issues serve as the candidate impact search pool
            candidate_ids = list(dataset.issues.keys())
            candidate_texts = [dataset.get_issue_text(cid) for cid in candidate_ids]
            
            # Select test queries deterministically from the seeded split
            query_ids = [qid for qid in dataset.test_issue_ids if qid in gt_map]

            if not query_ids:
                err_msg = "Deterministic test split produced 0 change requests with verified links"
                print(f"  ✗ Skipping {project_name}: {err_msg}")
                failed_projects.append({"project": project_name, "reason": err_msg})
                continue

            print(f"  - Candidate pool: {len(candidate_ids):,} issues | Test Queries: {len(query_ids):,}")

            # 2. Encode on GPU
            print("  - Encoding candidates on GPU...")
            cand_embeddings = embedder.encode_texts(
                candidate_texts,
                cache_key=f"{project_name}_candidates_{len(candidate_ids)}",
                show_progress=True
            )

            # Map query embeddings from candidate embeddings
            id_to_cand_idx = {cid: i for i, cid in enumerate(candidate_ids)}
            query_indices = [id_to_cand_idx[qid] for qid in query_ids if qid in id_to_cand_idx]
            query_embeddings = cand_embeddings[query_indices]

            # 3. GPU Top-K Retrieval
            print(f"  - Running GPU Top-{args.top_k} Retrieval (Chunked MatMul)...")
            retrieval_res = retriever.retrieve_top_k(
                query_embeddings=query_embeddings,
                candidate_embeddings=cand_embeddings,
                query_ids=query_ids,
                candidate_ids=candidate_ids,
                filter_self=True,
                top_k=args.top_k,
                show_progress=False
            )

            candidate_df = retriever.flatten_to_dataframe(retrieval_res, gt_map)

            # 4. Optional Stage-2 Cross-Encoder Re-Ranking
            score_col = "similarity"
            rank_col = "rank"
            if cross_encoder is not None and not candidate_df.empty:
                print("  - Stage-2 Cross-Encoder Re-Ranking on GPU...")
                issue_text_map = {cid: dataset.get_issue_text(cid) for cid in candidate_ids}
                candidate_df = cross_encoder.rerank_candidates(
                    candidate_df=candidate_df,
                    issue_text_map=issue_text_map,
                    show_progress=True
                )
                score_col = "cross_score"
                rank_col = "reranked_rank"

            # 5. Evaluate Metrics
            eval_results = evaluator.evaluate_pipeline(
                candidate_df=candidate_df,
                ground_truth_map=gt_map,
                threshold=args.threshold,
                score_column=score_col,
                rank_column=rank_col,
                total_candidates_pool=len(candidate_ids)
            )

            ranking_m = eval_results["ranking"]
            class_m = eval_results["classification"]
            p_elapsed = round(time.time() - p_start, 2)

            project_record = {
                "project": project_name,
                "total_issues": total_issues,
                "total_ground_truth_links": total_links,
                "evaluated_queries": ranking_m.get("Evaluated_Queries", len(query_ids)),
                "candidate_pool_size": len(candidate_ids),
                "elapsed_sec": p_elapsed,
                "classification_threshold": args.threshold,
                "Recall@1": ranking_m.get("Recall@1", 0.0),
                "Recall@3": ranking_m.get("Recall@3", 0.0),
                "Recall@5": ranking_m.get("Recall@5", 0.0),
                "Recall@10": ranking_m.get("Recall@10", 0.0),
                "Recall@20": ranking_m.get("Recall@20", 0.0),
                "Precision@1": ranking_m.get("Precision@1", 0.0),
                "Precision@5": ranking_m.get("Precision@5", 0.0),
                "Precision@10": ranking_m.get("Precision@10", 0.0),
                "MRR": ranking_m.get("MRR", 0.0),
                "MAP": ranking_m.get("MAP", 0.0),
                "Accuracy": class_m.get("Accuracy", 0.0),
                "Precision": class_m.get("Precision", 0.0),
                "Recall": class_m.get("Recall", 0.0),
                "F1": class_m.get("F1", 0.0),
                "F2": class_m.get("F2", 0.0),
            }
            project_results.append(project_record)

            print(f"  ✓ Finished in {p_elapsed}s | Recall@5: {project_record['Recall@5']} | "
                  f"Recall@10: {project_record['Recall@10']} | MAP: {project_record['MAP']} | "
                  f"F1: {project_record['F1']} | F2: {project_record['F2']}")

        except Exception as e:
            err_msg = f"Runtime exception: {str(e)}"
            print(f"  ✗ Error processing project '{project_name}': {e}")
            failed_projects.append({"project": project_name, "reason": err_msg})
            import traceback
            traceback.print_exc()

    # Export all records
    print("\n" + "=" * 70)
    print("AGGREGATING & EXPORTING BENCHMARK RESULTS")
    print("=" * 70)

    if project_results:
        df_all = exporter.save_project_metrics(project_results)
        summary = exporter.save_overall_summary(
            df_all,
            failed_projects=failed_projects,
            total_attempted=len(selected_projects)
        )
        exporter.generate_plots(df_all)

        print(f"\nResults successfully saved to: {os.path.abspath(args.output_dir)}")
        print(f"  - Per-project metrics: {os.path.join(args.output_dir, 'per_project_metrics.csv')}")
        print(f"  - Overall summary:     {os.path.join(args.output_dir, 'overall_summary.csv')}")
        print(f"  - Summary JSON:        {os.path.join(args.output_dir, 'overall_summary.json')}")
        print(f"  - Publication Plots:   {os.path.join(args.output_dir, 'plots')}")
        if failed_projects:
            print(f"  - Failed/Skipped log:  {os.path.join(args.output_dir, 'failed_projects.csv')}")

        # Display Final Summary Table
        print("\n" + "-" * 70)
        print("OVERALL MACRO-AVERAGE PERFORMANCE ACROSS EVALUATED PROJECTS:")
        print("-" * 70)
        macro = summary.get("Macro_Average", {})
        for k, v in macro.items():
            print(f"  {k:25s}: {v}")

        print("-" * 70)
        print("OVERALL MICRO-AVERAGE (QUERY-WEIGHTED) PERFORMANCE:")
        print("-" * 70)
        micro = summary.get("Micro_Average", {})
        for k, v in micro.items():
            print(f"  {k:25s}: {v}")
    else:
        # If no projects succeeded
        exporter.save_overall_summary(
            pd.DataFrame(),
            failed_projects=failed_projects,
            total_attempted=len(selected_projects)
        )

    total_time = round(time.time() - overall_start_time, 2)
    print(f"\nPipeline Run Finished in {total_time} seconds ({total_time/60:.2f} minutes).")

    # Processing Completion Report
    if failed_projects:
        print("\n" + "!" * 70)
        print(f"NOTICE: PARTIAL RUN. {len(failed_projects)} of {len(selected_projects)} projects were skipped or failed:")
        for f in failed_projects:
            print(f"  - {f['project']}: {f['reason']}")
        print(f"Details saved to: {os.path.join(args.output_dir, 'failed_projects.csv')}")
        print("!" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"SUCCESS: All {len(selected_projects)} requested projects were evaluated completely.")
        print("=" * 70)


if __name__ == "__main__":
    main()
