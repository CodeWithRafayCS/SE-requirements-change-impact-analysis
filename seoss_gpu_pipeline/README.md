# SEOSS Dataset: High-Throughput GPU Pipeline for Traceability & Change Impact Analysis

A production-grade, GPU-optimized pipeline designed to process the complete **3GB SEOSS Dataset** (*The SEOSS Dataset - Requirements, Bug Reports, Code History, and Trace Links for Entire Projects* by Rath & Mäder).

---

## Key Capabilities

1. **Full 3GB / 33 Project Database Support:** Directly queries all 33 SQLite `.sqlite3` databases without needing intermediate Cartesian CSV conversions or uncompressed data bloat.
2. **GPU Optimization for 8GB VRAM & Google Colab:**
   - **FP16 Mixed Precision Autocast:** Uses `torch.cuda.amp.autocast()` during embedding generation to utilize Tensor Core acceleration while preserving numerical stability in transformer normalization layers.
   - **Chunked GPU Matrix Multiplication:** Evaluates cosine similarities in manageable query chunks (default: 500) via `torch.matmul` and retrieves top candidates directly on GPU via `torch.topk`, bypassing the quadratic Cartesian pair memory bottleneck ($1.5\text{ Billion}$ candidate rows on large repositories like Hadoop).
   - **Embedding Disk Caching:** Computed embeddings can be safely cached to disk (`.pt`) so re-evaluations and parameter exploration run without re-encoding.
3. **Pure Empirical SE Metrics (Zero Hypothesis Testing):**
   - **Ranking & Retrieval (Threshold-Independent):** `Recall@1`, `Recall@3`, `Recall@5`, `Recall@10`, `Recall@20`, `Precision@K`, `MAP` (Mean Average Precision), `MRR` (Mean Reciprocal Rank).
   - **Binary Classification (At Explicit Threshold):** Evaluates `Accuracy`, `Precision`, `Recall`, `F1-Score`, `F2-Score` ($\beta=2$ prioritizing recall for safety-critical impact analysis), `Specificity`, `Balanced Accuracy` at an explicit, uncalibrated decision cutoff (default: `0.5`).
   - **Inspection Cost:** Candidate effort percentage ($K / \text{Total Repository Issues}$).
4. **Resilient & Transparent Reporting:**
   - Evaluates projects deterministically using seeded train/validation/test holdouts.
   - Accurately reports individual ground-truth links (`dataset.num_total_links`) separate from change request source counts.
   - If any project database contains no valid links or queries, it is recorded in `failed_projects.csv` with a clear explanation, preventing silent omission.
   - Summary tables strictly isolate evaluation metrics from operational metadata (e.g., elapsed time, pool size) during micro- and macro-averaging.

---

## Directory Structure

```text
seoss_gpu_pipeline/
│
├── SEOSS_Full_3GB_GPU_Pipeline.ipynb   # Google Colab Ready-to-Run Notebook
├── run_full_pipeline.py                 # Master CLI script
├── requirements.txt                     # Python dependencies
├── config.yaml                          # Default parameters (CLI flags override)
├── README.md                            # Documentation and execution guide
│
└── src/
    ├── __init__.py
    ├── data_loader.py                   # High-throughput SQLite loader for all 33 DBs
    ├── gpu_embedder.py                  # PyTorch CUDA FP16 dense embedder
    ├── gpu_retriever.py                 # Chunked GPU matrix multiplication & top-k
    ├── cross_encoder.py                 # Stage-2 neural cross-attention re-ranker
    ├── evaluator.py                     # Empirical metric computation engine
    └── exporter.py                      # CSV, JSON, and publication plot generator
```

---

## Local Setup (PC with NVIDIA GPU)

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12/3.13.
- NVIDIA GPU with CUDA driver installed.

### 2. Install Dependencies
For local GPU execution, ensure that PyTorch is installed with CUDA support from the official PyTorch wheel repository:
```bash
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies:
pip install -r requirements.txt
```

### 3. Execution Commands

**Quick Smoke Test (2 projects):**
```bash
python run_full_pipeline.py --data_dir ./dataverse_files --projects archiva,derby --device cuda
```

**Run Full 33 Projects (All 3GB):**
```bash
python run_full_pipeline.py --data_dir ./dataverse_files --projects all --device cuda --batch_size 256 --top_k 20 --threshold 0.5
```

**Run with Stage-2 Neural Cross-Encoder Re-Ranking:**
```bash
python run_full_pipeline.py --data_dir ./dataverse_files --projects all --device cuda --use_cross_encoder
```

---

## Google Colab Execution

1. Upload the `seoss_gpu_pipeline` folder or `SEOSS_Full_GPU_Pipeline.zip` to your Google Drive or Colab session.
2. Open `SEOSS_Full_3GB_GPU_Pipeline.ipynb` in Google Colab.
3. In Colab menu, select **Runtime > Change runtime type > GPU (T4 or A100)**.
4. Run the notebook cells sequentially:
   - **Step 1:** Verifies CUDA GPU and VRAM.
   - **Step 2:** Installs dependencies (Colab has CUDA PyTorch pre-installed).
   - **Step 3:** Detects the pipeline directory and locates `dataverse_files`.
   - **Step 4 & 5:** Executes smoke test and full 33-project benchmark.
   - **Step 6:** Displays result tables and plots inline.
   - **Step 7:** Downloads a zip archive containing all generated reports and figures.

---

## CLI & Configuration Reference

Parameters can be configured in `config.yaml` or overridden directly via CLI arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | `./config.yaml` | Path to YAML configuration file (CLI flags take precedence). |
| `--data_dir` | `str` | `./dataverse_files` | Directory containing the 33 SEOSS `.sqlite3` files. |
| `--output_dir` | `str` | `./results` | Target folder for output tables, JSON, and plots. |
| `--projects` | `str` | `all` | Comma-separated list (e.g. `derby,camel`) or `all`. |
| `--device` | `str` | `auto` | `auto` (CUDA if available), `cuda`, or `cpu`. |
| `--batch_size` | `int` | `256` | Batch size for GPU embedding generation. |
| `--chunk_size` | `int` | `500` | Query chunk size for GPU matrix multiplication. |
| `--top_k` | `int` | `20` | Candidate pool size retrieved per query. |
| `--threshold` | `float` | `0.5` | Fixed decision cutoff for binary classification metrics (uncalibrated zero-shot). |
| `--model_name` | `str` | `sentence-transformers/all-MiniLM-L6-v2` | Dense transformer embedding model. |
| `--use_cross_encoder` | `flag` | `False` | Enables Stage-2 joint cross-attention re-ranking. |
| `--cross_model_name` | `str` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for Stage-2 re-ranking. |
| `--cache_dir` | `str` | `./cache/embeddings` | Directory to cache computed embeddings. |
| `--no_cache` | `flag` | `False` | Disables disk caching of embeddings. |

---

## Output Artifacts

Upon completion, all outputs are stored in `--output_dir`:
- `per_project_metrics.csv`: Full breakdown for every project (Issues, Links, Evaluated Queries, Recall@1..20, Precision@1..10, MAP, MRR, Accuracy, Precision, Recall, F1, F2).
- `overall_summary.csv`: Macro-Average (equal project weight) and Micro-Average (query-weighted) metrics across evaluated projects.
- `overall_summary.json`: Structured machine-readable benchmark summary, including run status (`COMPLETE` or `PARTIAL`) and attempted/evaluated counts.
- `failed_projects.csv`: Detailed log of any skipped or failed projects with explicit failure reasons (generated only if issues occur).
- `plots/recall_at_k_curve.png`: Recall progression from K=1 to K=20.
- `plots/per_project_comparison.png`: Performance comparison across F1, MAP, and Recall@10 across projects.
- `plots/precision_recall_operating_curve.png`: Trade-off curve between precision and recall across inspection budgets.
