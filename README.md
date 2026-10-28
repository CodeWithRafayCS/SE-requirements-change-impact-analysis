# Large-Scale Requirements Change Impact Analysis Using GPU-Accelerated Dense Semantic Retrieval

**Authors:** Abdul Rafay Akhtar and Madeeha

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CUDA-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4%20(14.56GB)-76B900.svg?style=flat&logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-SEOSS--33%20Benchmark%20(3GB)-blueviolet.svg)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDDZ4Q)
[![Status](https://img.shields.io/badge/Benchmark-100%25%20Completed%20(33%2F33)-brightgreen.svg)]()

This repository contains the runnable codebase, empirical evaluation results, and research manuscript for our large-scale benchmark on **GPU-accelerated requirements Change Impact Analysis (CIA)** across the entire **SEOSS-33 dataset** (33 open-source software project databases from Apache and RedHat).

---

## Executive Summary & Verified Scale

| Benchmark Dimension | Value Across All 33 Open-Source Projects |
|---|---|
| **Evaluated Software Repositories** | **33 complete project databases** (Hadoop, Spark, Cassandra, Flink, Hive, Kafka, Lucene, etc.) |
| **Total Requirements & Issues** | **278,563** historical issue tickets and requirements |
| **Total Change Requests (CRs)** | **46,458** change requests possessing verified dependency links |
| **Evaluated Held-Out Test Queries** | **9,279** test queries evaluated under a strict zero-leakage 20% partition |
| **Verified Ground-Truth Dependency Links** | **96,750** true dependency trace links |
| **High-Risk Architectural Nodes** | **26,417** issues categorized as `Blocker` (11,148) or `Critical` (15,269) |
| **Total GPU Execution Runtime** | **243.15 seconds (~4.05 minutes)** on an NVIDIA Tesla T4 GPU (averaging **7.37s per repository**) |
| **Safety Recall at $\tau = 0.5$** | **93.14% Macro Recall** (unweighted) / **94.65% Micro Recall** (query-weighted) |
| **Early Ranking Discriminability** | **MRR: 0.3048** (first hit at rank ~3.28) • **MAP: 0.2749** • **Precision@1: 22.89%** |

---

## Key Breakthroughs & Architectural Contributions

1. **Eliminating the Quadratic $O(N^2)$ Cartesian Memory Crash:**
   In prior literature, computing similarity over massive candidate pools required generating all-pairs CSV files. On enterprise repositories like **Apache Hadoop** ($N = 39,081$ candidates, $M = 2,042$ test queries), Cartesian expansion produces **$1.52\text{ billion}$ candidate pairs** (~6 GB VRAM), causing immediate out-of-memory crashes. Our chunked matrix multiplication engine ($B = 500$ queries per chunk) evaluates similarities directly on GPU Tensor Cores, keeping peak VRAM strictly under **$500\text{ MB}$** regardless of repository scale.

2. **Severity-Aware Requirement Priority Triage:**
   Unlike prior tools that treat all tickets identically, our pipeline incorporates native Jira issue priority metadata (`Blocker`, `Critical`, `Major`, `Minor`, `Trivial`). By formulating a composite triage score:
   $$S_{\text{triage}}(q, c) = \alpha \cdot S_{\text{cos}}(q, c) + (1 - \alpha) \cdot w_{\text{priority}}(c)$$
   high-severity architectural bottlenecks are prioritized into the developer's immediate top-5 inspection window, ensuring high-risk impact paths are audited first.

3. **Pure Empirical Evaluation (Zero Data Leakage):**
   Operates under a strictly zero-shot retrieval regime without model parameter fine-tuning, guaranteeing absolute freedom from test-set leakage. Threshold-independent ranking metrics (`Recall@K`, `Precision@K`, `MAP`, `MRR`) are strictly decoupled from binary classification metrics (`Accuracy`, `Precision`, `Recall`, `F1`, `F2`).

---

## Main Benchmark Results Across All 33 Projects

| Metric | Macro-Average (Equal Project Weight) | Micro-Average (Query-Weighted) | Empirical Significance & Operational Interpretation |
|---|---|---|---|
| **Recall@1** | **19.79%** | **15.89%** | Nearly 1 in 5 queries finds a true dependency at rank 1. |
| **Recall@3** | **30.36%** | **25.31%** | Over 30% of dependencies captured in top 3 inspection slots. |
| **Recall@5** | **34.10%** | **29.56%** | Over one-third of all dependencies captured within top 5 candidates. |
| **Recall@10** | **41.85%** | **35.91%** | Over 40% of all dependencies retrieved in top 10 positions. |
| **Recall@20** | **48.78%** | **42.35%** | Nearly half of all dependencies retrieved within top 20 candidates. |
| **Precision@1** | **22.89%** | **19.40%** | Nearly 1 in 4 top-ranked recommendations is an exact true link. |
| **Precision@5** | **8.56%** | **7.89%** | Precision at developer inspection budget $K = 5$. |
| **Precision@10** | **5.37%** | **4.94%** | Precision at developer inspection budget $K = 10$. |
| **MRR** | **0.3048** | **0.2676** | First correct dependency appears at rank ~3.28 on average. |
| **MAP** | **0.2749** | **0.2293** | Strong global ranking precision across candidate cutoffs. |
| **Binary Recall ($\tau = 0.5$)** | **93.14%** | **94.65%** | Exceptional safety recall capturing 93.1%–94.7% of all impacts. |
| **Binary F2-Score ($\tau = 0.5$)** | **0.1558** | **0.1414** | Safety-weighted metric ($\beta = 2$) prioritizing recall over precision. |
| **Total Evaluated Projects** | **33** | **33** | 100% of projects successfully evaluated with 0 failures. |
| **Total Evaluated Queries** | **9,279** | **9,279** | Evaluated Change Requests across all held-out test splits. |
| **Total Execution Time** | **243.15s (4.05 min)** | **243.15s (4.05 min)** | Average runtime of 7.37s per repository on Tesla T4. |

---

## Repository Directory Structure

```text
├── research paper.docx                  # Full publication-ready manuscript (1.05 MB)
│
├── seoss_gpu_pipeline/                 # High-throughput GPU retrieval pipeline
│   ├── SEOSS_Full_3GB_GPU_Pipeline.ipynb # Ready-to-run Google Colab Notebook
│   ├── run_full_pipeline.py            # Master CLI execution script
│   ├── requirements.txt                # Python dependencies
│   ├── config.yaml                     # Default execution parameters
│   ├── README.md                       # Detailed GPU pipeline documentation
│   └── src/
│       ├── data_loader.py              # SQLite streaming data loader for all 33 DBs
│       ├── gpu_embedder.py             # PyTorch CUDA FP16 dense embedder (Sentence-BERT)
│       ├── gpu_retriever.py            # Chunked GPU matrix multiplication & top-k engine
│       ├── cross_encoder.py            # Stage-2 cross-encoder re-ranking module
│       ├── evaluator.py                # Ranking & classification empirical metrics engine
│       └── exporter.py                 # Summary aggregator and visualization engine
│
├── SEOSS_Results_Full_33/              # Benchmark output artifacts
│   ├── overall_summary.json            # Macro & micro average metrics in JSON format
│   ├── overall_summary.csv             # Formatted summary metrics table
│   ├── per_project_metrics.csv         # Granular per-project metrics across all 33 repos
│   └── plots/                          # High-resolution publication figures (300 DPI)
│       ├── scale_comparison.png        # Baseline vs. full 33-project scale comparison
│       ├── recall_at_k_curve.png       # Recall@K progression across inspection budgets
│       ├── per_project_comparison.png  # Per-project performance distribution
│       ├── precision_recall_operating_curve.png # Precision-Recall trade-off curve
│       ├── runtime_throughput.png      # O(N) linear runtime scalability scatter plot
│       └── priority_distribution.png   # Requirement Priority distribution & risk tiering
│
├── LICENSE                             # MIT License
└── README.md                           # Main repository documentation
```

---

## Visualizing the Benchmark

The high-resolution plots generated during execution are located in `SEOSS_Results_Full_33/plots/`:

* **`scale_comparison.png`**: Visualizes the **1,325× scale jump** in evaluated test queries and **3,714× jump** in requirements over prior toy baselines.
* **`recall_at_k_curve.png`**: Illustrates Recall@K scaling from $K=1$ ($19.79\%$) to $K=20$ ($48.78\%$).
* **`runtime_throughput.png`**: Confirms $O(N)$ linear scalability of chunked GPU matrix multiplication across all 33 software repositories.
* **`priority_distribution.png`**: Visualizes the distribution of the 278,563 issues across Jira priority levels and risk tiers.

---

## Quick Start & Reproduction

### Option A: 1-Click Execution on Google Colab (Recommended)

1. Open Google Colab and set runtime to **GPU (Tesla T4 or A100)**:  
   *Navigate to `Runtime` > `Change runtime type` > `T4 GPU`*.
2. Upload the `seoss_gpu_pipeline` directory or download from GitHub:
   ```bash
   git clone https://github.com/CodeWithRafayCS/SE-requirements-change-impact-analysis.git
   cd SE-requirements-change-impact-analysis/seoss_gpu_pipeline
   ```
3. Open `SEOSS_Full_3GB_GPU_Pipeline.ipynb` in Colab and execute cells sequentially.

---

### Option B: Local Execution (PC with NVIDIA GPU)

#### 1. Requirements
* Python 3.9, 3.10, or 3.11
* NVIDIA GPU with CUDA support (e.g., RTX 3060, RTX 4080, Tesla T4)
* PyTorch with CUDA:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install -r seoss_gpu_pipeline/requirements.txt
  ```

#### 2. Download the SEOSS Dataset
Download the 33 `.sqlite3` project databases from Harvard Dataverse:
* **DOI:** [10.7910/DVN/PDDZ4Q](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDDZ4Q)
* Place the `.sqlite3` files in `./dataverse_files/`.

#### 3. Run the Benchmark

* **Smoke Test (2 Projects — Archiva & Derby):**
  ```bash
  python seoss_gpu_pipeline/run_full_pipeline.py --data_dir ./dataverse_files --projects archiva,derby --device cuda
  ```

* **Full 33-Project Benchmark Execution:**
  ```bash
  python seoss_gpu_pipeline/run_full_pipeline.py --data_dir ./dataverse_files --projects all --device cuda --batch_size 256 --top_k 20 --threshold 0.5
  ```

---

## Citation & Reference

If you use this pipeline or benchmark in your research, please cite:

```bibtex
@article{rafay2026large,
  title={Large-Scale Empirical Requirements Change Impact Analysis Using GPU-Accelerated Dense Semantic Retrieval: A Comprehensive Benchmark Across 33 Open-Source Software Projects},
  author={Abdul Rafay Akhtar and Madeeha},
  journal={Software Engineering Research Repository},
  year={2026}
}
```

**Dataset Citation:**
* Rath, M., & Mäder, P. (2019). *The SEOSS Dataset - Requirements, Bug Reports, Code History, and Trace Links for Entire Projects*. Harvard Dataverse, DOI: [10.7910/DVN/PDDZ4Q](https://doi.org/10.7910/DVN/PDDZ4Q).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
