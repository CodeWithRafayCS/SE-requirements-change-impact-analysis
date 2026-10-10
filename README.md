# AI-Based Requirements Change Impact Analysis — Experiment Pipeline

This is the runnable implementation of the methodology in Section 6 of the research
paper. It reproduces Tables 8–13. Everything runs locally, CPU is enough (GPU speeds
up the NLI step).

## 0. About the dataset situation (read this first)

I looked for the real datasets used in Etezadi et al. (2026):

- **SAT-DLink** (the industrial dataset) is **not publicly released** — it is
  proprietary to that study.
- **WASP** (Goknil et al., 2014) is described in the paper as "publicly
  available," but I could not find an actual downloadable file, repository, or
  supplementary-materials link for it. Your best options are to (a) email the
  Etezadi et al. authors directly asking for the replication package (their
  contact is on the paper), or (b) check the ACM TOSEM article's official
  supplementary materials page once you have journal access.
- **SEOSS-33** (the paper's own designated fallback, Section 6.2.5) **is
  genuinely downloadable**: Harvard Dataverse, CC0 license,
  DOI [10.7910/DVN/PDDZ4Q](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDDZ4Q).
  It gives you real open-source requirements/issues and trace links, but **not**
  "change rationale → impacted requirements" labels in the same form — you'd
  build proxy labels from issue/requirement trace links, exactly the caveat
  already written into the paper's Section 6.2.5.

**Because of this, `data/pairs_SYNTHETIC_DEMO.csv` is included so you can smoke-test
the whole pipeline today.** It is a hand-built, clearly synthetic SRS: 52
requirements across 8 modules (auth, catalog, cart, payment, order,
notification, admin, reporting) and 15 change requests, where impacted
requirements are deliberately correlated with each change's primary module
and named cross-cutting dependencies (e.g. a payment-gateway change also
touches order and notification), so TF-IDF/cosine/NLI baselines behave
sensibly rather than looking at pure noise.

**Do not report numbers from this file in your paper.** It exists only so you
can verify the code runs correctly end to end before you have real data. Once
you get SEOSS-33 or the real WASP/SAT-DLink files, replace `data/pairs.csv`
and re-run.

A ready-made `config_demo.yaml` points at the demo file and lowers the
small-dataset threshold so it takes the ordinary holdout-split path (with a
real 15-CR dataset, the pipeline would otherwise correctly switch to 5-fold
CV, whose tuning step is intentionally left as a stub — see Section 4 below).
To smoke-test:

```bash
python run_pipeline.py --all --config config_demo.yaml
```

## 1. Install

```bash
pip install -r requirements.txt --break-system-packages
```

## 2. Put your real data in `data/pairs.csv`

You must first convert whatever the Etezadi et al. (or SEOSS-33 fallback) replication
package gives you into **one flat CSV** with exactly these columns:

| column             | meaning                                                              |
|--------------------|-----------------------------------------------------------------------|
| `cr_id`            | change-request identifier (same value for every candidate of one CR) |
| `change_text`      | the change rationale text for that CR                                |
| `requirement_id`   | candidate requirement's ID                                           |
| `requirement_text` | candidate requirement's text                                         |
| `label`            | 1 if this requirement is truly impacted by this CR, else 0            |
| `req_type`         | optional — module/section prefix (e.g. "SEC"), blank if not available |

**Every requirement in the project must appear as a candidate row for every CR** in
that project (i.e. this file is the full cross product of CRs × requirements, not
just the positives) — the pipeline needs the negatives to compute recall/precision/cost.

Look at `src/data_prep.py::EXAMPLE_ADAPTER` for a template showing how to turn a
typical "list of impacted IDs per CR" format into this flat file — you will almost
certainly need to edit that one function for whatever the real file structure turns
out to be. This is the only file-format-specific piece of the whole pipeline.

If you fall back to SEOSS-33, build `label` from requirement-to-requirement trace/
dependency links instead of expert impact annotations, and keep the caveat from
Section 6.2.5 in your report — it is a proxy label, not the same ground truth used
by Etezadi et al.

## 3. Run everything

```bash
python run_pipeline.py --all
```

This runs steps 01–10 from Appendix A in order and writes:

- `results/table8_main_results.csv`
- `results/table9_matched_cost.csv`
- `results/table10_signal_quality.csv`
- `results/table11_hypothesis_tests.csv`
- `results/table12_sensitivity.csv`
- `results/error_analysis_sample.csv`
- `results/*.png` (recall-vs-cost curve, PR curve, NLI score distributions)
- `splits/seed_*.json` (CR IDs per split, for reproducibility)
- `config_used.yaml` (frozen parameters, for the appendix)

### Mapping of Generated CSV Files to Paper Tables

The experiment scripts export filenames based on the project's internal stage numbering, which map directly to the research paper tables and sections:

| Generated CSV File | Paper Table / Section | Content Description |
|---|---|---|
| `results/table8_main_results.csv` | **Table 6** | Empirical performance of baselines and hybrid configurations on held-out test set |
| `results/table9_matched_cost.csv` | **Table 7** | Matched-cost recall comparison across fixed inspection budgets (Top 1, 2, 3, 4, 8) |
| `results/table10_signal_quality.csv` | **Table 8** | Stage-1 candidate score discriminability metrics (AUROC and AUPRC) |
| `results/table11_hypothesis_tests.csv` | **Table 9** | Formal pre-specified hypothesis testing outcomes (H1a, H1, H2, H3a, H3b, H4) |
| `results/table12_sensitivity.csv` | **Table 10** | Sensitivity analysis: Calibrated threshold ($\tau = 0.225$) vs. fixed threshold ($\tau = 0.500$) |
| `results/tau_calibration_diagnostics.csv` | **Table 11** | Stage-1 threshold ($\tau$) calibration diagnostics on validation data |
| `results/domain_annotation_experiment.csv` | **Section 6.7** | Exploratory component-level domain annotation diagnosis (`MRM-1050` $\to$ `MRM-1068`) |
| `results/error_analysis_sample.csv` | **Section 6.7** | Sampled false negative and false positive failure cases |

Copy the numbers straight into the paper's tables — every column name in the CSVs
matches a table column.

You can also run steps individually, matching Appendix A:

```bash
python run_pipeline.py --step 01_prepare_data
python run_pipeline.py --step 02_embed
python run_pipeline.py --step 03_baselines_val
python run_pipeline.py --step 04_nli_scores
python run_pipeline.py --step 05_features
python run_pipeline.py --step 06_tune
python run_pipeline.py --step 07_test_eval
python run_pipeline.py --step 08_stats
python run_pipeline.py --step 09_sensitivity_and_errors
python run_pipeline.py --step 10_report
```

Re-running a step reuses cached embeddings/NLI scores from `cache/` (they are the
slow part) unless you pass `--no-cache`.

## 4. Config

All tunable settings (grids, thresholds, seeds, model names, dataset size cutoffs)
live in `config.yaml` (`config_demo.yaml` is the same file pointed at the synthetic
dataset, for smoke-testing). Nothing else in the code should need editing except
the data adapter in step 2.

## 5. Small-dataset fallback

If your real dataset has fewer than `small_dataset_cr_threshold` (default 30)
change requests — likely, since WASP has 5 CRs and SAT-DLink has 5–11 —
`run_pipeline.py` switches from a fixed 60/20/20 split to 5-fold cross-validation
over change requests (Section 6.2.4). **`step_06` (tuning) currently raises
`NotImplementedError` on this path** rather than faking a shortcut — nested
per-fold tuning needs a small amount of extra code once you know your actual CR
count. Tell me the number and I'll fill it in.

## 6. What is NOT included

- Downloading the actual dataset (you must do that manually — see Section 6.2.5
  of the paper for the deadline/fallback logic).
- Fine-tuning the NLI model (mentioned as optional in Section 6.6.4 — a stub is
  provided in `src/nli_finetune_stub.py` if you have time for it).
- Re-running ProReFiCIA — its numbers are only read from `config.yaml` as
  published constants for the contextual comparison rows.
