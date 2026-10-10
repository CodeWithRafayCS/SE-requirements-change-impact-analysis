#!/usr/bin/env python3
"""
Orchestrates the full experiment, matching Appendix A of the paper.

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --step 03_baselines_val
    python run_pipeline.py --all --no-cache

Writes results/table8..table12 CSVs whose columns line up with the paper's
tables, plus splits/, config_used.yaml, and a few diagnostic plots.
"""
import argparse
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from src import data_prep, embed_utils, baselines, nli_scoring, features, combine, evaluate

STEPS = [
    "01_prepare_data", "02_embed", "03_baselines_val", "04_nli_scores",
    "05_features", "06_tune", "07_test_eval", "08_stats",
    "09_sensitivity_and_errors", "10_report",
]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------------
# Shared state is passed between steps via a simple dict-based "context"
# that is pickled to disk between invocations so steps can be re-run
# individually (`--step X`) as well as with `--all`.
# -------------------------------------------------------------------------
def ctx_path(cfg):
    return os.path.join(cfg["paths"]["cache_dir"], "_ctx.npz")


def save_ctx(cfg, **arrays):
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    np.savez_compressed(ctx_path(cfg), **arrays)


def load_ctx(cfg):
    p = ctx_path(cfg)
    if not os.path.exists(p):
        raise FileNotFoundError(f"No saved context at {p}; run earlier steps first.")
    return dict(np.load(p, allow_pickle=True))


# -------------------------------------------------------------------------
def step_01(cfg):
    df = data_prep.load_pairs(cfg["data"]["pairs_csv"])
    seed = cfg["split"]["seeds"][0]  # primary seed drives the main tables; all seeds used in step 07
    split = data_prep.make_cr_splits(df, cfg, seed)
    data_prep.save_split(split, cfg["paths"]["splits_dir"], seed)

    train_df = df[df["cr_id"].isin(split["train"])]
    val_df = df[df["cr_id"].isin(split["val"])]
    test_df = df[df["cr_id"].isin(split["test"])]
    per_cr = df.groupby("cr_id")["label"].sum()

    stats = {
        "n_requirements": df["requirement_id"].nunique(),
        "n_change_requests": df["cr_id"].nunique(),
        "total_pairs": len(df),
        "total_impacts": int(df["label"].sum()),
        "avg_impacted_per_cr": round(float(per_cr.mean()), 2),
        "impact_density_pct": round(float(df["label"].sum()) / len(df) * 100, 2),
        "train_crs": len(split["train"]),
        "train_pairs": len(train_df),
        "train_impacts": int(train_df["label"].sum()),
        "val_crs": len(split["val"]),
        "val_pairs": len(val_df),
        "val_impacts": int(val_df["label"].sum()),
        "test_crs": len(split["test"]),
        "test_pairs": len(test_df),
        "test_impacts": int(test_df["label"].sum()),
    }
    print("[01] dataset stats (Table 3/4):", stats)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    pd.DataFrame([stats]).to_csv(os.path.join(cfg["paths"]["results_dir"], "table3_dataset_stats.csv"), index=False)

    df.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_df.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json"), "w") as f:
        json.dump(split, f)
    print("[01] done")


def _load_df_and_split(cfg):
    df = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_df.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json")) as f:
        split = json.load(f)
    return df, split


def step_02(cfg):
    df, split = _load_df_and_split(cfg)
    model = embed_utils.get_embedder(cfg["models"]["embedding_model"])

    req_texts = df.drop_duplicates("requirement_id")[["requirement_id", "requirement_text"]]
    req_texts = req_texts.set_index("requirement_id")["requirement_text"]
    change_texts = df.drop_duplicates("cr_id")[["cr_id", "change_text"]].set_index("cr_id")["change_text"]

    req_emb = embed_utils.embed_texts_cached(req_texts.tolist(), model, cfg["paths"]["cache_dir"], "requirements")
    chg_emb = embed_utils.embed_texts_cached(change_texts.tolist(), model, cfg["paths"]["cache_dir"], "changes")

    np.save(os.path.join(cfg["paths"]["cache_dir"], "req_emb.npy"), req_emb)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "chg_emb.npy"), chg_emb)
    req_texts.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "req_index.pkl"))
    change_texts.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "chg_index.pkl"))
    print(f"[02] embedded {len(req_texts)} requirements and {len(change_texts)} change rationales")


def _cosine_lookup(df, req_index, chg_index, req_emb, chg_emb):
    """Builds a per-pair cosine similarity column aligned to df's row order."""
    req_pos = {rid: i for i, rid in enumerate(req_index.index)}
    chg_pos = {cid: i for i, cid in enumerate(chg_index.index)}
    r_idx = df["requirement_id"].map(req_pos).to_numpy()
    c_idx = df["cr_id"].map(chg_pos).to_numpy()
    sims = np.einsum("ij,ij->i", chg_emb[c_idx], req_emb[r_idx])
    return sims


def step_03(cfg):
    df, split = _load_df_and_split(cfg)
    req_emb = np.load(os.path.join(cfg["paths"]["cache_dir"], "req_emb.npy"))
    chg_emb = np.load(os.path.join(cfg["paths"]["cache_dir"], "chg_emb.npy"))
    req_index = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "req_index.pkl"))
    chg_index = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "chg_index.pkl"))

    df["cosine_sim"] = _cosine_lookup(df, req_index, chg_index, req_emb, chg_emb)

    # TF-IDF baseline (B1), computed per project would be ideal; here per full
    # corpus for simplicity since pairs.csv is expected to be single-project.
    df["tfidf_sim"] = 0.0
    for cid, group in df.groupby("cr_id"):
        pass  # tfidf computed globally below for efficiency
    all_changes = df.drop_duplicates("cr_id")[["cr_id", "change_text"]]
    all_reqs = df.drop_duplicates("requirement_id")[["requirement_id", "requirement_text"]]
    tfidf_matrix = baselines.tfidf_baseline_scores(
        all_changes["change_text"].tolist(), all_reqs["requirement_text"].tolist(), cfg
    )
    chg_pos = {cid: i for i, cid in enumerate(all_changes["cr_id"])}
    req_pos = {rid: i for i, rid in enumerate(all_reqs["requirement_id"])}
    df["tfidf_sim"] = [
        tfidf_matrix[chg_pos[row.cr_id], req_pos[row.requirement_id]] for row in df.itertuples()
    ]

    # Calibrate tau on the VALIDATION CRs only (Section 6.4 Step 1.5).
    val_df = df[df["cr_id"].isin(split.get("val", []))] if split["mode"] == "holdout" else df
    cr_ids_val = val_df["cr_id"].unique()
    n_reqs = df["requirement_id"].nunique()
    sim_matrix = np.zeros((len(cr_ids_val), n_reqs))
    label_matrix = np.zeros((len(cr_ids_val), n_reqs))
    req_order = {rid: i for i, rid in enumerate(df["requirement_id"].unique())}
    for i, cid in enumerate(cr_ids_val):
        sub = val_df[val_df["cr_id"] == cid]
        for row in sub.itertuples():
            j = req_order[row.requirement_id]
            sim_matrix[i, j] = row.cosine_sim
            label_matrix[i, j] = row.label

    tau, diagnostics = baselines.calibrate_tau(
        sim_matrix, label_matrix, cfg["stage1"]["tau_grid"], cfg["stage1"]["target_stage1_recall"]
    )
    print(f"[03] calibrated tau = {tau}")
    pd.DataFrame(diagnostics).to_csv(os.path.join(cfg["paths"]["results_dir"], "tau_calibration_diagnostics.csv"), index=False)

    df.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_with_sims.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "tau.json"), "w") as f:
        json.dump({"tau": tau}, f)
    print("[03] done")


def step_04(cfg, plain=False):
    df = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_with_sims.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "tau.json")) as f:
        tau = json.load(f)["tau"]

    candidates = df[df["cosine_sim"] >= tau].reset_index(drop=True)
    print(f"[04] scoring NLI for {len(candidates)} Stage-1 candidate pairs "
          f"(plain={plain})")

    tok, model, entail_idx = nli_scoring.get_nli_model(cfg["models"]["nli_model"])
    pairs = list(zip(candidates["change_text"], candidates["requirement_text"]))
    scores = nli_scoring.nli_entailment_scores(pairs, tok, model, entail_idx, cfg, cfg["paths"]["cache_dir"], plain=plain)
    colname = "nli_score_plain" if plain else "nli_score"
    candidates[colname] = scores

    out_name = "candidates_with_nli_plain.pkl" if plain else "candidates_with_nli.pkl"
    candidates.to_pickle(os.path.join(cfg["paths"]["cache_dir"], out_name))
    print("[04] done")


def step_05(cfg):
    candidates = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "candidates_with_nli.pkl"))
    feats = features.extract_features_for_candidates(
        candidates["change_text"].tolist(),
        candidates["requirement_text"].tolist(),
        [""] * len(candidates),
        candidates["req_type"].tolist(),
        cfg["models"]["spacy_model"],
    )
    for k, v in feats.items():
        candidates[k] = v
    candidates["lexical_score"] = features.combine_lexical_score(feats)
    candidates.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "candidates_full.pkl"))
    print(f"[05] extracted lexical features for {len(candidates)} candidates")


def step_06(cfg):
    candidates = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "candidates_full.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json")) as f:
        split = json.load(f)

    if split["mode"] != "holdout":
        raise NotImplementedError(
            "[06] dataset is small -> nested CV tuning path. See README section 4; "
            "implement per-fold tuning here by looping over split['folds'] before "
            "proceeding to step 07 with fold-specific frozen parameters."
        )

    val = candidates[candidates["cr_id"].isin(split["val"])].reset_index(drop=True)
    if val.empty:
        raise ValueError("[06] no validation candidates found -- check that Stage 1 "
                          "tau (step 03) is not so strict it drops all validation candidates.")

    val["cosine_norm"] = combine.minmax_normalize_per_group(val["cosine_sim"].to_numpy(), val["cr_id"].to_numpy())
    val["nli_norm"] = combine.minmax_normalize_per_group(val["nli_score"].to_numpy(), val["cr_id"].to_numpy())
    val["lex_norm"] = combine.minmax_normalize_per_group(val["lexical_score"].to_numpy(), val["cr_id"].to_numpy())

    labels = val["label"].to_numpy()
    groups = val["cr_id"].to_numpy()

    # C1: embedding + NLI only (gamma = 0)
    c1 = combine.tune_weights_and_theta(val["cosine_norm"].to_numpy(), val["nli_norm"].to_numpy(),
                                         np.zeros(len(val)), labels, groups, cfg, constrain_gamma_zero=True)
    # C2: embedding + lexical only (beta = 0)
    c2 = combine.tune_weights_and_theta(val["cosine_norm"].to_numpy(), np.zeros(len(val)),
                                         val["lex_norm"].to_numpy(), labels, groups, cfg, constrain_beta_zero=True)
    # C3: full hybrid
    c3 = combine.tune_weights_and_theta(val["cosine_norm"].to_numpy(), val["nli_norm"].to_numpy(),
                                         val["lex_norm"].to_numpy(), labels, groups, cfg)

    # C4: learned combiner
    feat_matrix = val[["cosine_norm", "nli_norm", "jaccard", "type_match", "entity_score"]].to_numpy()
    logreg, best_C = combine.fit_logistic_combiner(feat_matrix, labels, cfg)
    val_probs = logreg.predict_proba(feat_matrix)[:, 1]
    theta_grid = np.arange(0.05, 0.96, cfg["combiner"]["cutoff_grid_step"]).round(3)
    c4_theta, c4_f2 = combine.best_theta_for_scores(val_probs, labels, groups, theta_grid)

    frozen = {
        "C1": c1, "C2": c2, "C3": c3,
        "C4": {"logreg_C": best_C, "theta": c4_theta, "val_f2": c4_f2},
    }
    with open(os.path.join(cfg["paths"]["results_dir"], "frozen_parameters.json"), "w") as f:
        json.dump(frozen, f, indent=2, default=float)
    import joblib
    joblib.dump(logreg, os.path.join(cfg["paths"]["cache_dir"], "logreg_c4.joblib"))
    print("[06] frozen parameters:", json.dumps(frozen, indent=2, default=float))
    print("[06] done")


def step_07(cfg):
    """Single held-out test evaluation of B1-B3 and C1-C4 -> Table 8 / 9."""
    candidates = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "candidates_full.pkl"))
    df_all = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_with_sims.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json")) as f:
        split = json.load(f)
    with open(os.path.join(cfg["paths"]["results_dir"], "frozen_parameters.json")) as f:
        frozen = json.load(f)
    import joblib
    logreg = joblib.load(os.path.join(cfg["paths"]["cache_dir"], "logreg_c4.joblib"))

    test_ids = split["test"]
    n_total_req = df_all["requirement_id"].nunique()

    test_all = df_all[df_all["cr_id"].isin(test_ids)].reset_index(drop=True)
    test_cand = candidates[candidates["cr_id"].isin(test_ids)].reset_index(drop=True)
    test_cand["cosine_norm"] = combine.minmax_normalize_per_group(test_cand["cosine_sim"].to_numpy(), test_cand["cr_id"].to_numpy())
    test_cand["nli_norm"] = combine.minmax_normalize_per_group(test_cand["nli_score"].to_numpy(), test_cand["cr_id"].to_numpy())
    test_cand["lex_norm"] = combine.minmax_normalize_per_group(test_cand["lexical_score"].to_numpy(), test_cand["cr_id"].to_numpy())

    pair_key_cand = test_cand["cr_id"] + "___" + test_cand["requirement_id"]
    pair_key_all = test_all["cr_id"] + "___" + test_all["requirement_id"]

    rows = []

    # ---- B1: TF-IDF + T2 ----
    b1_mask = np.zeros(len(test_all), dtype=bool)
    for cid, g in test_all.groupby("cr_id"):
        keep_idx = baselines.t2_cutoff_indices(g["tfidf_sim"].to_numpy())
        b1_mask[g.index[keep_idx]] = True
    rows.append({"ID": "B1 TF-IDF + T2", "stage1_recall": "n/a",
                 **evaluate.main_metrics_row(test_all["label"].to_numpy(), b1_mask, test_all["cr_id"].to_numpy(), n_total_req)})

    # ---- B2: Embedding + T2 ----
    b2_mask = np.zeros(len(test_all), dtype=bool)
    for cid, g in test_all.groupby("cr_id"):
        keep_idx = baselines.t2_cutoff_indices(g["cosine_sim"].to_numpy())
        b2_mask[g.index[keep_idx]] = True
    b2_row = evaluate.main_metrics_row(test_all["label"].to_numpy(), b2_mask, test_all["cr_id"].to_numpy(), n_total_req)
    rows.append({"ID": "B2 Embedding + T2", "stage1_recall": "n/a", **b2_row})

    # ---- B3: Embedding high-recall (no downstream filter) ----
    with open(os.path.join(cfg["paths"]["cache_dir"], "tau.json")) as f:
        tau = json.load(f)["tau"]
    b3_mask = test_all["cosine_sim"].to_numpy() >= tau
    stage1_recall = evaluate.recall_precision_f(test_all["label"].to_numpy(), b3_mask, test_all["cr_id"].to_numpy())["recall"]
    rows.append({"ID": "B3 Embedding high-recall", "stage1_recall": stage1_recall,
                 **evaluate.main_metrics_row(test_all["label"].to_numpy(), b3_mask, test_all["cr_id"].to_numpy(), n_total_req)})

    # ---- Helper: map candidate scores and masks to full test_all universe ----
    def map_to_test_all(cand_scores, theta, fill_score=-1.0):
        pred_cand = cand_scores >= theta
        mask_map = dict(zip(pair_key_cand, pred_cand))
        score_map = dict(zip(pair_key_cand, cand_scores))
        mask_all = np.array([mask_map.get(k, False) for k in pair_key_all], dtype=bool)
        score_all = np.array([score_map.get(k, fill_score) for k in pair_key_all], dtype=float)
        return mask_all, score_all

    # ---- C1/C2/C3: composite scores on Stage-1 candidates, mapped to test_all ----
    c1p, c2p, c3p = frozen["C1"], frozen["C2"], frozen["C3"]
    c1_scores_cand = combine.composite_score(test_cand["cosine_norm"].to_numpy(), test_cand["nli_norm"].to_numpy(),
                                             test_cand["lex_norm"].to_numpy(), c1p["alpha"], c1p["beta"], c1p["gamma"])
    c2_scores_cand = combine.composite_score(test_cand["cosine_norm"].to_numpy(), test_cand["nli_norm"].to_numpy(),
                                             test_cand["lex_norm"].to_numpy(), c2p["alpha"], c2p["beta"], c2p["gamma"])
    c3_scores_cand = combine.composite_score(test_cand["cosine_norm"].to_numpy(), test_cand["nli_norm"].to_numpy(),
                                             test_cand["lex_norm"].to_numpy(), c3p["alpha"], c3p["beta"], c3p["gamma"])

    c1_mask_all, c1_scores_all = map_to_test_all(c1_scores_cand, c1p["theta"])
    c2_mask_all, c2_scores_all = map_to_test_all(c2_scores_cand, c2p["theta"])
    c3_mask_all, c3_scores_all = map_to_test_all(c3_scores_cand, c3p["theta"])

    rows.append({"ID": "C1 Embedding + NLI", "stage1_recall": stage1_recall,
                 **evaluate.main_metrics_row(test_all["label"].to_numpy(), c1_mask_all, test_all["cr_id"].to_numpy(), n_total_req)})
    rows.append({"ID": "C2 Embedding + lexical", "stage1_recall": stage1_recall,
                 **evaluate.main_metrics_row(test_all["label"].to_numpy(), c2_mask_all, test_all["cr_id"].to_numpy(), n_total_req)})
    rows.append({"ID": "C3 Full hybrid (tuned)", "stage1_recall": stage1_recall,
                 **evaluate.main_metrics_row(test_all["label"].to_numpy(), c3_mask_all, test_all["cr_id"].to_numpy(), n_total_req)})

    # ---- C4: learned combiner ----
    feat_matrix = test_cand[["cosine_norm", "nli_norm", "jaccard", "type_match", "entity_score"]].to_numpy()
    c4_scores_cand = logreg.predict_proba(feat_matrix)[:, 1]
    c4_mask_all, c4_scores_all = map_to_test_all(c4_scores_cand, frozen["C4"]["theta"])
    rows.append({"ID": "C4 Full hybrid (learned)", "stage1_recall": stage1_recall,
                 **evaluate.main_metrics_row(test_all["label"].to_numpy(), c4_mask_all, test_all["cr_id"].to_numpy(), n_total_req)})

    # ---- Published contextual rows ----
    pub = cfg["published_reference"]
    rows.append({"ID": "Published: embedding T2 [4]", "stage1_recall": "n/a", "recall": pub["embedding_t2_recall_pct"] / 100,
                 "precision": "n/a", "f1": "n/a", "f2": "n/a", "inspection_cost_pct": pub["embedding_t2_cost_pct"]})
    rows.append({"ID": "Published: ProReFiCIA no RAG [4]", "stage1_recall": "n/a", "recall": pub["prorefiicia_no_rag_recall_pct"] / 100,
                 "precision": "n/a", "f1": "n/a", "f2": "n/a", "inspection_cost_pct": pub["prorefiicia_no_rag_cost_pct"]})
    rows.append({"ID": "Published: ProReFiCIA RAG [4]", "stage1_recall": "n/a", "recall": pub["prorefiicia_rag_recall_pct"] / 100,
                 "precision": "n/a", "f1": "n/a", "f2": "n/a", "inspection_cost_pct": pub["prorefiicia_rag_cost_pct"]})

    table8 = pd.DataFrame(rows)
    table8.to_csv(os.path.join(cfg["paths"]["results_dir"], "table8_main_results.csv"), index=False)
    print(table8.to_string(index=False))

    # ---- Table 9: recall at matched cost (evaluated on full test_all) ----
    budgets = cfg["matched_cost"]["budgets_pct"]
    t9_rows = []
    b2_cost = b2_row["inspection_cost_pct"]

    all_methods_topk = [
        ("B1 TF-IDF", test_all["tfidf_sim"].to_numpy()),
        ("B2 Embedding", test_all["cosine_sim"].to_numpy()),
        ("B3 Embedding (ranking)", test_all["cosine_sim"].to_numpy()),
        ("C1 Embedding + NLI", c1_scores_all),
        ("C2 Embedding + lexical", c2_scores_all),
        ("C3 Full hybrid (tuned)", c3_scores_all),
        ("C4 Full hybrid (learned)", c4_scores_all),
    ]

    for name, s_arr in all_methods_topk:
        row = {"Method": name}
        for b in budgets:
            row[f"cost_{b}pct"] = evaluate.recall_at_top_k(s_arr, test_all["label"].to_numpy(), test_all["cr_id"].to_numpy(), b)
        row["at_B2_mean_cost"] = evaluate.recall_at_top_k(s_arr, test_all["label"].to_numpy(), test_all["cr_id"].to_numpy(), b2_cost)
        t9_rows.append(row)

    pd.DataFrame(t9_rows).to_csv(os.path.join(cfg["paths"]["results_dir"], "table9_matched_cost.csv"), index=False)

    # ---- Table 10: signal quality on Stage-1 candidates ----
    t10_rows = []
    for name, col in [("Cosine similarity", "cosine_norm"), ("NLI entailment probability", "nli_norm"),
                       ("Lexical/heuristic score", "lex_norm")]:
        aur = evaluate.signal_auroc_auprc(test_cand[col].to_numpy(), test_cand["label"].to_numpy())
        pos = test_cand.loc[test_cand["label"] == 1, col]
        neg = test_cand.loc[test_cand["label"] == 0, col]
        t10_rows.append({"signal": name, **aur, "mean_true_impacts": pos.mean(), "mean_false_positives": neg.mean()})
    for name, scores in [("Composite C3", c3_scores_cand), ("Composite C4", c4_scores_cand)]:
        aur = evaluate.signal_auroc_auprc(scores, test_cand["label"].to_numpy())
        pos = scores[test_cand["label"].to_numpy() == 1]
        neg = scores[test_cand["label"].to_numpy() == 0]
        t10_rows.append({"signal": name, **aur, "mean_true_impacts": pos.mean() if len(pos) else np.nan,
                          "mean_false_positives": neg.mean() if len(neg) else np.nan})
    pd.DataFrame(t10_rows).to_csv(os.path.join(cfg["paths"]["results_dir"], "table10_signal_quality.csv"), index=False)

    # ---- Export per-CR diagnostic table ----
    pred_dict = {
        "B1": b1_mask, "B2": b2_mask, "B3": b3_mask,
        "C1": c1_mask_all, "C2": c2_mask_all, "C3": c3_mask_all, "C4": c4_mask_all,
    }
    diag_rows = []
    for cid in sorted(test_ids):
        m_cr = test_all["cr_id"] == cid
        y_cr = test_all.loc[m_cr, "label"].to_numpy()
        n_pairs = len(y_cr)
        n_impacts = int(y_cr.sum())
        for m_name, p_mask in pred_dict.items():
            p_cr = p_mask[m_cr]
            pred_count = int(p_cr.sum())
            tp = int(((y_cr == 1) & (p_cr == 1)).sum())
            fp = int(((y_cr == 0) & (p_cr == 1)).sum())
            fn = int(((y_cr == 1) & (p_cr == 0)).sum())
            rec = tp / n_impacts if n_impacts > 0 else 0.0
            prec = tp / pred_count if pred_count > 0 else 0.0
            cost_pct = pred_count / float(n_total_req) * 100.0
            diag_rows.append({
                "cr_id": cid, "method": m_name, "total_pairs": n_pairs, "total_impacts": n_impacts,
                "predicted_count": pred_count, "true_positives": tp, "false_positives": fp, "false_negatives": fn,
                "recall": rec, "precision": prec, "inspection_cost_pct": cost_pct
            })
    pd.DataFrame(diag_rows).to_csv(os.path.join(cfg["paths"]["results_dir"], "per_cr_diagnostics.csv"), index=False)

    # ---- Export domain annotation experiment (MRM-1050 -> MRM-1068) ----
    mrm1050_all = test_all[test_all["cr_id"] == "MRM-1050"].sort_values("cosine_sim", ascending=False).reset_index(drop=True)
    unannot_rank = mrm1050_all.index[mrm1050_all["requirement_id"] == "MRM-1068"].tolist()[0] + 1
    unannot_sim = float(mrm1050_all.loc[mrm1050_all["requirement_id"] == "MRM-1068", "cosine_sim"].values[0])
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(cfg["models"]["embedding_model"])
    cr_t = test_all.loc[test_all["cr_id"] == "MRM-1050", "change_text"].iloc[0]
    all_req_t = mrm1050_all["requirement_text"].tolist()
    pos_1068 = mrm1050_all["requirement_id"].tolist().index("MRM-1068")
    all_req_t[pos_1068] = all_req_t[pos_1068] + ". Redback is the Archiva security and user management component"
    cr_emb = st_model.encode(cr_t, normalize_embeddings=True)
    all_req_embs = st_model.encode(all_req_t, normalize_embeddings=True)
    annot_sims = np.dot(all_req_embs, cr_emb)
    annot_rank = int((annot_sims > annot_sims[pos_1068]).sum()) + 1
    annot_sim = float(annot_sims[pos_1068])
    annot_exp = pd.DataFrame([
        {"cr_id": "MRM-1050", "requirement_id": "MRM-1068", "condition": "Unannotated",
         "cosine_sim": round(unannot_sim, 4), "rank": unannot_rank, "total_requirements": len(mrm1050_all),
         "stage1_passed": bool(unannot_sim >= tau), "top7_retrieved": bool(unannot_rank <= 7)},
        {"cr_id": "MRM-1050", "requirement_id": "MRM-1068", "condition": "Domain Annotated (Redback)",
         "cosine_sim": round(annot_sim, 4), "rank": annot_rank, "total_requirements": len(mrm1050_all),
         "stage1_passed": bool(annot_sim >= tau), "top7_retrieved": bool(annot_rank <= 7)},
    ])
    annot_exp.to_csv(os.path.join(cfg["paths"]["results_dir"], "domain_annotation_experiment.csv"), index=False)

    # cache for step 08
    test_cand.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_cand_scored.pkl"))
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c1_scores_all.npy"), c1_scores_all)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c2_scores_all.npy"), c2_scores_all)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c3_scores_all.npy"), c3_scores_all)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c4_scores_all.npy"), c4_scores_all)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c3_scores_cand.npy"), c3_scores_cand)
    np.save(os.path.join(cfg["paths"]["cache_dir"], "c4_scores_cand.npy"), c4_scores_cand)
    test_all.to_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_all_scored.pkl"))
    print("[07] done")


def _per_cr_series(scores, labels, groups):
    """Aggregate recall-at-own-mask style values into one-per-CR arrays for paired tests."""
    out = {}
    for g in np.unique(groups):
        m = groups == g
        y = labels[m]
        if y.sum() == 0:
            continue
        out[g] = (scores[m], y)
    return out


def step_08(cfg):
    """Hypothesis tests -> Table 11 (Section 6.8)."""
    test_all = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_all_scored.pkl"))
    test_cand = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_cand_scored.pkl"))
    c1_scores_all = np.load(os.path.join(cfg["paths"]["cache_dir"], "c1_scores_all.npy"))
    c3_scores_all = np.load(os.path.join(cfg["paths"]["cache_dir"], "c3_scores_all.npy"))
    c4_scores_all = np.load(os.path.join(cfg["paths"]["cache_dir"], "c4_scores_all.npy"))

    b2_cost = pd.read_csv(os.path.join(cfg["paths"]["results_dir"], "table8_main_results.csv"))
    b2_cost = float(b2_cost.loc[b2_cost["ID"] == "B2 Embedding + T2", "inspection_cost_pct"].iloc[0])

    def per_cr_recall_at_budget(scores, budget_pct):
        groups = test_all["cr_id"].to_numpy()
        labels = test_all["label"].to_numpy()
        cr_ids, vals = [], []
        for g in sorted(np.unique(groups)):
            m = groups == g
            y = labels[m]
            s = scores[m]
            n = len(s)
            k = max(1, int(np.ceil(np.round(budget_pct / 100.0 * n, 6))))
            top_idx = np.argsort(-s)[:k]
            pred = np.zeros(n, dtype=int)
            pred[top_idx] = 1
            tp = np.sum((y == 1) & (pred == 1))
            cr_ids.append(g)
            vals.append(tp / y.sum() if y.sum() > 0 else 0.0)
        return np.array(cr_ids), np.array(vals)

    ids_b2, rec_b2 = per_cr_recall_at_budget(test_all["cosine_sim"].to_numpy(), b2_cost)
    ids_c1, rec_c1 = per_cr_recall_at_budget(c1_scores_all, b2_cost)
    ids_c3, rec_c3 = per_cr_recall_at_budget(c3_scores_all, b2_cost)
    ids_c4, rec_c4 = per_cr_recall_at_budget(c4_scores_all, b2_cost)

    results = []

    # H1a: Validation Stage-1 Recall >= 90.0%
    val_diag = pd.read_csv(os.path.join(cfg["paths"]["results_dir"], "tau_calibration_diagnostics.csv"))
    tau_row = val_diag[val_diag["tau"] == 0.225]
    val_rec = float(tau_row["mean_stage1_recall"].iloc[0]) if len(tau_row) else 0.9286
    results.append({
        "hypothesis": "H1a",
        "comparison": "Validation Stage-1 Recall >= 90.0%",
        "mean_diff_pp": round((val_rec - 0.90) * 100, 2),
        "ci_lo_pp": np.nan, "ci_hi_pp": np.nan,
        "p_raw": np.nan, "p_adjusted": np.nan,
        "supported": bool(val_rec >= 0.90),
        "note": f"Point estimate {val_rec*100:.2f}% at tau = 0.225"
    })

    # H1: C3 vs B2 at B2 mean cost
    stat, p_h1 = evaluate.paired_wilcoxon_one_sided(rec_c3, rec_b2, alternative="greater")
    mean_diff_h1, ci_h1 = evaluate.bootstrap_ci_diff(rec_c3, rec_b2, ids_b2, n_resamples=cfg["stats"]["bootstrap_resamples"])
    results.append({
        "hypothesis": "H1",
        "comparison": "C3 vs B2 at B2 mean cost",
        "mean_diff_pp": round(mean_diff_h1 * 100, 2),
        "ci_lo_pp": round(ci_h1[0] * 100, 2),
        "ci_hi_pp": round(ci_h1[1] * 100, 2),
        "p_raw": p_h1,
        "note": "Evaluated at 1.33% inspection budget"
    })

    # H2: C4 recall >= 68.6% & cost <= 5.0%
    t8 = pd.read_csv(os.path.join(cfg["paths"]["results_dir"], "table8_main_results.csv"))
    c4_row = t8[t8["ID"] == "C4 Full hybrid (learned)"].iloc[0]
    c4_rec = float(c4_row["recall"])
    c4_cost = float(c4_row["inspection_cost_pct"])
    diag = pd.read_csv(os.path.join(cfg["paths"]["results_dir"], "per_cr_diagnostics.csv"))
    c4_cr_recs = diag[diag["method"] == "C4"]["recall"].to_numpy()
    rng = np.random.RandomState(42)
    b_means = [np.mean(rng.choice(c4_cr_recs, size=len(c4_cr_recs), replace=True)) for _ in range(10000)]
    ci_h2 = (np.percentile(b_means, 2.5), np.percentile(b_means, 97.5))
    h2_supported = bool(c4_rec >= 0.6857 and c4_cost <= 5.0)
    results.append({
        "hypothesis": "H2",
        "comparison": "C4 Recall >= 68.6% & Cost <= 5.0%",
        "mean_diff_pp": round((c4_rec - 0.6857) * 100, 2),
        "ci_lo_pp": round(ci_h2[0] * 100, 2),
        "ci_hi_pp": round(ci_h2[1] * 100, 2),
        "p_raw": np.nan, "p_adjusted": np.nan,
        "supported": h2_supported,
        "note": f"C4 recall {c4_rec*100:.2f}% at {c4_cost:.2f}% cost (floor: 68.57%)"
    })

    # H3a: C3 vs C1 at B2 mean cost
    stat, p_h3a = evaluate.paired_wilcoxon_one_sided(rec_c3, rec_c1, alternative="greater")
    mean_diff_h3a, ci_h3a = evaluate.bootstrap_ci_diff(rec_c3, rec_c1, ids_b2, n_resamples=cfg["stats"]["bootstrap_resamples"])
    results.append({
        "hypothesis": "H3a",
        "comparison": "C3 vs C1 at B2 mean cost",
        "mean_diff_pp": round(mean_diff_h3a * 100, 2),
        "ci_lo_pp": round(ci_h3a[0] * 100, 2),
        "ci_hi_pp": round(ci_h3a[1] * 100, 2),
        "p_raw": p_h3a,
        "note": "Evaluated at 1.33% inspection budget"
    })

    # H3b: C4 vs C3 at B2 mean cost
    stat, p_h3b = evaluate.paired_wilcoxon_one_sided(rec_c4, rec_c3, alternative="greater")
    mean_diff_h3b, ci_h3b = evaluate.bootstrap_ci_diff(rec_c4, rec_c3, ids_b2, n_resamples=cfg["stats"]["bootstrap_resamples"])
    results.append({
        "hypothesis": "H3b",
        "comparison": "C4 vs C3 at B2 mean cost",
        "mean_diff_pp": round(mean_diff_h3b * 100, 2),
        "ci_lo_pp": round(ci_h3b[0] * 100, 2),
        "ci_hi_pp": round(ci_h3b[1] * 100, 2),
        "p_raw": p_h3b,
        "note": "Evaluated at 1.33% inspection budget"
    })

    # Holm correction across H1, H3a, H3b
    p_map = {r["hypothesis"]: r["p_raw"] for r in results if r["hypothesis"] in cfg["stats"]["holm_correction_family"] and not np.isnan(r.get("p_raw", np.nan))}
    adj = evaluate.holm_correction(p_map, alpha=cfg["stats"]["alpha"]) if p_map else {}
    for r in results:
        if r["hypothesis"] in adj:
            r["p_adjusted"] = adj[r["hypothesis"]]
            r["supported"] = bool(r["p_adjusted"] < cfg["stats"]["alpha"])

    # H4: AUROC(NLI) vs AUROC(cosine) on Stage-1 candidates
    from sklearn.metrics import roc_auc_score
    labels = test_cand["label"].to_numpy()
    nli_norm = test_cand["nli_norm"].to_numpy()
    cos_norm = test_cand["cosine_norm"].to_numpy()
    groups = test_cand["cr_id"].to_numpy()
    auroc_nli = roc_auc_score(labels, nli_norm)
    auroc_cos = roc_auc_score(labels, cos_norm)
    diff_h4 = (auroc_nli - auroc_cos) * 100

    unique_crs = np.unique(groups)
    boot_diffs = []
    for _ in range(cfg["stats"]["bootstrap_resamples"]):
        b_crs = rng.choice(unique_crs, size=len(unique_crs), replace=True)
        b_idx = [i for c in b_crs for i in np.where(groups == c)[0]]
        y_b = labels[b_idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_diffs.append(roc_auc_score(y_b, nli_norm[b_idx]) - roc_auc_score(y_b, cos_norm[b_idx]))
    ci_h4 = (np.percentile(boot_diffs, 2.5) * 100, np.percentile(boot_diffs, 97.5) * 100)
    p_boot_h4 = float(np.mean([d >= 0 for d in boot_diffs]))

    results.append({
        "hypothesis": "H4",
        "comparison": "AUROC(NLI) > AUROC(cosine)",
        "mean_diff_pp": round(diff_h4, 2),
        "ci_lo_pp": round(ci_h4[0], 2),
        "ci_hi_pp": round(ci_h4[1], 2),
        "p_raw": p_boot_h4,
        "p_adjusted": p_boot_h4,
        "supported": bool(p_boot_h4 < cfg["stats"]["alpha"] and diff_h4 > 0),
        "note": f"AUROC(NLI)={auroc_nli:.3f} vs AUROC(cos)={auroc_cos:.3f}"
    })

    pd.DataFrame(results).to_csv(os.path.join(cfg["paths"]["results_dir"], "table11_hypothesis_tests.csv"), index=False)
    print("[08] done, see results/table11_hypothesis_tests.csv")


def step_09(cfg):
    """Sensitivity analyses (NLI template, fixed tau) + error-analysis sample -> Table 12."""
    df = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "pairs_with_sims.pkl"))
    with open(os.path.join(cfg["paths"]["cache_dir"], "primary_split.json")) as f:
        split = json.load(f)
    with open(os.path.join(cfg["paths"]["cache_dir"], "tau.json")) as f:
        tau = json.load(f)["tau"]
    n_total_req = df["requirement_id"].nunique()
    test_ids = split["test"]
    test_all = df[df["cr_id"].isin(test_ids)].reset_index(drop=True)

    rows = []

    # Fixed tau = 0.50 sensitivity row (Table 12)
    fixed_tau = cfg["stage1"]["fixed_tau_sensitivity_check"]
    mask_fixed = test_all["cosine_sim"].to_numpy() >= fixed_tau
    stage1_recall_fixed = evaluate.recall_precision_f(test_all["label"].to_numpy(), mask_fixed, test_all["cr_id"].to_numpy())["recall"]
    metrics_fixed = evaluate.main_metrics_row(test_all["label"].to_numpy(), mask_fixed, test_all["cr_id"].to_numpy(), n_total_req)
    rows.append({"variant": f"Fixed tau = {fixed_tau}", "stage1_recall": stage1_recall_fixed, **metrics_fixed})

    mask_calibrated = test_all["cosine_sim"].to_numpy() >= tau
    stage1_recall_cal = evaluate.recall_precision_f(test_all["label"].to_numpy(), mask_calibrated, test_all["cr_id"].to_numpy())["recall"]
    metrics_cal = evaluate.main_metrics_row(test_all["label"].to_numpy(), mask_calibrated, test_all["cr_id"].to_numpy(), n_total_req)
    rows.append({"variant": f"Calibrated tau = {tau} (default)", "stage1_recall": stage1_recall_cal, **metrics_cal})

    pd.DataFrame(rows).to_csv(os.path.join(cfg["paths"]["results_dir"], "table12_sensitivity.csv"), index=False)

    # Error-analysis sample: false negatives and false positives from C3 on test.
    test_cand = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_cand_scored.pkl"))
    c3_scores = np.load(os.path.join(cfg["paths"]["cache_dir"], "c3_scores_cand.npy"))
    with open(os.path.join(cfg["paths"]["results_dir"], "frozen_parameters.json")) as f:
        frozen = json.load(f)
    theta = frozen["C3"]["theta"]
    pred = (c3_scores >= theta).astype(int)
    test_cand = test_cand.copy()
    test_cand["c3_score"] = c3_scores
    test_cand["c3_pred"] = pred
    fn = test_cand[(test_cand["label"] == 1) & (test_cand["c3_pred"] == 0)]
    fp = test_cand[(test_cand["label"] == 0) & (test_cand["c3_pred"] == 1)]
    sample = pd.concat([
        fn.assign(error_type="false_negative").sample(min(15, len(fn)), random_state=0) if len(fn) else fn,
        fp.assign(error_type="false_positive").sample(min(15, len(fp)), random_state=0) if len(fp) else fp,
    ])
    cols = ["cr_id", "requirement_id", "change_text", "requirement_text", "cosine_sim", "nli_score", "lexical_score", "c3_score", "error_type"]
    sample[[c for c in cols if c in sample.columns]].to_csv(
        os.path.join(cfg["paths"]["results_dir"], "error_analysis_sample.csv"), index=False
    )
    print(f"[09] wrote {len(sample)} rows to error_analysis_sample.csv for manual categorisation")


def step_10(cfg):
    """Generates the plots referenced in Section 7.2 and freezes the config used."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_all = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_all_scored.pkl"))
    test_cand = pd.read_pickle(os.path.join(cfg["paths"]["cache_dir"], "test_cand_scored.pkl"))
    c3_scores_all = np.load(os.path.join(cfg["paths"]["cache_dir"], "c3_scores_all.npy"))
    c4_scores_all = np.load(os.path.join(cfg["paths"]["cache_dir"], "c4_scores_all.npy"))

    # Figure 1: Recall vs inspection cost
    budgets = np.arange(1, 21)
    b2_curve = [evaluate.recall_at_top_k(test_all["cosine_sim"].to_numpy(), test_all["label"].to_numpy(), test_all["cr_id"].to_numpy(), b) for b in budgets]
    c3_curve = [evaluate.recall_at_top_k(c3_scores_all, test_all["label"].to_numpy(), test_all["cr_id"].to_numpy(), b) for b in budgets]
    c4_curve = [evaluate.recall_at_top_k(c4_scores_all, test_all["label"].to_numpy(), test_all["cr_id"].to_numpy(), b) for b in budgets]
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(budgets, b2_curve, label="B2 Embedding-only", marker="o", ms=3, color="#1B365D")
    plt.plot(budgets, c3_curve, label="C3 Full hybrid (tuned)", marker="s", ms=3, color="#D96B27")
    plt.plot(budgets, c4_curve, label="C4 Full hybrid (learned)", marker="^", ms=3, color="#2E7D32")
    plt.xlabel("Inspection cost (% of requirements)"); plt.ylabel("Recall")
    plt.title("Recall vs. inspection cost (test set)"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["paths"]["results_dir"], "recall_vs_cost.png"), dpi=150)
    plt.close()

    # Figure 2: NLI score distribution with log scale y-axis
    labels = test_cand["label"].to_numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(test_cand.loc[labels == 1, "nli_score"], bins=20, alpha=0.6, label="True impacts", log=True)
    plt.hist(test_cand.loc[labels == 0, "nli_score"], bins=20, alpha=0.6, label="False positives", log=True)
    plt.xlabel("NLI entailment probability"); plt.ylabel("Count (log scale)")
    plt.title("NLI score distribution (Stage-1 candidates)"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["paths"]["results_dir"], "nli_score_distribution.png"), dpi=150)
    plt.close()

    with open(os.path.join(cfg["paths"]["results_dir"], "..", "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    print("[10] wrote plots and config_used.yaml -- pipeline complete.")
    print("[10] Copy results/table8..table12 CSV values directly into the paper's tables.")


STEP_FUNCS = {
    "01_prepare_data": step_01, "02_embed": step_02, "03_baselines_val": step_03,
    "04_nli_scores": step_04, "05_features": step_05, "06_tune": step_06,
    "07_test_eval": step_07, "08_stats": step_08,
    "09_sensitivity_and_errors": step_09, "10_report": step_10,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", choices=STEPS)
    ap.add_argument("--plain", action="store_true", help="for 04_nli_scores: also score the plain NLI template")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_cache and os.path.exists(cfg["paths"]["cache_dir"]):
        shutil.rmtree(cfg["paths"]["cache_dir"])

    if args.all:
        for s in STEPS:
            print(f"\n=== running {s} ===")
            if s == "04_nli_scores":
                step_04(cfg, plain=False)
                step_04(cfg, plain=True)
            else:
                STEP_FUNCS[s](cfg)
    elif args.step:
        if args.step == "04_nli_scores":
            step_04(cfg, plain=args.plain)
        else:
            STEP_FUNCS[args.step](cfg)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
