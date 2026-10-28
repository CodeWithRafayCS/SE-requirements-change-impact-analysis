# -*- coding: utf-8 -*-
"""
High-Performance SQLite Data Loader for the SEOSS 33-Project Dataset.
Extracts issues, descriptions, and traceability/impact links directly from SQLite.
"""

import glob
import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any


def clean_text(summary: Optional[str], description: Optional[str]) -> str:
    """Combines summary and description into a clean, normalized string."""
    s = str(summary or "").strip()
    d = str(description or "").strip()
    if s and d:
        text = f"{s}. {d}"
    elif s:
        text = s
    else:
        text = d
    # Truncate extremely long log dumps to 2048 chars for embedding efficiency
    if len(text) > 2048:
        text = text[:2048]
    return text if text else "No description provided."


class ProjectDataset:
    """Encapsulates all extracted data, links, and splits for a single SEOSS project."""
    def __init__(
        self,
        project_name: str,
        db_path: str,
        issues_df: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        splits: Dict[str, List[str]]
    ):
        self.project_name = project_name
        self.db_path = db_path
        self.issues_df = issues_df
        self.issue_ids = issues_df["issue_id"].tolist()
        self.issue_texts = issues_df["text"].tolist()
        self.id_to_idx = {iid: idx for idx, iid in enumerate(self.issue_ids)}
        self.id_to_text = dict(zip(self.issue_ids, self.issue_texts))
        
        # Ground truth mapping: source_id -> Set[target_id]
        self.ground_truth = ground_truth
        self.splits = splits
        
        # Summary numbers
        self.num_issues = len(self.issues_df)
        self.num_change_requests = len(self.ground_truth)
        self.num_total_links = sum(len(s) for s in self.ground_truth.values())
        self.avg_links_per_cr = (
            self.num_total_links / self.num_change_requests if self.num_change_requests > 0 else 0.0
        )

    @property
    def issues(self) -> Dict[str, str]:
        """Mapping from issue_id to cleaned text."""
        return self.id_to_text

    @property
    def links(self) -> Dict[str, Set[str]]:
        """Ground truth mapping: source_id -> Set[target_ids]."""
        return self.ground_truth

    @property
    def ground_truth_map(self) -> Dict[str, Set[str]]:
        """Alias for ground_truth."""
        return self.ground_truth

    @property
    def train_issue_ids(self) -> List[str]:
        return self.splits.get("train", [])

    @property
    def val_issue_ids(self) -> List[str]:
        return self.splits.get("val", [])

    @property
    def test_issue_ids(self) -> List[str]:
        return self.splits.get("test", [])

    def get_issue_text(self, issue_id: str) -> str:
        """Retrieves text representation of an issue by ID."""
        return self.id_to_text.get(str(issue_id), "")

    def __repr__(self):
        return (
            f"<ProjectDataset: {self.project_name} | Issues: {self.num_issues:,} | "
            f"CRs: {self.num_change_requests:,} | True Links: {self.num_total_links:,}>"
        )


def list_available_projects(data_dir: str) -> List[str]:
    """Finds all SQLite database files in the dataset folder."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.sqlite3")))
    projects = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return projects


def load_project_from_sqlite(
    db_path: str,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    random_seed: int = 42,
    min_text_len: int = 5
) -> Optional[ProjectDataset]:
    """
    Loads and indexes one SEOSS project SQLite database.
    
    Args:
        db_path: Path to the *.sqlite3 file
        train_frac: Fraction of change requests for training
        val_frac: Fraction of change requests for validation
        random_seed: Seed for reproducible CR splitting
        min_text_len: Minimum text length to qualify as a valid issue
        
    Returns:
        ProjectDataset or None if project has no valid issues/links
    """
    project_name = os.path.splitext(os.path.basename(db_path))[0]
    conn = sqlite3.connect(db_path)
    
    # 1. Load Issues
    query_issues = """
        SELECT issue_id, type, summary, description 
        FROM issue 
        WHERE summary IS NOT NULL OR description IS NOT NULL
    """
    raw_issues = pd.read_sql_query(query_issues, conn)
    
    if len(raw_issues) == 0:
        conn.close()
        return None

    # Construct combined text
    raw_issues["text"] = [
        clean_text(s, d) for s, d in zip(raw_issues["summary"], raw_issues["description"])
    ]
    raw_issues = raw_issues[raw_issues["text"].str.len() >= min_text_len].copy()
    raw_issues.drop_duplicates(subset=["issue_id"], inplace=True)
    raw_issues.reset_index(drop=True, inplace=True)

    valid_issue_set = set(raw_issues["issue_id"])

    # 2. Load Links
    query_links = "SELECT source_issue_id, target_issue_id FROM issue_link"
    raw_links = pd.read_sql_query(query_links, conn)
    conn.close()

    # Build ground-truth adjacency mapping
    ground_truth: Dict[str, Set[str]] = {}
    for _, row in raw_links.iterrows():
        src = str(row["source_issue_id"]).strip()
        tgt = str(row["target_issue_id"]).strip()
        
        # Avoid self-links and require both issues to exist in the issue table
        if src == tgt or src not in valid_issue_set or tgt not in valid_issue_set:
            continue
            
        if src not in ground_truth:
            ground_truth[src] = set()
        ground_truth[src].add(tgt)

    # Change Requests are issues that have at least one verified dependency link
    change_requests = sorted(list(ground_truth.keys()))
    if len(change_requests) == 0:
        return None

    # 3. Create Train / Validation / Test Partition across Change Requests
    rng = np.random.RandomState(random_seed)
    shuffled_crs = change_requests.copy()
    rng.shuffle(shuffled_crs)

    n_crs = len(shuffled_crs)
    if n_crs >= 5:
        n_train = max(1, int(round(train_frac * n_crs)))
        n_val = max(1, int(round(val_frac * n_crs)))
        if n_train + n_val >= n_crs:
            n_val = max(1, n_crs - n_train - 1)
        train_crs = shuffled_crs[:n_train]
        val_crs = shuffled_crs[n_train:n_train + n_val]
        test_crs = shuffled_crs[n_train + n_val:]
        if len(test_crs) == 0:
            test_crs = val_crs
    else:
        # For tiny projects, use all for evaluation
        train_crs = shuffled_crs
        val_crs = shuffled_crs
        test_crs = shuffled_crs

    splits = {
        "train": train_crs,
        "val": val_crs,
        "test": test_crs
    }

    return ProjectDataset(
        project_name=project_name,
        db_path=db_path,
        issues_df=raw_issues[["issue_id", "type", "text"]],
        ground_truth=ground_truth,
        splits=splits
    )


class SEOSSDataLoader:
    """
    High-level manager for querying and iterating over SEOSS projects.
    """
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)

    def list_available_projects(self) -> List[str]:
        return list_available_projects(self.data_dir)

    def load_project(
        self,
        project_name: str,
        train_frac: float = 0.60,
        val_frac: float = 0.20,
        random_seed: int = 42
    ) -> Optional[ProjectDataset]:
        """Loads a project dataset by name (with or without .sqlite3 extension)."""
        clean_name = project_name.replace(".sqlite3", "")
        db_path = os.path.join(self.data_dir, f"{clean_name}.sqlite3")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")
        return load_project_from_sqlite(
            db_path=db_path,
            train_frac=train_frac,
            val_frac=val_frac,
            random_seed=random_seed
        )
