# -*- coding: utf-8 -*-
"""
Generate markdown version of the publication-ready paper.
"""
import os
import paper_text
import paper_tables

md = []
title = paper_text.PAPER_METADATA["title"]
sub = paper_text.PAPER_METADATA["subtitle"]
p_type = paper_text.PAPER_METADATA["project_type"]
deg = paper_text.PAPER_METADATA["degree"]
grp = paper_text.PAPER_METADATA["group"]
students = paper_text.PAPER_METADATA["students"]
roll = paper_text.PAPER_METADATA["roll_nos"]
sup = paper_text.PAPER_METADATA["supervisor"]
dt = paper_text.PAPER_METADATA["date"]
repo = paper_text.PAPER_METADATA["repo"]

md.append(f"# {title}\n\n")
md.append(f"## {sub}\n\n")
md.append(f"**{p_type}**  \n")
md.append(f"*{deg}*  \n\n")
md.append(f"**{grp}**  \n")
md.append(f"**Students:** {students} (Roll No: {roll})  \n")
md.append(f"**Supervisor / Instructor:** {sup}  \n")
md.append(f"**Code Repository:** {repo}  \n")
md.append(f"**Date:** {dt}  \n\n")
md.append("---\n\n")

md.append("### Abstract\n\n" + paper_text.ABSTRACT_TEXT + "\n\n")
md.append("**Keywords:** *" + paper_text.KEYWORDS_TEXT + "*\n\n")
md.append("---\n\n")

def table_to_md(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(x).replace("\n", " ") for x in r) + " |\n"
    return out + "\n"

# Section 1
SECTION_1_TITLES = {
    "1.1": "1.1 Background",
    "1.2": "1.2 The Problem",
    "1.3": "1.3 Why This Matters",
    "1.4": "1.4 Existing Approaches and the Multi-Disciplinary Gap",
    "1.5": "1.5 Motivation",
    "1.6": "1.6 Research Objective",
    "1.7": "1.7 Main Contributions",
    "1.8": "1.8 Paper Organisation",
}
md.append("## 1. Introduction\n\n")
for k in sorted(paper_text.SECTION_1_TEXT.keys()):
    heading = SECTION_1_TITLES.get(k, k)
    md.append(f"### {heading}\n\n" + paper_text.SECTION_1_TEXT[k] + "\n\n")

# Section 2
SECTION_2_TITLES = {
    "2.1": "2.1 Foundational CIA Frameworks and Ontology-Driven Models",
    "2.2": "2.2 Structural and Network-Based Change Propagation",
    "2.3": "2.3 Neural Embeddings, Semantic Similarity, and Traceability Link Recovery",
    "2.4": "2.4 Machine Learning Classification and Agile Requirements Management",
    "2.5": "2.5 LLM Prompting, Cache-Augmented Generation, and Cost/Privacy Boundaries",
}
md.append("## 2. Related Work (Synthesizing R1 to R15)\n\n")
for k in sorted(paper_text.SECTION_2_TEXT.keys()):
    heading = SECTION_2_TITLES.get(k, k)
    md.append(f"### {heading}\n\n" + paper_text.SECTION_2_TEXT[k] + "\n\n")

md.append("### Table 1: Summary of All 15 Evaluated Studies (R1 to R15)\n\n")
md.append(table_to_md(paper_tables.TABLE_1_HEADERS, paper_tables.TABLE_1_ROWS))

# Section 3
SECTION_3_TITLES = {
    "3.1": "3.1 What the Literature Currently Does",
    "3.2": "3.2 What the Literature Does Not Adequately Address",
    "3.3": "3.3 Why This Research is Needed",
    "3.4": "3.4 Research Questions and Pre-Registered Hypotheses",
}
md.append("## 3. Research Gap and Problem Formulation\n\n")
for k in sorted(paper_text.SECTION_3_TEXT.keys()):
    heading = SECTION_3_TITLES.get(k, k)
    md.append(f"### {heading}\n\n" + paper_text.SECTION_3_TEXT[k] + "\n\n")

md.append("### Table 2: Research Questions and Pre-Registered Hypotheses\n\n")
md.append(table_to_md(paper_tables.TABLE_2_HEADERS, paper_tables.TABLE_2_ROWS))

# Section 4
SECTION_4_TITLES = {
    "4.1": "4.1 Overall Two-Stage Architecture",
    "4.2": "4.2 Stage 1: High-Recall Candidate Retrieval",
    "4.3": "4.3 Stage 2: Re-Ranking, Semantic Reasoning, and Feature Combination",
    "4.4": "4.4 The Seven Evaluated Local Configurations",
    "4.5": "4.5 Multi-Faceted Evaluation Metrics",
}
md.append("## 4. Proposed Methodology\n\n")
for k in sorted(paper_text.SECTION_4_TEXT.keys()):
    heading = SECTION_4_TITLES.get(k, k)
    md.append(f"### {heading}\n\n" + paper_text.SECTION_4_TEXT[k] + "\n\n")

md.append("### Table 3: The Seven Evaluated Local Configurations\n\n")
md.append(table_to_md(paper_tables.TABLE_3_HEADERS, paper_tables.TABLE_3_ROWS))

# Section 5
SECTION_5_TITLES = {
    "5.1": "5.1 Dataset Construction and Provenance",
    "5.2": "5.2 Toolchain, Pretrained Models, and Environment",
    "5.3": "5.3 Tuned and Frozen Parameters",
    "5.4": "5.4 Evaluation Protocol and Partitioning",
}
md.append("## 5. Experimental Setup\n\n")
for k in sorted(paper_text.SECTION_5_TEXT.keys()):
    heading = SECTION_5_TITLES.get(k, k)
    md.append(f"### {heading}\n\n" + paper_text.SECTION_5_TEXT[k] + "\n\n")

md.append("### Table 4: Dataset Statistics\n\n")
md.append(table_to_md(paper_tables.TABLE_4_HEADERS, paper_tables.TABLE_4_ROWS))
md.append("### Table 5a: Tuned Simplex Parameters (C1-C3)\n\n")
md.append(table_to_md(paper_tables.TABLE_5A_HEADERS, paper_tables.TABLE_5A_ROWS))
md.append("### Table 5b: Learned Combiner Parameters (C4)\n\n")
md.append(table_to_md(paper_tables.TABLE_5B_HEADERS, paper_tables.TABLE_5B_ROWS))

# Section 6
SECTION_6_TITLES = {
    "6.1": "6.1 Main Test Set Results Across Configurations",
    "6.2": "6.2 Recall at Matched Inspection Cost and Top-k Performance",
    "6.3": "6.3 Ranking Signal Quality and Feature Discriminability",
    "6.4": "6.4 Pre-Registered Hypothesis Testing",
    "6.5": "6.5 Sensitivity Analysis: Similarity Threshold Calibration",
    "6.6": "6.6 Validation Stage-1 Calibration Curve Diagnostics",
    "6.7": "6.7 Granular Error Analysis and Component Context Resolution",
}
md.append("## 6. Results\n\n")
md.append(f"### {SECTION_6_TITLES['6.1']}\n\n" + paper_text.SECTION_6_TEXT["6.1"] + "\n\n")
md.append("### Table 6: Comparative Performance Measures (Existing Methods vs. Proposed Hybrid Methods)\n\n")
md.append(table_to_md(paper_tables.TABLE_6_HEADERS, paper_tables.TABLE_6_ROWS))

if os.path.exists(os.path.join("results", "performance_comparison_chart.png")):
    md.append("![Figure 3: Performance Comparison](results/performance_comparison_chart.png)\n\n*Figure 3: Grouped comparison of Existing Methods vs. Proposed Hybrid Methods across Core Classification Measures (Recall, Precision, F1, F2) and Operational Inspection Cost.*\n\n")

md.append(f"### {SECTION_6_TITLES['6.2']}\n\n" + paper_text.SECTION_6_TEXT["6.2"] + "\n\n")
md.append("### Table 7: Matched-Cost Recall by Inspection Budget\n\n")
md.append(table_to_md(paper_tables.TABLE_7_HEADERS, paper_tables.TABLE_7_ROWS))

if os.path.exists(os.path.join("results", "recall_vs_cost.png")):
    md.append("![Figure 1: Recall vs. Cost](results/recall_vs_cost.png)\n\n*Figure 1: Recall vs. Inspection Cost curve comparing baseline B2 against hybrid configurations on the held-out test partition.*\n\n")

if os.path.exists(os.path.join("results", "matched_cost_topk_chart.png")):
    md.append("![Figure 4: Matched-Cost Recall](results/matched_cost_topk_chart.png)\n\n*Figure 4: Matched-Cost Recall across fixed inspection budgets (Top 1, Top 2, Top 3, Top 4, and Top 8 requirements) illustrating the Stage-1 candidate ceiling.*\n\n")

md.append(f"### {SECTION_6_TITLES['6.3']}\n\n" + paper_text.SECTION_6_TEXT["6.3"] + "\n\n")
md.append("### Table 8: Ranking Signal Quality on Stage-1 Candidates\n\n")
md.append(table_to_md(paper_tables.TABLE_8_HEADERS, paper_tables.TABLE_8_ROWS))

if os.path.exists(os.path.join("results", "nli_score_distribution.png")):
    md.append("![Figure 2: NLI Distribution](results/nli_score_distribution.png)\n\n*Figure 2: Distribution of zero-shot NLI entailment scores on Stage-1 candidate pairs (True Impacts vs. False Positives).*\n\n")

if os.path.exists(os.path.join("results", "signal_quality_chart.png")):
    md.append("![Figure 5: Signal Quality](results/signal_quality_chart.png)\n\n*Figure 5: Ranking signal discriminability (AUROC and AUPRC) on Stage-1 candidate pairs across feature modalities.*\n\n")

md.append(f"### {SECTION_6_TITLES['6.4']}\n\n" + paper_text.SECTION_6_TEXT["6.4"] + "\n\n")
md.append("### Table 9: Pre-Registered Hypothesis Testing\n\n")
md.append(table_to_md(paper_tables.TABLE_9_HEADERS, paper_tables.TABLE_9_ROWS))

if os.path.exists(os.path.join("results", "hypothesis_forest_plot.png")):
    md.append("![Figure 6: Hypothesis Forest Plot](results/hypothesis_forest_plot.png)\n\n*Figure 6: Forest plot of formal statistical hypothesis testing showing point estimates and 95% bootstrap confidence intervals.*\n\n")

md.append(f"### {SECTION_6_TITLES['6.5']}\n\n" + paper_text.SECTION_6_TEXT["6.5"] + "\n\n")
md.append("### Table 10: Sensitivity Analysis of Similarity Threshold\n\n")
md.append(table_to_md(paper_tables.TABLE_10_HEADERS, paper_tables.TABLE_10_ROWS))

md.append(f"### {SECTION_6_TITLES['6.6']}\n\n" + paper_text.SECTION_6_TEXT["6.6"] + "\n\n")
md.append("### Table 11: Validation Stage-1 Calibration Curve Diagnostics\n\n")
md.append(table_to_md(paper_tables.TABLE_11_HEADERS, paper_tables.TABLE_11_ROWS))

md.append(f"### {SECTION_6_TITLES['6.7']}\n\n" + paper_text.SECTION_6_TEXT["6.7"] + "\n\n")

# Section 7
SECTION_7_TITLES = {
    "7.1": "7.1 Question 1: What Did We Observe across Pipeline Stages?",
    "7.2": "7.2 Question 2: Did the Proposed Approach Perform Better than the Baselines?",
    "7.3": "7.3 Question 3: Why Did the Results Occur?",
    "7.4": "7.4 Question 4: What Are the Limitations and Practical Boundaries?",
    "7.5": "7.5 Question 5: How Do Findings Compare with Previous Research (R1–R15)?",
    "7.6": "7.6 Question 6: Do the Results Answer the Research Questions?",
}
md.append("## 7. Discussion and Synthesis\n\n")
for k in sorted(paper_text.SECTION_7_TEXT.keys()):
    heading = SECTION_7_TITLES.get(k, k)
    md.append(f"### {heading}\n\n" + paper_text.SECTION_7_TEXT[k] + "\n\n")

# Section 8-12
md.append("## 8. Research Gap Coverage\n\n" + paper_text.SECTION_8_TEXT + "\n\n")
md.append("### Table 12: Research Gap Coverage Mapping (All 15 Studies)\n\n")
md.append(table_to_md(paper_tables.TABLE_12_HEADERS, paper_tables.TABLE_12_ROWS))

md.append("## 9. Threats to Validity and Limitations\n\n" + paper_text.SECTION_9_TEXT + "\n\n")
md.append("## 10. Future Work\n\n" + paper_text.SECTION_10_TEXT + "\n\n")
md.append("## 11. Conclusion\n\n" + paper_text.SECTION_11_TEXT + "\n\n")

md.append("## 12. References\n\n")
for ref in paper_text.REFERENCES_LIST:
    md.append(f"- {ref}\n")

with open("AI_Based_Software_Requirements_CIA_Research_Paper.md", "w", encoding="utf-8") as f:
    f.write("".join(md))

print("Markdown document saved successfully to AI_Based_Software_Requirements_CIA_Research_Paper.md")
