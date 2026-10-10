# -*- coding: utf-8 -*-
"""
Full Word Document (.docx) Builder.
Compiles the comprehensive, publication-ready research paper into Word format.
"""
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

import paper_text
import paper_tables

def set_table_borders(table, color="1B365D", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def create_document():
    doc = Document()
    
    # Page setup: Standard Letter, 0.75-inch side margins (7.0-inch text width), 1-inch top/bottom
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        section.header_distance = Inches(0.5)
        section.footer_distance = Inches(0.5)

    # Style definitions
    COLOR_PRIMARY = RGBColor(27, 54, 93)     # #1B365D (Deep Navy)
    COLOR_SECONDARY = RGBColor(46, 91, 136)  # #2E5B88 (Steel Blue)
    COLOR_TEXT = RGBColor(34, 34, 34)        # #222222 (Dark Charcoal)
    COLOR_MUTED = RGBColor(100, 100, 100)

    def format_p(p, space_before=0, space_after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT):
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = line_spacing
        p.alignment = align

    def add_p(text, space_before=0, space_after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, italic=False, color=COLOR_TEXT, size=11):
        p = doc.add_paragraph()
        format_p(p, space_before, space_after, line_spacing, align)
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        return p

    def add_heading_1(text):
        p = doc.add_paragraph()
        format_p(p, space_before=14, space_after=6, align=WD_ALIGN_PARAGRAPH.LEFT)
        run = p.add_run(text)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(14)
        run.font.color.rgb = COLOR_PRIMARY
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        format_p(p, space_before=10, space_after=4, align=WD_ALIGN_PARAGRAPH.LEFT)
        run = p.add_run(text)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(12)
        run.font.color.rgb = COLOR_SECONDARY
        return p

    def add_styled_table(headers, rows, col_widths=None):
        table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        table.allow_autofit = False
        
        # Style table borders
        tblPr = table._tbl.tblPr
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="6" w:space="0" w:color="1B365D"/>'
            f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="1B365D"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
            f'<w:insideV w:val="none"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr.append(borders)

        # Repeat table header on subsequent pages
        header_tr = table.rows[0]._tr.get_or_add_trPr()
        header_tr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        # Header row cells
        hdr_cells = table.rows[0].cells
        for col_idx, text in enumerate(headers):
            hdr_cells[col_idx].text = text
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1B365D"/>')
            hdr_cells[col_idx]._tc.get_or_add_tcPr().append(shading)
            p = hdr_cells[col_idx].paragraphs[0]
            format_p(p, space_before=4, space_after=4, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
            for r in p.runs:
                r.bold = True
                r.font.name = "Calibri"
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(255, 255, 255)

        # Data rows
        for row_idx, row_data in enumerate(rows):
            row = table.rows[row_idx + 1]
            trPr = row._tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            row_cells = row.cells
            bg_color = "F7F9FC" if row_idx % 2 == 1 else "FFFFFF"
            for col_idx, val in enumerate(row_data):
                row_cells[col_idx].text = str(val)
                shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>')
                row_cells[col_idx]._tc.get_or_add_tcPr().append(shading)
                p = row_cells[col_idx].paragraphs[0]
                format_p(p, space_before=3, space_after=3, line_spacing=1.05, 
                         align=WD_ALIGN_PARAGRAPH.LEFT if col_idx > 0 and len(str(val)) > 15 else WD_ALIGN_PARAGRAPH.CENTER)
                for r in p.runs:
                    r.font.name = "Calibri"
                    r.font.size = Pt(9)
                    r.font.color.rgb = COLOR_TEXT

        # Apply exact column widths
        if col_widths:
            for c_idx, w in enumerate(col_widths):
                if c_idx < len(table.columns):
                    table.columns[c_idx].width = Inches(w)
            for row in table.rows:
                for c_idx, w in enumerate(col_widths):
                    if c_idx < len(row.cells):
                        row.cells[c_idx].width = Inches(w)

        spacer = doc.add_paragraph()
        format_p(spacer, space_before=0, space_after=6)
        return table

    # =========================================================================
    # 1. TITLE PAGE
    # =========================================================================
    p_top = doc.add_paragraph()
    format_p(p_top, space_before=36, space_after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_bar = p_top.add_run("―" * 45)
    r_bar.font.color.rgb = RGBColor(217, 107, 39)
    r_bar.bold = True

    add_p("RESEARCH PAPER", space_before=12, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, color=RGBColor(217, 107, 39), size=12)
    add_p(paper_text.PAPER_METADATA["title"], space_before=8, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, color=COLOR_PRIMARY, size=22)
    add_p(paper_text.PAPER_METADATA["subtitle"], space_before=2, space_after=18, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, color=COLOR_SECONDARY, size=15)
    
    add_p(paper_text.PAPER_METADATA["project_type"], space_before=6, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, color=COLOR_TEXT, size=12)
    add_p(paper_text.PAPER_METADATA["degree"], space_before=0, space_after=28, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, color=COLOR_MUTED, size=11)

    # Info Box Table
    info_table = doc.add_table(rows=5, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_table.autofit = False
    info_table.allow_autofit = False
    info_widths = [2.2, 4.8]
    for c_idx, w in enumerate(info_widths):
        info_table.columns[c_idx].width = Inches(w)
        
    info_data = [
        ("PROJECT GROUP:", paper_text.PAPER_METADATA["group"]),
        ("STUDENTS:", paper_text.PAPER_METADATA["students"] + f" (Roll No: {paper_text.PAPER_METADATA['roll_nos']})"),
        ("SUPERVISOR / INSTRUCTOR:", paper_text.PAPER_METADATA["supervisor"]),
        ("CODE REPOSITORY:", paper_text.PAPER_METADATA["repo"]),
        ("DATE:", paper_text.PAPER_METADATA["date"])
    ]
    for idx, (label, val) in enumerate(info_data):
        row_cells = info_table.rows[idx].cells
        row_cells[0].width = Inches(info_widths[0])
        row_cells[1].width = Inches(info_widths[1])
        row_cells[0].text = label
        row_cells[1].text = val
        for c in row_cells:
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F6F9"/>')
            c._tc.get_or_add_tcPr().append(shading)
            p = c.paragraphs[0]
            format_p(p, space_before=4, space_after=4, align=WD_ALIGN_PARAGRAPH.LEFT)
        for r in row_cells[0].paragraphs[0].runs:
            r.bold = True
            r.font.name = "Calibri"
            r.font.size = Pt(10)
            r.font.color.rgb = COLOR_PRIMARY
        for r in row_cells[1].paragraphs[0].runs:
            r.font.name = "Calibri"
            r.font.size = Pt(10)
            r.font.color.rgb = COLOR_TEXT

    set_table_borders(info_table, color="1B365D", sz="8")

    p_bot = doc.add_paragraph()
    format_p(p_bot, space_before=36, space_after=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_bar_bot = p_bot.add_run("―" * 45)
    r_bar_bot.font.color.rgb = RGBColor(217, 107, 39)
    r_bar_bot.bold = True

    doc.add_page_break()

    # =========================================================================
    # 2. TABLE OF CONTENTS
    # =========================================================================
    add_heading_1("Table of Contents")
    toc_entries = [
        ("Abstract", "4"),
        ("Keywords", "4"),
        ("1. Introduction", "5"),
        ("   1.1 Background", "5"),
        ("   1.2 The Problem", "5"),
        ("   1.3 Why This Matters", "5"),
        ("   1.4 Existing Approaches and the Multi-Disciplinary Gap", "6"),
        ("   1.5 Motivation", "6"),
        ("   1.6 Research Objective", "6"),
        ("   1.7 Main Contributions", "7"),
        ("   1.8 Paper Organisation", "7"),
        ("2. Related Work (Synthesizing R1 to R15)", "8"),
        ("   2.1 Foundational CIA Frameworks and Ontology-Driven Models", "8"),
        ("   2.2 Structural and Network-Based Change Propagation", "8"),
        ("   2.3 Neural Embeddings, Semantic Similarity, and Traceability Link Recovery", "9"),
        ("   2.4 Machine Learning Classification and Agile Requirements Management", "10"),
        ("   2.5 LLM Prompting, Cache-Augmented Generation, and Cost/Privacy Boundaries", "11"),
        ("   2.6 Summary Table of All 15 Evaluated Studies", "12"),
        ("3. Research Gap and Problem Formulation", "13"),
        ("   3.1 What the Literature Currently Does", "13"),
        ("   3.2 What the Literature Does Not Adequately Address", "13"),
        ("   3.3 Why This Research is Needed", "14"),
        ("   3.4 Research Questions and Pre-Registered Hypotheses", "14"),
        ("4. Proposed Methodology", "15"),
        ("   4.1 Overall Two-Stage Architecture", "15"),
        ("   4.2 Stage 1: High-Recall Candidate Retrieval", "15"),
        ("   4.3 Stage 2: Re-Ranking, Semantic Reasoning, and Feature Combination", "16"),
        ("   4.4 The Seven Evaluated Local Configurations", "17"),
        ("   4.5 Multi-Faceted Evaluation Metrics", "18"),
        ("5. Experimental Setup", "18"),
        ("   5.1 Dataset Construction and Provenance", "18"),
        ("   5.2 Toolchain, Pretrained Models, and Environment", "19"),
        ("   5.3 Tuned and Frozen Parameters", "19"),
        ("   5.4 Evaluation Protocol and Partitioning", "20"),
        ("6. Experimental Results", "21"),
        ("   6.1 Main Test Set Results Across Configurations", "21"),
        ("   6.2 Recall at Matched Inspection Cost and Top-k Performance", "22"),
        ("   6.3 Ranking Signal Quality and Feature Discriminability", "23"),
        ("   6.4 Pre-Registered Hypothesis Testing", "24"),
        ("   6.5 Sensitivity Analysis: Similarity Threshold Calibration", "25"),
        ("   6.6 Validation Stage-1 Calibration Curve Diagnostics", "25"),
        ("   6.7 Granular Error Analysis and Component Context Resolution", "26"),
        ("7. Discussion and Synthesis", "27"),
        ("   7.1 Question 1: What Did We Observe across Pipeline Stages?", "27"),
        ("   7.2 Question 2: Did the Proposed Approach Perform Better than the Baselines?", "28"),
        ("   7.3 Question 3: Why Did the Results Occur?", "29"),
        ("   7.4 Question 4: What Are the Limitations and Practical Boundaries?", "30"),
        ("   7.5 Question 5: How Do Findings Compare with Previous Research (R1–R15)?", "31"),
        ("   7.6 Question 6: Do the Results Answer the Research Questions?", "32"),
        ("8. Research Gap Coverage", "30"),
        ("9. Threats to Validity and Limitations", "31"),
        ("10. Future Work", "32"),
        ("11. Conclusion", "32"),
        ("12. References", "33")
    ]
    
    # Borderless 2-column TOC table to guarantee perfect alignment without wrapping
    toc_table = doc.add_table(rows=len(toc_entries), cols=2)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_table.autofit = False
    toc_table.allow_autofit = False
    toc_widths = [6.3, 0.7]
    for c_idx, w in enumerate(toc_widths):
        toc_table.columns[c_idx].width = Inches(w)
        
    for idx, (title, pg) in enumerate(toc_entries):
        row_cells = toc_table.rows[idx].cells
        row_cells[0].width = Inches(toc_widths[0])
        row_cells[1].width = Inches(toc_widths[1])
        
        p_t = row_cells[0].paragraphs[0]
        format_p(p_t, space_before=1, space_after=1, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.LEFT)
        r_t = p_t.add_run(title)
        r_t.font.name = "Calibri"
        r_t.font.size = Pt(10)
        if not title.startswith("   "):
            r_t.bold = True
            r_t.font.color.rgb = COLOR_PRIMARY
        else:
            r_t.font.color.rgb = COLOR_TEXT
            
        p_p = row_cells[1].paragraphs[0]
        format_p(p_p, space_before=1, space_after=1, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.RIGHT)
        r_p = p_p.add_run(pg)
        r_p.font.name = "Calibri"
        r_p.font.size = Pt(10)
        r_p.font.color.rgb = COLOR_PRIMARY if not title.startswith("   ") else COLOR_MUTED
        if not title.startswith("   "):
            r_p.bold = True

    doc.add_page_break()

    # =========================================================================
    # 3. ABSTRACT & KEYWORDS
    # =========================================================================
    add_heading_1("Abstract")
    add_p(paper_text.ABSTRACT_TEXT, space_before=4, space_after=10, line_spacing=1.15, size=10.5)

    add_heading_2("Keywords")
    add_p(paper_text.KEYWORDS_TEXT, space_before=2, space_after=14, italic=True, size=10)

    doc.add_page_break()

    # =========================================================================
    # 4. SECTION 1: INTRODUCTION
    # =========================================================================
    add_heading_1("1. Introduction")
    
    add_heading_2("1.1 Background")
    add_p(paper_text.SECTION_1_TEXT["1.1"])

    add_heading_2("1.2 The Problem")
    add_p(paper_text.SECTION_1_TEXT["1.2"])

    add_heading_2("1.3 Why This Matters")
    add_p(paper_text.SECTION_1_TEXT["1.3"])

    add_heading_2("1.4 Existing Approaches and the Multi-Disciplinary Gap")
    add_p(paper_text.SECTION_1_TEXT["1.4"])

    add_heading_2("1.5 Motivation")
    add_p(paper_text.SECTION_1_TEXT["1.5"])

    add_heading_2("1.6 Research Objective")
    add_p(paper_text.SECTION_1_TEXT["1.6"])

    add_heading_2("1.7 Main Contributions")
    add_p(paper_text.SECTION_1_TEXT["1.7"])

    add_heading_2("1.8 Paper Organisation")
    add_p(paper_text.SECTION_1_TEXT["1.8"])

    # =========================================================================
    # 5. SECTION 2: RELATED WORK (R1 TO R15)
    # =========================================================================
    add_heading_1("2. Related Work (Synthesizing R1 to R15)")
    
    add_heading_2("2.1 Foundational CIA Frameworks and Ontology-Driven Models")
    add_p(paper_text.SECTION_2_TEXT["2.1"])

    add_heading_2("2.2 Structural and Network-Based Change Propagation")
    add_p(paper_text.SECTION_2_TEXT["2.2"])

    add_heading_2("2.3 Neural Embeddings, Semantic Similarity, and Traceability Link Recovery")
    add_p(paper_text.SECTION_2_TEXT["2.3"])

    add_heading_2("2.4 Machine Learning Classification and Agile Requirements Management")
    add_p(paper_text.SECTION_2_TEXT["2.4"])

    add_heading_2("2.5 LLM Prompting, Cache-Augmented Generation, and Cost/Privacy Boundaries")
    add_p(paper_text.SECTION_2_TEXT["2.5"])

    add_heading_2("2.6 Summary Table of All 15 Evaluated Studies")
    add_p("Table 1 provides an exhaustive comparative synthesis of the fifteen related studies (R1 through R15) establishing the state of the art.", italic=True)
    add_styled_table(paper_tables.TABLE_1_HEADERS, paper_tables.TABLE_1_ROWS, col_widths=[1.4, 2.0, 1.8, 1.8])

    # =========================================================================
    # 6. SECTION 3: RESEARCH GAP AND PROBLEM FORMULATION
    # =========================================================================
    add_heading_1("3. Research Gap and Problem Formulation")

    add_heading_2("3.1 What the Literature Currently Does")
    add_p(paper_text.SECTION_3_TEXT["3.1"])

    add_heading_2("3.2 What the Literature Does Not Adequately Address")
    add_p(paper_text.SECTION_3_TEXT["3.2"])

    add_heading_2("3.3 Why This Research is Needed")
    add_p(paper_text.SECTION_3_TEXT["3.3"])

    add_heading_2("3.4 Research Questions and Pre-Registered Hypotheses")
    add_p(paper_text.SECTION_3_TEXT["3.4"])
    add_styled_table(paper_tables.TABLE_2_HEADERS, paper_tables.TABLE_2_ROWS, col_widths=[0.6, 2.2, 2.2, 2.0])

    # =========================================================================
    # 7. SECTION 4: PROPOSED METHODOLOGY
    # =========================================================================
    add_heading_1("4. Proposed Methodology")

    add_heading_2("4.1 Overall Two-Stage Architecture")
    add_p(paper_text.SECTION_4_TEXT["4.1"])

    add_heading_2("4.2 Stage 1: High-Recall Candidate Retrieval")
    add_p(paper_text.SECTION_4_TEXT["4.2"])

    add_heading_2("4.3 Stage 2: Re-Ranking, Semantic Reasoning, and Feature Combination")
    add_p(paper_text.SECTION_4_TEXT["4.3"])

    add_heading_2("4.4 The Seven Evaluated Local Configurations")
    add_p(paper_text.SECTION_4_TEXT["4.4"])
    add_styled_table(paper_tables.TABLE_3_HEADERS, paper_tables.TABLE_3_ROWS, col_widths=[0.5, 1.6, 1.4, 2.1, 1.4])

    add_heading_2("4.5 Multi-Faceted Evaluation Metrics")
    add_p(paper_text.SECTION_4_TEXT["4.5"])

    # =========================================================================
    # 8. SECTION 5: EXPERIMENTAL SETUP
    # =========================================================================
    add_heading_1("5. Experimental Setup")

    add_heading_2("5.1 Dataset Construction and Provenance")
    add_p(paper_text.SECTION_5_TEXT["5.1"])
    add_styled_table(paper_tables.TABLE_4_HEADERS, paper_tables.TABLE_4_ROWS, col_widths=[1.3, 1.1, 1.1, 1.1, 1.1, 1.3])

    add_heading_2("5.2 Toolchain, Pretrained Models, and Environment")
    add_p(paper_text.SECTION_5_TEXT["5.2"])

    add_heading_2("5.3 Tuned and Frozen Parameters")
    add_p(paper_text.SECTION_5_TEXT["5.3"])
    add_styled_table(paper_tables.TABLE_5A_HEADERS, paper_tables.TABLE_5A_ROWS, col_widths=[0.8, 1.2, 1.2, 1.2, 1.2, 1.4])
    add_styled_table(paper_tables.TABLE_5B_HEADERS, paper_tables.TABLE_5B_ROWS, col_widths=[0.8, 2.2, 1.4, 1.4, 1.2])

    add_heading_2("5.4 Evaluation Protocol and Partitioning")
    add_p(paper_text.SECTION_5_TEXT["5.4"])

    # =========================================================================
    # 9. SECTION 6: RESULTS
    # =========================================================================
    add_heading_1("6. Results")

    add_heading_2("6.1 Main Test Set Results Across Configurations")
    add_p(paper_text.SECTION_6_TEXT["6.1"])
    add_styled_table(paper_tables.TABLE_6_HEADERS, paper_tables.TABLE_6_ROWS, col_widths=[1.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.9, 0.8])

    fig3_path = os.path.join("results", "performance_comparison_chart.png")
    if os.path.exists(fig3_path):
        p_fig3 = doc.add_paragraph()
        format_p(p_fig3, space_before=6, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
        p_fig3.add_run().add_picture(fig3_path, width=Inches(6.2))
        add_p("Figure 3. Grouped comparison of Existing Methods vs. Proposed Hybrid Methods across Core Classification Measures (Recall, Precision, F1, F2) and Operational Inspection Cost.", 
              space_before=2, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9.5)

    add_heading_2("6.2 Recall at Matched Inspection Cost and Top-k Performance")
    add_p(paper_text.SECTION_6_TEXT["6.2"])
    add_styled_table(paper_tables.TABLE_7_HEADERS, paper_tables.TABLE_7_ROWS, col_widths=[1.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    fig1_path = os.path.join("results", "recall_vs_cost.png")
    if os.path.exists(fig1_path):
        p_fig1 = doc.add_paragraph()
        format_p(p_fig1, space_before=6, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
        p_fig1.add_run().add_picture(fig1_path, width=Inches(5.5))
        add_p("Figure 1. Recall vs. Inspection Cost curve comparing baseline B2 against hybrid configurations on the held-out test partition.", 
              space_before=2, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9.5)

    fig6_path = os.path.join("results", "matched_cost_topk_chart.png")
    if os.path.exists(fig6_path):
        p_fig6 = doc.add_paragraph()
        format_p(p_fig6, space_before=6, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
        p_fig6.add_run().add_picture(fig6_path, width=Inches(5.8))
        add_p("Figure 4. Matched-Cost Recall across fixed inspection budgets (Top 1, Top 2, Top 3, Top 4, and Top 8 requirements) illustrating the Stage-1 candidate ceiling.", 
              space_before=2, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9.5)

    add_heading_2("6.3 Ranking Signal Quality and Feature Discriminability")
    add_p(paper_text.SECTION_6_TEXT["6.3"])
    add_styled_table(paper_tables.TABLE_8_HEADERS, paper_tables.TABLE_8_ROWS, col_widths=[1.8, 1.0, 1.0, 1.6, 1.6])

    fig2_path = os.path.join("results", "nli_score_distribution.png")
    if os.path.exists(fig2_path):
        p_fig2 = doc.add_paragraph()
        format_p(p_fig2, space_before=6, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
        p_fig2.add_run().add_picture(fig2_path, width=Inches(5.2))
        add_p("Figure 2. Distribution of zero-shot NLI entailment scores on Stage-1 candidate pairs (True Impacts vs. False Positives).", 
              space_before=2, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9.5)

    fig5_path = os.path.join("results", "signal_quality_chart.png")
    if os.path.exists(fig5_path):
        p_fig5 = doc.add_paragraph()
        format_p(p_fig5, space_before=6, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
        p_fig5.add_run().add_picture(fig5_path, width=Inches(5.5))
        add_p("Figure 5. Ranking signal discriminability (AUROC and AUPRC) on Stage-1 candidate pairs across feature modalities.", 
              space_before=2, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9.5)

    add_heading_2("6.4 Pre-Registered Hypothesis Testing")
    add_p(paper_text.SECTION_6_TEXT["6.4"])
    add_styled_table(paper_tables.TABLE_9_HEADERS, paper_tables.TABLE_9_ROWS, col_widths=[0.6, 2.2, 1.1, 1.3, 0.9, 0.9])

    fig4_path = os.path.join("results", "hypothesis_forest_plot.png")
    if os.path.exists(fig4_path):
        p_fig4 = doc.add_paragraph()
        format_p(p_fig4, space_before=6, space_after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
        p_fig4.add_run().add_picture(fig4_path, width=Inches(6.0))
        add_p("Figure 6. Forest plot of formal statistical hypothesis testing showing point estimates and 95% bootstrap confidence intervals.", 
              space_before=2, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9.5)

    add_heading_2("6.5 Sensitivity Analysis: Similarity Threshold Calibration")
    add_p(paper_text.SECTION_6_TEXT["6.5"])
    add_styled_table(paper_tables.TABLE_10_HEADERS, paper_tables.TABLE_10_ROWS, col_widths=[1.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    add_heading_2("6.6 Validation Stage-1 Calibration Curve Diagnostics")
    add_p(paper_text.SECTION_6_TEXT["6.6"])
    add_styled_table(paper_tables.TABLE_11_HEADERS, paper_tables.TABLE_11_ROWS, col_widths=[1.6, 2.7, 2.7])

    add_heading_2("6.7 Granular Error Analysis and Component Context Resolution")
    add_p(paper_text.SECTION_6_TEXT["6.7"])

    # =========================================================================
    # 10. SECTION 7: DISCUSSION AND SYNTHESIS
    # =========================================================================
    add_heading_1("7. Discussion and Synthesis")

    add_heading_2("7.1 Question 1: What Did We Observe across Pipeline Stages?")
    add_p(paper_text.SECTION_7_TEXT["7.1"])

    add_heading_2("7.2 Question 2: Did the Proposed Approach Perform Better than the Baselines?")
    add_p(paper_text.SECTION_7_TEXT["7.2"])

    add_heading_2("7.3 Question 3: Why Did the Results Occur?")
    add_p(paper_text.SECTION_7_TEXT["7.3"])

    add_heading_2("7.4 Question 4: What Are the Limitations and Practical Boundaries?")
    add_p(paper_text.SECTION_7_TEXT["7.4"])

    add_heading_2("7.5 Question 5: How Do Findings Compare with Previous Research (R1–R15)?")
    add_p(paper_text.SECTION_7_TEXT["7.5"])

    add_heading_2("7.6 Question 6: Do the Results Answer the Research Questions?")
    add_p(paper_text.SECTION_7_TEXT["7.6"])

    # =========================================================================
    # 11. SECTION 8, 9, 10, 11
    # =========================================================================
    add_heading_1("8. Research Gap Coverage")
    add_p(paper_text.SECTION_8_TEXT)
    add_styled_table(paper_tables.TABLE_12_HEADERS, paper_tables.TABLE_12_ROWS, col_widths=[1.6, 1.4, 1.3, 1.8, 0.9])

    add_heading_1("9. Threats to Validity and Limitations")
    add_p(paper_text.SECTION_9_TEXT)

    add_heading_1("10. Future Work")
    add_p(paper_text.SECTION_10_TEXT)

    add_heading_1("11. Conclusion")
    add_p(paper_text.SECTION_11_TEXT)

    # =========================================================================
    # 12. SECTION 12: REFERENCES
    # =========================================================================
    add_heading_1("12. References")
    for ref_str in paper_text.REFERENCES_LIST:
        p_ref = doc.add_paragraph()
        format_p(p_ref, space_before=2, space_after=4, line_spacing=1.05)
        r = p_ref.add_run(ref_str)
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.color.rgb = COLOR_TEXT

    out_path_a1 = os.path.join("AI_Based_Software_Requirements_CIA_Research_Paper.docx")
    out_path_parent = os.path.join("..", "research paper.docx")
    
    saved_paths = []
    try:
        doc.save(out_path_a1)
        saved_paths.append(os.path.abspath(out_path_a1))
    except PermissionError:
        alt_path = os.path.join("AI_Based_Software_Requirements_CIA_Research_Paper_latest.docx")
        doc.save(alt_path)
        saved_paths.append(f"{os.path.abspath(alt_path)} (Main file currently locked in Word)")
        
    try:
        doc.save(out_path_parent)
        saved_paths.append(os.path.abspath(out_path_parent))
    except Exception as e:
        saved_paths.append(f"(Parent save note: {e})")
        
    print("Document successfully saved to:")
    for p in saved_paths:
        print(f"  - {p}")

if __name__ == "__main__":
    create_document()
