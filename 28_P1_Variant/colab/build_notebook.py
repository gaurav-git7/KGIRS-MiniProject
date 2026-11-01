"""Builds colab/P1_Variant_Colab.ipynb: a single self-contained notebook (no uploads needed).
The p1/*.py sources are embedded as %%writefile cells.   Run: python colab/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.strip("\n").splitlines(True)})


md("""
# P1 Chunking benchmark (Variant): NFCorpus and a HotpotQA subset

Compares **fixed-length, recursive and semantic chunking** at two chunk sizes (128 and 256 tokens)
against a no-chunking baseline, with the BGE-small embedder. Reports Recall@5, Recall@10, MRR@10 and NDCG@10
(T1), chunk counts and indexing time (T2), NDCG@10 bar chart (F1) and Recall@K curves K = 1..20 (F2).

**Before you start:** Runtime > Change runtime type > **T4 GPU**. Then Runtime > Run all (or run cells in order).
No uploads needed; everything is written by the cells below. Expected run time: roughly 10-20 minutes on a T4
(I have not timed it on Colab). Results are zipped and downloaded by the last cell.
""")

md("## 1. Check the GPU and install packages")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q sentence-transformers datasets tabulate matplotlib pandas
""")

md("## 2. Write the project code (p1 package)")
code("import os\nos.makedirs('/content/p1proj/p1', exist_ok=True)\n%cd /content/p1proj\nopen('p1/__init__.py','w').write('')")
for name in ["config", "data", "chunkers", "metrics", "run_experiment", "report"]:
    src = (ROOT / "p1" / f"{name}.py").read_text(encoding="utf-8")
    code(f"%%writefile p1/{name}.py\n{src}")

md("## 3. Self-test the metrics, then run the full experiment")
code("!python -m p1.metrics")
md("""
This is the main step. It prints one line per configuration (chunks, Recall@5/10, MRR, NDCG@10, indexing time)
for NFCorpus (once, deterministic) and HotpotQA (3 seeds = 3 different question samples).
""")
code("!python -m p1.run_experiment")

md("## 4. Build tables (T1, T2), figures (F1, F2) and failure cases")
code("""
!python -m p1.report
import pandas as pd
from IPython.display import display, Image
pd.set_option('display.width', 200)
print('T1: retrieval quality'); display(pd.read_csv('results/T1_quality.csv'))
print('T2: chunks and indexing time'); display(pd.read_csv('results/T2_cost.csv'))
print('NDCG@10 bootstrap CI'); display(pd.read_csv('results/T3_ndcg_ci.csv'))
print('Document type summary'); display(pd.read_csv('results/doc_type_summary.csv').T)
display(Image('results/F1_ndcg_by_strategy.png'))
display(Image('results/F2_recall_curves.png'))
display(Image('results/F3_hotpot_by_qtype.png'))
""")

md("## 5. Download everything (results.csv, tables, figures, logs)")
code("""
!cd /content/p1proj && zip -qr /content/p1_variant_results.zip results p1
from google.colab import files
files.download('/content/p1_variant_results.zip')
""")

md("""
## What to put in the report
- **T1, T2, F1, F2** are in `results/` (`T1_quality.csv`, `T2_cost.csv`, `F1_ndcg_by_strategy.png`, `F2_recall_curves.png`).
- `doc_type_summary.csv` gives document length statistics next to the best strategy per dataset: use it for the
  "which strategy suits which document type" note.
- `failure_cases.csv` lists the queries where the best and worst configuration differ most: pick 5 and explain them.
- `results.csv` holds every metric per seed (submit this). `run_info.json` records versions, GPU and settings.
- Report numbers exactly as printed. HotpotQA paragraphs are short, so chunking may change little there; say so if that is what you see.
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "P1_Variant_Colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
