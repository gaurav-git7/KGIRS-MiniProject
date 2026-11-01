"""Builds colab/P5_Colab.ipynb and colab/p5_code.zip (run from anywhere: python colab/build_notebook.py)."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

# ---- code bundle the notebook unpacks ------------------------------------
with zipfile.ZipFile(OUT / "p5_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for rel in ["config.py", "README.md", "EXPERIMENT_LOG.md"]:
        z.write(ROOT / rel, rel)
    for sub in ["scripts", "app"]:
        for p in sorted((ROOT / sub).glob("*.py")):
            z.write(p, f"{sub}/{p.name}")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.strip("\n").splitlines(True)})


md("""
# P5 (Primary): naive RAG vs Microsoft GraphRAG on MultiHop-RAG (Colab)

Runs the whole pipeline on a free Colab GPU. **Runtime > Change runtime type > T4 GPU** before you start.
Same code as the local project; the only differences are the model servers:
LLM = `Qwen/Qwen2.5-3B-Instruct` via **vLLM** (fast, handles many requests at once),
embedder = `BAAI/bge-small-en-v1.5` via a small local server. Both are open models within the 3B limit.

Before running: upload `p5_code.zip` (made by `colab/build_notebook.py`) when cell 2 asks for it.
Run cells top to bottom. Indexing (cell 10) is the long step. Keep this tab open.
""")

md("## 1. GPU check")
code("!nvidia-smi --query-gpu=name,memory.total --format=csv")

md("## 2. Upload and unpack the project code")
code("""
import os, zipfile
os.makedirs('/content/p5', exist_ok=True)
if not os.path.exists('/content/p5_code.zip'):
    from google.colab import files
    files.upload()                      # choose p5_code.zip
zipfile.ZipFile('/content/p5_code.zip').extractall('/content/p5')
%cd /content/p5
!ls
""")

md("## 3. Install client-side packages (GraphRAG 3.3.0 and friends)")
code("""
!pip install -q graphrag==3.3.0 datasets faiss-cpu openai tiktoken pyyaml tabulate networkx matplotlib fastapi uvicorn httpx sentence-transformers
""")

md("""
## 4. Install vLLM in an isolated environment
A separate venv keeps vLLM's pinned dependencies from clashing with GraphRAG. Takes a few minutes.
""")
code("""
!pip install -q uv
!uv venv /content/vllm_env
!uv pip install -q --python /content/vllm_env/bin/python vllm
!/content/vllm_env/bin/python -c "import vllm; print('vllm', vllm.__version__)"
""")

md("## 5. Configuration (environment variables read by `config.py`)")
code("""
import os
os.environ.update({
    'LLM_BASE_URL':   'http://localhost:8000/v1',     # vLLM
    'EMBED_BASE_URL': 'http://localhost:1236/v1',     # bge-small server
    'LLM_MODEL':      'Qwen/Qwen2.5-3B-Instruct',
    'EMBED_MODEL':    'BAAI/bge-small-en-v1.5',
    'GRAPHRAG_CONCURRENCY': '48',                      # vLLM batches concurrent requests (KV cache was only ~10% used at 16)
    'PYTHONIOENCODING': 'utf-8',
})
""")

md("## 6. Start the servers: vLLM (LLM), embedding server, token-counting proxy")
code("""
import subprocess, time, urllib.request, os

def wait(url, name, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3); print(name, 'ready after', int(time.time()-t0), 's'); return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f'{name} did not start; check /content/*.log')

vllm = subprocess.Popen(
    ['/content/vllm_env/bin/python', '-m', 'vllm.entrypoints.openai.api_server',
     '--model', 'Qwen/Qwen2.5-3B-Instruct', '--port', '8000',
     '--max-model-len', '8192', '--gpu-memory-utilization', '0.80', '--max-num-seqs', '64',
     '--dtype', 'half'],
    stdout=open('/content/vllm.log', 'w'), stderr=subprocess.STDOUT)
wait('http://localhost:8000/v1/models', 'vLLM')

emb = subprocess.Popen(['python', 'scripts/embed_server.py'], stdout=open('/content/embed.log', 'w'), stderr=subprocess.STDOUT)
wait('http://localhost:1236/v1/models', 'embedding server')

proxy = subprocess.Popen(['python', 'scripts/usage_proxy.py'], stdout=open('/content/proxy.log', 'w'), stderr=subprocess.STDOUT)
wait('http://localhost:1235/_stats', 'usage proxy')
""")

md("## 7. Sanity check the models")
code("""
import sys; sys.path.insert(0, 'scripts')
from common import chat, embed
print(chat([{'role': 'user', 'content': 'Say yes.'}]))
v, t = embed(['hello world']); print(v.shape, t)
""")

md("## 8. Build the subset (50 questions, 200 documents)")
code("!python scripts/01_build_subset.py")

md("## 9. Naive RAG baseline")
code("!python scripts/02_naive_rag.py 2>&1 | tail -5")

md("""
## 10. GraphRAG: setup and indexing (long step)
Indexing is cold (cache deleted) so time and tokens are honest. Progress is printed below.
If Colab disconnects mid-run, re-run cells 5-7 (servers) and then use `!python scripts/04_graphrag_index.py --resume`
(the time/tokens are then flagged as covering only the resumed part).
""")
code("""
!python scripts/03_graphrag_setup.py
!python scripts/04_graphrag_index.py
""")

md("## 11. GraphRAG queries: smoke test, then the full run")
code("!python scripts/05_graphrag_query.py --limit 3 2>&1 | tail -20")
md("Check the smoke-test answers and token counts above. If they look sane, run all 50 questions x 3 seeds:")
code("!python scripts/05_graphrag_query.py 2>&1 | tail -5")

md("## 12. Evaluate: results.csv, T1, T2, F1, F2")
code("""
!python scripts/06_evaluate.py
import pandas as pd
from IPython.display import display, Image
display(pd.read_csv('results/T1_accuracy.csv'))
display(pd.read_csv('results/T2_cost_and_graph.csv'))
display(Image('results/F1_accuracy_by_type.png'))
import os
if os.path.exists('results/F2_example_subgraph.png'):
    display(Image('results/F2_example_subgraph.png'))
""")

md("## 13. Download everything")
code("""
!cp /content/vllm.log /content/embed.log /content/proxy.log results/ 2>/dev/null
!cp -r graphrag_workspace/settings.yaml results/graphrag_settings.yaml
!cd /content/p5 && zip -qr /content/p5_results.zip results data/questions.json
from google.colab import files
files.download('/content/p5_results.zip')
""")

md("""
Put the downloaded `results/` folder into the local project (`1_P5_Primary/results/`) and run
`streamlit run app/streamlit_app.py` to browse it. Record in `EXPERIMENT_LOG.md` that the final run
used vLLM on a Colab GPU (note the GPU type), and report the local LM Studio naive baseline only as a cross-check.
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "P5_Colab.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT / "P5_Colab.ipynb", "and p5_code.zip")
