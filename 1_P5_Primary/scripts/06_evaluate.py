"""Step 6: score all answers and write results.csv, T1, T2, F1, F2 and failure cases.

Usage:  python scripts/06_evaluate.py [--example q07]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import analysis as A
from common import load_questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default=None, help="question id for the F2 subgraph")
    a = ap.parse_args()
    df = A.load_answers()
    if df.empty:
        sys.exit("No answer files found in results/. Run steps 2 and 5 first.")
    qs = load_questions()
    out = C.RESULTS_DIR

    A.results_long(df).to_csv(C.RESULTS_CSV, index=False)
    t1, t2 = A.table1(df), A.table2(df)
    t1.to_csv(out / "T1_accuracy.csv", index=False)
    t2.to_csv(out / "T2_cost_and_graph.csv", index=False)
    (out / "tables.md").write_text("## T1\n\n" + t1.to_markdown(index=False) + "\n\n## T2\n\n" +
                                   t2.to_markdown(index=False) + "\n", encoding="utf-8")
    print(t1.to_string(index=False), "\n")
    print(t2.to_string(index=False), "\n")

    A.figure1(df, out / "F1_accuracy_by_type.png")
    qid = a.example or A.pick_example(df, qs)
    if qid:
        A.figure2(df, qs, qid, out / "F2_example_subgraph.png")
        print(f"F2 uses question {qid}")
    else:
        print("F2 skipped: no GraphRAG-local subgraph available")
    A.failure_cases(df).to_csv(out / "failure_cases.csv", index=False)
    print(f"wrote {C.RESULTS_CSV} and figures/tables in {out}")


if __name__ == "__main__":
    main()
