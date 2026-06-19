from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

# The 10 representative questions
SELECTED_QKEYS = [
    "INEQ1_W54",
    "INEQ5_h_W54",
    "INEQ5_i_W54",
    "INEQ8_c_W54",
    "INEQ8_d_W54",
    "INEQ4_a_W54",
    "ECON5_d_W54",
    "FIN_SIT_W54",
    "WORRY2c_W54",
    "GOVPRIORITYc_W54",
]


def extract_questions(input_path: str = "data/wave_54/opinionqa.csv") -> list[dict]:
    df = pd.read_csv(input_path)
    
    # Parse list columns
    for col in ("responses", "ordinal", "options"):
        df[col] = df[col].apply(ast.literal_eval)
    
    # Filter to selected qkeys and Overall rows only (population distribution)
    df = df[df["qkey"].isin(SELECTED_QKEYS) & (df["attribute"] == "Overall")]
    
    # Reorder to match SELECTED_QKEYS order
    df["qkey_order"] = df["qkey"].map({q: i for i, q in enumerate(SELECTED_QKEYS)})
    df = df.sort_values("qkey_order").drop("qkey_order", axis=1)
    
    questions = []
    for _, row in df.iterrows():
        questions.append({
            "qkey": row["qkey"],
            "question": row["question"],
            "options": row["options"],
            "responses": row["responses"],
            "ordinal": row["ordinal"],
            "refusal_rate": float(row["refusal_rate"]),
        })
    
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="src/w54_selected.json")
    args = parser.parse_args()
    
    questions = extract_questions()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.out, "w") as f:
        json.dump(questions, f, indent=2)
    
    print(f"Extracted {len(questions)} questions → {args.out}")
    for q in questions:
        print(f"  {q['qkey']}: {q['question'][:70]}")


if __name__ == "__main__":
    main()
