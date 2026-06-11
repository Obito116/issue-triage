# LLM augmentation for the weak classes (question, documentation).
#
# SMOTE didn't help - interpolating tf-idf vectors gives garbage "documents".
# So instead just ask an LLM to write realistic issues for the rare classes,
# dedup them, and add them to the training set. Then re-run train.py and see
# if minority recall / macro F1 moves.
#
# Uses OpenRouter. Needs OPENROUTER_API_KEY.
#   python src/augment.py --label question --n 2000
import argparse
import json
import os

import pandas as pd

MODEL = os.getenv("SYNTHESIS_MODEL", "openai/gpt-4o-mini")

BRIEF = {
    "question": "someone asking how to do something or whether a behaviour is expected - "
                "not a bug report, not a feature request, just a question",
    "documentation": "a request to add/fix/clarify docs: README, docstrings, guides, "
                     "examples - not a code bug, not a feature",
}


def seeds(label, k=6):
    df = pd.read_parquet("data/train.parquet")
    return [t[:300] for t in df[df["labels"] == label]["text"].tolist()[:k]]


def generate(label, n, batch=20):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY first.")
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    seed_block = "\n".join(f"- {s}" for s in seeds(label))
    rows = []
    while len(rows) < n:
        want = min(batch, n - len(rows))
        r = client.chat.completions.create(
            model=MODEL,
            max_tokens=2000,
            messages=[
                {
                    "role": "system",
                    "content": "You write realistic, varied synthetic GitHub issues for one "
                    "category, to grow a classifier's training data. Vary length, tone and "
                    "domain (web, ML, CLI, mobile). JSON only.",
                },
                {
                    "role": "user",
                    "content": f"Label: {label}\nWhat it means: {BRIEF[label]}\n\n"
                    f"Real examples:\n{seed_block}\n\n"
                    f'Write {want} new, different issues. Return {{"issues": ["...", ...]}}.',
                },
            ],
        )
        txt = r.choices[0].message.content.strip()
        txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            items = json.loads(txt)["issues"]
        except Exception:
            continue
        rows += [i.strip() for i in items if isinstance(i, str) and len(i.strip()) > 15]
        print(f"  {label}: {len(rows)}/{n}")

    rows = list(dict.fromkeys(rows))[:n]  # dedup
    return pd.DataFrame({"labels": label, "text": rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=list(BRIEF))
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()

    aug = generate(a.label, a.n)
    out = "data/augmented.parquet"
    if os.path.exists(out):
        aug = pd.concat([pd.read_parquet(out), aug], ignore_index=True)
    aug.to_parquet(out, index=False)
    print(f"\nwrote {len(aug):,} rows to {out}. now merge into train.parquet and re-run train.py.")
