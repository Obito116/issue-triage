# GitHub Issue Auto-Triage

Paste a GitHub issue, get back what kind of issue it is: **bug**, **feature**,
**question**, or **documentation**. Plus a confidence breakdown, and an optional
one-line "why" from an LLM.

BUSS305 final project (Otgonbaatar L.).

## Why this

Every team that takes requests has the same problem: stuff comes in faster than
anyone can sort it. Bug reports, feature ideas, questions, doc fixes - someone
has to read each one and send it to the right place, by hand. It's slow and people
get it wrong. So I built a model that does the sorting automatically.

I trained it on GitHub issues because that's where you can actually get a huge pile
of *real, human-labeled* requests (1.27M of them). But the idea is general - it's
request triage; the dataset just happens to be developer issues. New issues keep
arriving unlabeled, so there's a real stream of text that genuinely needs sorting.

## Data

I used the NLBSE'23 issue classification dataset - 1.27M real GitHub issues,
labeled by actual developers (bug / feature / question / documentation). Link:
https://github.com/nlbse2023/issue-report-classification

It's a 1.6GB file so I don't train on all of it. `prepare_data.py` grabs a random
stratified slice (120k train / 30k test) and saves it as parquet.

Heads up: the classes are super imbalanced - bug 53%, feature 37%, question 6%,
documentation 4%. That imbalance ends up being the whole story (below).

## How it works

Pretty much straight out of the lectures:

- tf-idf with bigrams, train/test split (L3)
- Logistic Regression, also tried Linear SVM (L4)
- confusion matrix + macro vs micro F1 (L2)
- SMOTE to fight the imbalance (the SMOTE notebook)
- Streamlit app + an LLM "explain" button (L5)
- LLM augmentation for the rare classes (L6)

## Results (on the 30k test set I never touched)

| setup | micro F1 | macro F1 | question F1 | docs F1 |
|---|---|---|---|---|
| LogReg (baseline) | 82.6 | 68.1 | 47.3 | 54.7 |
| LogReg + class_weight | 75.3 | 61.6 | 44.6 | 40.7 |
| LogReg + SMOTE | 80.3 | 65.7 | 46.4 | 48.6 |
| Linear SVM | 81.7 | 68.2 | 48.2 | 54.0 |

For reference, the competition baselines on the full test set were fastText 0.851
and RoBERTa 0.891 micro F1.

The interesting part: micro F1 is 82.6 but macro is only 68.1. That 14-point gap
is the imbalance showing up - the model nails bug and feature (tons of examples)
and is weak on question and documentation (barely any). So I tried the textbook
fixes... and they made it *worse*. SMOTE basically interpolates between tf-idf
vectors, which doesn't mean anything in a 20k-dim sparse space, and the rare
classes aren't just rare, they're genuinely ambiguous. The thing that should
actually help is generating *realistic* rare-class examples with an LLM instead
of fake interpolated ones - that's what `augment.py` does.

## What the model learned

Top words per class, pulled straight from the LogReg weights:

- bug: bug, reproduce, fix, broken, fails, wrong
- feature: enhancement, feature, add, implement, allow
- question: question, how, can, is it, does
- documentation: docs, readme, docstring, clarify, guide

## Run it

```bash
pip install -r requirements.txt
python src/prepare_data.py     # downloads + slices the data (one time, ~few min)
python src/train.py            # trains + evaluates, writes models/
streamlit run app.py           # the web app
```

LLM stuff is optional - put your OpenRouter key in `.env` first:

```bash
python src/augment.py --label question --n 2000
```

## Files

```
src/prepare_data.py   download + subsample
src/train.py          train, compare configs, save model + metrics + confusion png
src/augment.py        LLM augmentation for rare classes
src/llm_explain.py    the "why this label" explainer
app.py                streamlit app
models/               classifier.pkl, metrics.json, confusion.png
```

The LLM never overrides the classifier - the model trained on real data picks the
label, the LLM just explains it or makes more training data.
