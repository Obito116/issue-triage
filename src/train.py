# Train + compare a few issue-triage classifiers and save the best one.
#
# From class: tf-idf + bigrams (L3), LogReg / Linear SVM (L4), confusion matrix +
# macro vs micro F1 (L2), SMOTE for the imbalance, and top-term interpretability.
#
# Vectorize once, then run every classifier on the same matrix - that's the only
# trick that keeps this fast. Compare on a test set we never touch and let the
# macro-F1 gap do the talking.  Run: python src/train.py
import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABELS = ["bug", "feature", "question", "documentation"]
SEED = 42


def evaluate(name, clf, Xte, yte):
    pred = clf.predict(Xte)
    micro = f1_score(yte, pred, average="micro")
    macro = f1_score(yte, pred, average="macro")
    rep = classification_report(yte, pred, labels=LABELS, output_dict=True, zero_division=0)
    pc = {l: round(rep[l]["f1-score"], 4) for l in LABELS}
    print(f"  {name:<14} micro {micro*100:5.2f}  macro {macro*100:5.2f}  | "
          + "  ".join(f"{l[:4]} {pc[l]*100:4.1f}" for l in LABELS))
    return {"micro_f1": round(micro, 4), "macro_f1": round(macro, 4), "per_class_f1": pc}


def main():
    os.makedirs("models", exist_ok=True)
    tr = pd.read_parquet("data/train.parquet")
    te = pd.read_parquet("data/test.parquet")
    ytr, yte = tr["labels"].values, te["labels"].values
    print(f"train {len(tr):,}  test {len(te):,}")
    print("class distribution:", tr["labels"].value_counts().to_dict())

    print("\nvectorizing (one time)...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000,
                            sublinear_tf=True, strip_accents="unicode")
    Xtr = tfidf.fit_transform(tr["text"])
    Xte = tfidf.transform(te["text"])
    print(f"  done -> {Xtr.shape}")

    print("\ntraining + evaluating:")
    results, fitted = {}, {}

    m = LogisticRegression(C=10, max_iter=300, solver="saga").fit(Xtr, ytr)
    results["baseline"] = evaluate("baseline", m, Xte, yte); fitted["baseline"] = m

    m = LogisticRegression(C=10, max_iter=200, solver="saga",
                           class_weight="balanced").fit(Xtr, ytr)
    results["class_weight"] = evaluate("class_weight", m, Xte, yte); fitted["class_weight"] = m

    Xr, yr = SMOTE(random_state=SEED, k_neighbors=5,
                   sampling_strategy={"question": 25000, "documentation": 25000}).fit_resample(Xtr, ytr)
    m = LogisticRegression(C=10, max_iter=300, solver="saga").fit(Xr, yr)
    results["smote"] = evaluate("smote", m, Xte, yte); fitted["smote"] = m

    m = LinearSVC(C=1.0, class_weight="balanced").fit(Xtr, ytr)
    results["linear_svm"] = evaluate("linear_svm", m, Xte, yte); fitted["linear_svm"] = m

    # deploy the best LogReg (keeps predict_proba + readable coefficients for the app)
    best = max(["baseline", "class_weight", "smote"], key=lambda n: results[n]["macro_f1"])
    print(f"\ndeploying: {best} (macro F1 {results[best]['macro_f1']*100:.2f})")
    clf = fitted[best]
    joblib.dump(Pipeline([("tfidf", tfidf), ("clf", clf)]), "models/classifier.pkl")

    pred = clf.predict(Xte)
    cm = confusion_matrix(yte, pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Issue triage - confusion matrix (test)")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    fig.savefig("models/confusion.png", dpi=130); plt.close(fig)

    vocab = np.array(tfidf.get_feature_names_out())
    terms = {c: [vocab[j] for j in np.argsort(clf.coef_[i])[::-1][:12]]
             for i, c in enumerate(clf.classes_)}

    json.dump({"selected_model": best, "train_size": len(tr), "test_size": len(te),
               "labels": LABELS, "results": results,
               "published_baselines_micro_f1": {"fastText": 0.851, "RoBERTa": 0.891},
               "top_terms_per_class": terms},
              open("models/metrics.json", "w"), indent=2)

    print("\ntop terms per class:")
    for c in LABELS:
        print(f"  {c:<14} {', '.join(terms[c][:8])}")
    print("\nsaved models/classifier.pkl, models/metrics.json, models/confusion.png")


if __name__ == "__main__":
    main()
