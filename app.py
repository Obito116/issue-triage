# Streamlit app for the issue triage model.
# Three tabs: Classify a single issue, run a live benchmark on unseen real
# issues, and a model card. Design follows the GitHub Primer issue-tracker look.
# Run: streamlit run app.py
import json
import os
import re

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score

from src.llm_explain import explain

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Issue Triage", page_icon="🏷️", layout="centered")

try:
    for _k in ("OPENROUTER_API_KEY", "EXPLAIN_MODEL"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass

COLOR = {"bug": "#f85149", "feature": "#3fb950", "question": "#f778ba", "documentation": "#58a6ff"}
DESC = {
    "bug": "Something is broken or behaves incorrectly.",
    "feature": "A request for new functionality or an enhancement.",
    "question": "Someone asking how to do something or whether behaviour is expected.",
    "documentation": "A request to add, fix, or clarify documentation.",
}
ROUTE = {
    "bug": "route to the bug board, ask for repro steps",
    "feature": "add to the backlog for triage",
    "question": "point to docs or move to Discussions",
    "documentation": "assign to a docs maintainer",
}
EXAMPLES = {
    "Bug": "Clicking export throws a 500 error when the table is empty. Stack trace points to a null pointer in ExportService.",
    "Feature": "It would be great to support dark mode in the settings page, with a toggle that remembers the choice.",
    "Question": "How do I configure the proxy settings when running behind a corporate firewall? Couldn't find it anywhere.",
    "Docs": "The README still references the old install command. Can someone update the getting-started section?",
}
LABELS = ["bug", "feature", "question", "documentation"]


@st.cache_resource
def load():
    pipe = joblib.load("models/classifier.pkl")
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    try:
        metrics = json.load(open("models/metrics.json"))
    except Exception:
        metrics = None
    return pipe, vocab, metrics


@st.cache_data
def load_samples():
    try:
        return pd.read_csv("data/test_sample.csv")
    except Exception:
        return None


pipe, VOCAB, METRICS = load()
clf = pipe.named_steps["clf"]
SAMPLES = load_samples()
VOCAB_UNI = set(w for w in VOCAB if " " not in w)


def build_text(title, body):
    return (str(title or "") + " " + str(body or "")).strip()[:4000]


def predict(text):
    pred = pipe.predict([text])[0]
    proba = pipe.predict_proba([text])[0]
    return pred, {c: float(p) for c, p in zip(clf.classes_, proba)}


def low_signal(text):
    toks = re.findall(r"\w+", text.lower())
    if not toks:
        return True, 0.0
    rec = sum(1 for w in toks if w in VOCAB_UNI)
    frac = rec / len(toks)
    vec = pipe.named_steps["tfidf"].transform([text])
    return (frac < 0.40 or vec.nnz < 4), frac


def evidence(text, label, k=8):
    vec = pipe.named_steps["tfidf"].transform([text])
    i = list(clf.classes_).index(label)
    coef = clf.coef_[i]
    scored = [(VOCAB[j], vec[0, j] * coef[j]) for j in vec.indices]
    scored = [(w, s) for w, s in scored if s > 0]
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:k]]


# Button callbacks. on_click runs before any widget instantiates on the rerun,
# so writing session_state here is what makes the text_area refresh reliably.
def use_example(body):
    st.session_state["issue"] = body
    st.session_state["src_true"] = None
    st.session_state["src_text"] = None


def use_random():
    pick = SAMPLES.sample(1)
    # never serve the same issue twice in a row
    if len(SAMPLES) > 1:
        while pick.index[0] == st.session_state.get("last_pick"):
            pick = SAMPLES.sample(1)
    st.session_state["last_pick"] = pick.index[0]
    row = pick.iloc[0]
    txt = build_text(row["title"], row["body"])
    st.session_state["issue"] = txt
    st.session_state["src_true"] = row["labels"]
    st.session_state["src_text"] = txt


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { max-width: 820px; padding-top: 2rem; }
    #MainMenu, footer { visibility:hidden; }
    [data-testid="stDecoration"] { display:none; }
    header[data-testid="stHeader"] { background:transparent; }
    @keyframes fadeUp { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
    @keyframes grow { from { width:0; } }
    .eyebrow { font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#8b949e;
               letter-spacing:.04em; }
    .h1 { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.9rem;
          color:#e6edf3; margin:.2rem 0 .3rem 0; }
    .sub { color:#8b949e; font-size:.92rem; margin-bottom:1rem; line-height:1.55; }
    .stat-row { display:flex; gap:.6rem; margin:.2rem 0 1rem 0; flex-wrap:wrap; }
    .stat { border:1px solid #30363d; border-radius:8px; background:#0d1117;
            padding:.5rem .8rem; min-width:120px; flex:1;
            transition:border-color .15s ease, transform .15s ease; }
    .stat:hover { border-color:#8b949e44; transform:translateY(-1px); }
    .stat .k { font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#e6edf3; font-weight:700; }
    .stat .l { font-size:.72rem; color:#8b949e; }
    .pill { display:inline-block; font-family:'JetBrains Mono',monospace; font-weight:600;
            font-size:.8rem; padding:.18rem .7rem; border-radius:2rem; }
    .card { border:1px solid #30363d; border-radius:10px; padding:1.05rem 1.15rem;
            background:#161b22; margin:.55rem 0; animation:fadeUp .25s ease;
            transition:border-color .15s ease; }
    .card:hover { border-color:#484f58; }
    .bar-track { background:#21262d; border-radius:4px; height:9px; flex:1; overflow:hidden; }
    .bar-track > div { animation:grow .5s ease; border-radius:4px; }
    .bar-row { display:flex; align-items:center; gap:.7rem; margin:.32rem 0;
               font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#8b949e; }
    .bar-name { width:120px; text-align:right; }
    .bar-val { width:42px; }
    .chip { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.78rem;
            padding:.14rem .5rem; margin:.15rem .25rem .15rem 0; border-radius:5px;
            border:1px solid #30363d; background:#0d1117; }
    .gh-table { width:100%; border-collapse:collapse; font-size:.8rem; }
    .gh-table th { text-align:left; color:#8b949e; font-weight:600; font-size:.74rem;
                   border-bottom:1px solid #30363d; padding:.4rem .5rem; }
    .gh-table td { border-bottom:1px solid #21262d; padding:.4rem .5rem; color:#c9d1d9;
                   font-family:'JetBrains Mono',monospace; font-size:.76rem; }
    .gh-table tr:hover td { background:#1c2128; }
    .ok { color:#3fb950; } .no { color:#f85149; }
    .note { border:1px solid #9e6a03; background:#211a02; color:#e3b341; border-radius:8px;
            padding:.6rem .8rem; font-size:.82rem; margin:.5rem 0; animation:fadeUp .25s ease; }
    .page-footer { margin-top:2.5rem; padding-top:.9rem; border-top:1px solid #21262d;
                   color:#6e7681; font-size:.76rem; font-family:'JetBrains Mono',monospace; }
    .page-footer a { color:#8b949e; text-decoration:none; }
    .page-footer a:hover { color:#58a6ff; }

    /* tabs — GitHub underline style */
    .stTabs [data-baseweb="tab-list"] { gap:.25rem; }
    .stTabs [data-baseweb="tab"] { font-size:.88rem; color:#8b949e; padding:.55rem .25rem;
                                   background:transparent; }
    .stTabs [data-baseweb="tab"]:hover { color:#e6edf3; }
    .stTabs [aria-selected="true"] { color:#e6edf3 !important; font-weight:600; }
    .stTabs [data-baseweb="tab-highlight"] { background-color:#f78166; height:2px; }
    .stTabs [data-baseweb="tab-border"] { background-color:#21262d; }

    /* buttons */
    .stButton > button { background:#21262d; color:#c9d1d9; border:1px solid #30363d;
                         border-radius:6px; font-weight:500; font-size:.85rem;
                         transition:all .15s ease; }
    .stButton > button:hover { background:#30363d; border-color:#8b949e; color:#e6edf3;
                               transform:translateY(-1px); }
    .stButton > button:active { transform:none; }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background:#238636; border:1px solid rgba(240,246,252,.1); color:#fff; font-weight:600; }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background:#2ea043; border-color:rgba(240,246,252,.1); }

    /* text area */
    .stTextArea [data-baseweb="textarea"] { background:#0d1117; border-color:#30363d;
                                            border-radius:8px; transition:border-color .15s ease,
                                            box-shadow .15s ease; }
    .stTextArea [data-baseweb="textarea"]:focus-within { border-color:#2f81f7;
                                            box-shadow:0 0 0 3px rgba(47,129,247,.15); }
    .stTextArea textarea { background:#0d1117; color:#e6edf3;
                           font-family:'JetBrains Mono',monospace; font-size:.84rem;
                           line-height:1.55; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">$ triage --classify</div>', unsafe_allow_html=True)
st.markdown('<div class="h1">GitHub Issue Triage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Reads an incoming issue and predicts its type — bug, feature, '
    'question, or documentation — so maintainers do not have to label by hand. '
    'A linear model over 1.27M real GitHub issues.</div>',
    unsafe_allow_html=True,
)

if METRICS:
    r = METRICS["results"][METRICS["selected_model"]]
    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat"><div class="k">{r["micro_f1"]*100:.1f}%</div><div class="l">micro F1 (30k test)</div></div>'
        f'<div class="stat"><div class="k">{r["macro_f1"]*100:.1f}%</div><div class="l">macro F1</div></div>'
        f'<div class="stat"><div class="k">1.27M</div><div class="l">real issues trained</div></div>'
        f'<div class="stat"><div class="k">4</div><div class="l">labels</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

tab1, tab2, tab3 = st.tabs(["Classify", "Live benchmark", "Model card"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    st.caption("Load an example, pull a random real issue, or paste your own.")
    cols = st.columns(5)
    for col, (name, body) in zip(cols[:4], EXAMPLES.items()):
        col.button(name, use_container_width=True, on_click=use_example, args=(body,))
    if SAMPLES is not None:
        cols[4].button("🎲 Random real", use_container_width=True, on_click=use_random)

    text = st.text_area(
        "Issue title + body", key="issue", height=170,
        placeholder="e.g. The export button throws a 500 error when the table is empty...",
    )
    use_llm = st.toggle("Explain with an LLM", value=False)
    go = st.button("Classify issue", type="primary", use_container_width=True)

    if go:
        issue = (text or "").strip()
        if not issue:
            st.warning("Paste an issue first.")
            st.stop()

        pred, conf = predict(issue)
        color = COLOR.get(pred, "#8b949e")

        weak, frac = low_signal(issue)
        if weak:
            st.markdown(
                f'<div class="note">⚠ Low signal — only {frac*100:.0f}% of the words match the '
                f'English issue vocabulary the model was trained on. This does not look like a '
                f'typical issue, so treat the prediction below as unreliable.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="card">
              <span class="pill" style="background:{color}22;color:{color};border:1px solid {color}66;">
                {pred}</span>
              <span style="color:#8b949e;font-family:'JetBrains Mono',monospace;font-size:.8rem;
                    margin-left:.6rem;">confidence {conf[pred]:.0%}</span>
              <div style="color:#c9d1d9;margin-top:.6rem;">{DESC[pred]}</div>
              <div style="color:#6e7681;font-family:'JetBrains Mono',monospace;font-size:.78rem;
                    margin-top:.4rem;">&rarr; {ROUTE[pred]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        src_true = st.session_state.get("src_true")
        if src_true and issue == (st.session_state.get("src_text") or "").strip():
            ok = (src_true == pred)
            mk = "✓ correct" if ok else "✗ wrong"
            cc = "#3fb950" if ok else "#f85149"
            st.markdown(
                f'<div class="card" style="border-color:{cc}55;">'
                f'<span style="color:#8b949e;font-size:.82rem;">Ground truth for this real issue: </span>'
                f'<span class="pill" style="background:{COLOR[src_true]}22;color:{COLOR[src_true]};'
                f'border:1px solid {COLOR[src_true]}66;">{src_true}</span>'
                f'<span style="color:{cc};font-family:JetBrains Mono,monospace;font-size:.8rem;'
                f'margin-left:.6rem;">model {mk}</span></div>',
                unsafe_allow_html=True,
            )

        rows = ""
        for c in sorted(conf, key=conf.get, reverse=True):
            cc = COLOR.get(c, "#8b949e")
            rows += (
                f'<div class="bar-row"><span class="bar-name">{c}</span>'
                f'<div class="bar-track"><div style="width:{conf[c]*100:.1f}%;height:100%;'
                f'background:{cc};"></div></div><span class="bar-val">{conf[c]*100:.0f}%</span></div>'
            )
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

        words = evidence(issue, pred)
        if words:
            chips = "".join(
                f'<span class="chip" style="color:{color};border-color:{color}66;">{w}</span>'
                for w in words
            )
            st.markdown(
                '<div style="color:#8b949e;font-size:.85rem;margin:.2rem 0 .4rem;">'
                f'Words that drove this label</div><div>{chips}</div>',
                unsafe_allow_html=True,
            )

        if use_llm:
            with st.spinner("Asking the LLM..."):
                st.markdown(
                    '<div style="color:#8b949e;font-size:.85rem;margin-top:1rem;">LLM take</div>',
                    unsafe_allow_html=True,
                )
                st.write(explain(issue, pred, conf[pred]))

# ---------------------------------------------------------------- TAB 2
with tab2:
    if SAMPLES is None:
        st.info("Benchmark sample not bundled.")
    else:
        st.markdown(
            '<div style="color:#c9d1d9;font-size:.9rem;margin-bottom:.3rem;">'
            'Run the model live on a random batch of <b>real GitHub issues it never saw in '
            'training</b>, and compare every prediction to the maintainer\'s true label. '
            'This is the honest test — not the curated examples.</div>',
            unsafe_allow_html=True,
        )
        n = st.slider("How many random unseen issues to test", 50, 800, 200, step=50)
        run = st.button("Run benchmark", type="primary", use_container_width=True)
        if run:
            samp = SAMPLES.sample(n).reset_index(drop=True)
            texts = [build_text(t, b) for t, b in zip(samp["title"], samp["body"])]
            preds = pipe.predict(texts)
            y = samp["labels"].values
            acc = accuracy_score(y, preds)
            macro = f1_score(y, preds, average="macro", labels=LABELS)
            per = f1_score(y, preds, average=None, labels=LABELS)

            st.markdown(
                f'<div class="stat-row" style="margin-top:.6rem;">'
                f'<div class="stat"><div class="k">{acc*100:.1f}%</div><div class="l">accuracy on {n} unseen</div></div>'
                f'<div class="stat"><div class="k">{macro*100:.1f}%</div><div class="l">macro F1</div></div>'
                + "".join(
                    f'<div class="stat"><div class="k" style="color:{COLOR[l]};">{s*100:.0f}%</div>'
                    f'<div class="l">{l} F1</div></div>'
                    for l, s in zip(LABELS, per)
                )
                + '</div>',
                unsafe_allow_html=True,
            )

            show = samp.head(16)
            trows = ""
            for i in range(len(show)):
                p = preds[i]; tr = show["labels"].values[i]
                ok = p == tr
                title = str(show["title"].values[i])[:58].replace("<", "&lt;")
                trows += (
                    f'<tr><td>{title}</td>'
                    f'<td style="color:{COLOR.get(p,"#8b949e")};">{p}</td>'
                    f'<td style="color:{COLOR.get(tr,"#8b949e")};">{tr}</td>'
                    f'<td class="{"ok" if ok else "no"}">{"✓" if ok else "✗"}</td></tr>'
                )
            st.markdown(
                '<div class="card"><table class="gh-table"><tr>'
                '<th>Issue title</th><th>predicted</th><th>true</th><th></th></tr>'
                f'{trows}</table>'
                '<div style="color:#6e7681;font-size:.72rem;margin-top:.5rem;">'
                'Showing 16 of the batch. Re-run for a fresh random sample.</div></div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------- TAB 3
with tab3:
    if METRICS:
        r = METRICS["results"][METRICS["selected_model"]]
        st.markdown(
            f"**Model.** TF-IDF (word + n-gram features) into a logistic-regression "
            f"classifier. Lightweight, no GPU, sub-second inference.\n\n"
            f"**Performance (held-out 30k test).** micro F1 {r['micro_f1']*100:.1f}% · "
            f"macro F1 {r['macro_f1']*100:.1f}%. Per-class F1: "
            f"bug {r['per_class_f1']['bug']:.2f} · feature {r['per_class_f1']['feature']:.2f} · "
            f"question {r['per_class_f1']['question']:.2f} · documentation {r['per_class_f1']['documentation']:.2f}.\n\n"
            f"**Data.** NLBSE'23 Issue Report Classification — {METRICS['train_size']:,} train / "
            f"{METRICS['test_size']:,} test sampled from 1.27M real, maintainer-labeled GitHub issues.\n\n"
            f"**Published baselines (full test).** fastText 85.1% · RoBERTa 89.1% micro F1."
        )
    st.markdown(
        "**Honest limitations.** The two majority classes (bug, feature) are strong; the rare "
        "classes (question, documentation) are weak because they overlap semantically with the "
        "others. Standard rebalancing (class weights, SMOTE) lowered macro F1 rather than raising "
        "it — the imbalance is a data problem, not a reweighting one. With more time: LLM-based "
        "augmentation of the minority classes, a fine-tuned transformer, and confidence-thresholded "
        "abstention.\n\nSource: github.com/Obito116/issue-triage"
    )

st.markdown(
    '<div class="page-footer">TF-IDF + logistic regression · trained on 1.27M real GitHub issues · '
    '<a href="https://github.com/Obito116/issue-triage">source</a></div>',
    unsafe_allow_html=True,
)
