# Streamlit app for the issue triage model.
# Paste an issue -> predicted type, confidence, and the words that drove it.
# Design leans on the GitHub issue-tracker look: real label colors, mono type,
# label pills. Run: streamlit run app.py
import json
import os

import joblib
import streamlit as st

from src.llm_explain import explain

# read a local .env if there is one (for the LLM toggle key)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Issue Triage", page_icon="🏷️", layout="centered")

# deployed on Streamlit Cloud the key lives in Secrets - copy it into the env
try:
    for _k in ("OPENROUTER_API_KEY", "EXPLAIN_MODEL"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass

# GitHub Primer label colors - one per class, used everywhere for consistency.
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


@st.cache_resource
def load():
    pipe = joblib.load("models/classifier.pkl")
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    try:
        metrics = json.load(open("models/metrics.json"))
    except Exception:
        metrics = None
    return pipe, vocab, metrics


pipe, VOCAB, METRICS = load()
clf = pipe.named_steps["clf"]


def evidence(text, label, k=8):
    # which words in *this* issue pushed it toward the predicted label.
    # score = the word's tf-idf weight in the issue x its weight for that class.
    vec = pipe.named_steps["tfidf"].transform([text])
    i = list(clf.classes_).index(label)
    coef = clf.coef_[i]
    scored = [(VOCAB[j], vec[0, j] * coef[j]) for j in vec.indices]
    scored = [(w, s) for w, s in scored if s > 0]
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:k]]


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { max-width: 760px; padding-top: 2.4rem; }
    .eyebrow { font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: #8b949e;
               letter-spacing: .04em; margin-bottom: .3rem; }
    .h1 { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 2rem;
          color: #e6edf3; margin: 0 0 .4rem 0; }
    .sub { color: #8b949e; font-size: .95rem; margin-bottom: 1.4rem; }
    .pill { display:inline-block; font-family:'JetBrains Mono',monospace; font-weight:600;
            font-size:.8rem; padding:.18rem .7rem; border-radius:2rem; }
    .card { border:1px solid #30363d; border-radius:10px; padding:1.1rem 1.2rem;
            background:#161b22; margin:.6rem 0; }
    .bar-track { background:#21262d; border-radius:4px; height:9px; flex:1; overflow:hidden; }
    .bar-row { display:flex; align-items:center; gap:.7rem; margin:.35rem 0;
               font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#8b949e; }
    .bar-name { width:110px; text-align:right; }
    .bar-val { width:42px; }
    .chip { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.78rem;
            padding:.14rem .5rem; margin:.15rem .25rem .15rem 0; border-radius:5px;
            border:1px solid #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">$ triage --input issue</div>', unsafe_allow_html=True)
st.markdown('<div class="h1">GitHub Issue Triage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Every team gets a flood of incoming requests — bug reports, feature '
    'ideas, questions, doc fixes. This sorts each one automatically and shows the words '
    'behind the call. Built on 1.27M real GitHub issues.</div>',
    unsafe_allow_html=True,
)

st.caption("Try an example")
cols = st.columns(len(EXAMPLES))
for col, (name, body) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state["issue"] = body

text = st.text_area(
    "Issue title + body",
    key="issue",
    height=160,
    placeholder="e.g. The export button throws a 500 error when the table is empty...",
)
use_llm = st.toggle("Explain with an LLM", value=False)
go = st.button("Classify issue", type="primary", use_container_width=True)

if go:
    issue = (text or "").strip()
    if not issue:
        st.warning("Paste an issue first.")
        st.stop()

    pred = pipe.predict([issue])[0]
    proba = pipe.predict_proba([issue])[0]
    conf = {c: float(p) for c, p in zip(clf.classes_, proba)}
    color = COLOR.get(pred, "#8b949e")

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

with st.expander("About this model"):
    if METRICS:
        r = METRICS["results"][METRICS["selected_model"]]
        st.markdown(
            f"- **Model:** TF-IDF + Logistic Regression\n"
            f"- **Test micro F1:** {r['micro_f1']*100:.1f}%  ·  "
            f"**macro F1:** {r['macro_f1']*100:.1f}%\n"
            f"- **Data:** NLBSE'23, 1.27M real GitHub issues "
            f"({METRICS['train_size']:,} train / {METRICS['test_size']:,} test sampled)\n"
            f"- **Published baselines:** fastText 85.1% · RoBERTa 89.1% micro F1"
        )
    st.markdown(
        "The four labels and their colors mirror real GitHub issue labels. Macro F1 sits "
        "well below micro F1 because the rare classes (question, documentation) are hard — "
        "the report covers why SMOTE didn't fix it."
    )
