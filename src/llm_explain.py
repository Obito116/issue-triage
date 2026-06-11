# Short LLM explanation for a prediction. The classifier already picked the
# label - this just says why in a sentence or two. Uses OpenRouter (OpenAI-compatible).
import os

MODEL = os.getenv("EXPLAIN_MODEL", "google/gemini-3.5-flash")


def explain(issue_text, label, confidence):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return "Set OPENROUTER_API_KEY to turn on explanations."

    try:
        from openai import OpenAI

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        # Reasoning models spend tokens thinking before they answer, so the
        # budget has to cover both - 160 was enough for gpt-4o-mini but cuts
        # Gemini off mid-sentence. Keep the visible thinking effort low.
        r = client.chat.completions.create(
            model=MODEL,
            max_tokens=10000,
            extra_body={"reasoning": {"effort": "low"}},
            messages=[
                {
                    "role": "system",
                    "content": "You explain why a GitHub issue got a given label. "
                    "Don't argue with the label. In 1-2 sentences, point at the words "
                    "or intent in the issue that justify it.",
                },
                {
                    "role": "user",
                    "content": f"Issue:\n{issue_text[:2000]}\n\n"
                    f"Label: {label} (confidence {confidence:.0%}). Why does it fit?",
                },
            ],
        )
        out = (r.choices[0].message.content or "").strip()
        return out or "(the model returned an empty explanation — try again)"
    except Exception as e:
        return f"(explanation unavailable: {e})"
