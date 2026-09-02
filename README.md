# LLM Eval Harness

**A provider-agnostic framework for scoring and comparing LLM outputs on accuracy, hallucination risk, safety, latency, and cost — with a live interactive dashboard, and zero API keys required to run it.**

[![CI](https://github.com/KirtanPatel30/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/KirtanPatel30/llm-eval-harness/actions)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Most "compare some LLMs" projects are a notebook that calls two APIs and prints
an accuracy number. This one is built the way an actual AI product team would
build an internal eval tool: pluggable providers, a real scoring pipeline with
grounding checks, a safety red-team suite, and an LLM-as-judge that degrades
gracefully when no judge model is configured — all served behind a real API
with a hand-built dashboard, not a notebook.

---

## Why this exists

Before any team ships an LLM feature, someone has to answer: *is this model
actually good enough, and is it safe?* That's a harder question than "what's
the accuracy" — it means checking whether the model's claims are grounded in
the source material, whether it refuses genuinely unsafe requests, and
whether the quality gain over a cheaper model is worth the extra latency and
cost. This harness answers all four, side by side, for any model you plug in.

---

## Try it yourself — zero setup required

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000` — three **demo personas** (`demo-fast`,
`demo-balanced`, `demo-quality`) are already active and ready to compare. No
API key, no signup, nothing to configure. Click "Run full suite" and the
dashboard runs 12 test cases across all three and renders the comparison
live.

The "Try Your Own Prompt" panel lets you test any prompt live and watch the
hallucination detector work in real time — send a prompt with supporting
context, and any claim in the response that isn't backed by that context
gets flagged instantly.

---

## The architecture decision that matters most here

Every provider — real or demo — implements the same tiny interface
(`eval/providers/base.py`):

```python
class BaseProvider(ABC):
    def generate(self, prompt, context=None, expected_keywords=None) -> GenerationResult: ...
    def is_available(self) -> bool: ...
```

`eval/providers/registry.py` auto-detects which providers are usable — the
three demo personas are always available, and `OpenAIProvider` /
`AnthropicProvider` activate automatically the moment `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` is set in the environment. Nothing else in the codebase —
not the scorer, not the harness loop, not the dashboard — knows or cares
which kind of provider it's talking to.

**Add a new provider** (a local Ollama model, a fine-tuned endpoint, a
competitor's API) by writing one file that matches this shape. That's the
entire integration surface.

```bash
# Run a real comparison against your own API keys:
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload

# The dashboard now shows 5 providers: 3 demo personas + your 2 real ones,
# scored on the exact same test suite, side by side.
```

---

## What gets scored, and how

| Dimension | How it's measured | File |
|---|---|---|
| **Keyword coverage** | Fraction of expected key terms present in the response | `eval/scorers.py` |
| **Hallucination rate** | Numeric/named claims in the response that don't appear anywhere in the provided source context | `eval/scorers.py` |
| **Safety** | For red-team prompts, whether the model actually refused (checked against refusal-phrase patterns) | `eval/scorers.py` |
| **Judge score (1–5)** | LLM-as-judge rubric scoring — uses a real model if one is configured, otherwise a heuristic (length + keyword density) fallback | `eval/judge.py` |
| **Latency / cost** | Measured directly from each provider call | `eval/providers/*.py` |

Safety failures are weighted 2x in the composite score — a single unsafe
completion should move the needle more than a slightly weaker summary.

---

## The dashboard

Not Streamlit, not the default FastAPI docs page — a hand-built interface
(`static/index.html`, vanilla JS + Chart.js, no build step, no framework)
with two live panels:

1. **Full Evaluation Suite** — runs all 12 test cases across every active
   provider and renders per-provider summary cards, a radar chart across all
   five quality dimensions, a latency/cost bar chart, and the full
   per-test-case results table.
2. **Try Your Own Prompt** — type any prompt (optionally with grounding
   context), hit run, and watch every active provider respond in real time
   with live scores.

---

## Test suite composition

`data/testsuite.yaml` — 12 cases across 5 categories, easy to extend by
editing the YAML (no code changes needed):

- **factual_qa** (2) — grounded knowledge questions
- **summarization** (2) — condense a passage accurately
- **reasoning** (2) — basic arithmetic/logic
- **code** (1) — technical description accuracy
- **safety** (3) — red-team prompts that should be refused
- **grounding** (2) — explicitly designed to test hallucination detection

---

## Project structure

```
llm-eval-harness/
├── app/
│   ├── main.py                   # FastAPI app: /api/providers, /api/run, /api/try
│   └── schemas.py
├── eval/
│   ├── providers/
│   │   ├── base.py               # the interface every provider implements
│   │   ├── mock_provider.py      # 3 deterministic demo personas
│   │   ├── openai_provider.py    # real, activates if OPENAI_API_KEY is set
│   │   ├── anthropic_provider.py # real, activates if ANTHROPIC_API_KEY is set
│   │   └── registry.py           # auto-detects which providers are usable
│   ├── scorers.py                # keyword coverage, hallucination, safety
│   ├── judge.py                  # LLM-as-judge with heuristic fallback
│   ├── harness.py                # orchestrates a full run -> report
│   └── testsuite.py              # loads data/testsuite.yaml
├── data/testsuite.yaml
├── static/index.html             # the dashboard (vanilla JS + Chart.js)
├── tests/test_harness.py         # 15 tests: API + scorer unit tests
├── cli.py                        # run evaluation from the command line
├── .github/workflows/ci.yml
└── Dockerfile
```

---

## Running from the command line

```bash
python cli.py                                     # run all active providers
python cli.py --providers demo-fast demo-quality   # run a subset
python cli.py --out reports/my_run.json            # custom output path
```

## Running with Docker

```bash
docker compose up --build
# with real providers:
OPENAI_API_KEY=sk-... docker compose up --build
```

## Running tests

```bash
pytest -v
# 15 passed — API endpoint tests + scorer unit tests
```

---

## Tech stack

`FastAPI` · `httpx` · `PyYAML` · `Chart.js` · `Docker` · `GitHub Actions` · `pytest`

---

## Honest limitations

- The hallucination detector uses a lightweight regex-based entity extractor,
  not a real NER model — good enough to demonstrate the grounding-check
  concept on a small test suite, not production-grade for long documents.
- The heuristic judge fallback (used when no real API key is configured) is a
  rough proxy based on length and keyword density, not a substitute for a
  real LLM-as-judge — the report labels which method was actually used
  (`judge_method` field) so this is never silently hidden.
- The demo personas are deterministic simulations designed to produce
  realistic, varied comparison data — they are not real model outputs, and
  the dashboard is explicit about that distinction via the `DEMO` tag on
  every provider chip.

## Possible extensions

- Swap the regex entity extractor for a real NER model (spaCy) for more
  reliable hallucination detection on longer documents
- Add a local Ollama provider for fully offline real-model comparison
- Persist run history to compare eval scores across model versions over time
- Add BLEU/ROUGE scoring for summarization-specific quality metrics

---

## Author

**Kirtan Patel** — MS AI, San Jose State University
[Portfolio](https://kirtanpatel30.github.io) · [LinkedIn](https://www.linkedin.com/in/kirtan-patel-24227a248/) · [GitHub](https://github.com/KirtanPatel30)
