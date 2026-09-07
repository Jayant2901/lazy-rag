# lazy-rag

Adaptive retrieval for RAG: only retrieve when the model actually needs to.

Most RAG pipelines retrieve on every query, even when the LLM already knows
the answer. This project builds a retrieval trigger based on the model's own
confidence, and benchmarks it against two baselines:

- **no-RAG** — pure LLM, no retrieval ever
- **always-RAG** — retrieve for every query
- **lazy-rag** — retrieve only when self-reported confidence falls below a threshold

The goal isn't just a working pipeline — it's the comparison: how much
retrieval can you skip before answer quality drops, and where does it break.

Related work this builds on / benchmarks against: Self-RAG (Asai et al.),
FLARE (Jiang et al.), Adaptive-RAG (Jeong et al.).

## Project structure

```
src/
  config.py       # env/config loading
  llm.py          # Anthropic API wrapper
  retriever.py    # embedding index + retrieval
  confidence.py   # self-verbalized confidence trigger
  pipeline.py     # AdaptiveRAG, AlwaysRAG, NoRAG pipelines
eval/
  metrics.py      # EM / F1 / retrieval-rate scoring
  run_eval.py     # runs all pipelines over a QA dataset, prints comparison table
data/
  corpus.jsonl    # retrieval corpus (id, text)
  qa.jsonl        # eval set (question, answer)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Data

Drop a corpus into `data/corpus.jsonl` (one JSON object per line: `{"id": ..., "text": ...}`)
and a QA eval set into `data/qa.jsonl` (`{"question": ..., "answer": ...}`).
Good starting benchmarks: PopQA, TriviaQA (favor adaptive retrieval), HotpotQA (favors always-retrieve).

## Running the eval

```bash
python eval/run_eval.py --threshold 0.6
```

Prints a comparison table: EM/F1, retrieval rate, and estimated cost per pipeline.

## Roadmap

- [x] Repo scaffold
- [ ] Load a benchmark QA set + corpus
- [ ] Confidence-threshold sweep, quality-vs-retrieval-rate plot
- [ ] Token-level uncertainty trigger (FLARE-style) as a second variant
- [ ] Error analysis on adaptive-RAG failures
- [ ] Write-up (method, results, limitations)
