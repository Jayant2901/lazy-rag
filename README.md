# lazy-rag

[![CI](https://github.com/Jayant2901/lazy-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Jayant2901/lazy-rag/actions/workflows/ci.yml)

Adaptive retrieval for RAG: only retrieve when the model actually needs to.

Most RAG pipelines retrieve on every query, even when the LLM already knows
the answer. This project builds a retrieval trigger based on the model's own
self-reported confidence, and benchmarks it against two baselines:

- **no-RAG** — pure LLM, no retrieval ever
- **always-RAG** — retrieve for every query
- **lazy-rag** — retrieve only when self-reported confidence falls below a threshold
- **lazy-rag-consistency** — same gate, but the trigger is self-consistency
  (sample agreement across repeated draws) instead of self-verbalized confidence

Related work this builds on / benchmarks against: Self-RAG (Asai et al.),
FLARE (Jiang et al.), Adaptive-RAG (Jeong et al.), SelfCheckGPT (Manakul et al.).

## Finding: self-verbalized confidence is a weak retrieval trigger

Benchmark: 40 questions sampled from [PopQA](https://huggingface.co/datasets/akariasai/PopQA)
(long-tail, low-popularity entity facts — deliberately hard for a model to know
from pretraining alone), retrieving against a corpus of Wikipedia summaries for
the relevant entities. Model: `qwen/qwen3.8-27b` via Groq. All EM figures below
are percentile bootstrap means with 95% CIs (1000 resamples); pairwise
comparisons use an exact two-sided sign test on paired per-question outcomes.
EM/F1 are scored against the *best-matching* answer in PopQA's full alias list
per question (not just the first alias — see caveat below).

| pipeline | EM | 95% CI | F1 | retrieval rate |
|---|---|---|---|---|
| no-rag | 0.000 | [0.00, 0.00] | 0.048 | 0% |
| always-rag | 0.375 | [0.23, 0.53] | 0.458 | 100% |
| lazy-rag (τ=0.6) | 0.175 | [0.08, 0.30] | 0.258 | 0% |
| lazy-rag-consistency (τ=0.6) | 0.300 | [0.18, 0.45] | 0.408 | 100% |

- lazy-rag vs always-rag: p=0.057 (not significant at n=40) — always-rag won
  11 of 14 discordant pairs. At τ=0.6 the gate is badly miscalibrated in this
  run: it never triggered retrieval at all (see root cause below).
- lazy-rag vs no-rag: **p=0.016** (significant) — lazy-rag won all 7
  discordant pairs.
- lazy-rag-consistency vs always-rag: p=0.375 (not significant) — both
  retrieved on 100% of questions this run, so the 5 discordant pairs are LLM
  sampling noise in the RAG generation call itself, not a gating difference.
- lazy-rag-consistency vs no-rag: **p<0.001** (significant) — consistency
  won all 12 discordant pairs.

**Self-consistency is a strictly more conservative (and better-behaved)
trigger than self-verbalized confidence.** At the same τ=0.6, self-verbalized
confidence skipped retrieval on *every* question this run (0% retrieval,
matching the "chronically overconfident" pattern below), while self-consistency
triggered retrieval on *every* question (100%) — statistically
indistinguishable from always-rag. That's the practical middle ground the
roadmap called for: since Groq's free tier doesn't expose `logprobs` (so true
FLARE-style token-probability triggering isn't available), self-consistency
is at least a *safe* substitute — it fails toward "retrieve too much" rather
than "confidently skip retrieval when wrong," which is the more expensive
failure mode of the two. It does not yet demonstrate the gate can be both
selective *and* accurate; that needs a threshold sweep for this variant too
(open item below).

**Retrieval isn't the bottleneck.** `Recall@1 = 0.975, Recall@3 = Recall@5 =
1.000` on this corpus (`python -m eval.eval_retrieval`) — the gold document
is almost always retrieved when retrieval is triggered. Nearly every failure
of lazy-rag is therefore a *calibration* failure (the gate deciding not to
retrieve when it should have), not a retrieval-quality failure.

Sweeping the confidence threshold from 0.0 to 1.0 shows why the gate
struggles, across two different models:

![quality vs retrieval rate, qwen vs gpt-oss-20b](data/sweep_plot_multimodel.png)

| τ | qwen EM | qwen 95% CI | qwen retrieval | gpt-oss-20b EM | gpt-oss-20b 95% CI | gpt-oss-20b retrieval |
|---|---|---|---|---|---|---|
| 0.0 | 0.175 | [0.08, 0.30] | 0% | — | — | 0% |
| 0.3–0.9 | ~0.20 | [0.08, 0.33] | 5–10% | ~0.10–0.15 | wide, overlapping | ~70% |
| 0.95 | 0.225 | [0.10, 0.35] | 22.5% | ~0.10–0.15 | wide, overlapping | ~70–85% |
| 1.0 | 0.325 | [0.20, 0.48] | 65% | ~0.10–0.15 | wide, overlapping | 85% |

**Honest read, not oversold:** for qwen, EM is essentially flat across
τ=0.3–0.9 — the 95% CIs for every threshold in that range overlap almost
completely, so the apparent trend is noise around one plateau at n=40, not a
real effect. Only τ=1.0 stands out with a higher point estimate, and even
that overlaps substantially with τ=0.95's interval. For gpt-oss-20b the
pattern is non-monotonic and, again, every threshold's CI overlaps every
other's — at this sample size the confidence signal doesn't reliably
discriminate thresholds from each other for either model. The one thing both
models agree on: retrieval rate only really moves once τ approaches 1.0,
meaning the gate has to become "almost always-rag" before it behaves
differently from a flat baseline. (The gpt-oss-20b sweep predates the
answers-list scoring fix below and hasn't been re-run — its qualitative
"flat, then rises near τ=1.0" shape is what's load-bearing here, not its
absolute EM values.)

**Root cause, confirmed directly:** of 40 questions, 29 had the model
self-report confidence ≥0.7 while giving a wrong answer — including several
at confidence *1.00*. Examples (`data/overconfident_failures.json` has all
29; exact figures shift run-to-run since LLM sampling isn't deterministic,
but the qualitative pattern is stable):

| question | confidence | model said (no retrieval) | gold | correct w/ retrieval? |
|---|---|---|---|---|
| Who is the author of *So Disdained*? | 1.00 | Steven Erikson | Nevil Shute | yes |
| Who was the director of *Tum Bin*? | 1.00 | I. S. Johar | Anubhav Sinha | yes |
| Who was the screenwriter for *Exam*? | 0.95 | Jeff Natteford | Stuart Hazeldine | yes |

The model isn't being deceptive — it's poorly calibrated. Asking an LLM to
verbalize a confidence score doesn't reliably distinguish "I actually know
this" from "this sounds like something I should know." That's consistent
with the broader finding in the calibration literature that natural-language
self-reported confidence correlates weakly with actual correctness.

**Found and fixed a scoring bug:** `eval/prepare_popqa.py` originally kept
only the first entry of PopQA's `possible_answers` list, so EM/F1 were scored
against a single alias even though PopQA accepts several per question (113 of
150 questions in this dataset have more than one accepted answer, e.g.
"Catholic Church" / "Roman Catholic Church" / "Church" / "Roman Apostolic
Catholic Church" for one question). `eval/prepare_popqa.py` now keeps the
full list and `eval/metrics.py` scores EM/F1 as the max over all accepted
aliases. This changed the headline numbers — e.g. `lazy-rag` EM moved from
0.100 to 0.175 on the same threshold — so treat every number in this README
as post-fix. The qualitative finding (self-verbalized confidence is a weak,
overconfident trigger) is unchanged; the absolute scores were an
underestimate before, not an overestimate.

**Implication for future work:** self-consistency (above) is a workable
middle ground, but a trigger grounded in the generation itself — e.g.
token-level output probability (FLARE-style) — would likely discriminate
known-vs-unknown better than either self-verbalized confidence or sample
agreement. Neither Groq free-tier model used here exposes `logprobs`, so
that remains untested.

## Project structure

```
demo.py                    # 30-second CLI: shows confidence, gate decision, and answer
src/
  config.py            # env/config loading
  llm.py                # Groq API wrapper, retries on rate limits
  retriever.py           # embedding index + retrieval (FAISS, cached by corpus hash) + Recall@k
  confidence.py           # self-verbalized confidence trigger
  consistency.py           # self-consistency (sample-agreement) trigger
  pipeline.py               # no-rag / always-rag / lazy-rag / lazy-rag-consistency pipelines
eval/
  metrics.py              # EM / F1 scoring (max over gold-answer aliases), bootstrap_ci
  significance.py          # exact paired sign test
  provenance.py             # generated_at / git_sha / model metadata for result JSONs
  run_eval.py                # runs the selected pipelines over a QA set, prints comparison + significance
  threshold_sweep.py         # sweeps lazy-rag across confidence thresholds (with bootstrap CIs)
  plot_sweep.py                # renders quality-vs-retrieval-rate chart, supports multi-model overlay
  eval_retrieval.py             # standalone Recall@k reporter
  error_analysis.py              # finds confident-but-wrong cases
  prepare_popqa.py                # builds corpus.jsonl / qa.jsonl from PopQA + Wikipedia
tests/
  test_metrics.py                 # normalize/EM/F1/bootstrap_ci unit tests
  test_confidence.py               # confidence-parsing unit tests
  test_consistency.py               # self-consistency scoring unit tests
  test_significance.py               # sign_test unit tests
  test_retriever.py                   # retrieval/caching/recall_at_k unit tests (mocked model/index)
  test_pipeline.py                     # lazy_rag threshold-gating unit tests
  test_llm.py                           # retry/backoff parsing unit tests
data/
  corpus.jsonl              # retrieval corpus (id, title, text)
  qa.jsonl                    # full eval set (question, answers, gold_title)
  qa_subset.jsonl               # smaller subset for fast iteration under free-tier rate limits
  eval_results_n40.json          # 4-pipeline eval output (EM/F1/CI/significance)
  sweep_results.json              # qwen threshold sweep output
  sweep_results_gpt-oss-20b.json   # gpt-oss-20b threshold sweep output (predates the scoring fix)
  sweep_plot_multimodel.png         # quality vs retrieval-rate chart, both models overlaid
  overconfident_failures.json        # confidence>=0.7-but-wrong cases
  corpus.example.jsonl                # tiny offline demo corpus (no network/HF auth needed)
  qa.example.jsonl                     # tiny offline demo QA set
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY` (free at
[console.groq.com](https://console.groq.com)) and `HF_TOKEN` (free at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
avoids a rate-limit warning on the embedding-model download).

## Try it without the full dataset

`data/corpus.example.jsonl` / `data/qa.example.jsonl` are a tiny hand-written
corpus + QA set that don't require network access or Hugging Face auth, so
you can try the pipeline immediately after setup:

```bash
python -m demo "What year was the Eiffel Tower completed?" --corpus data/corpus.example.jsonl
```

`demo.py` prints the model's self-reported confidence, the gate's
retrieve-or-skip decision, and the final answer — a 30-second look at what
the eval scripts below measure at scale.

## Reproducing the results

```bash
python eval/prepare_popqa.py        # builds data/corpus.jsonl + data/qa.jsonl from PopQA
python -m eval.run_eval --qa data/qa_subset.jsonl \
    --pipelines no-rag always-rag lazy-rag lazy-rag-consistency \
    --out data/eval_results_n40.json
python -m eval.threshold_sweep --qa data/qa_subset.jsonl   # sweeps thresholds, writes sweep_results.json
python -m eval.plot_sweep --results data/sweep_results.json data/sweep_results_gpt-oss-20b.json \
    --labels qwen3.8-27b gpt-oss-20b --out data/sweep_plot_multimodel.png
python -m eval.eval_retrieval --qa data/qa_subset.jsonl    # Recall@k
python -m eval.error_analysis                              # finds confident-but-wrong cases
pytest tests/ -v                                            # unit tests (no network/API calls)
```

Note on model choice: Groq's free tier has per-model daily token caps, and
larger models (`gpt-oss-120b`/`20b`) get exhausted quickly across iterative
runs. Results above use two models with separate quota buckets —
`qwen/qwen3.8-27b` and `openai/gpt-oss-20b` — specifically so the finding
could be checked for consistency across models rather than resting on one.
Set `LAZY_RAG_MODEL` in `.env` to try another.

Evaluation was run at n=40 rather than the full 150-question set: Groq's
free-tier tokens-per-minute limit degrades sharply on sustained runs (call
latency climbs from ~1s to 100s+ per item after ~20-30 calls in a window),
making a 450-call full run impractical to complete without multi-hour waits.
The n=40 subset was chosen deliberately small enough to finish reliably
while still supporting bootstrap CIs and a paired significance test.

## License

[MIT](LICENSE)

## Roadmap

- [x] Repo scaffold
- [x] Load a benchmark QA set + corpus (PopQA + Wikipedia)
- [x] Confidence-threshold sweep, quality-vs-retrieval-rate plot
- [x] Error analysis on adaptive-RAG failures
- [x] Bootstrap confidence intervals + paired significance testing
- [x] Recall@k to isolate retrieval quality from generation quality
- [x] Second-model replication sweep (gpt-oss-20b) to check the finding holds
      across models
- [x] Unit tests + CI
- [x] Embedding-index caching
- [x] Fixed EM/F1 to score against PopQA's full answer-alias list, not just
      the first one
- [x] Self-consistency (sample-agreement) trigger as a second variant,
      evaluated and reported above
- [ ] Threshold sweep for lazy-rag-consistency (only τ=0.6 reported so far)
- [ ] Token-level uncertainty trigger (FLARE-style) as a third variant, to
      test whether it discriminates known-vs-unknown better than either
      self-verbalized confidence or self-consistency — blocked on a Groq
      model that exposes `logprobs`
- [ ] A second QA dataset beyond PopQA, to check the finding isn't specific
      to long-tail entity questions
