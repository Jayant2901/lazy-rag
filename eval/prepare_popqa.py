import ast
import json
import time
from pathlib import Path

import requests
from datasets import load_dataset

N_SAMPLES = 150
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_answers(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    return ast.literal_eval(raw)


def wiki_summary(title: str) -> str | None:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "lazy-rag-dataprep/1.0"})
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    extract = resp.json().get("extract")
    return extract if extract and len(extract) > 40 else None


def main():
    print("Loading PopQA from Hugging Face...")
    ds = load_dataset("akariasai/PopQA", split="test").shuffle(seed=42)

    corpus: dict[str, str] = {}
    qa = []
    seen_titles: dict[str, bool] = {}

    for row in ds:
        if len(qa) >= N_SAMPLES:
            break
        title = row["s_wiki_title"]
        answers = parse_answers(row["possible_answers"])
        if not title or not answers:
            continue

        if title not in seen_titles:
            summary = wiki_summary(title)
            seen_titles[title] = summary is not None
            if summary:
                corpus[title] = summary
            time.sleep(0.05)  # be polite to the Wikipedia API

        if not seen_titles[title]:
            continue

        qa.append({"question": row["question"], "answer": answers[0]})

    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for i, (title, text) in enumerate(corpus.items()):
            f.write(json.dumps({"id": str(i), "text": f"{title}: {text}"}) + "\n")

    with open(DATA_DIR / "qa.jsonl", "w", encoding="utf-8") as f:
        for item in qa:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(corpus)} corpus docs -> data/corpus.jsonl")
    print(f"Wrote {len(qa)} QA pairs -> data/qa.jsonl")


if __name__ == "__main__":
    main()
