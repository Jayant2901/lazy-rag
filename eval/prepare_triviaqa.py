"""Builds a second, independent QA benchmark from TriviaQA to check whether the
lazy-rag finding is specific to PopQA's long-tail entity questions or holds more
generally. Unlike PopQA, TriviaQA ships its own evidence text and answer aliases,
so no Wikipedia API calls are needed here.
"""
import json
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

N_SAMPLES = 40
CONTEXT_CHARS = 1500  # keep corpus entries summary-sized, comparable to PopQA's
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    print("Loading TriviaQA (rc.wikipedia, streaming) from Hugging Face...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation", streaming=True).shuffle(
        seed=42, buffer_size=1000
    )

    corpus: dict[str, str] = {}
    qa = []

    for row in ds:
        if len(qa) >= N_SAMPLES:
            break
        titles = row["entity_pages"]["title"]
        contexts = row["entity_pages"]["wiki_context"]
        if not titles or not contexts:
            continue
        title, context = titles[0], contexts[0]
        if not context or len(context) < 100:
            continue

        answers = [row["answer"]["value"], *row["answer"]["aliases"]]
        answers = list(dict.fromkeys(a for a in answers if a))  # dedupe, keep order
        if not answers:
            continue

        corpus[title] = context[:CONTEXT_CHARS]
        qa.append({"question": row["question"], "answers": answers, "gold_title": title})

    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "triviaqa_corpus.jsonl", "w", encoding="utf-8") as f:
        for i, (title, text) in enumerate(corpus.items()):
            f.write(json.dumps({"id": str(i), "title": title, "text": f"{title}: {text}"}) + "\n")

    with open(DATA_DIR / "triviaqa_qa.jsonl", "w", encoding="utf-8") as f:
        for item in qa:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(corpus)} corpus docs -> data/triviaqa_corpus.jsonl")
    print(f"Wrote {len(qa)} QA pairs -> data/triviaqa_qa.jsonl")


if __name__ == "__main__":
    main()
