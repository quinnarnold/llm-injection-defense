"""
Generates synthetic Q&A pairs from scraped Bryant University corpus.

Reads all JSON files from data/scraped/, chunks the text, and calls Groq
to produce question-answer pairs suitable for fine-tuning. Decays through
models in preference order when rate limited.

Output: data/instruction/raw_qa.jsonl
Each line: {"question": "...", "answer": "...", "source": "url_or_path", "chunk_id": "..."}

Usage:
    export GROQ_API_KEY=...
    python -m src.finetune.generate_qa
    python -m src.finetune.generate_qa --input data/scraped --output data/instruction/raw_qa.jsonl
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Decay order: best model first, never falls below 17B
_GROQ_PREFERENCE = [
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 300
CHUNKS_PER_BATCH = 3
QA_PER_CHUNK = 2
CALL_DELAY = 1.0

GARBAGE_PATTERNS = re.compile(
    r"not provided|not mentioned|not specified|no information|"
    r"this passage|this text|the passage|the text does not|"
    r"please refer to|various subjects|not available in",
    re.IGNORECASE,
)


def _pick_available_models(client: OpenAI) -> list[str]:
    try:
        available = {m.id for m in client.models.list().data}
        return [m for m in _GROQ_PREFERENCE if m in available]
    except Exception:
        return list(_GROQ_PREFERENCE)


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _load_documents(input_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            source = data.get("url") or data.get("source_path") or path.stem
            title = data.get("title", "")
            text = data.get("text", "")
            if len(text.strip()) >= MIN_CHUNK_LEN:
                docs.append({
                    "source": source,
                    "title": title,
                    "text": text,
                    "filename": path.stem,
                })
        except Exception:
            continue
    return docs


def _make_prompt(chunks_with_ids: list[tuple[str, str, str]]) -> str:
    parts = []
    for chunk_id, title, chunk in chunks_with_ids:
        header = f"[PASSAGE {chunk_id}]"
        if title:
            header += f" (Source: {title})"
        parts.append(f"{header}\n{chunk}")
    passages = "\n\n".join(parts)
    return (
        f"Below are text passages from Bryant University's website and official documents.\n\n"
        f"{passages}\n\n"
        f"For each passage, generate exactly {QA_PER_CHUNK} high-quality question-answer pairs "
        f"that a Bryant University student might realistically ask.\n\n"
        f"Requirements:\n"
        f"- Questions must be specific and self-contained (no 'this course' without naming it)\n"
        f"- Answers must be complete sentences, 1-3 sentences long\n"
        f"- Answers must directly answer the question using only facts from the passage\n"
        f"- Do NOT generate a pair if the passage lacks enough info to give a concrete answer\n"
        f"- Never say 'the passage states' or 'according to the text' -- answer as if you know the facts\n\n"
        f"Output a single JSON array. Each object must have: "
        f"\"chunk_id\" (the passage ID), \"question\", \"answer\".\n"
        f"Output only valid JSON with no markdown fences."
    )


def _call_api(client: OpenAI, model: str, prompt: str) -> tuple[str | None, bool]:
    """Returns (response_text, rate_limited)."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dataset generator for a university chatbot. "
                        "Generate high-quality, self-contained Q&A pairs. "
                        "Every answer must be a complete sentence that makes sense without seeing the source. "
                        "Output only valid JSON arrays as instructed. Never add commentary or markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content, False
    except RateLimitError:
        return None, True
    except Exception as e:
        print(f"  API error: {e}")
        return None, False


def _is_garbage(question: str, answer: str) -> bool:
    combined = question + " " + answer
    if GARBAGE_PATTERNS.search(combined):
        return True
    if len(answer.split()) < 4:
        return True
    if answer.rstrip(".!").lower() == question.rstrip("?").lower():
        return True
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    if len(q_words) > 5 and len(a_words - q_words) < 3:
        return True
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "of", "at", "in", "for", "and",
                 "to", "or", "by", "with", "it", "its", "this", "that", "be", "been", "has",
                 "have", "do", "does", "did", "not", "can", "will", "would", "should", "may",
                 "on", "from", "as", "but", "if", "than", "also", "called", "known", "named"}
    q_content = set(question.lower().split()) - stopwords
    a_content = set(answer.lower().split()) - stopwords
    new_words = a_content - q_content
    q_joined = question.lower().replace("-", " ").replace("'", "")
    new_words = {w for w in new_words if w.replace("-", " ") not in q_joined}
    if len(a_content) > 3 and len(new_words) < 2:
        return True
    if answer.lower().startswith("no,") and re.search(r"\b(but|however|while|although|it does|they do|it can|they can)\b", answer.lower()):
        return True
    return False


def _parse_pairs(raw: str, chunk_ids: set[str]) -> list[dict]:
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        items = json.loads(raw)
        pairs = []
        for item in items:
            if not isinstance(item, dict):
                continue
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            cid = str(item.get("chunk_id", ""))
            if q and a and len(q) > 10 and len(a) > 15 and not _is_garbage(q, a):
                pairs.append({"question": q, "answer": a, "chunk_id": cid})
        return pairs
    except Exception:
        return []


def generate_qa(input_dir: Path, output_path: Path) -> int:
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise SystemExit("Set GROQ_API_KEY before running.")

    client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)
    models = _pick_available_models(client)
    print(f"Groq models available (decay order): {models}")
    model_idx = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_chunks: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    done_chunks.add(item.get("chunk_id", ""))
        print(f"Resuming: {len(done_chunks)} chunk IDs already processed")

    docs = _load_documents(input_dir)
    print(f"Loaded {len(docs)} documents from {input_dir}")

    all_chunks: list[tuple[str, str, str, str]] = []
    for doc in docs:
        for i, chunk in enumerate(_chunk_text(doc["text"])):
            chunk_id = f"{doc['filename']}:{i}"
            if chunk_id not in done_chunks:
                all_chunks.append((chunk_id, doc["title"], chunk, doc["source"]))

    print(f"Chunks to process: {len(all_chunks)}")

    total_pairs = 0
    with open(output_path, "a") as out_f:
        batch: list[tuple[str, str, str, str]] = []
        for i, (chunk_id, title, chunk_text, source) in enumerate(all_chunks):
            batch.append((chunk_id, title, chunk_text, source))
            if len(batch) < CHUNKS_PER_BATCH and i < len(all_chunks) - 1:
                continue

            chunks_with_ids = [(c[0], c[1], c[2]) for c in batch]
            sources_map = {c[0]: c[3] for c in batch}
            prompt = _make_prompt(chunks_with_ids)

            raw = None
            while model_idx < len(models):
                model = models[model_idx]
                raw, rate_limited = _call_api(client, model, prompt)
                if raw is not None:
                    break
                if rate_limited:
                    if model_idx + 1 < len(models):
                        model_idx += 1
                        print(f"  Rate limited on {model}, decaying to {models[model_idx]}")
                    else:
                        print("  Rate limited on all models, waiting 60s...")
                        time.sleep(60)
                        model_idx = 0
                else:
                    break

            if raw:
                pairs = _parse_pairs(raw, {c[0] for c in chunks_with_ids})
                for pair in pairs:
                    cid = pair["chunk_id"]
                    pair["source"] = sources_map.get(cid, "")
                    out_f.write(json.dumps(pair) + "\n")
                    total_pairs += 1
                out_f.flush()
                print(f"[{i + 1}/{len(all_chunks)}] +{len(pairs)} pairs (total: {total_pairs}) [{models[model_idx]}]")
            else:
                print(f"[{i + 1}/{len(all_chunks)}] skipped (no response)")

            batch = []
            time.sleep(CALL_DELAY)

    print(f"\nDone: {total_pairs} Q&A pairs written to {output_path}")
    return total_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/scraped"))
    parser.add_argument("--output", type=Path, default=Path("data/instruction/raw_qa.jsonl"))
    args = parser.parse_args()
    generate_qa(args.input, args.output)
