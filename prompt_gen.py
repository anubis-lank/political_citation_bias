import argparse
import json
import logging
import re
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("./data/processed")
CONFIG_PATH = Path("./config.json")

def parse_query_json(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Try strict JSON first
    try:
        data = json.loads(text)
        return str(data.get("query", "")).strip()
    except Exception:
        pass

    # Fallback: extract first JSON object from mixed text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return ""
    try:
        data = json.loads(m.group(0))
        return str(data.get("query", "")).strip()
    except Exception:
        return ""
    
def generate_queries(n: int, seed: int, variant: str = "balanced") -> None:
    """
    Generate queries for either 'binary' (left+right) or 'balanced' (left+center+right) datasets.
    """

    if variant not in ["binary", "balanced"]:
        raise ValueError(f"variant must be 'binary' or 'balanced', got: {variant}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = config["models"]["main_model"]

    articles_path = PROCESSED_DIR / f"articles_anonymized_{variant}.csv"
    if not articles_path.exists():
        raise FileNotFoundError(f"Missing input: {articles_path}")
    
    articles = pd.read_csv(articles_path)

    # One row per event
    events = articles[["event_id", "title"]].drop_duplicates()
    np.random.seed(seed)
    sample = events.sample(n=min(n, len(events)), random_state=seed).reset_index(drop=True)

    base_url = os.getenv("BASE_URL")
    token = os.getenv("HF_TOKEN")
    if not base_url:
        raise ValueError("BASE_URL is missing in .env")
    if not token:
        raise ValueError("HF_TOKEN is missing in .env")
    
    client = OpenAI(base_url=base_url, api_key=token)

    rows = []
    failures = []

    logger.info(f"Generating queries for {len(sample)} events ({variant} variant) with model: {model}")

    for i, ev in sample.iterrows():
        event_id = ev["event_id"]
        title = ev["title"]

        ev_rows = articles[articles["event_id"] == event_id]
        left = ev_rows[ev_rows["bias_rating"] == "left"].head(1)
        right = ev_rows[ev_rows["bias_rating"] == "right"].head(1)

        # For balanced variant, also require center
        if variant == "balanced":
            center = ev_rows[ev_rows["bias_rating"] == "center"].head(1)
            if left.empty or center.empty or right.empty:
                failures.append({
                    "event_id": event_id,
                    "title": title,
                    "reason": "missing_left_center_or_right_article"
                })
                continue
            center_text = str(center.iloc[0]["text_anonymized"])[:2000]
        else: #binary variant
            if left.empty or right.empty:
                failures.append({
                    "event_id": event_id,
                    "title": title,
                    "reason": "missing_left_or_right_article"
                })
                continue
            center_text = ""

        left_text = str(left.iloc[0]["text_anonymized"])[:2000]
        right_text = str(right.iloc[0]["text_anonymized"])[:2000]

        passages_section = f"""Passage [1]: {left_text}
Passage [2]: {center_text}
Passage [3]: {right_text}""" if variant == "balanced" else f"""Passage [1]: {left_text}
Passage [2]: {right_text}"""

        prompt = f"""
### Instruction:
Below is an event and several related news passages.
Please generate a natural and concise query based on
the following requirements:
1. The query should focus on the core topic of the
event.
2. The query should be as concise as possible while
all the provided passages can answer it directly.
3. Ensure your output strictly adheres to the
following JSON format: {{"query":"..."}}
### Event: {title}
### Relevant Passages:
{passages_section}
""".strip()
        
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=96,
            )
            content = completion.choices[0].message.content
            query = parse_query_json(content)

            if not query:
                failures.append({
                    "event_id": event_id,
                    "title": title,
                    "reason": "empty_or_unparseable_query",
                    "raw_response": content[:500]
                })
                continue

            rows.append({
                "event_id": event_id,
                "title": title,
                "query": query
            })
            
            if i < 3:
                logger.info(f"Sample query {i+1}: {query}")

        except Exception as e:
            failures.append({
                "event_id": event_id,
                "title": title,
                "reason": f"api_error:{type(e).__name__}:{e}"
            })

    out_q = PROCESSED_DIR / f"queries_subset_{n}_{variant}.csv"
    out_f = PROCESSED_DIR / f"query_generation_failures_{n}_{variant}.csv"

    pd.DataFrame(rows).to_csv(out_q, index=False)
    pd.DataFrame(failures).to_csv(out_f, index=False)

    logger.info(f"Saved: {out_q}")
    logger.info(f"Saved: {out_f}")
    logger.info(f"Success: {len(rows)} / {len(sample)}")
    logger.info(f"Failures: {len(failures)} / {len(sample)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["generate-queries"])
    parser.add_argument("--subset-n", type=int, default=100, help="Number of queries to generate.")
    parser.add_argument("--variant", choices=["binary", "balanced"], default="balanced", help="Dataset variant: 'binary' (left+right) or 'balanced' (left+center+right)")
    args = parser.parse_args()

    if args.stage == "generate-queries":
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        subset_n = args.subset_n if args.subset_n else config["subset"]["size"]
        seed = config["subset"]["seed"]
        generate_queries(subset_n, seed, variant=args.variant)