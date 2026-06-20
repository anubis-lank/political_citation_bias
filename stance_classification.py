import argparse
import json
#import logging
import re
from pathlib import Path
import os
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Uncomment for logging
# Set up logging
#logging.basicConfig(
    #level=logging.INFO,
    #format="%(levelname)s: %(message)s"
#)
#logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("./data/processed")
CONFIG_PATH = Path("./config.json")

def parse_json_block(text: str) -> dict:
    if not isinstance(text, str):
        return {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    
def classify_once() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = config["classifier_model"]
    threshold = float(config["thresholds"]["stance_confidence"])

    articles_path = PROCESSED_DIR / "articles_anonymized.csv"
    out_path = PROCESSED_DIR / "stance_predictions.csv"

    if not articles_path.exists():
        raise FileNotFoundError(f"Missing input: {articles_path}")
    
    df = pd.read_csv(articles_path)

    # Uncomment for logging
    #logger.info(f"Loaded {len(df)} articles from {articles_path}")
    #logger.info(f"Using classifier model: {model}")

    client = OpenAI(base_url=os.getenv('BASE_URL'), api_key=os.getenv("HF_TOKEN"))

    rows = []
    error_count = 0

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Classifying")):
        text = str(row["text_anonymized"])[:5000]

        prompt = f"""
You are a strict political stance classifier.
Classify the article text as one of: left, right, center.
Return only JSON:
{{
  "stance": "left|right|center",
  "confidence": 0.0
}}

Article:
{text}
""".strip()
        
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=128,
            )
            content = completion.choices[0].message.content

            # Uncomment for logging
            # Log first 3 raw responses for inspection
            #if idx < 3:
                #logger.info(f"Article {idx} raw response: {content[:200]}")
                
            parsed = parse_json_block(content)

            stance = str(parsed.get("stance", "center")).lower()
            confidence = float(parsed.get("confidence", 0.0))

            # Uncomment for logging
            #if idx < 3:
                #logger.info(f"Article {idx} parsed stance: {stance}, confidence: {confidence}")
        
        except Exception as e:
            #error_count += 1
            #logger.error(f"Article {idx} failed: {type(e).__name__}: {e}")
            stance = "center"
            confidence = 0.0

        if stance not in {"left", "right", "center"}:
            stance = "center"

        rows.append(
            {
                "article_id": row["article_id"],
                "predicted_stance": stance,
                "confidence": confidence,
                "passes_threshold": bool(confidence >= threshold and stance in {"left", "center", "right"}),
            }
        )

    output = pd.DataFrame(rows)
    output.to_csv(out_path, index=False)
    
    # Uncomment for logging
    #logger.info(f"Saved: {out_path}")
    #logger.info(f"Total errors: {error_count} / {len(df)}")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("stage", choices=["classify"])
    args = parse.parse_args()

    if args.stage == "classify":
        classify_once()