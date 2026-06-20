import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

RAW_PATH = Path("./data/raw/allsides_balanced_news_headlines-texts.csv")
PROCESSED_DIR = Path("./data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def stable_id(value: str, prefix: str) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"

def anonymize_text(text: str, outlet_hints: list[str]) -> str:
    if not isinstance(text, str):
        return ""

    x = text

    # URLs, emails, social handles
    x = re.sub(r"https?://\S+|www\.\S+", "[URL]", x, flags=re.IGNORECASE)
    x = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", x, flags=re.IGNORECASE)
    x = re.sub(r"@\w+", "[HANDLE]", x)

    # Typical byline patterns
    x = re.sub(r"\bBy\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", "By [AUTHOR]", x)
    x = re.sub(r"\bReported by\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", "Reported by [AUTHOR]", x, flags=re.IGNORECASE)

    # Remove outlet hints if present
    for outlet in outlet_hints:
        if outlet and isinstance(outlet, str):
            pattern = re.escape(outlet.strip())
            if pattern:
                x = re.sub(pattern, "[OUTLET]", x, flags=re.IGNORECASE)

    # Collapse whitespace
    x = re.sub(r"\s+", " ", x).strip()
    return x

def preprocess() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")
    
    df = pd.read_csv(RAW_PATH)

    required = {"title", "text", "bias_rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Keep only left, center, right and valid text
    df = df[df["bias_rating"].isin(["left", "center", "right"])].copy()
    df = df.dropna(subset=["title", "text"]).copy()

    # BINARY PATH (left + right only)
    binary_df = df[df["bias_rating"].isin(["left", "right"])].copy()
    binary_balanced = binary_df.groupby("title").filter(
        lambda g: {"left", "right"}.issubset(set(g["bias_rating"]))
    ).copy()

    # BALANCED PATH (left + center + right)
    balanced = df.groupby("title").filter(
        lambda g: {"left", "center", "right"}.issubset(set(g["bias_rating"]))
    ).copy()

    for data, suffix in [(binary_balanced, "_binary"), (balanced, "_balanced")]:
        if len(data) == 0:
            print(f"Warning: No data for {suffix} variant")
            continue

        # Build IDs
        data["event_id"] = data["title"].apply(lambda t: stable_id(str(t), "evt"))
        data["article_id"] = (
            data["event_id"]
            +"_"
            + data.groupby("event_id").cumcount().astype(str)
        )

        # Optional outlet column, if present in your CSV
        outlet_col = "source" if "source" in data.columns else None
        data["outlet_name"] = data[outlet_col].fillna("").astype(str) if outlet_col else ""

        outlet_hints = sorted(
            [x for x in data["outlet_name"].dropna().astype(str).unique().tolist() if x.strip()]
        )

        # Build anonymised text column
        data["text_anonymized"] = data["text"].astype(str).apply(
            lambda t: anonymize_text(t, outlet_hints)
        )

        # Save outputs
        articles_path = PROCESSED_DIR / f"articles_clean{suffix}.csv"
        anon_path = PROCESSED_DIR / f"articles_anonymized{suffix}.csv"
        event_path = PROCESSED_DIR / f"events_balanced{suffix}.csv"
        audit_path = PROCESSED_DIR / f"anonymization_audit{suffix}.json"

        data.to_csv(articles_path, index=False)

        anon_cols = ["article_id", "event_id", "title", "bias_rating", "outlet_name", "text_anonymized"]
        data[anon_cols].to_csv(anon_path, index=False)

        events = data[["event_id", "title"]].drop_duplicates()
        events.to_csv(event_path, index=False)

        audit = {
            "total_articles": int(len(data)),
            "total_events": int(events.shape[0]),
            "left_articles": int((data["bias_rating"] == "left").sum()),
            "center_articles": int((data["bias_rating"] == "center").sum()),
            "right_articles": int((data["bias_rating"] == "right").sum())
        }
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

        print(f"Saved: {articles_path}")
        print(f"Saved: {anon_path}")
        print(f"Saved: {event_path}")
        print(f"Saved: {audit_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preprocess"])
    args = parser.parse_args()

    if args.stage == "preprocess":
        preprocess()