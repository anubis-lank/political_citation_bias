import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("./data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_ALIASES = {
    "article_id": ["article_id", "uuid", "id", "doc_id", "news_id"],
    "title": ["title", "headline", "head_line", "article_title"],
    "text": ["text", "content", "body", "article_text", "article_body", "story", "article_content", "article_correction", "description"],
    "source": ["source", "outlet", "publisher", "news_source", "domain", "site", "origine"],
    "url": ["url", "link", "article_url"],
    "published_at": ["published_at", "date", "publish_date", "timestamp", "datetime", "article_date"],
    "event_id": ["event_id", "cluster_id", "group_id", "story_id"],
    "bias_rating": ["bias_rating", "bias", "stance", "political_leaning"],
    "veracity_label": ["label", "veracity_label", "fact_label", "truth_label"],
}


def stable_id(value: str, prefix: str) -> str:
    """Generate a stable hash-based ID."""
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def detect_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized_lookup = {normalize_name(col): col for col in df.columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized_lookup:
            return normalized_lookup[key]
    return None


def anonymize_text(text: str, outlet_hints: list[str]) -> str:
    """Remove URLs, emails, handles, bylines and outlet hints."""
    if not isinstance(text, str):
        return ""

    x = text

    x = re.sub(r"https?://\S+|www\.\S+", "[URL]", x, flags=re.IGNORECASE)
    x = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", x, flags=re.IGNORECASE)
    x = re.sub(r"@\w+", "[HANDLE]", x)

    x = re.sub(r"\bBy\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", "By [AUTHOR]", x)
    x = re.sub(
        r"\bReported by\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
        "Reported by [AUTHOR]",
        x,
        flags=re.IGNORECASE,
    )

    for outlet in outlet_hints:
        if outlet and isinstance(outlet, str):
            pattern = re.escape(outlet.strip())
            if pattern:
                x = re.sub(pattern, "[OUTLET]", x, flags=re.IGNORECASE)

    x = re.sub(r"\s+", " ", x).strip()
    return x


def load_config(config_path: str) -> dict:
    """Load formatter config."""
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


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


DEFAULT_ENCODINGS = ["utf-8-sig", "utf-8", "cp1256", "cp1252", "latin1"]

def read_csv_with_fallback(path: Path, encodings: list[str] | None = None) -> pd.DataFrame:
    encodings = encodings or DEFAULT_ENCODINGS
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:
            last_error = error
            logger.warning(f"Failed reading {path} with encoding={encoding}")

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 0,
        f"Could not decode {path} using encodings: {encodings}",
    )

def load_input_csv(raw_path: str, encoding: str | None = None) -> pd.DataFrame:
    """Read the input CSV with encoding fallback."""
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    if encoding:
        df = pd.read_csv(path, encoding=encoding, low_memory=False)
        logger.info(f"Loaded {len(df)} rows from {path} with encoding={encoding}")
        return df

    df = read_csv_with_fallback(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to a standard schema."""
    df = df.copy()

    detected = {
        standard_name: detect_column(df, aliases)
        for standard_name, aliases in INPUT_ALIASES.items()
    }

    if detected["text"] is None:
        raise ValueError(
            "No text column found. Expected one of: " + ", ".join(INPUT_ALIASES["text"])
        )

    rename_map = {
        original_name: standard_name
        for standard_name, original_name in detected.items()
        if original_name is not None
    }
    df = df.rename(columns=rename_map)

    if "article_id" not in df.columns:
        if "url" in df.columns:
            df["article_id"] = df["url"].astype(str).apply(lambda value: stable_id(value, "art"))
        else:
            df["article_id"] = df.index.map(lambda value: stable_id(str(value), "art"))

    if "title" not in df.columns:
        df["title"] = ""

    if "source" not in df.columns:
        df["source"] = ""

    if "event_id" not in df.columns:
        non_empty_titles = df["title"].astype(str).str.strip().ne("")
        if non_empty_titles.any():
            df["event_id"] = df["title"].astype(str).apply(lambda value: stable_id(value, "evt"))
        else:
            df["event_id"] = df["article_id"].astype(str).apply(lambda value: stable_id(value, "evt"))

    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)

    if "published_at" in df.columns:
        df["published_at"] = df["published_at"].fillna("").astype(str)

    if "bias_rating" in df.columns:
        df["bias_rating"] = df["bias_rating"].fillna("").astype(str).str.lower().str.strip()

    if "veracity_label" in df.columns:
        df["veracity_label"] = df["veracity_label"].fillna("").astype(str).str.lower().str.strip()

    return df


def load_classifier_client() -> tuple[OpenAI, str, float]:
    """Load the stance classifier client and threshold from config.json."""
    config_path = Path("./config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing classifier config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["classifier_model"]
    threshold = float(config["thresholds"]["stance_confidence"])

    base_url = os.getenv("BASE_URL")
    token = os.getenv("HF_TOKEN")
    if not base_url:
        raise ValueError("BASE_URL is missing in .env")
    if not token:
        raise ValueError("HF_TOKEN is missing in .env")

    client = OpenAI(base_url=base_url, api_key=token)
    return client, model, threshold


def classify_bias_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Predict left/right/center stance for each row and store it in bias_rating."""
    df = df.copy()
    client, model, threshold = load_classifier_client()

    predicted_rows = []
    debug_limit = 100
    center_from_exception = 0
    center_from_parse = 0
    center_from_invalid_label = 0
    center_from_model = 0

    for row_index, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Classifying stance")):
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

        raw_response = ""
        parse_reason = "ok"

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=128,
            )
            raw_response = completion.choices[0].message.content or ""
            parsed = parse_json_block(raw_response)

            if not parsed:
                parse_reason = "parse_failed"
                stance = "center"
                confidence = 0.0
                center_from_parse += 1
            else:
                stance = str(parsed.get("stance", "center")).lower().strip()
                confidence = float(parsed.get("confidence", 0.0))

                if stance not in {"left", "right", "center"}:
                    parse_reason = f"invalid_label:{stance}"
                    stance = "center"
                    center_from_invalid_label += 1
                elif stance == "center":
                    center_from_model += 1
        except Exception as error:
            parse_reason = f"exception:{type(error).__name__}"
            stance = "center"
            confidence = 0.0
            center_from_exception += 1

        if row_index < debug_limit:
            logger.info(
                "stance_debug row=%s article_id=%s parse_reason=%s stance=%s confidence=%.3f raw_response=%s",
                row_index,
                row.get("article_id", ""),
                parse_reason,
                stance,
                confidence,
                raw_response[:300].replace("\n", " "),
            )

        predicted_rows.append(
            {
                "bias_rating": stance,
                "predicted_stance": stance,
                "confidence": confidence,
                "passes_threshold": bool(confidence >= threshold and stance in {"left", "right", "center"}),
            }
        )

    prediction_df = pd.DataFrame(predicted_rows)
    for column in prediction_df.columns:
        df[column] = prediction_df[column].values

    logger.info(
        "stance_summary total=%s center_model=%s center_parse=%s center_invalid_label=%s center_exception=%s",
        len(df),
        center_from_model,
        center_from_parse,
        center_from_invalid_label,
        center_from_exception,
    )

    return df


def build_anonymized_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create anonymized article text using available source hints."""
    df = df.copy()
    outlet_hints = sorted([value for value in df["source"].dropna().astype(str).unique() if value.strip()])

    tqdm.pandas(desc="Anonymizing articles")
    df["text_anonymized"] = df["text"].progress_apply(lambda value: anonymize_text(value, outlet_hints))

    return df


def write_variant_outputs(df: pd.DataFrame, suffix: str) -> None:
    """Write one set of output files with the given suffix."""
    clean_path = PROCESSED_DIR / f"articles_clean{suffix}.csv"
    anon_path = PROCESSED_DIR / f"articles_anonymized{suffix}.csv"
    events_path = PROCESSED_DIR / f"events_balanced{suffix}.csv"
    audit_path = PROCESSED_DIR / f"anonymization_audit{suffix}.json"

    clean_cols = [
        "article_id",
        "event_id",
        "title",
        "text",
        "source",
        "url",
        "published_at",
        "veracity_label",
        "bias_rating",
        "predicted_stance",
        "confidence",
        "passes_threshold",
    ]
    clean_cols = [column for column in clean_cols if column in df.columns]

    anon_cols = [
        "article_id",
        "event_id",
        "title",
        "text_anonymized",
        "source",
        "url",
        "published_at",
        "veracity_label",
        "bias_rating",
        "predicted_stance",
        "confidence",
        "passes_threshold",
    ]
    anon_cols = [column for column in anon_cols if column in df.columns]

    df[clean_cols].to_csv(clean_path, index=False)
    df[anon_cols].to_csv(anon_path, index=False)

    events = df[["event_id", "title"]].drop_duplicates().sort_values("event_id")
    events.to_csv(events_path, index=False)

    audit = {
        "total_articles": int(len(df)),
        "total_events": int(df["event_id"].nunique()),
        "missing_source": int(df["source"].eq("").sum()) if "source" in df.columns else int(len(df)),
        "missing_title": int(df["title"].eq("").sum()) if "title" in df.columns else int(len(df)),
        "has_bias_rating": bool("bias_rating" in df.columns),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    logger.info(f"Saved: {clean_path}")
    logger.info(f"Saved: {anon_path}")
    logger.info(f"Saved: {events_path}")
    logger.info(f"Saved: {audit_path}")


def build_outputs(df: pd.DataFrame) -> None:
    """Write rag_setup-ready processed files and compatibility aliases."""
    write_variant_outputs(df, "")
    write_variant_outputs(df, "_balanced")
    write_variant_outputs(df, "_binary")


def format_documents(config_path: str, interactive: bool = False) -> None:
    """Main formatter workflow: normalize, anonymize, and write outputs."""
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    raw_path = config["raw_path"]
    df = load_input_csv(raw_path, config.get("encoding"))
    df = standardize_columns(df)

    df = df[df["text"].astype(str).str.strip().ne("")].copy()
    df = build_anonymized_text(df)
    df = classify_bias_rating(df)

    if interactive:
        preview_path = PROCESSED_DIR / "formatter_preview.csv"
        df.head(50).to_csv(preview_path, index=False)
        input(f"\nReview {preview_path} then press Enter to write final outputs...")

    build_outputs(df)
    logger.info("Document formatting complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize any news CSV into rag_setup-ready processed files."
    )
    parser.add_argument("--config", type=str, default="formatter_config.json", help="Path to formatter config JSON")
    parser.add_argument("--interactive", action="store_true", help="Write a preview and pause before final output")

    args = parser.parse_args()
    format_documents(args.config, args.interactive)
