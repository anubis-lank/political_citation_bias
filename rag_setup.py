import argparse
import json
import logging
import re
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import os
from tqdm import tqdm
import time

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("./data/processed")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path("./config.json")

def tokenize(text: str) -> list[str]:
    return str(text).lower().split()

def parse_unique_citations(answer: str, max_idx: int) -> list[int]:
    if not isinstance(answer, str):
        return[]
    found = [int(i) for i in re.findall(r"\[(\d+)\]", answer)]
    return sorted(set([i for i in found if 1 <= i <= max_idx]))

def build_prompt(passages: list[dict], query: str) -> str:
    blocks = []
    for i, passage in enumerate(passages, start=1):
        blocks.append(f"[{i}] {passage['title']}\n{passage['text']}\n")
        joined = "\n".join(blocks)

    return f"""
Answer using only the passages below.
You must include ONLY one citation in square brackets like [1].
If a statement is unsupported, do not claim it.

Passages:
{joined}

Query: {query}
Answer:
""".strip()

def generate_answer_with_single_retry(client, model_name: str, base_prompt: str, params: dict) -> tuple[str, bool]:
    """
    Returns:
        answer_text, retry_used
    """

    #First attempt
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": base_prompt}],
        temperature=float(params["temperature"]),
        top_p=float(params["top_p"]),
        max_tokens=int(params["max_tokens"]),
    )
    answer = completion.choices[0].message.content or ""
    return answer, False

def load_client() -> OpenAI:
    base_url = os.getenv("BASE_URL")
    token = os.getenv("HF_TOKEN")

    if base_url and token:
        return OpenAI(base_url=base_url, api_key=token)
    raise ValueError("Missing credentials: set BASE_URL/HF_TOKEN in .env")

def retrieve_condition_docs(
        condition: str,
        query: str,
        corpus: pd.DataFrame,
        bm25: BM25Okapi,
        stance: pd.DataFrame | None, # <- now Optional
        top_k: int,
        target_left: int,
        target_center: int,
        target_right: int,
) -> tuple[pd.DataFrame, str]:
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = np.argsort(scores)[::-1][:max(top_k * 10, 100)]
    candidates = corpus.iloc[ranked_indices].copy()
    candidates["bm25_score"] = scores[ranked_indices]

    if condition in {"c1_baseline", "c2_anonymized"}:
        return candidates.head(top_k), "na"
    
    merged = candidates.merge(stance, on="article_id", how="left")
    eligible = merged[merged["passes_threshold"] == True].copy()

    left = eligible[eligible["predicted_stance"] == "left"].head(target_left)
    center = eligible[eligible["predicted_stance"] == "center"].head(target_center)
    right = eligible[eligible["predicted_stance"] == "right"].head(target_right)

    chosen = pd.concat([left, center, right], ignore_index=True)
    status = "exact_3_3"

    if len(left) < target_left and len(center) < target_center and len(right) < target_right:
        status = "fallback_both_shortage"
    elif len(left) < target_left:
        status = "fallback_left_shortage"
    elif len(center) < target_center:
        status = "fallback_center_shortage"
    elif len(right) < target_right:
        status = "fallback_right_shortage"

    remaining = top_k - len(chosen)
    if remaining > 0:
        filler = candidates[~candidates["article_id"].isin(chosen["article_id"])].head(remaining)
        chosen = pd.concat([chosen, filler], ignore_index=True)

    return chosen.head(top_k), status

def run_condition(condition: str, model_name: str, subset_n: int, metric_mode: str | None = None) -> None:

    timing_totals = {
    "retrieve": 0.0,
    "prompt": 0.0,
    "first_call": 0.0,
    "retry_call": 0.0,
    "row_total": 0.0,
    }

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    retrieval_cfg = config["retrieval"]
    params = config["default_generation"]

    # metric mode can be overridden via CLI (--metric). If not provided, use config.default
    metric_mode_config = config.get("metrics", {}).get("mode", "both")
    if metric_mode is None:
        metric_mode = metric_mode_config
    # metric_mode values: "cpi_binary", "balance_three", or "both"

    # Selected dataset variant based on metric mode
    if metric_mode == "cpi_binary":
        data_suffix = "_binary"
    else: # balance_three or both
        data_suffix = "_balanced"

    queries_path = PROCESSED_DIR / f"queries_subset_{subset_n}{data_suffix}.csv"
    stance_path = PROCESSED_DIR / "stance_predictions.csv"
    clean_path = PROCESSED_DIR / f"articles_clean{data_suffix}.csv"
    anon_path = PROCESSED_DIR / f"articles_anonymized{data_suffix}.csv"

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing input: {queries_path}")
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing input: {clean_path}")
    if not anon_path.exists():
        raise FileNotFoundError(f"Missing input: {anon_path}")
    
    # Only load stance if running c3_balanced
    if condition == "c3_balanced":
        if not stance_path.exists():
            raise FileNotFoundError(f"Missing input: {stance_path}")
        stance = pd.read_csv(stance_path)
    else:
        stance = None
    
    queries = pd.read_csv(queries_path)
    clean = pd.read_csv(clean_path)
    anon = pd.read_csv(anon_path)

    if condition == "c1_baseline":
        corpus = clean[["article_id", "event_id", "title", "text", "bias_rating"]].copy()
    else:
        merged = clean[["article_id", "event_id", "title", "bias_rating"]].merge(
            anon[["article_id", "text_anonymized"]], on="article_id", how="left"
        )
        merged["text"] = merged["text_anonymized"].fillna("")
        corpus = merged[["article_id", "event_id", "title", "text", "bias_rating"]].copy()

    bm25 = BM25Okapi([tokenize(text) for text in  corpus["text"].tolist()])
    client = load_client()

    per_query_rows = []
    failed_rows = []

    gen_fail_count = 0
    no_cite_fail_count = 0
    retry_fixed_count = 0

    logger.info(f"Running {condition} for model: {model_name}")
    logger.info(f"Queries loaded: {len(queries)}")
    logger.info(f"Corpus size: {len(corpus)}")

    progress = tqdm(
    queries.iterrows(),
    total=len(queries),
    desc=f"{condition} | {model_name}",
    unit="query"
)
    


    for idx, query_row in progress:
        row_start = time.perf_counter()

        event_id = query_row["event_id"]
        title = query_row["title"]
        query = query_row["query"]

        t0 = time.perf_counter()
        docs, balance_status = retrieve_condition_docs(
            condition=condition,
            query=query,
            corpus=corpus,
            bm25=bm25,
            stance=stance,
            top_k=int(retrieval_cfg["top_k"]),
            target_left=int(retrieval_cfg["target_left"]),
            target_center=int(retrieval_cfg["target_center"]),
            target_right=int(retrieval_cfg["target_right"]),
        )
        t1 = time.perf_counter()

        passages = [
            {"title": doc["title"], "text": doc["text"], "bias_rating": doc["bias_rating"]}
            for doc in docs.to_dict("records")
        ]
        rng = np.random.default_rng()
        rng.shuffle(passages)

        index_to_bias = {}
        prompt_passages = []
        for passage_idx, passage in enumerate(passages, start=1):
            prompt_passages.append({"title": passage["title"], "text": passage["text"]})
            index_to_bias[passage_idx] = passage["bias_rating"]

        prompt = build_prompt(prompt_passages, query)
        t2 = time.perf_counter()

        try:
            answer, retry_used = generate_answer_with_single_retry(client, model_name, prompt, params)
            t3 = time.perf_counter()

        except Exception as e:
            t3 = time.perf_counter()
            failed_rows.append({
                "event_id": event_id,
                "title": title,
                "query": query,
                "condition": condition,
                "model": model_name,
                "failure_reason": f"generation_error:{type(e).__name__}:{e}",
            })
            logger.error(f"[{idx + 1}/{len(queries)}] generation failed for event_id={event_id}: {e}")
            gen_fail_count += 1
            continue
        
        cited_indices = parse_unique_citations(answer, len(passages))

        # Single retry only when zero citations
        if len(cited_indices) == 0:
            retry_prompt = (
                prompt
                + "\n\nIMPORTANT: Your final answer must include ONLY one numeric citation like [1]. "
                  "If uncertain, state uncertainty but still cite the most relevant passage."
            )
            no_cite_fail_count += 1

            try:
                retry_t0 = time.perf_counter()
                completion_retry = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": retry_prompt}],
                    temperature=float(params["temperature"]),
                    top_p=float(params["top_p"]),
                    max_tokens=int(params["max_tokens"]),
                )
                answer_retry = completion_retry.choices[0].message.content or ""
                t_retry_done = time.perf_counter()
                timing_totals["retry_call"] += t_retry_done - retry_t0
                cited_indices_retry = parse_unique_citations(answer_retry, len(passages))

                if len(cited_indices_retry) > 0:
                    answer = answer_retry
                    cited_indices = cited_indices_retry
                    retry_used = True
                    logger.info(f"[{idx + 1}/{len(queries)}] retry fixed no-citation")
                    retry_fixed_count += 1
                else:
                    failed_rows.append({
                        "event_id": event_id,
                        "title": title,
                        "query": query,
                        "condition": condition,
                        "model": model_name,
                        "failure_reason": "no_citations_after_retry",
                    })
                    logger.info(f"[{idx + 1}/{len(queries)}] failed: no citations after retry")
                    continue

            except Exception as e:
                failed_rows.append({
                    "event_id": event_id,
                    "title": title,
                    "query": query,
                    "condition": condition,
                    "model": model_name,
                    "failure_reason": f"retry_generation_error:{type(e).__name__}:{e}",
                })
                logger.error(f"[{idx + 1}/{len(queries)}] retry generation failed: {e}")
                continue

        cited_biases = [index_to_bias[i] for i in cited_indices if i in index_to_bias]
        left_cites = cited_biases.count("left")
        center_cites = cited_biases.count("center")
        right_cites = cited_biases.count("right")

        row_end = time.perf_counter()  # Add this line
        timing_totals["retrieve"] += t1 - t0  # Add these
        timing_totals["prompt"] += t2 - t1
        timing_totals["first_call"] += t3 - t2
        timing_totals["row_total"] += row_end - row_start

        per_query_rows.append({
            "event_id": event_id,
            "title": title,
            "query": query,
            "conditions": condition,
            "model": model_name,
            "answer": answer,
            "unique_citations": str(cited_indices),
            "unique_left_citations": left_cites,
            "unique_center_citations": center_cites,
            "unique_right_citations": right_cites,
            "balanced_status": balance_status,
            "retry_used": bool(retry_used)
        })

        if idx < 3:
            logger.info(f"[{idx + 1}/{len(queries)}] sample answer: {answer[:250]}")

    per_query_df = pd.DataFrame(per_query_rows)
    failed_df = pd.DataFrame(failed_rows)

    left_total = int(per_query_df["unique_left_citations"].sum()) if not per_query_df.empty else 0
    center_total = int(per_query_df["unique_center_citations"].sum()) if not per_query_df.empty else 0
    right_total = int(per_query_df["unique_right_citations"].sum()) if not per_query_df.empty else 0

    # Three-way denominator and balance score (Left - Right) / (Left + Centre + Right)
    denom_three = left_total + center_total + right_total
    balance_score = ((left_total - right_total) / denom_three) * 100 if denom_three > 0 else None

    # Binary CPI (legacy): (Left - Right) / (Left + Right)
    denom_binary = left_total + right_total
    cpi_binary = ((left_total - right_total) / denom_binary) * 100 if denom_binary > 0 else None

    run_slug = f"{condition}_{model_name.replace('/', '_').replace(':', '_')}"
    per_query_out = RESULTS_DIR / f"per_query_results__{run_slug}.csv"
    failed_out = RESULTS_DIR / f"failed_queries__{run_slug}.csv"
    summary_out = RESULTS_DIR / f"model_condition_summary__{run_slug}.csv"

    per_query_df.to_csv(per_query_out, index=False)
    failed_df.to_csv(failed_out, index=False)

    retry_used_count = int(per_query_df["retry_used"].sum()) if ("retry_used" in per_query_df.columns and not per_query_df.empty) else 0

    # Choose which metrics to expose based on metric_mode
    # metric_mode: "cpi_bbinary", "balance_three", or "both"
    summary_record = {
        "model": model_name,
        "condition": condition,
        "attempted_queries": int(len(queries)),
        "successful_queries": int(len(per_query_df)),
        "failed_queries": int(len(failed_df)),
        "left_unique_citations_total": left_total,
        "center_unique_citations_total": center_total,
        "right_unique_citations_total": right_total,
        "retry_used_count": retry_used_count
    }

    if metric_mode == "cpi_binary":
        summary_record["cpi"] = cpi_binary
        summary_record["balance_score"] = None
    elif metric_mode == "balance_three":
        summary_record["cpi"] = None
        summary_record["balance_score"] = balance_score
    else: # both
        summary_record["cpi"] = cpi_binary
        summary_record["balance_score"] = balance_score

    summary_df = pd.DataFrame([summary_record])
    summary_df.to_csv(summary_out, index=False)

    logger.info(f"Saved: {per_query_out}")
    logger.info(f"Saved: {failed_out}")
    logger.info(f"Saved: {summary_out}")
    logger.info(f"Balanced Score: {balance_score}")
    logger.info(f"CPI: {cpi_binary}")

    
    progress.set_postfix({
        "ok": len(per_query_rows),
        "fail": len(failed_rows),
        "retry_fixed": retry_fixed_count
    })

    print(f"Generation failures: {gen_fail_count}")
    print(f"No-citation failures: {no_cite_fail_count}")
    print(f"Retry fixed: {retry_fixed_count}")

    n = max(len(per_query_rows), 1)
    logger.info(
        "Timing avg per successful row (sec): "
        f"retrieve={timing_totals['retrieve']/n:.3f}, "
        f"prompt={timing_totals['prompt']/n:.3f}, "
        f"first_call={timing_totals['first_call']/n:.3f}, "
        f"retry_call={timing_totals['retry_call']/n:.3f}, "
        f"row_total={timing_totals['row_total']/n:.3f}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run-condition"])
    parser.add_argument("--condition", required=True, choices=["c1_baseline", "c2_anonymized", "c3_balanced"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset-n", type=int, default=100)
    parser.add_argument("--metric", choices=["cpi_binary", "balance_three", "both"],
                        help = 'Which metric(s) to compute/save for this run. If omitted, uses config.metrics.mode.')
    args = parser.parse_args()

    if args.stage == "run-condition":
        run_condition(args.condition, args.model, args.subset_n, args.metric)

