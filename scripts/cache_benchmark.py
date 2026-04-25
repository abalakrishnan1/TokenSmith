"""
scripts/cache_benchmark.py
"""

import faiss
import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import RAGConfig
from src.instrumentation.logging import get_logger
from src.ranking.ranker import EnsembleRanker
from src.retriever import FAISSRetriever, BM25Retriever, load_artifacts
from src.semantic_cache import SemanticCache
from src.main import get_answer


# Benchmark workloads
# TRAIN_QUERIES: these used for Runs A (no cache) and B (cold cache)
#   20 queries, 10 per topic, with strong semantic locality within topics of ACID and serializability
# PARAPHRASE_QUERIES: these used for run C (warm cache).
#   Tests whether the spatial thresholds generalize beyond exact repeats.

TRAIN_QUERIES = [
    ("ACID", "What is ACID in databases?"),
    ("ACID", "Explain the atomicity property of transactions."),
    ("ACID", "What happens if a transaction violates consistency?"),
    ("ACID", "How does durability protect against system crashes?"),
    ("ACID", "What is the isolation property and why does it matter?"),
    ("ACID", "Give an example of an atomicity failure."),
    ("ACID", "How are ACID properties enforced by a DBMS?"),
    ("ACID", "What is the relationship between consistency and integrity constraints?"),
    ("ACID", "Can a transaction be durable but not atomic? Explain."),
    ("ACID", "Describe a scenario where isolation prevents incorrect results."),

    ("SERIAL", "What is serializability in databases?"),
    ("SERIAL", "Explain conflict serializability."),
    ("SERIAL", "What is a schedule in transaction processing?"),
    ("SERIAL", "How do you test if a schedule is conflict serializable?"),
    ("SERIAL", "What is the difference between serial and serializable schedules?"),
    ("SERIAL", "What is a precedence graph and how is it used?"),
    ("SERIAL", "Can a non-serial schedule be correct? Why?"),
    ("SERIAL", "What is view serializability and how does it differ from conflict serializability?"),
    ("SERIAL", "Give an example of a non-serializable schedule."),
    ("SERIAL", "Why is serializability important for concurrent transactions?"),
]

PARAPHRASE_QUERIES = [
    ("ACID", "Can you describe what the ACID properties mean for a database?"),
    ("ACID", "What does it mean for a transaction to be atomic?"),
    ("ACID", "If a transaction breaks consistency, what does the database do?"),
    ("ACID", "How does a DBMS keep committed data safe after a crash?"),
    ("ACID", "Why do we need isolation between concurrent transactions?"),
    ("ACID", "Show me a case where atomicity would fail."),
    ("ACID", "Which mechanisms does a database use to guarantee ACID?"),
    ("ACID", "How is consistency related to integrity rules in a database?"),
    ("ACID", "Is it possible for a transaction to be durable without being atomic?"),
    ("ACID", "When would the isolation property stop a wrong outcome from happening?"),

    ("SERIAL", "What does it mean for a schedule to be serializable?"),
    ("SERIAL", "What is conflict serializability about?"),
    ("SERIAL", "Define a schedule in the context of database transactions."),
    ("SERIAL", "How can I check whether a given schedule is conflict serializable?"),
    ("SERIAL", "How is a serial schedule different from a serializable one?"),
    ("SERIAL", "What role does the precedence graph play in serializability testing?"),
    ("SERIAL", "Is it ever okay for a schedule to be non-serial?"),
    ("SERIAL", "How does view serializability compare to conflict serializability?"),
    ("SERIAL", "Show me a schedule that is not serializable."),
    ("SERIAL", "Why does concurrency control care about serializability?"),
]


def setup():
    cfg = RAGConfig.from_yaml(pathlib.Path("config/config.yaml"))
    cfg.disable_streaming = True
    cfg.enable_history = False 
    logger = get_logger()
    args = argparse.Namespace(
        mode="chat",
        system_prompt_mode=cfg.system_prompt_mode,
        double_prompt=False,
        index_prefix="textbook_index",
    )

    artifacts_dir = cfg.get_artifacts_directory()
    faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(artifacts_dir, args.index_prefix)
    retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx)]
    
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )

    artifacts = {
        "chunks": chunks, "sources": sources,
        "retrievers": retrievers, "ranker": ranker, "meta": meta,
    }
    
    print("\nWarming up models (loading into VRAM)...")
    get_answer("warmup query", cfg, args, logger, console=None, artifacts=artifacts, semantic_cache=None)
    print("Warmup complete.\n")
    
    return cfg, args, logger, artifacts


def run_queries(label, queries, cfg, args, logger, artifacts, cache):
    """run a query workload once, used for all runs. returns list of result dicts."""
    print(f"\n{'=' * 80}\n  {label}\n{'=' * 80}")
    results = []
    prev_hits = cache.stats["hits"] if cache else 0

    for topic, q in queries:
        t0 = time.time()
        ans = get_answer(q, cfg, args, logger, console=None, artifacts=artifacts, semantic_cache=cache)
        elapsed = time.time() - t0

        if cache is not None:
            cur_hits = cache.stats["hits"]
            hit = cur_hits > prev_hits
            prev_hits = cur_hits
        else:
            hit = None

        results.append({
            "topic": topic,
            "query": q,
            "latency": round(elapsed, 3),
            "cache_hit": hit,
            "answer_preview": (ans[:150] + "...") if isinstance(ans, str) and len(ans) > 150 else str(ans)[:150],
        })

        flag = "HIT " if hit else ("MISS" if cache else "----")
        print(f"  [{topic:6s}] [{flag}] {elapsed:6.2f}s  {q[:65]}", flush=True)

    return results


def summarize(label, results):
    latencies = [r["latency"] for r in results]
    n = len(latencies)
    mean = sum(latencies)/n
    total = sum(latencies)
    hits = sum(1 for r in results if r["cache_hit"] is True)
    miss_lat = [r["latency"] for r in results if r["cache_hit"] is not True]
    hit_lat = [r["latency"] for r in results if r["cache_hit"] is True]

    print(f"\n  {label}")
    print(f"mean latency: {mean:.2f}s")
    print(f"total time: {total:.1f}s")
    print(f"hits: {hits}/{n}")
    if hit_lat:
        print(f"hit mean: {sum(hit_lat)/len(hit_lat):.2f}s")
    if miss_lat:
        print(f"miss mean: {sum(miss_lat)/len(miss_lat):.2f}s")

    # want to see per topic breakdown of results
    by_topic = {}
    for r in results:
        by_topic.setdefault(r["topic"], []).append(r)
    for t, rs in by_topic.items():
        t_hits = sum(1 for r in rs if r["cache_hit"] is True)
        t_mean = sum(r["latency"] for r in rs) / len(rs)
        print(f"    {t:6s}: {t_hits}/{len(rs)} hits, mean {t_mean:.2f}s")


def main():
    cfg, args, logger, artifacts = setup()
    all_results = {}

    # run A, no cache and just the training queries
    cfg.use_semantic_cache = False
    all_results["no_cache"] = run_queries(
        "RUN A — NO CACHE (baseline)",
        TRAIN_QUERIES, cfg, args, logger, artifacts, cache=None,
    )

    cfg.use_semantic_cache = True

    cache = SemanticCache(capacity=cfg.cache_capacity, alpha=cfg.cache_alpha, deviation=cfg.cache_deviation, embed_model=cfg.embed_model, d_reduced=4, n_buckets=2)

    all_results["cache_cold"] = run_queries("RUN B — CACHE COLD (training queries, spatial neighbors)", TRAIN_QUERIES, cfg, args, logger, artifacts, cache=cache)
    all_results["cache_warm_paraphrase"] = run_queries("RUN C — CACHE WARM (paraphrased queries, held-out)", PARAPHRASE_QUERIES, cfg, args, logger, artifacts, cache=cache)

    # summary text
    print(f"\n{'=' * 80}\n  SUMMARY\n{'=' * 80}")
    for label, results in all_results.items():
        summarize(label, results)

    # speedup
    no_cache_mean = sum(r["latency"] for r in all_results["no_cache"]) / len(all_results["no_cache"])
    warm_mean = sum(r["latency"] for r in all_results["cache_warm_paraphrase"]) / len(all_results["cache_warm_paraphrase"])
    if warm_mean > 0:
        print(f"\n  Speedup (divide by the warm paraphrase): {no_cache_mean / warm_mean:.2f}x")

    out_path = pathlib.Path("scripts/cache_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "capacity": cfg.cache_capacity,
                "alpha": cfg.cache_alpha,
                "deviation": cfg.cache_deviation,
                "top_k": cfg.top_k,
                "rerank_top_k": cfg.rerank_top_k,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # dump cache state for the visualization script
    all_qs = TRAIN_QUERIES + PARAPHRASE_QUERIES
    topics = [t for t, _ in all_qs]
    query_texts = [q for _, q in all_qs]
    query_embeddings = cache.embed_model.encode(query_texts).astype("float32")

    viz_state = {
        "topics": topics,
        "queries": query_texts,
        "query_embeddings": query_embeddings,
        "thresholds": {k: float(v) for k, v in cache._thresholds.items()},
        "bucket_boundaries": (
            np.asarray(cache._bucket_boundaries)
        ),
        "pca_components": cache.pca.components_,
        "pca_mean": cache.pca.mean_ ,
        "d_reduced": cache.d_reduced,
        "n_buckets": cache.n_buckets,
    }

    viz_path = pathlib.Path("scripts/cache_viz_state.pkl")
    with open(viz_path, "wb") as f:
        pickle.dump(viz_state, f)

if __name__ == "__main__":
    main()
