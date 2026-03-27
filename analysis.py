"""
Comparative analysis of search engine configurations.

Compares different index construction methods, compression methods,
and scoring/retrieval strategies across three dimensions:
  1. Index size on disk after indexing
  2. Retrieval speed (latency per query)
  3. Effectiveness (evaluation metrics)

Usage:
    python analysis.py
"""

import os
import re
import time

from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings, EliasGammaPostings
from evaluation import load_qrels, rbp, dcg, ndcg, ap


# list of method configuration to be compared
CONFIGURATIONS = [
    # BSBI configurations
    {
        "name": "BSBI + VBE + TF-IDF",
        "index_method": "bsbi",
        "postings_encoding": VBEPostings,
        "scoring": "tfidf",
    },
    {
        "name": "BSBI + VBE + BM25 Okapi",
        "index_method": "bsbi",
        "postings_encoding": VBEPostings,
        "scoring": "bm25",
    },
    {
        "name": "BSBI + Elias Gamma + TF-IDF",
        "index_method": "bsbi",
        "postings_encoding": EliasGammaPostings,
        "scoring": "tfidf",
    },
    {
        "name": "BSBI + Elias Gamma + BM25 Okapi",
        "index_method": "bsbi",
        "postings_encoding": EliasGammaPostings,
        "scoring": "bm25",
    },
    {
        "name": "BSBI + VBE + WAND BM25",
        "index_method": "bsbi",
        "postings_encoding": VBEPostings,
        "scoring": "wand_bm25",
    },
    {
        "name": "BSBI + Elias Gamma + WAND BM25",
        "index_method": "bsbi",
        "postings_encoding": EliasGammaPostings,
        "scoring": "wand_bm25",
    },
    # SPIMI configurations
    {
        "name": "SPIMI + VBE + TF-IDF",
        "index_method": "spimi",
        "postings_encoding": VBEPostings,
        "scoring": "tfidf",
    },
    {
        "name": "SPIMI + VBE + BM25 Okapi",
        "index_method": "spimi",
        "postings_encoding": VBEPostings,
        "scoring": "bm25",
    },
    {
        "name": "SPIMI + Elias Gamma + TF-IDF",
        "index_method": "spimi",
        "postings_encoding": EliasGammaPostings,
        "scoring": "tfidf",
    },
    {
        "name": "SPIMI + Elias Gamma + BM25 Okapi",
        "index_method": "spimi",
        "postings_encoding": EliasGammaPostings,
        "scoring": "bm25",
    },
    {
        "name": "SPIMI + VBE + WAND BM25",
        "index_method": "spimi",
        "postings_encoding": VBEPostings,
        "scoring": "wand_bm25",
    },
    {
        "name": "SPIMI + Elias Gamma + WAND BM25",
        "index_method": "spimi",
        "postings_encoding": EliasGammaPostings,
        "scoring": "wand_bm25",
    },
]


# Helper functions

def get_index_instance(config):
    """
    Create a BSBIIndex or SPIMIIndex instance from a configuration dict.
    If the index does not exist yet, it will be built automatically.

    Parameters
    ----------
    config : dict
        A configuration dictionary from CONFIGURATIONS

    Returns
    -------
    BSBIIndex or SPIMIIndex
        A configured index instance (with index built if needed)
    """
    enc = config["postings_encoding"]
    method = config["index_method"]

    if method == "spimi":
        instance = SPIMIIndex(
            data_dir="collection",
            postings_encoding=enc,
            output_dir=os.path.join("index", "spimi", enc.name),
            tmp_dir=os.path.join("tmp", "spimi", enc.name),
        )
    else:
        instance = BSBIIndex(
            data_dir="collection",
            postings_encoding=enc,
            output_dir=os.path.join("index", "bsbi", enc.name),
            tmp_dir=os.path.join("tmp", "bsbi", enc.name),
        )

    build_key = f"{method}_{enc.name}"
    if build_key not in _built_indices:
        print(f"  Building index for {method}/{enc.name}...")
        instance.index()
        _built_indices.add(build_key)

    return instance

_built_indices = set()

def retrieve(instance, query, scoring, k=10):
    """
    Run retrieval on an index instance using the specified scoring method.

    Parameters
    ----------
    instance : BSBIIndex or SPIMIIndex
        The index to search
    query : str
        The query string
    scoring : str
        The scoring method identifier (e.g., "tfidf", "bm25")
    k : int
        Number of top results to return

    Returns
    -------
    List[Tuple[float, str]]
        Top-K results as (score, doc_path) tuples
    """
    if scoring == "tfidf":
        return instance.retrieve_tfidf(query, k=k)
    elif scoring == "bm25":
        return instance.retrieve_bm25(query, k=k)
    elif scoring == "bm25_alt2":
        return instance.retrieve_bm25_alt2(query, k=k)
    elif scoring == "bm25_alt3":
        return instance.retrieve_bm25_alt3(query, k=k)
    elif scoring == "wand_bm25":
        return instance.retrieve_wand_bm25(query, k=k)
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")


# 1. Index size comparison

def measure_index_size(config):
    """
    Measure total index size on disk for a configuration.
    Sums the sizes of all files in the output directory.

    Parameters
    ----------
    config : dict
        A configuration dictionary from CONFIGURATIONS

    Returns
    -------
    int
        Total size in bytes
    """
    # Ensure index is built
    get_index_instance(config)

    enc = config["postings_encoding"]
    method = config["index_method"]
    index_dir = os.path.join("index", method, enc.name)
    total = 0
    for f in os.listdir(index_dir):
        total += os.path.getsize(os.path.join(index_dir, f))
    return total

def compare_index_sizes(configs):
    """
    Print a table comparing index sizes across configurations.

    Parameters
    ----------
    configs : List[dict]
        List of configuration dictionaries
    """
    print("=" * 60)
    print("1. INDEX SIZE COMPARISON")
    print("=" * 60)

    results = []
    for config in configs:
        size = measure_index_size(config)
        results.append((config["name"], size))

    # find the smallest for relative comparison
    min_size = min(size for _, size in results)

    print(f"  {'Configuration':<40} {'Size':>10} {'Relative':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    for name, size in results:
        relative = size / min_size
        print(f"  {name:<40} {size:>8} B  {relative:>9.2f}x")
    print()


# 2. Retrieval speed comparison

def measure_retrieval_speed(config, queries, k=10, n_runs=3):
    """
    Measure average retrieval latency per query for a configuration.
    Runs each query n_runs times and reports the mean.

    Parameters
    ----------
    config : dict
        A configuration dictionary from CONFIGURATIONS
    queries : List[str]
        List of query strings
    k : int
        Number of top results to retrieve
    n_runs : int
        Number of repetitions for averaging

    Returns
    -------
    tuple(float, float)
        (average time per query in ms, total time in ms)
    """
    instance = get_index_instance(config)
    scoring = config["scoring"]

    # Pre-load all data structures into memory before timing
    instance.load()

    # warm-up run
    for q in queries:
        retrieve(instance, q, scoring, k=k)

    # timed runs
    total_time = 0.0
    total_queries = 0
    for _ in range(n_runs):
        for q in queries:
            start = time.perf_counter()
            retrieve(instance, q, scoring, k=k)
            total_time += time.perf_counter() - start
            total_queries += 1

    total_ms = total_time * 1000
    avg_ms = total_ms / total_queries
    return avg_ms, total_ms

def compare_retrieval_speed(configs, queries, k=10):
    """
    Print a table comparing retrieval speed across configurations.

    Parameters
    ----------
    configs : List[dict]
        List of configuration dictionaries
    queries : List[str]
        List of query strings
    k : int
        Number of top results to retrieve
    """
    print("=" * 60)
    print("2. RETRIEVAL SPEED COMPARISON")
    print("=" * 60)

    results = []
    for config in configs:
        avg_ms, total_ms = measure_retrieval_speed(config, queries, k=k)
        results.append((config["name"], avg_ms, total_ms))

    min_avg = min(avg for _, avg, _ in results)
    min_total = min(total for _, _, total in results)

    print(f"  {'Configuration':<40} {'Avg/query':>12} {'Rel Avg':>10} {'Total':>12} {'Rel Total':>10}")
    print(f"  {'-'*40} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    for name, avg, total in results:
        rel_avg = avg / min_avg
        rel_total = total / min_total
        print(f"  {name:<40} {avg:>9.2f} ms  {rel_avg:>9.2f}x {total:>9.2f} ms  {rel_total:>9.2f}x")
    print()


# 3. Effectiveness comparison

def measure_effectiveness(config, qrels, query_file="queries.txt", k=1000):
    """
    Compute evaluation metrics for a configuration over all queries.

    Parameters
    ----------
    config : dict
        A configuration dictionary from CONFIGURATIONS
    qrels : dict
        Query relevance judgments as loaded by load_qrels()
    query_file : str
        Path to the queries file
    k : int
        Number of top documents to retrieve per query

    Returns
    -------
    dict
        Dictionary of metric_name -> score
    """
    instance = get_index_instance(config)
    scoring = config["scoring"]

    metrics = {}
    with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ndcg_scores = []
        ap_scores = []
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            num_relevant = sum(qrels[qid].values())

            ranking = []
            for (score, doc) in retrieve(instance, query, scoring, k=k):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking.append(qrels[qid][did])
            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking, num_relevant))
            ap_scores.append(ap(ranking, num_relevant))

    metrics["RBP"] = sum(rbp_scores) / len(rbp_scores)
    metrics["DCG"] = sum(dcg_scores) / len(dcg_scores)
    metrics["NDCG"] = sum(ndcg_scores) / len(ndcg_scores)
    metrics["AP"] = sum(ap_scores) / len(ap_scores)
    return metrics

def compare_effectiveness(configs, qrels):
    """
    Print a table comparing effectiveness metrics across configurations.

    Parameters
    ----------
    configs : List[dict]
        List of configuration dictionaries
    qrels : dict
        Query relevance judgments
    """
    print("=" * 60)
    print("3. EFFECTIVENESS COMPARISON")
    print("=" * 60)

    results = []
    all_metric_names = set()
    for config in configs:
        metrics = measure_effectiveness(config, qrels)
        results.append((config["name"], metrics))
        all_metric_names.update(metrics.keys())

    metric_names = sorted(all_metric_names)

    header = f"  {'Configuration':<40}"
    for m in metric_names:
        header += f" {m:>10}"
    print(header)
    print(f"  {'-'*40}" + f" {'-'*10}" * len(metric_names))
    for name, metrics in results:
        row = f"  {name:<40}"
        for m in metric_names:
            row += f" {metrics.get(m, 0):.6f}  "
        print(row)
    print()


if __name__ == "__main__":

    print()
    print("Search Engine Configuration Analysis")
    print("=" * 60)
    print()

    # load queries for speed test
    queries = []
    with open("queries.txt") as f:
        for line in f:
            parts = line.strip().split()
            queries.append(" ".join(parts[1:]))

    # 1. Index size
    compare_index_sizes(CONFIGURATIONS)

    # 2. Retrieval speed
    compare_retrieval_speed(CONFIGURATIONS, queries, k=10)

    # 3. Effectiveness
    qrels = load_qrels()
    compare_effectiveness(CONFIGURATIONS, qrels)
