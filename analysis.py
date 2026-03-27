"""
Comparative analysis of search engine configurations.

Compares different compression methods and scoring/retrieval
strategies across three dimensions:
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
from compression import VBEPostings, EliasGammaPostings
from evaluation import load_qrels, rbp, dcg, ndcg, ap


# list of method configuration to be compared
CONFIGURATIONS = [
    {
        "name": "VBE + TF-IDF",
        "postings_encoding": VBEPostings,
        "scoring": "tfidf",
    },
    {
        "name": "VBE + BM25 Okapi",
        "postings_encoding": VBEPostings,
        "scoring": "bm25",
    },
    # {
    #     "name": "VBE + BM25 Alt 2",
    #     "postings_encoding": VBEPostings,
    #     "scoring": "bm25_alt2",
    # },
    # {
    #     "name": "VBE + BM25 Alt 3",
    #     "postings_encoding": VBEPostings,
    #     "scoring": "bm25_alt3",
    # },
    {
        "name": "Elias Gamma + TF-IDF",
        "postings_encoding": EliasGammaPostings,
        "scoring": "tfidf",
    },
    {
        "name": "Elias Gamma + BM25 Okapi",
        "postings_encoding": EliasGammaPostings,
        "scoring": "bm25",
    },
    # {
    #     "name": "Elias Gamma + BM25 Alt 2",
    #     "postings_encoding": EliasGammaPostings,
    #     "scoring": "bm25_alt2",
    # },
    # {
    #     "name": "Elias Gamma + BM25 Alt 3",
    #     "postings_encoding": EliasGammaPostings,
    #     "scoring": "bm25_alt3",
    # },
    {
        "name": "VBE + WAND BM25",
        "postings_encoding": VBEPostings,
        "scoring": "wand_bm25",
    },
    {
        "name": "Elias Gamma + WAND BM25",
        "postings_encoding": EliasGammaPostings,
        "scoring": "wand_bm25",
    },
]


# Helper functions

def get_bsbi_instance(config):
    """
    Create a BSBIIndex instance from a configuration dict.
    If the index does not exist yet, it will be built automatically.

    Parameters
    ----------
    config : dict
        A configuration dictionary from CONFIGURATIONS

    Returns
    -------
    BSBIIndex
        A configured BSBIIndex instance (with index built if needed)
    """
    enc = config["postings_encoding"]
    bsbi = BSBIIndex(
        data_dir="collection",
        postings_encoding=enc,
        output_dir=os.path.join("index", enc.name),
        tmp_dir=os.path.join("tmp", enc.name),
    )

    # Rebuild once per encoding (tracked by _built_encodings)
    if enc.name not in _built_encodings:
        print(f"  Building index for {enc.name}...")
        bsbi.index()
        _built_encodings.add(enc.name)

    return bsbi

_built_encodings = set()

def retrieve(bsbi_instance, query, scoring, k=10):
    """
    Run retrieval on a BSBIIndex instance using the specified scoring method.

    Parameters
    ----------
    bsbi_instance : BSBIIndex
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
        return bsbi_instance.retrieve_tfidf(query, k=k)
    elif scoring == "bm25":
        return bsbi_instance.retrieve_bm25(query, k=k)
    elif scoring == "bm25_alt2":
        return bsbi_instance.retrieve_bm25_alt2(query, k=k)
    elif scoring == "bm25_alt3":
        return bsbi_instance.retrieve_bm25_alt3(query, k=k)
    elif scoring == "wand_bm25":
        return bsbi_instance.retrieve_wand_bm25(query, k=k)
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")


# 1. Index size comparison

def measure_index_size(config):
    """
    Measure total index size on disk for a configuration.
    Sums the sizes of all files in the output directory (index/<method>/).

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
    get_bsbi_instance(config)

    enc = config["postings_encoding"]
    index_dir = os.path.join("index", enc.name)
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

    print(f"  {'Configuration':<30} {'Size':>10} {'Relative':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    for name, size in results:
        relative = size / min_size
        print(f"  {name:<30} {size:>8} B  {relative:>9.2f}x")
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
    bsbi = get_bsbi_instance(config)
    scoring = config["scoring"]

    # Pre-load all data structures into memory before timing,
    # so I/O cost is not included in the speed measurement.
    bsbi.load()

    # warm-up run (prime OS file cache and any lazy state)
    for q in queries:
        retrieve(bsbi, q, scoring, k=k)

    # timed runs
    total_time = 0.0
    total_queries = 0
    for _ in range(n_runs):
        for q in queries:
            start = time.perf_counter()
            retrieve(bsbi, q, scoring, k=k)
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

    print(f"  {'Configuration':<30} {'Avg/query':>12} {'Rel Avg':>10} {'Total':>12} {'Rel Total':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    for name, avg, total in results:
        rel_avg = avg / min_avg
        rel_total = total / min_total
        print(f"  {name:<30} {avg:>9.2f} ms  {rel_avg:>9.2f}x {total:>9.2f} ms  {rel_total:>9.2f}x")
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
    bsbi = get_bsbi_instance(config)
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
            for (score, doc) in retrieve(bsbi, query, scoring, k=k):
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

    header = f"  {'Configuration':<30}"
    for m in metric_names:
        header += f" {m:>10}"
    print(header)
    print(f"  {'-'*30}" + f" {'-'*10}" * len(metric_names))
    for name, metrics in results:
        row = f"  {name:<30}"
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
