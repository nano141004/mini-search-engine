"""
Comparative analysis of search engine configurations.

Compares different index construction methods, compression methods,
and scoring/retrieval strategies across four dimensions:
  1. Index build time
  2. Index size on disk after indexing
  3. Retrieval speed (latency per query)
  4. Effectiveness (evaluation metrics)

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
    # FST dictionary configurations (prefix+suffix sharing compression)
    {
        "name": "BSBI + FST + VBE + BM25 Okapi",
        "index_method": "bsbi",
        "dict_type": "fst",
        "postings_encoding": VBEPostings,
        "scoring": "wand_bm25",
    },
    {
        "name": "BSBI + FST + Elias Gamma + BM25 Okapi",
        "index_method": "bsbi",
        "dict_type": "fst",
        "postings_encoding": EliasGammaPostings,
        "scoring": "wand_bm25",
    },
    {
        "name": "SPIMI + FST + VBE + BM25 Okapi",
        "index_method": "spimi",
        "dict_type": "fst",
        "postings_encoding": VBEPostings,
        "scoring": "wand_bm25",
    },
    {
        "name": "SPIMI + FST + Elias Gamma + BM25 Okapi",
        "index_method": "spimi",
        "dict_type": "fst",
        "postings_encoding": EliasGammaPostings,
        "scoring": "wand_bm25",
    },
    # LSI + FAISS (HNSW) configurations
    {
        "name": "LSI (k=100) + FAISS HNSW",
        "index_method": "lsi",
        "postings_encoding": VBEPostings,
        "scoring": "lsi",
        "n_components": 100,
    },
    {
        "name": "LSI (k=200) + FAISS HNSW",
        "index_method": "lsi",
        "postings_encoding": VBEPostings,
        "scoring": "lsi",
        "n_components": 200,
    },
]


# Helper functions

def get_index_instance(config):
    """
    Create a BSBIIndex, SPIMIIndex, or LSIIndex instance from a configuration dict.
    If the index does not exist yet, it will be built automatically.

    Parameters
    ----------
    config : dict
        A configuration dictionary from CONFIGURATIONS

    Returns
    -------
    BSBIIndex, SPIMIIndex, or LSIIndex
        A configured index instance (with index built if needed)
    """
    method = config["index_method"]

    if method == "lsi":
        from lsi import LSIIndex
        n_comp = config.get("n_components", 100)
        lsi_dir = f'index/lsi_{n_comp}' if n_comp != 100 else 'index/lsi'
        instance = LSIIndex(n_components=n_comp, output_dir=lsi_dir)
        build_key = f"lsi_{n_comp}"
        if build_key not in _built_indices:
            # LSI needs a source inverted index; build BSBI+VBE if needed
            enc = config["postings_encoding"]
            source = BSBIIndex(
                data_dir="collection", postings_encoding=enc,
                output_dir=os.path.join("index", "bsbi", enc.name),
                tmp_dir=os.path.join("tmp", "bsbi", enc.name),
            )
            source_key = f"bsbi_{enc.name}"
            if source_key not in _built_indices:
                print(f"  Building source index for bsbi/{enc.name}...")
                s0 = time.perf_counter()
                source.index()
                _built_indices[source_key] = time.perf_counter() - s0

            print(f"  Building LSI index (k={n_comp})...")
            start = time.perf_counter()
            instance.build(source)
            _built_indices[build_key] = time.perf_counter() - start
        return instance

    enc = config["postings_encoding"]
    dict_type = config.get("dict_type", "idmap")
    method_dir = f"{method}_fst" if dict_type == "fst" else method

    if method == "spimi":
        instance = SPIMIIndex(
            data_dir="collection",
            postings_encoding=enc,
            output_dir=os.path.join("index", method_dir, enc.name),
            tmp_dir=os.path.join("tmp", method_dir, enc.name),
            dict_type=dict_type,
        )
    else:
        instance = BSBIIndex(
            data_dir="collection",
            postings_encoding=enc,
            output_dir=os.path.join("index", method_dir, enc.name),
            tmp_dir=os.path.join("tmp", method_dir, enc.name),
            dict_type=dict_type,
        )

    build_key = f"{method_dir}_{enc.name}"
    if build_key not in _built_indices:
        print(f"  Building index for {method_dir}/{enc.name}...")
        start = time.perf_counter()
        instance.index()
        elapsed = time.perf_counter() - start
        _built_indices[build_key] = elapsed

    return instance

_built_indices = {}  # build_key -> elapsed seconds

def retrieve(instance, query, scoring, k=10):
    """
    Run retrieval on an index instance using the specified scoring method.

    Parameters
    ----------
    instance : BSBIIndex, SPIMIIndex, or LSIIndex
        The index to search
    query : str
        The query string
    scoring : str
        The scoring method identifier (e.g., "tfidf", "bm25", "lsi")
    k : int
        Number of top results to return

    Returns
    -------
    List[Tuple[float, str]]
        Top-K results as (score, doc_path) tuples
    """
    if scoring == "lsi":
        return instance.retrieve(query, k=k)
    elif scoring == "tfidf":
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


# 1. Index build time comparison

def compare_build_times(configs):
    """
    Print a table comparing index build times across configurations.
    Build time depends only on (index_method, encoding), not scoring,
    so duplicates are shown only once.

    Parameters
    ----------
    configs : List[dict]
        List of configuration dictionaries
    """
    print("=" * 60)
    print("1. INDEX BUILD TIME COMPARISON")
    print("=" * 60)

    # Ensure all indices are built (triggers get_index_instance)
    seen = set()
    results = []
    for config in configs:
        get_index_instance(config)
        method = config["index_method"]
        if method == "lsi":
            n_comp = config.get("n_components", 100)
            build_key = f"lsi_{n_comp}"
            if build_key not in seen:
                seen.add(build_key)
                label = f"LSI (k={n_comp}) + FAISS HNSW"
                elapsed = _built_indices.get(build_key, 0.0)
                results.append((label, elapsed))
        else:
            dict_type = config.get("dict_type", "idmap")
            method_dir = f"{method}_fst" if dict_type == "fst" else method
            build_key = f"{method_dir}_{config['postings_encoding'].name}"
            if build_key not in seen:
                seen.add(build_key)
                dict_label = " + FST" if dict_type == "fst" else ""
                label = f"{method.upper()}{dict_label} + {config['postings_encoding'].__name__}"
                elapsed = _built_indices.get(build_key, 0.0)
                results.append((label, elapsed))

    min_time = min(t for _, t in results) if results else 1.0

    print(f"  {'Configuration':<40} {'Time':>12} {'Relative':>10}")
    print(f"  {'-'*40} {'-'*12} {'-'*10}")
    for name, t in results:
        relative = t / min_time if min_time > 0 else 0
        print(f"  {name:<40} {t:>9.3f} s   {relative:>9.2f}x")
    print()


# 2. Index size comparison (on disk)

def dir_size(path):
    """Sum the sizes of all files in a directory (non-recursive)."""
    total = 0
    if os.path.isdir(path):
        for f in os.listdir(path):
            total += os.path.getsize(os.path.join(path, f))
    return total

def measure_index_size(config):
    """
    Measure total index size on disk for a configuration.
    Returns both the final index size and the intermediate (tmp) size.

    Parameters
    ----------
    config : dict
        A configuration dictionary from CONFIGURATIONS

    Returns
    -------
    tuple(int, int)
        (final index size in bytes, tmp intermediate size in bytes)
    """
    # Ensure index is built
    get_index_instance(config)

    method = config["index_method"]
    if method == "lsi":
        n_comp = config.get("n_components", 100)
        lsi_dir = f'index/lsi_{n_comp}' if n_comp != 100 else 'index/lsi'
        return dir_size(lsi_dir), 0

    enc = config["postings_encoding"]
    dict_type = config.get("dict_type", "idmap")
    method_dir = f"{method}_fst" if dict_type == "fst" else method
    index_dir = os.path.join("index", method_dir, enc.name)
    tmp_dir = os.path.join("tmp", method_dir, enc.name)
    return dir_size(index_dir), dir_size(tmp_dir)

def compare_index_sizes(configs):
    """
    Print a table comparing index sizes across configurations.
    Shows both the final index size and intermediate (tmp) size,
    since SPIMI's pickle intermediates are larger than BSBI's
    compressed intermediates.

    Size depends only on (index_method, encoding), not scoring,
    so duplicate rows are collapsed.

    Parameters
    ----------
    configs : List[dict]
        List of configuration dictionaries
    """
    print("=" * 80)
    print("2. INDEX SIZE COMPARISON")
    print("=" * 80)

    seen = set()
    results = []
    for config in configs:
        method = config["index_method"]
        if method == "lsi":
            n_comp = config.get("n_components", 100)
            build_key = f"lsi_{n_comp}"
            if build_key in seen:
                continue
            seen.add(build_key)
            final_size, tmp_size = measure_index_size(config)
            label = f"LSI (k={n_comp}) + FAISS HNSW"
            results.append((label, final_size, tmp_size))
        else:
            dict_type = config.get("dict_type", "idmap")
            method_dir = f"{method}_fst" if dict_type == "fst" else method
            build_key = f"{method_dir}_{config['postings_encoding'].name}"
            if build_key in seen:
                continue
            seen.add(build_key)
            final_size, tmp_size = measure_index_size(config)
            dict_label = " + FST" if dict_type == "fst" else ""
            label = f"{method.upper()}{dict_label} + {config['postings_encoding'].__name__}"
            results.append((label, final_size, tmp_size))

    min_final = min(f for _, f, _ in results)
    min_tmp = min(t for _, _, t in results) if any(t > 0 for _, _, t in results) else 1

    print(f"  {'Configuration':<35} {'Final':>10} {'Rel':>7} {'Tmp (intermediate)':>18} {'Rel':>7} {'Total':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*7} {'-'*18} {'-'*7} {'-'*10}")
    for name, final, tmp in results:
        rel_f = final / min_final
        rel_t = tmp / min_tmp if min_tmp > 0 else 0
        total = final + tmp
        print(f"  {name:<35} {final:>8} B {rel_f:>6.2f}x {tmp:>16} B {rel_t:>6.2f}x {total:>8} B")
    print()


# 3. Retrieval speed comparison

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
    print("3. RETRIEVAL SPEED COMPARISON")
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


# 4. Effectiveness comparison

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
    print("4. EFFECTIVENESS COMPARISON")
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

    # 1. Build time (triggers index building for all configs)
    compare_build_times(CONFIGURATIONS)

    # 2. Index size
    compare_index_sizes(CONFIGURATIONS)

    # 3. Retrieval speed
    compare_retrieval_speed(CONFIGURATIONS, queries, k=10)

    # 4. Effectiveness
    qrels = load_qrels()
    compare_effectiveness(CONFIGURATIONS, qrels)
