"""
End-to-end search engine pipeline: index -> search -> evaluate.

Runs a single configuration end-to-end instead of the full comparative
analysis from analysis.py.

Usage:
    python run-end-to-end.py
    python run-end-to-end.py --index-method spimi --encoding elias_gamma --scoring bm25
    python run-end-to-end.py --index-method bsbi --dict-type fst --scoring wand_bm25
    python run-end-to-end.py --index-method lsi --lsi-components 200 --hnsw-M 64
    python run-end-to-end.py --index-method lsi --encoding elias_gamma --dict-type fst
    python run-end-to-end.py --skip-indexing --skip-search

Defaults (no args): BSBI + VBE + TF-IDF + idmap (matches original template)

Config dimensions:
    --index-method   bsbi | spimi | lsi
    --encoding       standard | vbe | elias_gamma
    --dict-type      idmap (python hash) | fst (FST prefix+suffix sharing)
    --scoring        tfidf | bm25 | bm25_alt2 | bm25_alt3 | wand_bm25
    --lsi-*          LSI/FAISS HNSW params (only when --index-method lsi)
    --skip-*         skip indexing / search / eval steps
"""

import argparse
import os
import re
import time

from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings, StandardPostings
from evaluation import load_qrels, rbp, dcg, ndcg, ap


ENCODINGS = {
    "standard": StandardPostings,
    "vbe": VBEPostings,
    "elias_gamma": EliasGammaPostings,
}

SCORING_METHODS = {
    "tfidf": "retrieve_tfidf",
    "bm25": "retrieve_bm25",
    "bm25_alt2": "retrieve_bm25_alt2",
    "bm25_alt3": "retrieve_bm25_alt3",
    "wand_bm25": "retrieve_wand_bm25",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end search engine pipeline: index -> search -> evaluate"
    )

    # core config
    parser.add_argument("--index-method", choices=["bsbi", "spimi", "lsi"], default="bsbi",
                        help="Index construction method (default: bsbi)")
    parser.add_argument("--encoding", choices=list(ENCODINGS.keys()), default="vbe",
                        help="Postings list compression (default: vbe)")
    parser.add_argument("--scoring", choices=list(SCORING_METHODS.keys()), default="tfidf",
                        help="Scoring/retrieval method (default: tfidf)")
    parser.add_argument("--dict-type", choices=["idmap", "fst"], default="idmap",
                        help="Dictionary structure: idmap (python hash) or fst (default: idmap)")

    # retrieval / eval params
    parser.add_argument("--k", type=int, default=1000,
                        help="Top-K for evaluation (default: 1000)")
    parser.add_argument("--search-k", type=int, default=10,
                        help="Top-K for demo search (default: 10)")

    # LSI-specific
    parser.add_argument("--lsi-components", type=int, default=100,
                        help="Number of SVD components for LSI (default: 100)")
    parser.add_argument("--hnsw-M", type=int, default=32,
                        help="HNSW connections per node (default: 32)")
    parser.add_argument("--hnsw-ef-construction", type=int, default=200,
                        help="HNSW construction-time search depth (default: 200)")
    parser.add_argument("--hnsw-ef-search", type=int, default=128,
                        help="HNSW query-time search depth (default: 128)")

    # step control
    parser.add_argument("--skip-indexing", action="store_true",
                        help="Skip indexing (assumes index already built)")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip demo search step")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation step")

    return parser.parse_args()


def make_instance(index_method, encoding, dict_type):
    """Create a BSBIIndex or SPIMIIndex instance."""
    enc = ENCODINGS[encoding]
    method_dir = f"{index_method}_fst" if dict_type == "fst" else index_method

    kwargs = dict(
        data_dir="collection",
        postings_encoding=enc,
        output_dir=os.path.join("index", method_dir, enc.name),
        tmp_dir=os.path.join("tmp", method_dir, enc.name),
        dict_type=dict_type,
    )

    if index_method == "spimi":
        from spimi import SPIMIIndex
        return SPIMIIndex(**kwargs)
    return BSBIIndex(**kwargs)


def run_evaluation(retrieve_fn, k):
    """Evaluate using all metrics over the 30 queries."""
    print(f"\n[3/3] Evaluation (top-{k})...")
    qrels = load_qrels()

    rbp_scores, dcg_scores, ndcg_scores, ap_scores = [], [], [], []
    with open("queries.txt") as f:
        for line in f:
            parts = line.strip().split()
            qid, query = parts[0], " ".join(parts[1:])
            num_relevant = sum(qrels[qid].values())

            ranking = []
            for score, doc in retrieve_fn(query, k=k):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking.append(qrels[qid][did])

            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking, num_relevant))
            ap_scores.append(ap(ranking, num_relevant))

    print(f"\n  Mean scores over {len(rbp_scores)} queries:")
    print(f"  RBP  = {sum(rbp_scores) / len(rbp_scores):.6f}")
    print(f"  DCG  = {sum(dcg_scores) / len(dcg_scores):.6f}")
    print(f"  NDCG = {sum(ndcg_scores) / len(ndcg_scores):.6f}")
    print(f"  AP   = {sum(ap_scores) / len(ap_scores):.6f}")


def run_demo_search(retrieve_fn, search_k):
    """Run demo search on 3 sample queries."""
    print(f"\n[2/3] Demo search (top-{search_k})...")
    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
    ]
    for query in queries:
        print(f"\n  Query: {query}")
        for score, doc in retrieve_fn(query, k=search_k):
            print(f"    {doc:30} {score:.3f}")


def run_lsi(args):
    """End-to-end flow for LSI + FAISS."""
    from lsi import LSIIndex

    n = args.lsi_components
    lsi_dir = f"index/lsi_{n}" if n != 100 else "index/lsi"
    lsi = LSIIndex(
        n_components=n,
        output_dir=lsi_dir,
        hnsw_M=args.hnsw_M,
        ef_construction=args.hnsw_ef_construction,
        ef_search=args.hnsw_ef_search,
    )

    enc = ENCODINGS[args.encoding]
    dict_label = " + FST" if args.dict_type == "fst" else ""
    source_label = f"{args.index_method.upper()}{dict_label} + {enc.__name__}"
    config_name = f"LSI (k={n}, M={args.hnsw_M}) + FAISS HNSW  [source: {source_label}]"

    print(f"\nConfiguration: {config_name}")
    print("=" * 60)

    # 1. Build
    if not args.skip_indexing:
        print(f"\n[1/3] Building index...")

        # LSI needs a source inverted index
        source = make_instance(args.index_method, args.encoding, args.dict_type)
        source_dir = source.output_dir
        if not os.path.exists(os.path.join(source_dir, "main_index.index")):
            print(f"  Building source inverted index ({source_label})...")
            source.index()

        start = time.perf_counter()
        lsi.build(source)
        elapsed = time.perf_counter() - start
        print(f"  LSI index built in {elapsed:.3f}s")
    else:
        print(f"\n[1/3] Indexing... skipped")

    # 2. Demo search
    if not args.skip_search:
        run_demo_search(lsi.retrieve, args.search_k)
    else:
        print(f"\n[2/3] Demo search... skipped")

    # 3. Evaluation
    if not args.skip_eval:
        run_evaluation(lsi.retrieve, args.k)
    else:
        print(f"\n[3/3] Evaluation... skipped")

    print()


def run_inverted_index(args):
    """End-to-end flow for BSBI / SPIMI inverted index."""
    enc = ENCODINGS[args.encoding]
    dict_label = " + FST" if args.dict_type == "fst" else ""
    config_name = (f"{args.index_method.upper()}{dict_label} + {enc.__name__} "
                   f"+ {args.scoring.upper()}")

    print(f"\nConfiguration: {config_name}")
    print("=" * 60)

    instance = make_instance(args.index_method, args.encoding, args.dict_type)

    # 1. Indexing
    if not args.skip_indexing:
        print(f"\n[1/3] Indexing...")
        start = time.perf_counter()
        instance.index()
        elapsed = time.perf_counter() - start
        print(f"  Done in {elapsed:.3f}s")
    else:
        print(f"\n[1/3] Indexing... skipped")
        instance.load()

    retrieve_fn = getattr(instance, SCORING_METHODS[args.scoring])

    # 2. Demo search
    if not args.skip_search:
        run_demo_search(retrieve_fn, args.search_k)
    else:
        print(f"\n[2/3] Demo search... skipped")

    # 3. Evaluation
    if not args.skip_eval:
        run_evaluation(retrieve_fn, args.k)
    else:
        print(f"\n[3/3] Evaluation... skipped")

    print()


def main():
    args = parse_args()

    if args.index_method == "lsi":
        run_lsi(args)
    else:
        run_inverted_index(args)


if __name__ == "__main__":
    main()
