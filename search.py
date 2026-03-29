import os
from bsbi import BSBIIndex
from spimi import SPIMIIndex
from compression import VBEPostings, EliasGammaPostings

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

# indexing has been performed previously
# BSBIIndex / SPIMIIndex serve only as abstractions for the index
for dict_type in ["idmap", "fst"]:
  for index_method, IndexClass, method_name in [("bsbi", BSBIIndex, "BSBI"),
                                                 ("spimi", SPIMIIndex, "SPIMI")]:
    for postings_encoding in [VBEPostings, EliasGammaPostings]:
        method_dir = f"{index_method}_fst" if dict_type == "fst" else index_method
        dict_label = " + FST" if dict_type == "fst" else ""
        instance = IndexClass(data_dir = 'collection', \
                              postings_encoding = postings_encoding, \
                              output_dir = os.path.join('index', method_dir, postings_encoding.name), \
                              tmp_dir = os.path.join('tmp', method_dir, postings_encoding.name), \
                              dict_type = dict_type)

        scoring_methods = [
            ("TF-IDF", instance.retrieve_tfidf),
            ("BM25 Okapi", instance.retrieve_bm25),
            ("BM25 Alt 2", instance.retrieve_bm25_alt2),
            ("BM25 Alt 3", instance.retrieve_bm25_alt3),
            ("WAND BM25", instance.retrieve_wand_bm25),
        ]

        for scoring_name, retrieve_fn in scoring_methods:
            print(f"===== {method_name}{dict_label} + {postings_encoding.__name__} + {scoring_name} =====")
            for query in queries:
                print("Query  : ", query)
                print("Results:")
                for (score, doc) in retrieve_fn(query, k = 10):
                    print(f"  {doc:30} {score:>.3f}")
                print()
            print()

# LSI + FAISS (HNSW) retrieval
from lsi import LSIIndex
lsi = LSIIndex(n_components=100, output_dir='index/lsi')
print("===== LSI + FAISS (HNSW, k=100) =====")
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in lsi.retrieve(query, k = 10):
        print(f"  {doc:30} {score:>.3f}")
    print()
print()
