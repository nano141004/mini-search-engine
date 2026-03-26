import os
from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

# indexing has been performed previously
# BSBIIndex serves only as an abstraction for that index
for postings_encoding in [VBEPostings, EliasGammaPostings]:
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = postings_encoding, \
                              output_dir = os.path.join('index', postings_encoding.name), \
                              tmp_dir = os.path.join('tmp', postings_encoding.name))

    scoring_methods = [
        ("TF-IDF", BSBI_instance.retrieve_tfidf),
        ("BM25 Okapi", BSBI_instance.retrieve_bm25),
        ("BM25 Alt 2", BSBI_instance.retrieve_bm25_alt2),
        ("BM25 Alt 3", BSBI_instance.retrieve_bm25_alt3),
    ]

    for scoring_name, retrieve_fn in scoring_methods:
        print(f"===== {postings_encoding.__name__} + {scoring_name} =====")
        for query in queries:
            print("Query  : ", query)
            print("Results:")
            for (score, doc) in retrieve_fn(query, k = 10):
                print(f"  {doc:30} {score:>.3f}")
            print()
        print()
