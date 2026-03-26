import os
from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

# indexing has been performed previously
# BSBIIndex serves only as an abstraction for that index
for postings_encoding in [VBEPostings, EliasGammaPostings]:
    print(f"===== Retrieval using {postings_encoding.__name__} =====")
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = postings_encoding, \
                              output_dir = os.path.join('index', postings_encoding.name), \
                              tmp_dir = os.path.join('tmp', postings_encoding.name))

    for query in queries:
        print("Query  : ", query)
        print("Results:")
        for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
            print(f"{doc:30} {score:>.3f}")
        print()
    print()