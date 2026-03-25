from bsbi import BSBIIndex
from compression import VBEPostings

# indexing has been performed previously
# BSBIIndex serves only as an abstraction for that index
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index', \
                          tmp_dir = 'tmp')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]
           
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
        print(f"{doc:30} {score:>.3f}")
    print()