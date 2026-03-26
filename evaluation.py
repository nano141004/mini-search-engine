import os
import re
from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings

######## >>>>> an IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ Calculate search effectiveness metric score using
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         binary vector such as [1, 0, 1, 1, 1, 0]
         gold standard relevance of documents at rank 1, 2, 3, etc.
         Example: [1, 0, 1, 1, 1, 0] means the document at rank-1 is relevant,
                  at rank-2 not relevant, at rank-3,4,5 relevant, and
                  at rank-6 not relevant

      Returns
      -------
      Float
        RBP score
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


######## >>>>> loading qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ Load query relevance judgments (qrels)
      in the format of dictionary of dictionary
      qrels[query id][document id]

      where, for example, qrels["Q3"][12] = 1 means Doc 12
      is relevant to Q3; and qrels["Q3"][10] = 0 means
      Doc 10 is not relevant to Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUATION !

def eval(qrels, postings_encoding = VBEPostings, query_file = "queries.txt", k = 1000):
  """
    Loop over all 30 queries, compute the score for each query,
    then compute the MEAN SCORE over those 30 queries.
    For each query, return the top-1000 documents.

    Parameters
    ----------
    qrels : dict
        Query relevance judgments as loaded by load_qrels()
    postings_encoding : class
        Postings encoding class (e.g., VBEPostings, EliasGammaPostings)
    query_file : str
        Path to the file containing queries
    k : int
        Number of top documents to retrieve per query
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = postings_encoding, \
                          output_dir = os.path.join('index', postings_encoding.name), \
                          tmp_dir = os.path.join('tmp', postings_encoding.name))

  with open(query_file) as file:
    rbp_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # BE CAREFUL, the doc id during indexing may differ from the doc id
      # listed in qrels
      ranking = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))

  print("TF-IDF evaluation results over 30 queries")
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels incorrect"
  assert qrels["Q1"][300] == 0, "qrels incorrect"

  for postings_encoding in [VBEPostings, EliasGammaPostings]:
    print(f"\n===== Evaluation using {postings_encoding.__name__} =====")
    eval(qrels, postings_encoding)