import math
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
  for i in range(1, len(ranking) + 1): #fixed from the base code, cause the len(ranking) is not included means it wasnt checking the last ranked doc before
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


######## >>>>> IR metric: DCG (Discounted Cumulative Gain)

def dcg(ranking):
  """ Calculate Discounted Cumulative Gain (DCG).
      Uses the formula from Jarvelin & Kekalainen (Microsoft modified):
        DCG@K = sum_{i=1}^{K} r_i / log2(i + 1)

      Parameters
      ----------
      ranking: List[int]
         binary relevance vector, e.g. [1, 0, 1, 1, 1, 0]

      Returns
      -------
      Float
        DCG score
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    if ranking[pos] == 1:
      score += 1.0 / math.log2(i + 1)
  return score


######## >>>>> IR metric: NDCG (Normalized DCG)

def ndcg(ranking, num_relevant):
  """ Calculate Normalized Discounted Cumulative Gain (NDCG).
        NDCG@K = DCG@K / IDCG@K
      where IDCG@K is DCG@K of the ideal ranking (all relevant docs at top).

      Parameters
      ----------
      ranking: List[int]
         binary relevance vector, e.g. [1, 0, 1, 1, 1, 0]
      num_relevant: int
         total number of relevant documents in the collection for this query

      Returns
      -------
      Float
        NDCG score (between 0 and 1)
  """
  k = len(ranking)
  dcg_score = dcg(ranking)

  # ideal ranking: min(num_relevant, k) ones at the top, rest zeros
  ideal_ranking = [1] * min(num_relevant, k) + [0] * max(0, k - num_relevant)
  idcg_score = dcg(ideal_ranking)

  if idcg_score == 0:
    return 0.0
  return dcg_score / idcg_score


######## >>>>> IR metric: AP (Average Precision)

def ap(ranking, num_relevant):
  """ Calculate Average Precision (AP).
        AP@K = sum_{i=1}^{K} (Prec@i / R) * r_i
      where R is the total number of relevant documents in the collection,
      and Prec@i = (# relevant docs in top i) / i.

      Only positions with r_i = 1 contribute to the sum.

      Parameters
      ----------
      ranking: List[int]
         binary relevance vector, e.g. [1, 0, 1, 1, 1, 0]
      num_relevant: int
         total number of relevant documents in the collection for this query (R)

      Returns
      -------
      Float
        AP score (between 0 and 1)
  """
  if num_relevant == 0:
    return 0.0

  score = 0.
  relevant_count = 0
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    if ranking[pos] == 1:
      relevant_count += 1
      prec_at_i = relevant_count / i  # Prec@i = (# relevant in top i) / i
      score += prec_at_i / num_relevant
  return score


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

def eval(qrels, postings_encoding = VBEPostings, scoring = "tfidf", index_method = "bsbi", dict_type = "idmap", query_file = "queries.txt", k = 1000, lsi_n_components = 100):
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
    scoring : str
        Scoring method: "tfidf", "bm25", "bm25_alt2", "bm25_alt3", "wand_bm25", or "lsi"
    index_method : str
        Index construction method: "bsbi" or "spimi"
    dict_type : str
        Dictionary type: "idmap" (default) or "fst" (FST compression)
    query_file : str
        Path to the file containing queries
    k : int
        Number of top documents to retrieve per query
    lsi_n_components : int
        Number of LSI components (only used when scoring="lsi")
  """
  if scoring == "lsi":
      from lsi import LSIIndex
      lsi_dir = f'index/lsi_{lsi_n_components}' if lsi_n_components != 100 else 'index/lsi'
      lsi_instance = LSIIndex(n_components=lsi_n_components, output_dir=lsi_dir)
      retrieve_fn = lsi_instance.retrieve
  else:
      method_dir = f"{index_method}_fst" if dict_type == "fst" else index_method
      if index_method == "spimi":
          from spimi import SPIMIIndex
          instance = SPIMIIndex(data_dir = 'collection',
                                postings_encoding = postings_encoding,
                                output_dir = os.path.join('index', method_dir, postings_encoding.name),
                                tmp_dir = os.path.join('tmp', method_dir, postings_encoding.name),
                                dict_type = dict_type)
      else:
          instance = BSBIIndex(data_dir = 'collection',
                               postings_encoding = postings_encoding,
                               output_dir = os.path.join('index', method_dir, postings_encoding.name),
                               tmp_dir = os.path.join('tmp', method_dir, postings_encoding.name),
                               dict_type = dict_type)

      retrieve_fn = {
          "tfidf": instance.retrieve_tfidf,
          "bm25": instance.retrieve_bm25,
          "bm25_alt2": instance.retrieve_bm25_alt2,
          "bm25_alt3": instance.retrieve_bm25_alt3,
          "wand_bm25": instance.retrieve_wand_bm25,
      }[scoring]

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ndcg_scores = []
    ap_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # total number of relevant docs for this query (R)
      num_relevant = sum(qrels[qid].values())

      # BE CAREFUL, the doc id during indexing may differ from the doc id
      # listed in qrels
      ranking = []
      for (score, doc) in retrieve_fn(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ndcg_scores.append(ndcg(ranking, num_relevant))
      ap_scores.append(ap(ranking, num_relevant))

  print(f"{scoring.upper()} evaluation results over 30 queries")
  print("RBP score  =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score  =", sum(dcg_scores) / len(dcg_scores))
  print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
  print("AP score   =", sum(ap_scores) / len(ap_scores))

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels incorrect"
  assert qrels["Q1"][300] == 0, "qrels incorrect"

  for dict_type in ["idmap", "fst"]:
    for index_method in ["bsbi", "spimi"]:
      for postings_encoding in [VBEPostings, EliasGammaPostings]:
        dict_label = " + FST" if dict_type == "fst" else ""
        print(f"\n===== Evaluation using {index_method.upper()}{dict_label} + {postings_encoding.__name__} =====")
        for scoring in ["tfidf", "bm25", "bm25_alt2", "bm25_alt3", "wand_bm25"]:
          eval(qrels, postings_encoding, scoring=scoring, index_method=index_method, dict_type=dict_type)

  # LSI + FAISS evaluation
  print(f"\n===== Evaluation using LSI + FAISS (HNSW, k=100) =====")
  eval(qrels, scoring="lsi", lsi_n_components=100)