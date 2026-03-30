import os
import pickle
import heapq
import math
from bisect import bisect_left
from collections import Counter

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs


class BaseIndex:
    """
    Base class for inverted index construction and retrieval.

    Subclasses (BSBIIndex, SPIMIIndex) implement the indexing pipeline:
      parse_block, invert/write, merge, index

    This class provides all shared functionality:
      __init__, save, load, and all retrieval/scoring methods.
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index", tmp_dir="tmp", dict_type="idmap"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.dict_type = dict_type
        self.intermediate_indices = []
        self.avdl = None
        self.wand_ub = None

    def save(self):
        """Save doc_id_map and term_id_map to the output directory via pickle.
        If dict_type is 'fst', converts term_id_map to an FST-based
        dictionary before saving (compresses via prefix+suffix sharing).
        """
        term_map = self.term_id_map
        if self.dict_type == 'fst':
            from fst import FSTIdMap
            term_map = FSTIdMap.from_id_map(self.term_id_map)

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(term_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Load doc_id_map, term_id_map, and avdl from the output directory"""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        avdl_path = os.path.join(self.output_dir, 'avdl.pkl')
        if os.path.exists(avdl_path):
            with open(avdl_path, 'rb') as f:
                self.avdl = pickle.load(f)
        else:
            with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as reader:
                self.avdl = sum(reader.doc_length.values()) / len(reader.doc_length)
            with open(avdl_path, 'wb') as f:
                pickle.dump(self.avdl, f)

        wand_ub_path = os.path.join(self.output_dir, 'wand_upper_bounds.pkl')
        if os.path.exists(wand_ub_path):
            with open(wand_ub_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and data and isinstance(next(iter(data)), int):
                self.wand_ub = data
            else:
                self.wand_ub = None

    def _ensure_loaded(self):
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

    def index(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Retrieval methods                                                  #
    # ------------------------------------------------------------------ #

    def retrieve_tfidf(self, query, k=10):
        """
        TaaT TF-IDF retrieval.

        w(t, D) = (1 + log tf(t, D))   if tf > 0, else 0
        w(t, Q) = IDF = log(N / df(t))
        Score   = Σ w(t, Q) * w(t, D)
        """
        self._ensure_loaded()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Okapi BM25.
        RSV = Σ log(N/df_t) · (k1+1)·tf / (k1·((1-b)+b·dl/avdl) + tf)
        """
        self._ensure_loaded()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avdl = self.avdl

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)
                    idf = math.log(N / df)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        dl = merged_index.doc_length[doc_id]
                        tf_norm = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += idf * tf_norm

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25_alt2(self, query, k=10, k1=1.2, b=0.75):
        """
        BM25 Alt 2 — alternative IDF: log((N-df+0.5)/(df+0.5)).
        """
        self._ensure_loaded()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avdl = self.avdl

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)
                    idf = math.log((N - df + 0.5) / (df + 0.5))
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        dl = merged_index.doc_length[doc_id]
                        tf_norm = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += idf * tf_norm

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25_alt3(self, query, k=10, k1=1.2, b=0.75, k2=100):
        """
        BM25 Alt 3 — adds query term frequency scaling:
        · (k2+1)·tf(t,Q) / (k2+tf(t,Q))
        """
        self._ensure_loaded()

        terms = [self.term_id_map[word] for word in query.split()]
        query_tf = Counter(terms)

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avdl = self.avdl

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)
                    idf = math.log(N / df)
                    qtf = query_tf[term]
                    qtf_scale = ((k2 + 1) * qtf) / (k2 + qtf)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        dl = merged_index.doc_length[doc_id]
                        tf_norm = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += idf * tf_norm * qtf_scale

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def precompute_wand_upper_bounds(self, k1=1.2, b=0.75):
        """Precompute max BM25 score each term can contribute to any doc."""
        self._ensure_loaded()

        ub = {}
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as reader:
            N = len(reader.doc_length)
            avdl = self.avdl
            for term_id in reader.postings_dict:
                df = reader.postings_dict[term_id][1]
                idf = math.log(N / df)
                postings, tf_list = reader.get_postings_list(term_id)
                best = 0.0
                for i in range(len(postings)):
                    tf = tf_list[i]
                    dl = reader.doc_length[postings[i]]
                    s = idf * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
                    if s > best:
                        best = s
                ub[term_id] = best

        with open(os.path.join(self.output_dir, 'wand_upper_bounds.pkl'), 'wb') as f:
            pickle.dump(ub, f)
        self.wand_ub = ub

    def retrieve_wand_bm25(self, query, k=10, k1=1.2, b=0.75):
        """WAND Top-K retrieval using BM25 scoring."""
        self._ensure_loaded()
        if self.wand_ub is None:
            self.precompute_wand_upper_bounds()

        query_terms = [self.term_id_map[word] for word in query.split()]

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avdl = self.avdl

            # init(queryTerms): load postings, set each pointer to first item
            # Each entry: [postings_list, tf_list, idf (α_t), UB_t, ptr]
            terms = []
            for t in query_terms:
                if t not in merged_index.postings_dict or t not in self.wand_ub:
                    continue
                posting, tf_list = merged_index.get_postings_list(t)
                if not posting:
                    continue
                df = merged_index.postings_dict[t][1]
                terms.append([posting, tf_list, math.log(N / df), self.wand_ub[t], 0])

            if not terms:
                return []

            topk = []   # min-heap
            theta = 0.0 # θ — minimum score in current Top-K

            while terms:
                # sort(terms, posting) — sort by current DID
                terms.sort(key=lambda t: t[0][t[4]])

                # findPivotTerm(terms, θ) — first pTerm where accumulated UB >= θ
                cumul_ub = 0.0
                pTerm = -1
                for i in range(len(terms)):
                    cumul_ub += terms[i][3]
                    if cumul_ub > theta:
                        pTerm = i
                        break

                if pTerm == -1:
                    break  # no more candidates can beat θ

                # pivot ← posting[pTerm].DID
                pivot = terms[pTerm][0][terms[pTerm][4]]

                # if posting[0].DID == pivot → all preceding terms point at pivot
                if terms[0][0][terms[0][4]] == pivot:
                    # Full evaluation: compute exact BM25 score for pivot doc
                    dl = merged_index.doc_length.get(pivot, 0)
                    score = 0.0
                    for t in terms:
                        if t[0][t[4]] != pivot:
                            break  # sorted → rest are at later docs
                        tf = t[1][t[4]]
                        score += t[2] * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
                        t[4] += 1  # advance past pivot

                    # Update Top-K heap and θ
                    if score > theta:
                        heapq.heappush(topk, (score, pivot))
                        if len(topk) > k:
                            heapq.heappop(topk)
                        if len(topk) == k:
                            theta = topk[0][0]
                else:
                    # Advance preceding terms to pivot (iterator.next(pivot))
                    for t in terms[:pTerm]:
                        if t[0][t[4]] < pivot:
                            t[4] = bisect_left(t[0], pivot, t[4])

                # Remove exhausted posting lists
                terms = [t for t in terms if t[4] < len(t[0])]

            return [(s, self.doc_id_map[d]) for s, d in sorted(topk, reverse=True)]
