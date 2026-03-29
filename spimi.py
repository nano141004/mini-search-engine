import os
import pickle
import heapq
import math
from bisect import bisect_left
from collections import Counter

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm

class SPIMIIndex:
    """
    Single-Pass In-Memory Indexing (SPIMI) implementation.

    Key difference from BSBI:
      - BSBI maintains a global term-to-termID mapping across all blocks,
        generates (termID, docID) pairs, and sorts them.
      - SPIMI builds a separate dictionary per block, mapping term strings
        directly to postings lists (hash table). No global term-to-termID
        mapping is needed during block processing. Only the dictionary keys
        (term strings) are sorted at the end of each block — NOT the
        termID-docID pairs like in BSBI.
      - Term IDs are assigned only during the final merge step.
      - Time complexity per block is O(T) where T = number of tokens,
        since there is no sorting of termID-docID pairs.

    Attributes
    ----------
    term_id_map(IdMap): For mapping terms to termIDs (built during merge)
    doc_id_map(IdMap): For mapping relative paths of documents to docIDs
    data_dir(str): Path to data
    output_dir(str): Path to output final index files
    tmp_dir(str): Path to output intermediate index files
    postings_encoding: See compression.py
    index_name(str): Name of the file containing the inverted index
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

    def parse_block(self, block_dir_relative):
        """
        Produce a token stream of (term_string, doc_id) pairs.

        Unlike BSBI's parse_block which returns (termID, docID) pairs
        using a global term_id_map, SPIMI keeps terms as raw strings.
        Each block will build its own local dictionary from these strings.

        The doc_id_map IS still global (shared across blocks) since we
        need consistent document IDs throughout.

        Parameters
        ----------
        block_dir_relative : str
            Relative path to a sub-directory within the collection folder.

        Returns
        -------
        List[Tuple[str, int]]
            Token stream of (term_string, doc_id) pairs.
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        token_stream = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                doc_id = self.doc_id_map[docname]
                for token in f.read().split():
                    token_stream.append((token, doc_id))
        return token_stream

    def spimi_invert(self, token_stream):
        """
        SPIMI-INVERT algorithm (textbook pseudocode).

        Builds an in-memory hash table (Python dict) mapping each term
        string directly to its postings with term frequencies. Postings
        are appended directly to the list — no need to first collect
        (termID, docID) pairs and then sort them like BSBI does.

        At the end, only the dictionary keys (terms) are sorted. This is
        much cheaper than sorting all termID-docID pairs, since the number
        of unique terms << number of termID-docID pairs.

        Parameters
        ----------
        token_stream : List[Tuple[str, int]]
            List of (term_string, doc_id) pairs from parse_block.

        Returns
        -------
        sorted_terms : List[str]
            Sorted list of unique term strings in this block.
        dictionary : Dict[str, Dict[int, int]]
            Mapping of term_string -> {doc_id: tf}.
        """
        dictionary = {}                             # 1  dictionary = NEWHASH()
        for term, doc_id in token_stream:           # 2-3  while (free memory) token <- next(stream)
            if term not in dictionary:              # 4  if term(token) not in dictionary
                dictionary[term] = {}               # 5    postings_list = ADDTODICTIONARY(dictionary, term)
            postings = dictionary[term]             # 7  postings_list = GETPOSTINGSLIST(dictionary, term)
            if doc_id not in postings:
                postings[doc_id] = 0
            postings[doc_id] += 1                   # 11 ADDTOPOSTINGSLIST(postings_list, docID(token))
        sorted_terms = sorted(dictionary.keys())    # 12 sorted_terms <- SORTTERMS(dictionary)
        return sorted_terms, dictionary             # 13 return sorted_terms, dictionary

    def write_block_to_disk(self, sorted_terms, dictionary, index_id):
        """
        WRITEBLOCKTODISK: persist one block's inverted index as a pickle file.

        Each block is stored as a list of (term_string, sorted_doc_ids, tf_list)
        tuples, pre-sorted by term string so that the merge step can use
        heapq.merge directly.

        Parameters
        ----------
        sorted_terms : List[str]
            Terms in sorted order.
        dictionary : Dict[str, Dict[int, int]]
            {term_string: {doc_id: tf}}.
        index_id : str
            Identifier for this intermediate index file.
        """
        block = []
        for term in sorted_terms:
            postings = dictionary[term]
            sorted_doc_ids = sorted(postings.keys())
            tf_list = [postings[doc_id] for doc_id in sorted_doc_ids]
            block.append((term, sorted_doc_ids, tf_list))

        path = os.path.join(self.tmp_dir, index_id + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(block, f)

    def merge(self, merged_index):
        """
        MERGEBLOCKS: merge all intermediate SPIMI blocks into a single index.

        Loads all intermediate pickle blocks and uses heapq.merge to produce
        a single sorted stream of (term_string, postings, tf_list) tuples.
        When the same term appears in multiple blocks, their postings lists
        are merged using sorted_merge_posts_and_tfs.

        During merge, global term IDs are assigned via self.term_id_map —
        this is the first (and only) time term-to-termID mapping happens.

        Parameters
        ----------
        merged_index : InvertedIndexWriter
            The final merged index writer.
        """
        blocks = []
        for index_id in self.intermediate_indices:
            path = os.path.join(self.tmp_dir, index_id + '.pkl')
            with open(path, 'rb') as f:
                blocks.append(pickle.load(f))

        merged_iter = heapq.merge(*blocks, key=lambda x: x[0])
        curr_term, postings, tf_list = next(merged_iter)
        for term, postings_, tf_list_ in merged_iter:
            if term == curr_term:
                zip_p_tf = sorted_merge_posts_and_tfs(
                    list(zip(postings, tf_list)),
                    list(zip(postings_, tf_list_))
                )
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                term_id = self.term_id_map[curr_term]
                merged_index.append(term_id, postings, tf_list)
                curr_term, postings, tf_list = term, postings_, tf_list_
        term_id = self.term_id_map[curr_term]
        merged_index.append(term_id, postings, tf_list)

    def index(self):
        """
        SPIMI index construction (follows the SPIMIINDEXCONSTRUCTION pseudocode).

        For each block (sub-directory in collection):
          1. PARSENEXTBLOCK  — produce token stream (term strings, not termIDs)
          2. SPIMI-INVERT    — build per-block dictionary via hash table
          3. WRITEBLOCKTODISK — write sorted block to disk as pickle

        Finally:
          4. MERGEBLOCKS — merge all blocks, assign global term IDs, write
             final compressed index using InvertedIndexWriter
        """
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            # PARSENEXTBLOCK
            token_stream = self.parse_block(block_dir_relative)
            # SPIMI-INVERT
            sorted_terms, dictionary = self.spimi_invert(token_stream)
            # WRITEBLOCKTODISK
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            self.write_block_to_disk(sorted_terms, dictionary, index_id)

        # MERGEBLOCKS
        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            self.merge(merged_index)

            self.avdl = sum(merged_index.doc_length.values()) / len(merged_index.doc_length)
            with open(os.path.join(self.output_dir, 'avdl.pkl'), 'wb') as f:
                pickle.dump(self.avdl, f)

        # Save term_id_map (built during merge) and doc_id_map
        self.save()

        # Pre-compute WAND upper bounds (needs a reader, so after writer closes)
        self.precompute_wand_upper_bounds()

    # ------------------------------------------------------------------ #
    #  Retrieval methods (identical to BSBIIndex — shared final index     #
    #  format means retrieval logic is the same)                          #
    # ------------------------------------------------------------------ #

    def retrieve_tfidf(self, query, k=10):
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

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
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

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
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

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
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

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
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

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
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()
        if self.wand_ub is None:
            self.precompute_wand_upper_bounds()

        query_terms = [self.term_id_map[word] for word in query.split()]

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avdl = self.avdl

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

            topk = []
            theta = 0.0

            while terms:
                terms.sort(key=lambda t: t[0][t[4]])

                cumul_ub = 0.0
                pTerm = -1
                for i in range(len(terms)):
                    cumul_ub += terms[i][3]
                    if cumul_ub > theta:
                        pTerm = i
                        break

                if pTerm == -1:
                    break

                pivot = terms[pTerm][0][terms[pTerm][4]]

                if terms[0][0][terms[0][4]] == pivot:
                    dl = merged_index.doc_length.get(pivot, 0)
                    score = 0.0
                    for t in terms:
                        if t[0][t[4]] != pivot:
                            break
                        tf = t[1][t[4]]
                        score += t[2] * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * dl / avdl) + tf)
                        t[4] += 1

                    if score > theta:
                        heapq.heappush(topk, (score, pivot))
                        if len(topk) > k:
                            heapq.heappop(topk)
                        if len(topk) == k:
                            theta = topk[0][0]
                else:
                    for t in terms[:pTerm]:
                        if t[0][t[4]] < pivot:
                            t[4] = bisect_left(t[0], pivot, t[4])

                terms = [t for t in terms if t[4] < len(t[0])]

            return [(s, self.doc_id_map[d]) for s, d in sorted(topk, reverse=True)]


if __name__ == "__main__":

    for dict_type in ['idmap', 'fst']:
        method_dir = 'spimi_fst' if dict_type == 'fst' else 'spimi'
        for postings_encoding in [VBEPostings, EliasGammaPostings]:
            SPIMI_instance = SPIMIIndex(data_dir='collection',
                                        postings_encoding=postings_encoding,
                                        output_dir=os.path.join('index', method_dir, postings_encoding.name),
                                        tmp_dir=os.path.join('tmp', method_dir, postings_encoding.name),
                                        dict_type=dict_type)
            SPIMI_instance.index()  # start indexing!
