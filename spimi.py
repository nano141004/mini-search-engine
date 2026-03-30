import os
import pickle
import heapq

from index import InvertedIndexWriter
from util import sorted_merge_posts_and_tfs
from compression import VBEPostings, EliasGammaPostings
from retrieval import BaseIndex
from tqdm import tqdm


class SPIMIIndex(BaseIndex):
    """
    Single-Pass In-Memory Indexing (SPIMI).

    Key difference from BSBI:
      - No global term-to-termID mapping during block processing.
      - Each block builds its own hash table mapping term strings
        directly to postings lists.
      - Only dictionary keys (terms) are sorted per block — NOT
        all termID-docID pairs.
      - Term IDs are assigned only during the final merge step.
      - Time complexity per block is O(T) where T = number of tokens.

    Inherits all retrieval methods from BaseIndex.
    """

    def parse_block(self, block_dir_relative):
        """
        Produce a token stream of (term_string, doc_id) pairs.

        Unlike BSBI which returns (termID, docID) using a global term_id_map,
        SPIMI keeps terms as raw strings. Each block builds its own local
        dictionary from these strings.

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
        SPIMI-INVERT: build in-memory hash table from token stream.

        Returns
        -------
        sorted_terms : List[str]
            Sorted unique term strings in this block.
        dictionary : Dict[str, Dict[int, int]]
            term_string -> {doc_id: tf}.
        """
        dictionary = {}
        for term, doc_id in token_stream:
            if term not in dictionary:
                dictionary[term] = {}
            postings = dictionary[term]
            if doc_id not in postings:
                postings[doc_id] = 0
            postings[doc_id] += 1
        sorted_terms = sorted(dictionary.keys())
        return sorted_terms, dictionary

    def write_block_to_disk(self, sorted_terms, dictionary, index_id):
        """
        WRITEBLOCKTODISK: persist one block as a pickle of
        [(term_string, sorted_doc_ids, tf_list), ...].
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
        MERGEBLOCKS: merge all intermediate pickle blocks, assigning
        global term IDs via self.term_id_map during the merge.

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
        SPIMI index construction.

        Per block: parse_block -> spimi_invert -> write_block_to_disk.
        Then merge all blocks, assign global term IDs, write final index.
        """
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            token_stream = self.parse_block(block_dir_relative)
            sorted_terms, dictionary = self.spimi_invert(token_stream)
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            self.write_block_to_disk(sorted_terms, dictionary, index_id)

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            self.merge(merged_index)

            self.avdl = sum(merged_index.doc_length.values()) / len(merged_index.doc_length)
            with open(os.path.join(self.output_dir, 'avdl.pkl'), 'wb') as f:
                pickle.dump(self.avdl, f)

        self.save()
        self.precompute_wand_upper_bounds()


if __name__ == "__main__":

    for dict_type in ['idmap', 'fst']:
        method_dir = 'spimi_fst' if dict_type == 'fst' else 'spimi'
        for postings_encoding in [VBEPostings, EliasGammaPostings]:
            SPIMI_instance = SPIMIIndex(data_dir='collection',
                                        postings_encoding=postings_encoding,
                                        output_dir=os.path.join('index', method_dir, postings_encoding.name),
                                        tmp_dir=os.path.join('tmp', method_dir, postings_encoding.name),
                                        dict_type=dict_type)
            SPIMI_instance.index()
