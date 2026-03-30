import os
import contextlib
import heapq

from index import InvertedIndexReader, InvertedIndexWriter
from util import sorted_merge_posts_and_tfs
from compression import VBEPostings, EliasGammaPostings
from retrieval import BaseIndex
from tqdm import tqdm


class BSBIIndex(BaseIndex):
    """
    Blocked Sort-Based Indexing (BSBI).

    Maintains a global term-to-termID mapping across all blocks.
    Each block produces sorted (termID, docID) pairs which are
    inverted and written as intermediate indices, then merged.

    Inherits all retrieval methods from BaseIndex.
    """

    def parse_block(self, block_dir_relative):
        """
        Parse text files into a sequence of <termID, docID> pairs.

        Parameters
        ----------
        block_dir_relative : str
            Relative path to a sub-directory within the collection folder.

        Returns
        -------
        List[Tuple[Int, Int]]
            All (termID, docID) pairs extracted from the block.
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Invert td_pairs and write to the index using a hashtable (SPIMI-style
        storage within BSBI's global-ID framework).

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index on disk associated with a block
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Merge all intermediate inverted indices via external merge sort.

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            Intermediate index readers (one per block).
        merged_index: InvertedIndexWriter
            The final merged index writer.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)
        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)),
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def index(self):
        """
        BSBI index construction.

        For each block: parse_block -> invert_write -> intermediate index.
        Then merge all intermediate indices into the final index.
        """
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.tmp_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.tmp_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

            self.avdl = sum(merged_index.doc_length.values()) / len(merged_index.doc_length)
            with open(os.path.join(self.output_dir, 'avdl.pkl'), 'wb') as f:
                import pickle
                pickle.dump(self.avdl, f)

        self.precompute_wand_upper_bounds()


if __name__ == "__main__":

    for dict_type in ['idmap', 'fst']:
        method_dir = 'bsbi_fst' if dict_type == 'fst' else 'bsbi'
        for postings_encoding in [VBEPostings, EliasGammaPostings]:
            BSBI_instance = BSBIIndex(data_dir='collection',
                                      postings_encoding=postings_encoding,
                                      output_dir=os.path.join('index', method_dir, postings_encoding.name),
                                      tmp_dir=os.path.join('tmp', method_dir, postings_encoding.name),
                                      dict_type=dict_type)
            BSBI_instance.index()
