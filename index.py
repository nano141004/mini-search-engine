import pickle
import os

class InvertedIndex:
    """
    Class that implements how to efficiently scan or read an
    Inverted Index stored in a file; and also provides a mechanism
    to write an Inverted Index to a file (storage) during indexing.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list)

        postings_dict is the "Dictionary" concept that is part of the
        Inverted Index. This postings_dict is assumed to fit entirely
        in memory.

        As the name suggests, the "Dictionary" is implemented as a Python dictionary
        that maps a term ID (integer) to a 4-tuple:
           1. start_position_in_index_file : (in bytes) the position where
              the corresponding postings are located in the file (storage). We can
              use the "seek" operation to reach it.
           2. number_of_postings_in_list : how many docIDs are in the
              postings (Document Frequency)
           3. length_in_bytes_of_postings_list : the length of the postings list
              in bytes.
           4. length_in_bytes_of_tf_list : the length of the list of term frequencies
              for the associated postings list in bytes

    terms: List[int]
        List of term IDs, to remember the order of terms inserted into
        the Inverted Index.

    """
    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Name used to store files containing the index
        postings_encoding : See compression.py, candidates are StandardPostings,
                        GapBasedPostings, etc.
        directory (str): directory where the index file is located
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # To keep track of the order of terms inserted into the index
        self.doc_length = {}    # key: doc ID (int), value: document length (number of tokens)
                                # This will later be useful for normalizing the Score by document
                                # length when computing scores with TF-IDF or BM25

    def __enter__(self):
        """
        Load all metadata when entering the context.
        Metadata:
            1. Dictionary ---> postings_dict
            2. Iterator for the List containing the order of terms entered into
                the index during construction. ---> term_iter
            3. doc_length, a Python dictionary with key = doc id, and
                value = the number of tokens in that document (document length).
                Useful for length normalization when using TF-IDF or BM25
                scoring regime; useful for knowing the value of N when computing IDF,
                where N is the number of documents in the collection

        Metadata is saved to file using the "pickle" library.

        You also need to understand the special method __enter__(..) in Python and
        the concept of Context Manager in Python. Please study the following link:

        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        # Open the index file
        self.index_file = open(self.index_file_path, 'rb+')

        # Load postings dict and terms iterator from the metadata file
        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms, self.doc_length = pickle.load(f)
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Close index_file and save postings_dict and terms when exiting context"""
        # Close the index file
        self.index_file.close()

        # Save metadata (postings dict and terms) to the metadata file using pickle
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length], f)


class InvertedIndexReader(InvertedIndex):
    """
    Class that implements how to efficiently scan or read an
    Inverted Index stored in a file.
    """
    def __iter__(self):
        return self

    def reset(self):
        """
        Reset the file pointer to the beginning, and reset the term
        iterator pointer to the beginning
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()  # reset term iterator

    def __next__(self):
        """
        The InvertedIndexReader class is also iterable (has an iterator).
        Please study:
        https://stackoverflow.com/questions/19151/how-to-build-a-basic-iterator

        When an instance of this InvertedIndexReader class is used
        as an iterator in a loop scheme, the special method __next__(...)
        is responsible for returning the next (term, postings_list, tf_list) tuple
        in the inverted index.

        ATTENTION! This method must return only a small portion of data from
        the large index file. Why only a small portion? So that it fits
        in memory for processing. DO NOT LOAD THE ENTIRE INDEX INTO MEMORY!
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[curr_term]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Return a postings list (list of docIDs) along with the associated
        list of term frequencies for a term (stored as a
        tuple (postings_list, tf_list)).

        ATTENTION! This method must NOT iterate over the entire index
        from beginning to end. This method must jump directly to the specific
        byte position in the file (index file) where the postings list (and also
        the list of TFs) for the term is stored.
        """
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[term]
        self.index_file.seek(pos)
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """
    Class that implements how to efficiently write an
    Inverted Index stored in a file.
    """
    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """
        Append a term, postings_list, and associated TF list
        to the end of the index file.

        This method does 4 things:
        1. Encode postings_list using self.postings_encoding (encode method),
        2. Encode tf_list using self.postings_encoding (encode_tf method),
        3. Store metadata in the form of self.terms, self.postings_dict, and self.doc_length.
           Recall that self.postings_dict maps a termID to
           a 4-tuple: - start_position_in_index_file
                       - number_of_postings_in_list
                       - length_in_bytes_of_postings_list
                       - length_in_bytes_of_tf_list
        4. Append the bytestream of the encoded postings_list and
           the encoded tf_list to the end of the index file on disk.

        Don't forget to update self.terms and self.doc_length as well!

        SEARCH ON YOUR FAVORITE SEARCH ENGINE:
        - You may want to read about Python I/O
          https://docs.python.org/3/tutorial/inputoutput.html
          This link also covers how to append information
          to the end of a file.
        - Some file object methods that may be useful such as seek(...)
          and tell()

        Parameters
        ----------
        term:
            term or termID that is the unique identifier of a term
        postings_list: List[Int]
            List of docIDs where the term appears
        tf_list: List[Int]
            List of term frequencies
        """
        self.terms.append(term) # update self.terms

        # update self.doc_length
        for i in range(len(postings_list)):
            doc_id, freq = postings_list[i], tf_list[i]
            if doc_id not in self.doc_length:
                self.doc_length[doc_id] = 0
            self.doc_length[doc_id] += freq

        self.index_file.seek(0, os.SEEK_END)
        curr_position_in_byte = self.index_file.tell()
        compressed_postings = self.postings_encoding.encode(postings_list)
        compressed_tf_list = self.postings_encoding.encode_tf(tf_list)
        self.index_file.write(compressed_postings)
        self.index_file.write(compressed_tf_list)
        self.postings_dict[term] = (curr_position_in_byte, len(postings_list), \
                                    len(compressed_postings), len(compressed_tf_list))


if __name__ == "__main__":

    from compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1,2], "terms incorrect"
        assert index.doc_length == {2:2, 3:38, 4:25, 5:56, 8:3, 10:30}, "doc_length incorrect"
        assert index.postings_dict == {1: (0, \
                                           5, \
                                           len(VBEPostings.encode([2,3,4,8,10])), \
                                           len(VBEPostings.encode_tf([2,4,2,3,30]))),
                                       2: (len(VBEPostings.encode([2,3,4,8,10])) + len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           3, \
                                           len(VBEPostings.encode([3,4,5])), \
                                           len(VBEPostings.encode_tf([34,23,56])))}, "postings dictionary incorrect"
        
        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3,4,5])))) == [3,4,5], "an error was found"
        assert VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34,23,56])))) == [34,23,56], "an error was found"
