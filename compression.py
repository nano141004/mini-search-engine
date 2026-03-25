import array

class StandardPostings:
    """
    Class with static methods, to convert the representation of a postings list
    from a List of integers into a sequence of bytes.
    We use Python's array library.

    ASSUMPTION: the postings_list for a term fits in memory!

    Please study:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray representing the sequence of integers in postings_list
        """
        # For standard encoding, use L for unsigned long, since docIDs
        # will not be negative. And we assume the largest docID
        # can be stored in a 4-byte unsigned representation.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list from a stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray representing the encoded postings list as output
            from the static method encode above.

        Returns
        -------
        List[int]
            list of docIDs resulting from decoding encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies into a stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray representing the raw TF values of term occurrences in each
            document in the list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies from a stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray representing the encoded term frequencies list as output
            from the static method encode_tf above.

        Returns
        -------
        List[int]
            List of term frequencies resulting from decoding encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """
    Unlike StandardPostings, where for a postings list,
    the original sequence of integers from the postings list
    is stored on disk as-is.

    In VBEPostings, what is stored is the gap, except for
    the first posting. After that, it is encoded with the Variable-Byte
    Encoding algorithm into a bytestream.

    Example:
    postings list [34, 67, 89, 454] will first be converted to gap-based,
    i.e., [34, 33, 22, 365]. Then it is encoded with the Variable-Byte
    Encoding compression algorithm, and then converted to a bytestream.

    ASSUMPTION: the postings_list for a term fits in memory!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        See our textbook!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend to the front
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # the leading bit of the last byte is set to 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """
        Perform encoding (with compression) on a
        list of numbers, using Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a stream of bytes (using Variable-Byte
        Encoding). DON'T FORGET to convert to a gap-based list first, before
        encoding and converting to a bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray representing the sequence of integers in postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies into a stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray representing the raw TF values of term occurrences in each
            document in the list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decode a bytestream that was previously encoded with
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list from a stream of bytes. DON'T FORGET that
        the bytestream decoded from encoded_postings_list is still a
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray representing the encoded postings list as output
            from the static method encode above.

        Returns
        -------
        List[int]
            list of docIDs resulting from decoding encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies from a stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray representing the encoded term frequencies list as output
            from the static method encode_tf above.

        Returns
        -------
        List[int]
            List of term frequencies resulting from decoding encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("bytes from encoding postings: ", encoded_postings_list)
        print("size of encoded postings   : ", len(encoded_postings_list), "bytes")
        print("bytes from encoding TF list: ", encoded_tf_list)
        print("size of encoded TF list    : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("decoding result (postings): ", decoded_posting_list)
        print("decoding result (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "decoding result does not match original postings"
        assert decoded_tf_list == tf_list, "decoding result does not match original TF list"
        print()
