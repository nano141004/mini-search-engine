import array
import math

class StandardPostings:
    """
    Class with static methods, to convert the representation of a postings list
    from a List of integers into a sequence of bytes.
    We use Python's array library.

    ASSUMPTION: the postings_list for a term fits in memory!

    Please study:
        https://docs.python.org/3/library/array.html
    """

    name = "standard"

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

    name = "vbe"

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

class EliasGammaPostings:
    """
    Bit-level compression using Elias Gamma Coding.

    Like VBEPostings, the postings list is first converted to a gap-based
    representation before encoding. Each gap value is then encoded using
    Elias Gamma coding, which is a universal code for positive integers.

    Since Elias Gamma can only encode positive integers (>= 1), all values
    are shifted by +1 before encoding and shifted back by -1 after decoding
    to correctly handle zero values (e.g., docID 0).

    Encoding algorithm for a positive integer X:
        1. Find N = floor(log2(X)), i.e., the largest N with 2^N <= X
        2. Write N zeros followed by a 1 (unary code for N)
        3. Write (X - 2^N) in N-bit binary (K)

    Decoding algorithm:
        1. Read leading zeros from left and count them; the count is N
        2. Skip the '1' delimiter
        3. Read the next N bits and convert to decimal; this is K
        4. X = 2^N + K

    Example:
        19 = 2^4 + 3 → N=4 → unary "00001" + binary "0011" → "000010011"

    The resulting bit stream is packed into bytes (padded to byte boundary
    with trailing zeros). No length prefix is needed because Elias Gamma is
    self-delimiting: trailing zero-padding is detected during decoding since
    it never leads to a '1' delimiter bit.

    ASSUMPTION: the postings_list for a term fits in memory!

    """

    name = "elias_gamma"

    @staticmethod
    def eg_encode_number(number):
        """
        Encode a single positive integer (>= 1) using Elias Gamma coding.

        Parameters
        ----------
        number : int
            A positive integer >= 1 to encode

        Returns
        -------
        List[int]
            List of bits (0s and 1s) representing the Elias Gamma code
        """
        if number < 1:
            raise ValueError("Elias Gamma coding requires positive integers (>= 1)")

        # N = floor(log2(number))
        N = number.bit_length() - 1

        # Unary code for N: N zeros followed by a 1
        bits = [0] * N + [1]

        # Binary representation of (number - 2^N) in exactly N bits
        remainder = number - (1 << N)
        for i in range(N - 1, -1, -1):
            bits.append((remainder >> i) & 1)

        return bits

    @staticmethod
    def eg_encode(list_of_numbers):
        """
        Encode a list of non-negative integers using Elias Gamma coding.
        Each number is shifted by +1 before encoding (to handle zeros,
        since Elias Gamma requires values >= 1).

        No length prefix is stored because Elias Gamma is self-delimiting:
        the decoder can distinguish real codes from zero-padding since
        padding zeros will never be followed by a '1' delimiter bit.

        Parameters
        ----------
        list_of_numbers : List[int]
            List of non-negative integers to encode

        Returns
        -------
        bytes
            Encoded bytestream of packed bits (padded to byte boundary)
        """
        all_bits = []
        for number in list_of_numbers:
            all_bits.extend(EliasGammaPostings.eg_encode_number(number + 1))

        # Pad bits to a multiple of 8
        padding = (8 - len(all_bits) % 8) % 8
        all_bits.extend([0] * padding)

        # Pack bits into bytes (MSB first)
        result = bytearray()
        for i in range(0, len(all_bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | all_bits[i + j]
            result.append(byte)

        return bytes(result)

    @staticmethod
    def eg_decode(encoded_bytestream):
        """
        Decode a bytestream that was previously encoded with Elias Gamma coding.
        Each decoded number is shifted back by -1 to reverse the +1 shift
        applied during encoding.

        Exploits the self-delimiting property of Elias Gamma: trailing
        zero-padding bits are detected because they never lead to a '1'
        delimiter, so decoding stops naturally without needing a count prefix.

        Parameters
        ----------
        encoded_bytestream : bytes
            Bytestream of packed Elias Gamma coded bits

        Returns
        -------
        List[int]
            List of decoded non-negative integers
        """
        # Convert bytes to a list of bits (MSB first)
        bits = []
        for byte in encoded_bytestream:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        numbers = []
        pos = 0
        total_bits = len(bits)

        while pos < total_bits:
            # Count leading zeros to find N
            N = 0
            while pos < total_bits and bits[pos] == 0:
                N += 1
                pos += 1

            # If we reached the end without finding a '1' delimiter,
            # the remaining bits were zero-padding — stop decoding
            if pos >= total_bits:
                break

            pos += 1  # skip the '1' delimiter

            # Read N bits for the remainder K
            K = 0
            for _ in range(N):
                K = (K << 1) | bits[pos]
                pos += 1

            # X = 2^N + K, then shift back by -1
            numbers.append((1 << N) + K - 1)

        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list into a stream of bytes using Elias Gamma coding.
        The postings list is first converted to a gap-based representation,
        then each gap value is encoded with Elias Gamma.

        Parameters
        ----------
        postings_list : List[int]
            List of docIDs (postings), sorted in ascending order

        Returns
        -------
        bytes
            Bytestream representing the encoded postings list
        """
        # Convert to gap-based representation
        gap_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_list.append(postings_list[i] - postings_list[i - 1])
        return EliasGammaPostings.eg_encode(gap_list)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list from a stream of bytes encoded with Elias Gamma.
        The decoded gap-based list is converted back to absolute docIDs
        by computing cumulative sums.

        Parameters
        ----------
        encoded_postings_list : bytes
            Bytestream representing the encoded postings list

        Returns
        -------
        List[int]
            List of docIDs resulting from decoding
        """
        gap_list = EliasGammaPostings.eg_decode(encoded_postings_list)
        # Convert gaps back to absolute docIDs
        result = [gap_list[0]]
        for i in range(1, len(gap_list)):
            result.append(result[-1] + gap_list[i])
        return result

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies into a stream of bytes using
        Elias Gamma coding. No gap conversion is applied for TF values.

        Parameters
        ----------
        tf_list : List[int]
            List of term frequencies (each >= 1)

        Returns
        -------
        bytes
            Bytestream representing the encoded TF list
        """
        return EliasGammaPostings.eg_encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode list of term frequencies from a stream of bytes
        encoded with Elias Gamma coding.

        Parameters
        ----------
        encoded_tf_list : bytes
            Bytestream representing the encoded TF list

        Returns
        -------
        List[int]
            List of term frequencies resulting from decoding
        """
        return EliasGammaPostings.eg_decode(encoded_tf_list)


if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
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
        assert decoded_posting_list == postings_list, \
            "decoding result does not match original postings"
        assert decoded_tf_list == tf_list, \
            "decoding result does not match original TF list"
        print()
