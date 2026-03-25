class IdMap:
    """
    Recall from the lecture that, in practice, a document and
    a term will be represented as an integer. Therefore,
    we need to maintain a mapping between a string term (or
    document) to its corresponding integer, and vice versa. This IdMap class
    will handle that.
    """

    def __init__(self):
        """
        Mapping from string (term or document name) to id is stored in
        a Python dictionary; quite efficient. The reverse mapping is stored in
        a Python list.

        example:
            str_to_id["hello"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "hello"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Return the number of terms (or documents) stored in IdMap."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Return the string associated with index i."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Return the integer id i corresponding to a string s.
        If s is not in IdMap, assign a new integer id and return
        that new integer id.
        """
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str) - 1
        return self.str_to_id[s]

    def __getitem__(self, key):
        """
        __getitem__(...) is a special method in Python that allows a
        collection class (like this IdMap) to have element access or
        modification mechanisms using the [..] syntax like lists and dictionaries in Python.

        Please search for this information on your favorite Web search engine. I found
        the following link:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        If key is an integer, use __get_str;
        if key is a string, use __get_id
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Merge two lists of tuples (doc id, tf) and return the merged
    result (TF needs to be accumulated for all tuples
    with the same doc id), following these rules:

    example: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
             posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

             return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                    = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int)]
        Two sorted lists of tuples to be merged.

    Returns
    -------
    List[(Comparable, int)]
        The sorted merged result
    """
    i, j = 0, 0
    merge = []
    while (i < len(posts_tfs1)) and (j < len(posts_tfs2)):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            freq = posts_tfs1[i][1] + posts_tfs2[j][1]
            merge.append((posts_tfs1[i][0], freq))
            i += 1
            j += 1
        elif posts_tfs1[i][0] < posts_tfs2[j][0]:
            merge.append(posts_tfs1[i])
            i += 1
        else:
            merge.append(posts_tfs2[j])
            j += 1
    while i < len(posts_tfs1):
        merge.append(posts_tfs1[i])
        i += 1
    while j < len(posts_tfs2):
        merge.append(posts_tfs2[j])
        j += 1
    return merge

def test(output, expected):
    """ simple function for testing """
    return "PASSED" if output == expected else "FAILED"

if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id incorrect"
    assert term_id_map[1] == "semua", "term_id incorrect"
    assert term_id_map[0] == "halo", "term_id incorrect"
    assert term_id_map["selamat"] == 2, "term_id incorrect"
    assert term_id_map["pagi"] == 3, "term_id incorrect"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id incorrect"

    assert sorted_merge_posts_and_tfs([(1, 34), (3, 2), (4, 23)], \
                                      [(1, 11), (2, 4), (4, 3 ), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "sorted_merge_posts_and_tfs incorrect"
