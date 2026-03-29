"""
Latent Semantic Indexing (LSI) with FAISS vector search.

Pipeline:
  Index time:
    1. Read TF data from existing inverted index -> sparse TF-IDF term-doc matrix C
    2. Truncated SVD: C ~ U_k S_k V_k^T  (rank-k approximation)
    3. Doc vectors = (S_k V_k^T)^T = V_k S_k  -> L2-normalize -> FAISS HNSW index

  Query time:
    1. query -> tokenize -> TF-IDF vector q
    2. Fold into LSI space: q_lsi = U_k^T @ q  -> normalize
    3. FAISS nearest neighbor search -> top-k documents

Math (from lecture slides):
  C = U S V^T                          (SVD of term-doc matrix, terms x docs)
  doc embedding   = columns of S_k V_k^T   (or rows of V_k S_k)
  query folding   = U_k^T @ q              (project query into LSI space)
  similarity      = cosine(q_lsi, doc_j)   (via FAISS on normalized vectors)

Usage:
  python lsi.py
"""

import os
import pickle
import math
import numpy as np
import faiss
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm

from bsbi import BSBIIndex
from index import InvertedIndexReader
from compression import VBEPostings


class LSIIndex:
    """LSI + FAISS HNSW retrieval. Builds on an existing inverted index."""

    def __init__(self, n_components=100, output_dir='index/lsi',
                 hnsw_M=32, ef_construction=200, ef_search=128):
        """
        Parameters
        ----------
        n_components : int
            Number of SVD components (latent dimensions). Typical: 100-300.
        output_dir : str
            Directory to store FAISS index and LSI metadata.
        hnsw_M : int
            HNSW connections per node. Higher = better recall, more memory.
        ef_construction : int
            HNSW construction-time search depth. Higher = slower build, better graph.
        ef_search : int
            HNSW query-time search depth. Higher = slower query, better recall.
        """
        self.n_components = n_components
        self.output_dir = output_dir
        self.hnsw_M = hnsw_M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Filled during build / load
        self.U_k = None          # (n_terms, k) left singular vectors
        self.s_k = None          # (k,) singular values
        self.idf = None          # (n_terms,) IDF values for query weighting
        self.faiss_index = None  # FAISS HNSW index
        self.doc_id_map = None   # IdMap: doc_id <-> doc path
        self.term_id_map = None  # IdMap: term_id <-> term string
        self.n_terms = None      # vocabulary size
        self.N = None            # number of documents
        self.k = None            # actual number of components used

    def build(self, source):
        """
        Build LSI + FAISS index from an existing BSBIIndex or SPIMIIndex.

        Parameters
        ----------
        source : BSBIIndex or SPIMIIndex
            Must have been indexed already (index files on disk).
        """
        source.load()
        self.term_id_map = source.term_id_map
        self.doc_id_map = source.doc_id_map

        # --- Step 1: Build sparse TF-IDF term-document matrix ---
        print("  Building TF-IDF term-document matrix from inverted index...")
        with InvertedIndexReader(source.index_name, source.postings_encoding,
                                 directory=source.output_dir) as reader:
            N = len(reader.doc_length)
            self.N = N
            n_terms = max(reader.postings_dict.keys()) + 1
            self.n_terms = n_terms

            rows, cols, vals = [], [], []
            idf = np.zeros(n_terms)

            for term_id in tqdm(reader.postings_dict, desc="  Reading postings"):
                df = reader.postings_dict[term_id][1]
                term_idf = math.log(N / df)
                idf[term_id] = term_idf
                postings, tf_list = reader.get_postings_list(term_id)
                for i in range(len(postings)):
                    rows.append(term_id)
                    cols.append(postings[i])
                    vals.append(math.log(1 + tf_list[i]) * term_idf)

            self.idf = idf

        C = csr_matrix((vals, (rows, cols)), shape=(n_terms, N))
        print(f"  Matrix: {n_terms} terms x {N} docs, {C.nnz} non-zeros")

        # --- Step 2: Truncated SVD ---
        k = min(self.n_components, min(C.shape) - 1)
        self.k = k
        print(f"  Computing truncated SVD (k={k})...")
        U, s, Vt = svds(C, k=k)

        # svds returns ascending order -> sort descending by singular value
        idx = np.argsort(-s)
        U, s, Vt = U[:, idx], s[idx], Vt[idx, :]
        self.U_k = U   # (n_terms, k)
        self.s_k = s    # (k,)

        print(f"  Top-5 singular values: {s[:5]}")

        # --- Step 3: Document vectors = V_k @ S_k, then L2-normalize ---
        # doc_vectors[j] = j-th row of V_k @ diag(S_k) = j-th col of S_k @ V_k^T
        doc_vectors = (Vt.T * s).astype(np.float32)  # (N, k) broadcast multiply
        norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_vectors /= norms

        # --- Step 4: Build FAISS HNSW index ---
        print(f"  Building FAISS HNSW index (d={k}, M={self.hnsw_M})...")
        self.faiss_index = faiss.IndexHNSWFlat(k, self.hnsw_M)
        self.faiss_index.hnsw.efConstruction = self.ef_construction
        self.faiss_index.hnsw.efSearch = self.ef_search
        self.faiss_index.add(doc_vectors)

        self.save()
        print(f"  LSI index saved to {self.output_dir}/")

    def save(self):
        """Save FAISS index and LSI metadata to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        faiss.write_index(self.faiss_index,
                          os.path.join(self.output_dir, 'lsi.faiss'))
        with open(os.path.join(self.output_dir, 'lsi_meta.pkl'), 'wb') as f:
            pickle.dump({
                'U_k': self.U_k, 's_k': self.s_k, 'idf': self.idf,
                'doc_id_map': self.doc_id_map, 'term_id_map': self.term_id_map,
                'n_terms': self.n_terms, 'N': self.N, 'k': self.k,
            }, f)

    def load(self):
        """Load FAISS index and LSI metadata from disk."""
        self.faiss_index = faiss.read_index(
            os.path.join(self.output_dir, 'lsi.faiss'))
        self.faiss_index.hnsw.efSearch = self.ef_search
        with open(os.path.join(self.output_dir, 'lsi_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        self.U_k = meta['U_k']
        self.s_k = meta['s_k']
        self.idf = meta['idf']
        self.doc_id_map = meta['doc_id_map']
        self.term_id_map = meta['term_id_map']
        self.n_terms = meta['n_terms']
        self.N = meta['N']
        self.k = meta['k']

    def retrieve(self, query, k=10):
        """
        Retrieve top-k documents using LSI + FAISS HNSW.

        Query folding-in (from lecture slides):
          q_lsi = U_k^T @ q_tfidf
        Then cosine similarity via FAISS (L2 on normalized vectors).

        Parameters
        ----------
        query : str
            Query tokens separated by spaces.
        k : int
            Number of top documents to return.

        Returns
        -------
        List[(float, str)]
            (cosine_similarity, doc_path) sorted descending.
        """
        if self.faiss_index is None:
            self.load()

        # Build TF-IDF query vector in term space
        q_vec = np.zeros(self.n_terms)
        for token in query.split():
            term_id = self.term_id_map[token]
            if term_id < self.n_terms:
                q_vec[term_id] += 1

        # TF-IDF weight: log(1+tf) * idf
        nz = q_vec > 0
        q_vec[nz] = np.log(1 + q_vec[nz]) * self.idf[nz]

        # Fold into LSI space: q_lsi = U_k^T @ q
        q_lsi = self.U_k.T @ q_vec  # (k,)

        # L2-normalize for cosine similarity
        norm = np.linalg.norm(q_lsi)
        if norm > 0:
            q_lsi /= norm

        q_lsi = q_lsi.astype(np.float32).reshape(1, -1)

        # FAISS search (L2 on normalized vectors: d^2 = 2 - 2*cos)
        D, I = self.faiss_index.search(q_lsi, k)

        results = []
        for i in range(len(I[0])):
            doc_id = int(I[0][i])
            if doc_id < 0:  # FAISS returns -1 for missing results
                continue
            cosine_sim = 1.0 - float(D[0][i]) / 2.0
            results.append((cosine_sim, self.doc_id_map[doc_id]))

        return results


if __name__ == "__main__":
    # Build LSI index from existing BSBI + VBE inverted index
    bsbi = BSBIIndex(
        data_dir='collection',
        postings_encoding=VBEPostings,
        output_dir=os.path.join('index', 'bsbi', VBEPostings.name),
        tmp_dir=os.path.join('tmp', 'bsbi', VBEPostings.name),
    )

    lsi = LSIIndex(n_components=100, output_dir='index/lsi')
    print("Building LSI + FAISS HNSW index...")
    lsi.build(bsbi)

    # Demo queries
    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
    ]

    print("\n===== LSI + FAISS (HNSW) Retrieval =====")
    for query in queries:
        print(f"Query  :  {query}")
        print("Results:")
        for score, doc in lsi.retrieve(query, k=10):
            print(f"  {doc:30} {score:>.3f}")
        print()
