# Mini Search Engine

A search engine built from scratch using only Python standard libraries (plus numpy/scipy/faiss for LSI), implementing index construction, compression, multiple scoring methods, and evaluation metrics.
Built on top of [the given template](https://github.com/nano141004/mini-search-engine/commit/a14d053105742b8138253d43aa15e3ccabd0b0d6).

## Setup

Run the environment setup script:

```bash
setup_env.bat
```

This creates a virtual environment and installs all dependencies from `requirements.txt`. After that, activate the environment:

```bash
venv\Scripts\activate
```

## How to Run

There are three ways to run the search engine, depending on what you need.

### 1. Step by step

Run each step manually in order:

```bash
# 1. Build the index (BSBI)
python bsbi.py

# 2. Run search queries
python search.py

# 3. Evaluate search effectiveness
python evaluation.py
```

`search.py` loops over all index method + encoding + dict type + scoring combinations and prints results for 3 sample queries. `evaluation.py` does the same but computes RBP, DCG, NDCG, and AP scores over all 30 queries.

### 2. End-to-end (single configuration)

`run-end-to-end.py` runs the full pipeline (index, search, evaluate) for a single configuration. With no arguments, it defaults to BSBI + VBE + TF-IDF + idmap.

```bash
python run-end-to-end.py
```

All configuration dimensions are exposed via command-line arguments:

| Argument | Choices | Default |
|---|---|---|
| `--index-method` | `bsbi`, `spimi`, `lsi` | `bsbi` |
| `--encoding` | `standard`, `vbe`, `elias_gamma` | `vbe` |
| `--scoring` | `tfidf`, `bm25`, `bm25_alt2`, `bm25_alt3`, `wand_bm25` | `tfidf` |
| `--dict-type` | `idmap`, `fst` | `idmap` |
| `--k` | int | `1000` (eval top-K) |
| `--search-k` | int | `10` (demo search top-K) |

LSI-specific arguments (only relevant when `--index-method lsi`):

| Argument | Default | Description |
|---|---|---|
| `--lsi-components` | `100` | Number of SVD components |
| `--hnsw-M` | `32` | HNSW connections per node |
| `--hnsw-ef-construction` | `200` | HNSW construction-time search depth |
| `--hnsw-ef-search` | `128` | HNSW query-time search depth |

Step control flags: `--skip-indexing`, `--skip-search`, `--skip-eval`.

Examples:

```bash
# SPIMI + Elias Gamma + BM25
python run-end-to-end.py --index-method spimi --encoding elias_gamma --scoring bm25

# BSBI + FST dictionary + WAND BM25
python run-end-to-end.py --index-method bsbi --dict-type fst --scoring wand_bm25

# LSI with 200 SVD components
python run-end-to-end.py --index-method lsi --lsi-components 200 --hnsw-M 64

# LSI using Elias Gamma source index + FST dict
python run-end-to-end.py --index-method lsi --encoding elias_gamma --dict-type fst

# Only evaluate (skip indexing and search)
python run-end-to-end.py --skip-indexing --skip-search
```

### 3. Comparative analysis

`analysis.py` runs all 18 configurations and compares them across four dimensions:

1. Index build time
2. Index size on disk (final + intermediate)
3. Retrieval speed (latency per query)
4. Effectiveness (RBP, DCG, NDCG, AP)

```bash
python analysis.py
```

To add or remove configurations for comparison, edit the `CONFIGURATIONS` list in `analysis.py`. Each entry is a dict specifying `index_method`, `postings_encoding`, `scoring`, and optionally `dict_type` or `n_components`.

## Implementation Details

### Index Construction

#### BSBI (Blocked Sort-Based Indexing)

The default index construction method. Maintains a global term-to-termID mapping across all blocks. For each block, generates (termID, docID) pairs, sorts them, and writes them to disk as intermediate indices. The final index is produced by merging all intermediate indices.

Implementation: `bsbi.py`

#### SPIMI (Single-Pass In-Memory Indexing)

An alternative index construction method. Instead of maintaining a global term-to-termID mapping, SPIMI builds a separate dictionary (hash table) per block, mapping term strings directly to postings lists. Only the dictionary keys are sorted at the end of each block -- not the (termID, docID) pairs. Term IDs are assigned only during the final merge step. This avoids the overhead of sorting termID-docID pairs per block.

Implementation: `spimi.py`

### Postings Compression

#### Standard

No compression. DocIDs and TFs are stored as raw 4-byte unsigned integers.

Implementation: `compression.py` -- `StandardPostings`

#### Variable Byte Encoding (VBE)

Gap-based encoding. The postings list is first converted to gaps (differences between consecutive docIDs), then each gap is encoded using variable-byte encoding. TF values are encoded directly (no gap conversion).

Implementation: `compression.py` -- `VBEPostings`

#### Elias Gamma

Bit-level compression. Like VBE, the postings list is gap-encoded first. Each value is then encoded using Elias Gamma coding: N leading zeros + a 1-bit delimiter + N bits for the remainder, where N = floor(log2(value)). Values are shifted by +1 before encoding to handle zeros. The encoding is self-delimiting, so no length prefix is needed -- trailing zero-padding from byte alignment is naturally detected during decoding.

Implementation: `compression.py` -- `EliasGammaPostings`

### Dictionary Structure

#### IdMap

Default dictionary. A plain Python hash-based mapping between terms/doc paths and integer IDs. Simple and fast, but no compression of the dictionary itself.

Implementation: `util.py` -- `IdMap`

#### FST (Finite State Transducer)

A compact dictionary structure that shares both prefixes and suffixes between terms. Built using the Mihov & Maurel (2001) algorithm for direct construction of minimal acyclic subsequential transducers. Outputs (term IDs) are accumulated along edge traversals. Serialized with compact binary + zlib compression.

Implementation: `fst.py` -- `FST`, `FSTIdMap`

### Scoring Methods

#### TF-IDF

Standard TF-IDF cosine similarity scoring. The baseline retrieval method from the template.

#### BM25

Three BM25 variants are implemented with constants k1=1.2, b=0.75:
- **BM25 Okapi** -- the standard BM25 formula
- **BM25 Alt 2** -- the alternative BM25 scoring formulation
- **BM25 Alt 3** -- uses an additional k2 parameter (k2=100) for query term frequency weighting

All three require pre-computed average document length (avdl), which is computed and stored during indexing.

#### WAND BM25

WAND (Weighted AND) top-K retrieval using BM25 Okapi scoring. Instead of scoring every document, WAND skips documents that cannot make it into the top-K results by maintaining per-term upper bound scores. Upper bounds are pre-computed during indexing and stored in `wand_upper_bounds.pkl` per index directory.

### LSI (Latent Semantic Indexing) + FAISS

Builds on top of an existing inverted index. Constructs a sparse TF-IDF term-document matrix, then applies truncated SVD to reduce it to a low-rank approximation. Document vectors are L2-normalized and indexed using FAISS HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search. At query time, the query is projected into the LSI space via the learned U_k matrix, normalized, and searched against the FAISS index.

Implementation: `lsi.py` -- `LSIIndex`

### Evaluation Metrics

Four evaluation metrics are implemented, computed over 30 queries against human-annotated relevance judgments (`qrels.txt`):

- **RBP** (Rank-Biased Precision, p=0.8) -- from the template
- **DCG** (Discounted Cumulative Gain) -- using the formula DCG@K = sum of r_i / log2(i+1)
- **NDCG** (Normalized DCG) -- DCG normalized by the ideal DCG
- **AP** (Average Precision) -- mean of precision at each relevant document position, divided by R

Implementation: `evaluation.py`
