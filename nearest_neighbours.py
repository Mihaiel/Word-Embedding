# Nearest-neighbour query over the Brown-trained embeddings.
#
# Loads output/brown/embeddings.npy and output/brown/vocab.json, then for
# each query word computes cosine similarity against every other word and
# prints the top-K most similar.
#
# Usage:
#   python3 nearest_neighbours.py king queen dog cat
#   python3 nearest_neighbours.py            # runs a default curated batch
#   python3 nearest_neighbours.py --k 20 coffee
#   python3 nearest_neighbours.py --pair king queen  # cosine between two words

import argparse
import json
import os
import sys

import numpy as np


DEFAULT_QUERIES = [
    # Royalty / people
    "king", "queen", "man", "woman", "child",
    # Animals
    "dog", "cat", "horse",
    # Time
    "year", "monday", "january", "morning",
    # Drinks / food
    "coffee", "water", "bread",
    # Places / abstract
    "city", "country", "world",
    # Common verbs (to see if they cluster)
    "said", "thought", "made",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_embeddings(embeddings_path, vocab_path):
    """Load the saved embeddings + vocabulary and return arrays + dictionaries."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"{embeddings_path} not found. Run `python3 train_brown.py` first."
        )
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"{vocab_path} not found. Run `python3 train_brown.py` first."
        )

    embeddings = np.load(embeddings_path)
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    id_to_word = vocab_data["id_to_word"]
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    return embeddings, word_to_id, id_to_word


# ---------------------------------------------------------------------------
# Pre-compute L2-normalised embeddings once. With normalised vectors,
# cosine similarity reduces to a single dot product (much cheaper than
# the unnormalised version on each query).
# ---------------------------------------------------------------------------
def l2_normalise(E, eps=1e-12):
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(norms, eps)


# ---------------------------------------------------------------------------
# Core queries
# ---------------------------------------------------------------------------
def top_k_nearest(E_norm, word_to_id, id_to_word, query, k=10):
    """Return top-k (word, cosine) pairs for `query`, excluding the query itself.
    Returns None if the query is OOV."""
    if query not in word_to_id:
        return None
    qid = word_to_id[query]
    cosines = E_norm @ E_norm[qid]
    # Mask out the query itself
    cosines[qid] = -np.inf
    top_idx = np.argpartition(-cosines, k)[:k]
    top_idx = top_idx[np.argsort(-cosines[top_idx])]
    return [(id_to_word[i], float(cosines[i])) for i in top_idx]


def cosine_pair(E_norm, word_to_id, w1, w2):
    """Cosine similarity between two words. Returns None if either is OOV."""
    if w1 not in word_to_id or w2 not in word_to_id:
        return None
    a = E_norm[word_to_id[w1]]
    b = E_norm[word_to_id[w2]]
    return float(a @ b)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_top_k(query, results, k):
    print(f"\n  {query}")
    if results is None:
        print(f"    (not in vocabulary)")
        return
    for word, sim in results:
        print(f"    {word:<20} {sim:+.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Nearest-neighbour queries over the Brown-trained embeddings."
    )
    parser.add_argument(
        "queries", nargs="*",
        help="Query words. If none given, a curated default batch is used."
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of nearest neighbours to return per query (default: 10)."
    )
    parser.add_argument(
        "--embeddings", default="output/brown/embeddings.npy",
        help="Path to embeddings .npy file."
    )
    parser.add_argument(
        "--vocab", default="output/brown/vocab.json",
        help="Path to vocab .json file."
    )
    parser.add_argument(
        "--pair", nargs=2, metavar=("WORD1", "WORD2"),
        help="Print cosine similarity between WORD1 and WORD2 and exit."
    )
    args = parser.parse_args()

    E, word_to_id, id_to_word = load_embeddings(args.embeddings, args.vocab)
    print(f"Loaded {len(id_to_word):,} embeddings from {args.embeddings}")
    E_norm = l2_normalise(E)

    if args.pair:
        w1, w2 = args.pair
        sim = cosine_pair(E_norm, word_to_id, w1, w2)
        if sim is None:
            missing = [w for w in (w1, w2) if w not in word_to_id]
            print(f"OOV: {missing}")
            sys.exit(1)
        print(f"\ncos({w1}, {w2}) = {sim:+.4f}")
        return

    queries = args.queries if args.queries else DEFAULT_QUERIES

    print(f"\nNearest neighbours (top {args.k}):")
    print("=" * 60)
    for q in queries:
        results = top_k_nearest(E_norm, word_to_id, id_to_word, q, k=args.k)
        print_top_k(q, results, args.k)


if __name__ == "__main__":
    main()
