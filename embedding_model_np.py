# NumPy-backed equivalent of embedding_model.py.
#
# The §4 pedagogical implementation in embedding_model.py uses pure Python
# lists and per-coordinate loops. That style is great for explaining what
# the algorithm does, but it cannot keep up with a corpus the size of
# Brown (1.1M tokens, 10k vocab, hundreds of millions of updates per
# epoch). This file mirrors the same API with NumPy arrays, which speeds
# up the inner loops by roughly two orders of magnitude.
#
# Public API (matches embedding_model.EmbeddingModel exactly):
#   - EmbeddingModelNP(vocab_size, embedding_dim=5)
#   - get_vector(word_id)
#   - train_on_pair(center, context, learning_rate=0.01)
#   - train_on_negative(center, negative, learning_rate=0.01)
#   - train_on_negative_margin(center, negative, margin, learning_rate=0.01)
#
# The numerical behaviour is identical to embedding_model.py up to
# floating-point rounding — same algorithm, same update rules, just
# vectorized. The §5 evaluation prose can refer the reader to the pure-
# Python version in §4 as the canonical explanation.

import numpy as np


class EmbeddingModelNP:
    """NumPy-backed word embedding model.

    Internally stores embeddings as one (vocab_size, embedding_dim) array
    so that the pull/push updates can be done with array slicing instead
    of Python-level loops.
    """

    def __init__(self, vocab_size, embedding_dim=5, rng=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Same initialisation distribution as the pure-Python version:
        # uniform on [-0.5, +0.5] per coordinate.
        if rng is None:
            rng = np.random.default_rng()
        self.embeddings = rng.uniform(
            -0.5, 0.5, size=(vocab_size, embedding_dim)
        ).astype(np.float64)

    # -----------------------------------------------------------------------
    # Read-only access. Returns a copy so callers cannot accidentally
    # mutate the model's internal storage.
    # -----------------------------------------------------------------------
    def get_vector(self, word_id):
        return self.embeddings[word_id].copy()

    # -----------------------------------------------------------------------
    # Attraction. Matches embedding_model.train_on_pair exactly:
    #     v_c <- v_c + lr * (v_o - v_c)
    #     v_o <- v_o + lr * (v_c - v_o)   (using pre-update v_c)
    # Both sides read pre-update values, then both write back, exactly like
    # the per-coordinate version. NumPy lets us do it in two array ops.
    # -----------------------------------------------------------------------
    def train_on_pair(self, center_word, context_word, learning_rate=0.01):
        vc = self.embeddings[center_word]
        vo = self.embeddings[context_word]
        # Snapshot pre-update values so the second write does not see the
        # first one. This mirrors the per-coordinate read pattern in
        # embedding_model.py.
        vc_old = vc.copy()
        vo_old = vo.copy()
        self.embeddings[center_word]  = vc_old + learning_rate * (vo_old - vc_old)
        self.embeddings[context_word] = vo_old + learning_rate * (vc_old - vo_old)

    # -----------------------------------------------------------------------
    # §4.7 sign-flipped push. Same arithmetic shape as attraction, with
    # the sign of the displacement flipped. Kept for parity with the §4.7
    # pedagogical material; §4.9 training uses train_on_negative_margin.
    # -----------------------------------------------------------------------
    def train_on_negative(self, center_word, negative_word, learning_rate=0.01):
        vc = self.embeddings[center_word]
        vn = self.embeddings[negative_word]
        vc_old = vc.copy()
        vn_old = vn.copy()
        self.embeddings[center_word]   = vc_old - learning_rate * (vn_old - vc_old)
        self.embeddings[negative_word] = vn_old - learning_rate * (vc_old - vn_old)

    # -----------------------------------------------------------------------
    # §4.8 margin-gated push. The off-switch makes this the version actual
    # training uses. Same gradient-descent step on the hinge loss
    # L = (1/2) max(0, m - d)^2 that §4.8 derives.
    # -----------------------------------------------------------------------
    def train_on_negative_margin(self, center_word, negative_word,
                                 margin, learning_rate=0.01):
        vc = self.embeddings[center_word]
        vn = self.embeddings[negative_word]

        diff = vc - vn
        # np.linalg.norm gives the Euclidean length of the diff vector
        dist = float(np.linalg.norm(diff))

        # Off-switch: already at least `margin` apart, no update
        if dist >= margin:
            return
        # Defensive: vectors essentially identical, gradient undefined
        if dist < 1e-12:
            return

        scale = learning_rate * (margin - dist) / dist
        # Single-line array updates: each gets the scaled displacement,
        # one positive, one negative. Both read the pre-update diff.
        self.embeddings[center_word]   = vc + scale * diff
        self.embeddings[negative_word] = vn - scale * diff
