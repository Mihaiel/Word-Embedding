# Larger-corpus loader for §5 experiments.
#
# The §4 training scripts use a small custom corpus (corpus.txt) where every
# word is included. For §5 we want to train on a real-world corpus, so we
# add three things on top of what TextPreprocessor does:
#
#   1) A loader for the Brown corpus via NLTK. First call downloads the
#      corpus into the user's local nltk_data directory (a few MB).
#   2) Token normalization (lowercase, strip punctuation and apostrophes)
#      so that "King's" and "king" collapse to the same token.
#   3) Vocabulary capping. Real corpora have a long tail of rare words,
#      most of which we cannot reasonably train on. We keep only the
#      `top_n` most frequent words and drop the rest from training pairs.
#
# The return shape matches TextPreprocessor.preprocess() except that we
# do not return the tokenized sentences (the encoded form is all the
# training code needs).

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Token-level normalization. Lowercase, strip non-letters. Apostrophes are
# removed (so "it's" -> "its", which is mildly lossy but easier to reason
# about than splitting clitics). Tokens that become empty after
# normalization (pure punctuation, numbers) are dropped by the caller.
# ---------------------------------------------------------------------------
def normalize_token(token):
    token = token.lower()
    token = re.sub(r"[’']", "", token)  # straight + curly apostrophe
    token = re.sub(r"[^a-z]", "", token)
    return token


def normalize_sentences(raw_sentences):
    """Apply `normalize_token` to every token, dropping empties."""
    out = []
    for sent in raw_sentences:
        normalized = [normalize_token(t) for t in sent]
        normalized = [t for t in normalized if t]
        if normalized:
            out.append(normalized)
    return out


# ---------------------------------------------------------------------------
# Vocabulary construction. Counts every word, keeps only the `top_n` most
# frequent. Returns the same three structures every other script in this
# project consumes.
# ---------------------------------------------------------------------------
def build_vocab_capped(tokenized_sentences, top_n):
    counts = Counter()
    for sent in tokenized_sentences:
        for word in sent:
            counts[word] += 1

    top_words = counts.most_common(top_n)

    word_to_id = {}
    id_to_word = {}
    word_counts = {}
    for idx, (word, count) in enumerate(top_words):
        word_to_id[word] = idx
        id_to_word[idx] = word
        word_counts[idx] = count

    return word_to_id, id_to_word, word_counts


# ---------------------------------------------------------------------------
# Encode sentences using the capped vocabulary. Words that did not make
# the cut are simply skipped — they leave a small hole in the sentence
# but do not produce training pairs. This is the standard treatment for
# OOV ("out of vocabulary") words in word2vec-style training.
# ---------------------------------------------------------------------------
def encode_sentences(tokenized_sentences, word_to_id):
    encoded = []
    for sent in tokenized_sentences:
        encoded_sent = []
        for word in sent:
            if word in word_to_id:
                encoded_sent.append(word_to_id[word])
        if encoded_sent:
            encoded.append(encoded_sent)
    return encoded


# ---------------------------------------------------------------------------
# Convenience pipeline for an arbitrary iterable of tokenized sentences.
# Used by `load_brown` below, and reusable for any other corpus you might
# want to swap in later.
# ---------------------------------------------------------------------------
def prepare_corpus(raw_sentences, top_n=10000):
    """Normalize, build vocab capped at top_n, encode.

    Parameters
    ----------
    raw_sentences : iterable of list[str]
        Sentences as lists of tokens. Tokens can have any casing or
        punctuation; normalization happens inside.
    top_n : int
        Keep only the `top_n` most frequent words. The rest are dropped.

    Returns
    -------
    (word_to_id, id_to_word, encoded_sentences, word_counts)
    """
    tokenized = normalize_sentences(raw_sentences)
    word_to_id, id_to_word, word_counts = build_vocab_capped(tokenized, top_n)
    encoded = encode_sentences(tokenized, word_to_id)
    return word_to_id, id_to_word, encoded, word_counts


# ---------------------------------------------------------------------------
# Brown corpus loader. First call requires network access to fetch the
# corpus from NLTK; subsequent calls read from the cached copy in the
# user's ~/nltk_data directory.
# ---------------------------------------------------------------------------
def load_brown(top_n=10000, sample_size=None):
    """Load the Brown corpus from NLTK, normalize, cap vocab, encode.

    Parameters
    ----------
    top_n : int, default 10000
        Vocabulary cap. Brown has roughly 50k unique tokens after
        normalization; keeping the top 10k covers about 95% of total
        occurrences while keeping training tractable.
    sample_size : int or None
        If given, only the first `sample_size` sentences are used.
        Useful for quick smoke tests.

    Returns
    -------
    (word_to_id, id_to_word, encoded_sentences, word_counts)
    """
    import nltk
    try:
        from nltk.corpus import brown
        # Force the corpus to load; this is what triggers the
        # LookupError on a fresh install.
        _ = brown.fileids()
    except LookupError:
        nltk.download("brown")
        from nltk.corpus import brown

    raw = brown.sents()
    if sample_size is not None:
        raw = list(raw)[:sample_size]

    return prepare_corpus(raw, top_n=top_n)


# ---------------------------------------------------------------------------
# Stand-alone smoke test. Run this file directly to verify the loader
# works on your machine before training anything.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading Brown (this downloads ~5 MB on first run)...")
    w2id, id2w, encoded, counts = load_brown(top_n=10000)

    print(f"\nVocabulary size: {len(w2id)} (capped at 10000)")
    print(f"Sentences kept:   {len(encoded)}")
    print(f"Total tokens:     {sum(len(s) for s in encoded)}")

    print("\nTop 20 most-frequent words:")
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    for wid, count in ranked[:20]:
        print(f"  {id2w[wid]:<15} {count}")

    print("\nFirst 3 encoded sentences (decoded back to words):")
    for sent in encoded[:3]:
        words = [id2w[wid] for wid in sent]
        print(f"  {' '.join(words[:25])}{' ...' if len(words) > 25 else ''}")
