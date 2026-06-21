# Production-scale training script for §5.
#
# Trains the §4.9 algorithm (attraction + margin-gated repulsion +
# frequency-biased negative sampling) on the Brown corpus, using the
# NumPy-backed EmbeddingModelNP from embedding_model_np.py for speed.
#
# Outputs land in output/brown/:
#   - embeddings.npy      The (vocab_size, embedding_dim) NumPy array.
#   - vocab.json          word_to_id and id_to_word dictionaries.
#   - metadata.json       Hyperparameters and timings, for reproducibility.
#   - metrics.png         Mean vector norm over epochs.
#
# The pure-Python EmbeddingModel from embedding_model.py is unchanged and
# remains the canonical reference for the algorithm itself. This script
# uses the NumPy port for speed only; the numerical behaviour is identical
# to within floating-point rounding.

import json
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt

from corpus_loader import load_brown
from training_data import TrainingDataGenerator
from negative_sampler import UnigramSampler
from embedding_model_np import EmbeddingModelNP


# ---------------------------------------------------------------------------
# Hyperparameters. These match the §4.9 simulator defaults except the
# embedding dimension (bumped from 2 to 50 so semantic structure has room
# to emerge) and the learning rate (bumped from 0.001 to 0.025, which is a
# more typical word2vec-scale value for a larger corpus).
# ---------------------------------------------------------------------------
SEED               = 20
TOP_N              = 10_000
EMBEDDING_DIM      = 50
WINDOW_SIZE        = 2
NEGATIVES_PER_PAIR = 5
MARGIN             = 1.0
EXPONENT           = 0.75
LEARNING_RATE      = 0.025
EPOCHS             = 10

OUTPUT_DIR = "output/brown"


def mean_norm(E):
    """Average Euclidean norm across all rows of E."""
    norms = np.linalg.norm(E, axis=1)
    return float(np.mean(norms))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_overall = time.time()

    # -----------------------------------------------------------------------
    # 1) Load and preprocess Brown.
    # -----------------------------------------------------------------------
    print("Loading Brown corpus (first run downloads ~5 MB from NLTK)...")
    t0 = time.time()
    word_to_id, id_to_word, encoded, word_counts = load_brown(top_n=TOP_N)
    vocab_size = len(word_to_id)
    total_tokens = sum(len(s) for s in encoded)
    print(
        f"  Vocab size:    {vocab_size:,} (top {TOP_N:,} words)\n"
        f"  Sentences:     {len(encoded):,}\n"
        f"  Total tokens:  {total_tokens:,}\n"
        f"  Load time:     {time.time() - t0:.1f}s"
    )

    print("\nMost-frequent words (sanity check):")
    for wid, count in sorted(word_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {id_to_word[wid]:<15} {count:,}")

    # -----------------------------------------------------------------------
    # 2) Generate training pairs.
    # -----------------------------------------------------------------------
    print("\nGenerating training pairs...")
    t0 = time.time()
    gen = TrainingDataGenerator()
    pairs = gen.generate_pairs(encoded, window_size=WINDOW_SIZE)
    print(f"  Pairs:        {len(pairs):,}")
    print(f"  Gen time:     {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # 3) Build the frequency-biased negative sampler.
    # -----------------------------------------------------------------------
    sampler = UnigramSampler(word_counts, exponent=EXPONENT)
    print(f"\nUnigramSampler built (exponent = {EXPONENT})")

    # -----------------------------------------------------------------------
    # 4) Create the embedding model. Use a seeded numpy RNG so the
    # initial weights are reproducible across runs.
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    model = EmbeddingModelNP(vocab_size, embedding_dim=EMBEDDING_DIM, rng=rng)
    print(f"\nModel: {vocab_size:,} x {EMBEDDING_DIM}D embeddings")
    print(f"Initial mean norm: {mean_norm(model.embeddings):.4f}")

    # -----------------------------------------------------------------------
    # 5) Training loop. We DO NOT compute mean pairwise distance every
    # epoch the way the §4 scripts did — that is O(V^2) and would take
    # minutes per epoch on a 10k vocabulary. Mean norm (O(V)) is fast
    # enough to track per epoch.
    # -----------------------------------------------------------------------
    random.seed(SEED)
    epoch_norms = [mean_norm(model.embeddings)]
    epoch_times = []

    print(
        f"\nTraining: epochs={EPOCHS}, lr={LEARNING_RATE}, "
        f"K={NEGATIVES_PER_PAIR}, margin={MARGIN}"
    )
    print("-" * 60)
    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()
        random.shuffle(pairs)
        for center, context in pairs:
            model.train_on_pair(center, context, learning_rate=LEARNING_RATE)
            for _ in range(NEGATIVES_PER_PAIR):
                neg = sampler.sample(exclude_id=center)
                model.train_on_negative_margin(
                    center, neg, MARGIN, learning_rate=LEARNING_RATE
                )
        et = time.time() - t_epoch
        epoch_times.append(et)
        norm = mean_norm(model.embeddings)
        epoch_norms.append(norm)
        print(f"  Epoch {epoch:2d}/{EPOCHS}  mean norm = {norm:.4f}  ({et:.1f}s)")

    train_total = sum(epoch_times)
    print("-" * 60)
    print(f"Training done in {train_total:.1f}s "
          f"({train_total / 60:.1f} min, "
          f"avg {train_total / EPOCHS:.1f}s/epoch)")

    # -----------------------------------------------------------------------
    # 6) Save outputs. Embeddings as .npy, vocab as JSON, metadata as JSON.
    # We also save id_to_word as a list-of-words (index = id) to keep the
    # JSON compact; reading code can reconstruct word_to_id easily.
    # -----------------------------------------------------------------------
    emb_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    np.save(emb_path, model.embeddings)
    print(f"\nSaved embeddings -> {emb_path}")

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    id2w_list = [id_to_word[i] for i in range(vocab_size)]
    with open(vocab_path, "w") as f:
        json.dump({"id_to_word": id2w_list}, f, indent=2)
    print(f"Saved vocab      -> {vocab_path}")

    meta = {
        "seed": SEED,
        "top_n": TOP_N,
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "window_size": WINDOW_SIZE,
        "negatives_per_pair": NEGATIVES_PER_PAIR,
        "margin": MARGIN,
        "exponent": EXPONENT,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "total_tokens": total_tokens,
        "training_pairs": len(pairs),
        "epoch_norms": epoch_norms,
        "epoch_times_seconds": epoch_times,
        "total_training_seconds": train_total,
        "total_wall_seconds": time.time() - t_overall,
    }
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata   -> {meta_path}")

    # -----------------------------------------------------------------------
    # 7) Quick training-curve plot.
    # -----------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(epoch_norms)), epoch_norms,
             color="#6471E9", linewidth=2, marker="o", markersize=4)
    plt.axhline(MARGIN, color="grey", linestyle="--", linewidth=1,
                label=f"margin m = {MARGIN}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean vector norm")
    plt.title(f"Mean vector norm over training (Brown, "
              f"V={vocab_size:,}, d={EMBEDDING_DIM})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "metrics.png")
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"Saved curve      -> {curve_path}")

    print(f"\nAll done in {time.time() - t_overall:.1f}s")


if __name__ == "__main__":
    main()
