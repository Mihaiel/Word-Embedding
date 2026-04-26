# Simple training script for the word embedding model.
# Trains for a fixed number of epochs with the direct-attraction rule
# plus K random-repulsion updates per positive pair, and plots how the
# vectors move in the embedding space.

import math
import os
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from text_preprocessor import TextPreprocessor
from training_data import TrainingDataGenerator
from embedding_model import EmbeddingModel


# Hyperparameters
SEED = 20
WINDOW_SIZE = 1
EMBEDDING_DIM = 2
LEARNING_RATE = 0.001
EPOCHS = 300
NEGATIVES_PER_PAIR = 1


# Helper: average Euclidean distance between all pairs of word vectors
def mean_pairwise_distance(vectors):
    total = 0.0
    count = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            s = 0.0
            for k in range(len(vectors[i])):
                diff = vectors[i][k] - vectors[j][k]
                s += diff * diff
            total += math.sqrt(s)
            count += 1
    return total / count


# Helper: average Euclidean norm across all word vectors
def mean_norm(vectors):
    total = 0.0
    for v in vectors:
        s = 0.0
        for x in v:
            s += x * x
        total += math.sqrt(s)
    return total / len(vectors)


# Helper: scatter points and write each word next to its vector
def draw_labelled_scatter(ax, snap, id_to_word):
    xs = [v[0] for v in snap]
    ys = [v[1] for v in snap]
    ax.scatter(xs, ys, s=10)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x + 0.01, y + 0.01, id_to_word[idx], fontsize=6, alpha=0.8)


# Helper: uniform random negative that is not the center itself
def sample_negative(center_id, vocab_size):
    while True:
        neg = random.randrange(vocab_size)
        if neg != center_id:
            return neg


random.seed(SEED)


# 1. Preprocess the corpus
with open("corpus.txt") as f:
    text = f.read()

preprocessor = TextPreprocessor()
_, word_to_id, id_to_word, encoded = preprocessor.preprocess(text)
vocab_size = len(word_to_id)
print("Vocabulary size:", vocab_size)


# 2. Build training pairs
generator = TrainingDataGenerator()
pairs = generator.generate_pairs(encoded, window_size=WINDOW_SIZE)
print("Training pairs:", len(pairs))


# 3. Create the embedding model
model = EmbeddingModel(vocab_size, embedding_dim=EMBEDDING_DIM)


# 4. Training loop
# We store a copy of all vectors at every epoch so we can animate afterwards.
distances = [mean_pairwise_distance(model.embeddings)]
norms = [mean_norm(model.embeddings)]
history = [[list(v) for v in model.embeddings]]

for epoch in range(1, EPOCHS + 1):
    random.shuffle(pairs)
    for center, context in pairs:
        model.train_on_pair(center, context, learning_rate=LEARNING_RATE)
        for _ in range(NEGATIVES_PER_PAIR):
            neg = sample_negative(center, vocab_size)
            model.train_on_negative(center, neg, learning_rate=LEARNING_RATE)

    distances.append(mean_pairwise_distance(model.embeddings))
    norms.append(mean_norm(model.embeddings))
    history.append([list(v) for v in model.embeddings])

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d}: "
            f"mean pairwise distance = {distances[-1]:.4f}, "
            f"mean norm = {norms[-1]:.4f}"
        )


# 5. Make the output/direct-and-repulsion folder
os.makedirs("output/direct-and-repulsion", exist_ok=True)


# 6. Plot the two metrics over time
plt.figure()
plt.plot(distances, label="mean pairwise distance")
plt.plot(norms, label="mean vector norm")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Attraction + repulsion: distance and norm over training")
plt.legend()
plt.grid(True)
plt.savefig("output/direct-and-repulsion/metrics.png")
plt.close()


# Figure out axis limits from the full run so nothing falls off the frame.
all_x = [v[0] for snap in history for v in snap]
all_y = [v[1] for snap in history for v in snap]
xlim = (min(all_x) - 0.5, max(all_x) + 0.5)
ylim = (min(all_y) - 0.5, max(all_y) + 0.5)


# 7. Plot the embedding space at several checkpoints
checkpoints = [0, 1, 10, 50, 100, 150, 200, EPOCHS]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, epoch in enumerate(checkpoints):
    ax = axes[i // 4][i % 4]
    snap = history[epoch]
    draw_labelled_scatter(ax, snap, id_to_word)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"Epoch {epoch}")
    ax.grid(True)
plt.tight_layout()
plt.savefig("output/direct-and-repulsion/snapshots.png")
plt.close()


# 8. Plot the final state on its own for easy inspection
fig, ax = plt.subplots(figsize=(8, 8))
draw_labelled_scatter(ax, history[EPOCHS], id_to_word)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_title(f"Final state  (Epoch {EPOCHS}, K = {NEGATIVES_PER_PAIR})")
ax.grid(True)
plt.tight_layout()
plt.savefig("output/direct-and-repulsion/final.png")
plt.close()


# 9. Animate the full training
# Keep the output size modest: smaller figure and every-5-epochs sampling.
# Frame count = EPOCHS // FRAME_STEP + 1.
FRAME_STEP = 5
fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=90)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.grid(True)
scatter = ax.scatter([], [])

# One text object per word, updated every frame
label_artists = []
for word_id in range(len(id_to_word)):
    v = history[0][word_id]
    t = ax.text(v[0] + 0.01, v[1] + 0.01, id_to_word[word_id],
                fontsize=6, alpha=0.8)
    label_artists.append(t)

title = ax.set_title("")

def update(epoch):
    snap = history[epoch]
    scatter.set_offsets([[v[0], v[1]] for v in snap])
    for word_id, t in enumerate(label_artists):
        v = snap[word_id]
        t.set_position((v[0] + 0.01, v[1] + 0.01))
    title.set_text(f"Epoch {epoch}")
    return [scatter, title, *label_artists]

anim = animation.FuncAnimation(fig, update, frames=range(0, EPOCHS + 1, FRAME_STEP))
anim.save("output/direct-and-repulsion/repulsion.gif", writer="pillow", fps=10)
plt.close()


print("Done. Plots saved to output/direct-and-repulsion/")
