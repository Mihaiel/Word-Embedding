# PCA visualization of the Brown-trained embeddings for §5.2.
#
# Loads output/brown/embeddings.npy plus output/brown/vocab.json, picks a
# curated subset of words organised by semantic category, and produces two
# artefacts:
#
#   1. output/brown/embedding-pca.png
#      A static 2D scatter, used as the page's fallback figure for readers
#      without WebGL.
#   2. output/brown/embedding-3d.json
#      A normalised 3D projection that the in-browser Three.js viewer
#      consumes. Format: a list of {word, x, y, z, category, color}.
#
# PCA is implemented from scratch via NumPy SVD so there is no scikit-learn
# dependency. The maths is one line: center the data, then take the first
# two (or three) right-singular vectors as the projection axes.

import json
import os

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Curated word list. Each entry is (category_label, list of words).
# Categories chosen because the nearest-neighbour evidence in §5.1
# suggests they cluster cleanly under this model, plus six more that are
# common in Brown and tend to cluster well in any decent word2vec setup.
# ---------------------------------------------------------------------------
CATEGORIES = [
    # First 10: same as before. Brand-aligned colours, all confirmed to
    # cluster reasonably on Brown by the §5.1 nearest-neighbour evidence.
    ("Days of the week", [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ]),
    ("Months", [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]),
    ("Time of day", [
        "morning", "afternoon", "evening", "night",
    ]),
    ("Food", [
        "bread", "butter", "cheese", "soup", "milk", "cream",
    ]),
    ("Family", [
        "father", "mother", "brother", "sister",
        "son", "daughter", "husband", "wife",
    ]),
    ("Body", [
        "head", "hand", "face", "eyes", "arm", "foot",
    ]),
    ("Numbers", [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
    ]),
    ("Colors", [
        "red", "blue", "green", "white", "black", "yellow",
    ]),
    ("Nature", [
        "sea", "mountain", "river", "sky", "sun", "moon", "tree",
    ]),
    ("Politics", [
        "president", "kennedy", "nixon", "government", "washington", "senate",
    ]),

    # Next 10: added for the 3D viewer in §5.2.
    ("Verbs", [
        "said", "made", "told", "came", "took", "knew", "found",
    ]),
    ("Emotions", [
        "love", "fear", "hope", "joy", "anger",
    ]),
    ("Buildings", [
        "house", "room", "office", "building", "garden",
    ]),
    ("Transport", [
        "car", "bus", "train", "plane", "ship",
    ]),
    ("Clothing", [
        "coat", "hat", "shirt", "dress", "shoes",
    ]),
    ("Music", [
        "music", "song", "dance", "voice",
    ]),
    ("Religion", [
        "god", "faith", "prayer", "soul", "church",
    ]),
    ("Places", [
        "city", "town", "country", "state", "world",
    ]),
    ("Education", [
        "school", "student", "teacher", "book", "class",
    ]),
    ("Animals", [
        "dog", "cat", "horse", "bird",
    ]),
]


# Twenty-colour palette. The first 10 are the same brand-aligned colours
# we used before (so the §5.2 figure stays continuous with §3.3, §4.8 and
# §4.9). The next 10 are chosen for contrast against each other and
# against the existing 10. Twenty colours is genuinely difficult to
# disambiguate by eye, so we lean on the hover tooltip and the legend
# rather than colour alone.
COLOURS = {
    "Days of the week": "#6471E9",  # brand primary blue
    "Months":           "#766C82",  # brand secondary purple-grey
    "Time of day":      "#2E9D5F",  # green (matches §4.9 delta-good)
    "Food":             "#D9534F",  # warm red (matches §3.4 distance line)
    "Family":           "#E8A33D",  # warm orange
    "Body":             "#C8479D",  # pink-magenta
    "Numbers":          "#2BABB7",  # teal
    "Colors":           "#8C5E2A",  # brown
    "Nature":           "#6A8B3F",  # olive green
    "Politics":         "#5D4377",  # deep plum
    "Verbs":            "#F95D6A",  # salmon
    "Emotions":         "#FFB000",  # gold
    "Buildings":        "#2F4B7C",  # navy
    "Transport":        "#A05195",  # mauve
    "Clothing":         "#D45087",  # raspberry
    "Music":            "#00B5B8",  # cyan
    "Religion":         "#708090",  # slate grey
    "Places":           "#3CB44B",  # vivid green
    "Education":        "#4B0082",  # indigo
    "Animals":          "#FF8C00",  # dark orange
}

OUTPUT_PATH    = "output/brown/embedding-pca.png"
JSON_3D_PATH   = "output/brown/embedding-3d.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_embeddings(embeddings_path="output/brown/embeddings.npy",
                    vocab_path="output/brown/vocab.json"):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"{embeddings_path} not found. Run train_brown.py first."
        )
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"{vocab_path} not found. Run train_brown.py first."
        )
    E = np.load(embeddings_path)
    with open(vocab_path) as f:
        id_to_word = json.load(f)["id_to_word"]
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    return E, word_to_id, id_to_word


# ---------------------------------------------------------------------------
# PCA from scratch. Given an array X of shape (n_samples, n_features),
# return the k-dimensional projection in shape (n_samples, k). The maths
# is one line: center the data, run SVD, take the first k right-singular
# vectors as the new axes, project onto them.
# ---------------------------------------------------------------------------
def pca_kd(X, k):
    X_centered = X - X.mean(axis=0)
    _U, _S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:k]  # shape (k, n_features)
    return X_centered @ components.T  # shape (n_samples, k)


def pca_2d(X):
    return pca_kd(X, 2)


def pca_3d(X):
    return pca_kd(X, 3)


# Normalise coordinates so the cloud is centered at the origin and fits
# inside the unit sphere. This makes the Three.js camera positioning
# predictable regardless of the absolute scale of the trained vectors.
def normalise_for_viewer(coords):
    coords = coords - coords.mean(axis=0)
    max_dist = float(np.max(np.linalg.norm(coords, axis=1)))
    return coords / max_dist if max_dist > 0 else coords


# ---------------------------------------------------------------------------
# Build the (words, vectors, categories) tuple from CATEGORIES, filtering
# out anything that is not in the trained vocabulary.
# ---------------------------------------------------------------------------
def gather_word_vectors(E, word_to_id):
    words = []
    vectors = []
    categories = []
    missing = {}
    for cat, ws in CATEGORIES:
        for w in ws:
            if w in word_to_id:
                words.append(w)
                vectors.append(E[word_to_id[w]])
                categories.append(cat)
            else:
                missing.setdefault(cat, []).append(w)
    if missing:
        print("Skipped (OOV):")
        for cat, ws in missing.items():
            print(f"  {cat}: {', '.join(ws)}")
    return words, np.array(vectors), categories


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pca(words, coords, categories, output_path):
    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_alpha(0)

    seen_categories = []
    for word, (x, y), cat in zip(words, coords, categories):
        colour = COLOURS.get(cat, "#999999")
        label = cat if cat not in seen_categories else None
        seen_categories.append(cat)
        ax.scatter(x, y, color=colour, s=60, alpha=0.85,
                   edgecolors="#1d1f28", linewidth=0.6, label=label, zorder=3)
        # Word label, offset slightly to the upper-right of the point
        ax.annotate(
            word, (x, y),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, color="#1d1f28", fontweight="500",
        )

    ax.set_xlabel("PC 1", color="#1d1f28", fontsize=11)
    ax.set_ylabel("PC 2", color="#1d1f28", fontsize=11)
    ax.set_title(
        "Brown-trained word embeddings, projected to 2D via PCA",
        color="#1d1f28", fontsize=13, pad=12,
    )
    ax.tick_params(colors="#1d1f28", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#dddddd")
    ax.grid(True, linestyle=":", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # Legend, deduplicated and styled
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(),
              loc="best", frameon=True, fontsize=10,
              facecolor="#fafbff", edgecolor="#dddddd")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def export_3d_json(words, coords_3d, categories, output_path):
    """Write the 3D-projected points to JSON for the Three.js viewer."""
    coords_norm = normalise_for_viewer(coords_3d)
    payload = []
    for word, (x, y, z), cat in zip(words, coords_norm, categories):
        payload.append({
            "word":     word,
            "x":        round(float(x), 4),
            "y":        round(float(y), 4),
            "z":        round(float(z), 4),
            "category": cat,
            "color":    COLOURS.get(cat, "#999999"),
        })
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {output_path}  ({len(payload)} points)")


def main():
    E, word_to_id, id_to_word = load_embeddings()
    print(f"Loaded {len(id_to_word):,} embeddings, {E.shape[1]}D each")

    words, vectors, categories = gather_word_vectors(E, word_to_id)
    print(f"\nPlotting {len(words)} words across "
          f"{len(set(categories))} categories")

    # 2D PCA -> the static PNG fallback
    coords_2d = pca_2d(vectors)
    plot_pca(words, coords_2d, categories, OUTPUT_PATH)

    # 3D PCA -> the JSON consumed by the in-browser Three.js viewer
    coords_3d = pca_3d(vectors)
    export_3d_json(words, coords_3d, categories, JSON_3D_PATH)


if __name__ == "__main__":
    main()
