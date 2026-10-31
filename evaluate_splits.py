"""
Repeated-split evaluation for the anime recommender.

Why this exists: a single 70/30 split of ~30 held-out anime means one extra
hit changes precision@10 by 0.10. This script repeats the same evaluation
over many random splits and reports the average and spread for every method,
so no number depends on one lucky split.

Run it from your project folder (the one with anime_dataset.csv,
mal_export.xml and anime_embeddings.npy):

    python evaluate_splits.py

Or point it at another folder:

    python evaluate_splits.py path/to/folder
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
N_SPLITS = 50            # number of random train/test splits
TOP_K = 10               # precision@K
PER_CLUSTER_POOL = 5     # candidates taken from each cluster (same as recommender.py)
CLUSTER_OPTIONS = [2, 3, 4]
SIM_WEIGHT, POP_WEIGHT = 0.75, 0.25   # same blend as recommender.py


def path(name):
    return os.path.join(DATA_DIR, name)


# ---------- load data (same cleaning as recommender.py) ----------
df = pd.read_csv(path("anime_dataset.csv"))
df = df[df["is_adult"] == False]
df = df.dropna(subset=["description"])
df = df.reset_index(drop=True)

EMBEDDINGS_PATH = path("anime_embeddings.npy")
if os.path.exists(EMBEDDINGS_PATH):
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)

assert len(embeddings) == len(df), (
    f"embeddings ({len(embeddings)}) and dataset ({len(df)}) have different sizes; "
    "delete anime_embeddings.npy and run again so it is regenerated"
)


def parse_mal_export(xml_path):
    root = ET.parse(xml_path).getroot()
    records = []
    for anime in root.findall("anime"):
        records.append({
            "mal_id": anime.findtext("series_animedb_id"),
            "my_score": anime.findtext("my_score"),
        })
    out = pd.DataFrame(records)
    out["mal_id"] = pd.to_numeric(out["mal_id"], errors="coerce")
    out["my_score"] = pd.to_numeric(out["my_score"], errors="coerce")
    return out


my_list = parse_mal_export(path("mal_export.xml"))
my_list = my_list[my_list["my_score"] > 0]

idx_of = {mid: i for i, mid in enumerate(df["mal_id"])}
my_list = my_list[my_list["mal_id"].isin(idx_of)]
rated_idx = np.array([idx_of[m] for m in my_list["mal_id"]])
rated_scores = my_list["my_score"].to_numpy(dtype=float)

strong = rated_scores >= 8
strong_idx = rated_idx[strong]
strong_scores = rated_scores[strong]
print(f"Rated anime found in dataset: {len(rated_idx)}")
print(f"Strong likes (score >= 8):    {len(strong_idx)}")
print(f"Splits: {N_SPLITS}, top-k: {TOP_K}\n")


# ---------- scoring helpers ----------
def minmax(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x, dtype=float)


members = df["members"].fillna(0).to_numpy(dtype=float)
pop_norm = minmax(np.log1p(members))


def blended_scores(taste_vector):
    sims = cosine_similarity(taste_vector.reshape(1, -1), embeddings)[0]
    return SIM_WEIGHT * minmax(sims) + POP_WEIGHT * pop_norm


def taste_vector(idx, scores):
    return np.average(embeddings[idx], axis=0, weights=scores)


def recommend(train_idx, train_scores, n_clusters, top_k=TOP_K, pool=PER_CLUSTER_POOL):
    """n_clusters == 1 is the single taste vector; otherwise K-Means taste clusters."""
    exclude = np.zeros(len(df), dtype=bool)
    exclude[train_idx] = True

    if n_clusters == 1:
        groups = [np.arange(len(train_idx))]
        take = top_k
    else:
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
            embeddings[train_idx]
        )
        groups = [np.where(labels == c)[0] for c in range(n_clusters)]
        take = pool

    candidates = {}
    for g in groups:
        score = blended_scores(taste_vector(train_idx[g], train_scores[g]))
        score[exclude] = -np.inf
        for i in np.argsort(-score)[:take]:
            candidates[i] = max(candidates.get(i, -np.inf), score[i])

    ranked = sorted(candidates, key=candidates.get, reverse=True)
    return ranked[:top_k]


def popularity_recommend(train_idx, top_k=TOP_K):
    score = pop_norm.copy()
    score[train_idx] = -np.inf
    return list(np.argsort(-score)[:top_k])


# ---------- repeated evaluation ----------
method_names = ["Popularity baseline", "Single taste vector"] + [
    f"Clustered, n={n}" for n in CLUSTER_OPTIONS
]
precision = {m: [] for m in method_names}
random_baseline = []

for seed in range(N_SPLITS):
    train_pos, test_pos = train_test_split(
        np.arange(len(strong_idx)), test_size=0.3, random_state=seed
    )
    train_idx, train_scores = strong_idx[train_pos], strong_scores[train_pos]
    test_set = set(strong_idx[test_pos].tolist())

    def p_at_k(recs):
        return sum(int(i) in test_set for i in recs) / TOP_K

    precision["Popularity baseline"].append(p_at_k(popularity_recommend(train_idx)))
    precision["Single taste vector"].append(p_at_k(recommend(train_idx, train_scores, 1)))
    for n in CLUSTER_OPTIONS:
        precision[f"Clustered, n={n}"].append(p_at_k(recommend(train_idx, train_scores, n)))

    random_baseline.append(len(test_set) / (len(df) - len(train_idx)))

# ---------- report ----------
print(f"Precision@{TOP_K} over {N_SPLITS} random splits")
print("-" * 62)
print(f"{'Method':<24}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
for m in method_names:
    a = np.array(precision[m])
    print(f"{m:<24}{a.mean():>8.3f}{a.std():>8.3f}{a.min():>8.2f}{a.max():>8.2f}")
print(f"{'Random chance':<24}{np.mean(random_baseline):>8.3f}")

print("\nPaired comparison vs popularity baseline (same splits)")
print("-" * 62)
pop = np.array(precision["Popularity baseline"])
for m in method_names[1:]:
    a = np.array(precision[m])
    diff = a - pop
    print(
        f"{m:<24} mean diff {diff.mean():+.3f}   "
        f"wins {int((diff > 0).sum())} / ties {int((diff == 0).sum())} / "
        f"losses {int((diff < 0).sum())}"
    )

print(
    "\nNote: splits overlap heavily (same ~30 anime reused), so treat the spread as a "
    "rough guide, not a formal confidence interval."
)

pd.DataFrame(precision).to_csv(path("split_results.csv"), index_label="split")
print("Per-split results saved to split_results.csv")