# Anime Recommender: Content-Based Recommendations from My Own MAL Ratings

A personal project exploring whether a content-based recommender (built from anime synopsis embeddings) can outperform simple popularity-based recommendations for matching *my own* taste — using my real MyAnimeList rating history as ground truth.

## Motivation

MyAnimeList's built-in recommendations often felt generic — driven heavily by "users who liked X also liked Y" collaborative signals rather than understanding *why* I liked something. This project tests an alternative: represent each anime by the actual meaning of its synopsis (via text embeddings), and build a recommender purely from content similarity, rather than aggregate user behavior.

## Data

- **Anime metadata**: ~4,900 titles from a Kaggle-hosted MyAnimeList dataset (synopsis, genres, tags, studio, score, member count).
- **Personal ratings**: my own MAL list, exported as XML (128 rated titles).


## Method

1. **Cleaning**: dropped adult-flagged titles and rows with missing synopses (7 out of ~4,985 — checked these individually and confirmed they were genuinely missing on MAL too, not a parsing bug).
2. **Embeddings**: generated 384-dimensional sentence embeddings for every synopsis using `all-MiniLM-L6-v2` (a small, fast, local sentence-transformer model — chosen to avoid API costs/rate limits and keep the pipeline fully offline-capable). Cached to disk after first generation to avoid recomputation.
3. **Taste vector**: computed a weighted average of the embeddings for my rated anime, weighted by my score (higher-rated anime pull the vector more strongly toward their "direction" in embedding space).
4. **Popularity blending**: combined content similarity with a normalized, log-transformed member count signal (75% similarity / 25% popularity), to avoid surfacing thematically-matching but extremely obscure titles.
5. **Clustering (key iteration)**: discovered my rated anime span genuinely distinct genre profiles (Action, Drama, Comedy, Romance, Suspense, Sports all represented in meaningful numbers) — meaning a single averaged taste vector likely blurred distinct preferences together. Used K-Means to split my ratings into taste "clusters," generating a separate taste vector per cluster, then merged per-cluster top candidates into one final ranked list (capped per cluster, to prevent one large cluster from dominating).

## Evaluation

Used a hold-out methodology: split my highly-rated anime (score ≥ 8) into train/test sets (70/30). Rebuilt the taste vector(s) using only the train set, then checked whether the held-out (test) anime — pretending I'd never rated them — were successfully "rediscovered" in the top 10 recommendations. This is measured as **precision@10**.

| Approach | Precision@10 |
|---|---|
| Popularity baseline (most-members-first) | 0.30 |
| Single taste vector (no clustering) | 0.10 |
| Clustered, n=4 | 0.10 |
| Clustered, n=3 | 0.10 |
| **Clustered, n=2** | **0.40** |

**Random-chance baseline** (for context): ~0.063, given ~31 positives in a ~4,900-item pool.


### Qualitative check

Several top recommendations from the final (n=2) model were anime I had actually watched or partially watched but never logged/scored on MAL (e.g. dropped partway through) — a good informal signal that the model is picking up genuine taste signal beyond what the formal precision metric captures.

## Limitations

- Single-user dataset (121 rated anime) — small by ML standards; findings are personal, not necessarily generalizable to other users' taste distributions.
- Small test set (30 held-out anime) means the gap between 0.30 and 0.40 precision (3 vs 4 hits) should be read as a promising signal, not a statistically robust result.
- Static dataset snapshot — doesn't include newest releases.
- No true collaborative filtering — recommendations are based solely on my own ratings and item content, not on patterns across other users.

## What I'd try next

- True multi-user collaborative filtering, using a public multi-user MAL ratings dataset alongside personal content-based similarity.
- Extend to accept any user's uploaded MAL export (currently hardcoded to my own list) via a simple frontend.
- Try a larger/more expressive embedding model and compare retrieval quality.
- Re-run against a live, current dataset rather than a static snapshot.

## How to run this yourself

**1. Clone the repo and install dependencies**
```bash
git clone <your-repo-url>
cd anime-recommender
pip install pandas numpy scikit-learn sentence-transformers
```

**2. Get the anime metadata dataset**
- Download the dataset from Kaggle: [Anime Dataset 2026 — 5,000 Titles from MyAnimeList](https://www.kaggle.com/datasets/soulforge/anime-dataset-2026-5000-titles-from-myanimelist)
- Unzip it and place `anime_dataset.csv` in the project's root folder

**3. Export your own MyAnimeList list**
- Log in to MyAnimeList, go to `myanimelist.net/panel.php?go=export`
- Export your anime list (downloads as a `.xml.gz` file)
- Unzip it, then **rename the resulting file to `mal_export.xml`**
- Place it in the project's root folder, alongside `anime_dataset.csv`

**4. Run the script**
```bash
python recommender.py
```
- First run will take a few minutes to generate embeddings (downloads the MiniLM model automatically, then embeds ~4,900 synopses). This is cached to `anime_embeddings.npy` afterward, so future runs are near-instant.
- The script will print your top 10 unified recommendations, plus the evaluation results (precision@10 for the popularity baseline, single-vector, and clustered approaches).

**Note**: this currently only works with *your own* exported MAL list (`my_score`/`mal_id` matching is done against your specific ratings). Supporting arbitrary uploaded lists is a planned extension — see "What I'd try next" below.

## Stack

Python, pandas, numpy, scikit-learn (KMeans, cosine similarity, train_test_split), sentence-transformers (`all-MiniLM-L6-v2`).
