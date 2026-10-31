# Anime Recommender: Content-Based Recommendations from My Own MAL Ratings

A personal project exploring whether a content-based recommender (built from anime synopsis embeddings) can match *my own* taste better than simple baselines, using my real MyAnimeList rating history as ground truth.

## Motivation

MyAnimeList's built-in recommendations often felt generic, driven heavily by "users who liked X also liked Y" collaborative signals rather than understanding *why* I liked something. This project tests an alternative: represent each anime by the meaning of its synopsis (via text embeddings) and build a recommender purely from content similarity, rather than aggregate user behavior.

## Data

- **Anime metadata**: ~4,900 titles from a Kaggle-hosted MyAnimeList dataset (synopsis, genres, tags, studio, score, member count).
- **Personal ratings**: my own MAL list, exported as XML (128 rated titles, 121 of which matched the Kaggle dataset).

## Method

1. **Cleaning**: dropped adult-flagged titles and rows with missing synopses (7 out of ~4,985, checked individually and confirmed they were genuinely missing on MAL too, not a parsing bug).
2. **Embeddings**: generated 384-dimensional sentence embeddings for every synopsis using `all-MiniLM-L6-v2` (a small, fast, local sentence-transformer model, chosen to avoid API costs/rate limits and keep the pipeline fully offline-capable). Cached to disk after first generation to avoid recomputation.
3. **Taste vector**: computed a weighted average of the embeddings of my rated anime, weighted by my score, so higher-rated anime pull the vector more strongly toward their "direction" in embedding space.
4. **Popularity blending**: combined content similarity with a normalized, log-transformed member count (75% similarity / 25% popularity), to avoid surfacing thematically matching but extremely obscure titles.
5. **Clustering (key iteration)**: my rated anime span distinct genre profiles (Action, Drama, Comedy, Romance, Suspense and Sports are all represented in meaningful numbers), so a single averaged taste vector likely blurs distinct preferences together. I used K-Means on the synopsis embeddings of my rated anime to split them into taste "clusters", built a separate taste vector per cluster, then merged each cluster's top candidates into one final ranked list (capped per cluster so one large cluster can't dominate). I use n=3 clusters; results for n=2, 3 and 4 were similar (see Evaluation).

## Evaluation

I used a hold-out methodology: I split my highly rated anime (score ≥ 8, 100 titles) into 70 train and 30 test. I rebuilt the taste vector(s) using only the train set, then checked whether the held-out anime, which the model had never seen, appeared in its top 10 recommendations. This is measured as **precision@10**.

A single split of only 30 test anime is very noisy (one extra hit moves the score by 0.10), so I repeated the whole procedure over **50 random splits** (`evaluate_splits.py`) and report the mean ± standard deviation.

| Approach | Precision@10 (mean ± std, 50 splits) |
| --- | --- |
| Popularity baseline (most-members-first) | 0.304 ± 0.100 |
| Single taste vector (no clustering) | 0.186 ± 0.100 |
| Clustered, n=2 | 0.314 ± 0.125 |
| **Clustered, n=3** | **0.338 ± 0.118** |
| Clustered, n=4 | 0.324 ± 0.121 |

**Random-chance baseline** (for context): ~0.006 (about 30 held-out positives among ~4,900 titles).

Paired against the popularity baseline on the same splits, clustered n=3 won 19 splits, tied 20 and lost 11 (mean difference +0.034).

### Takeaways

- Clustering clearly improves on a single taste vector (0.19 → 0.34), which supports the idea that one averaged vector blurs together different tastes.
- The advantage of clustering over the popularity baseline is small (+0.03) and within the split-to-split noise, so I read it as "comparable to, possibly slightly better than, popularity" rather than a clear win.
- n=2, 3 and 4 perform within noise of each other, so n=3 is a reasonable choice, not a sharply tuned optimum.
- An earlier single-split run showed 0.40 for n=2 (and 0.10 for n=3). Averaging over 50 splits shows that was a favourable split for n=2 and an unfavourable one for n=3, which is why I switched to repeated evaluation.

### Qualitative check (informal)

Several top recommendations from the final model were anime I had actually watched or partially watched but never logged or scored on MAL (for example, dropped partway through). This is an informal signal that the model picks up some real taste signal beyond what the formal precision metric captures.

## Limitations

- Single-user dataset (121 matched rated anime) is small by ML standards; findings are personal and may not generalize to other people's taste distributions.
- The 30-title test set is small, and the 50 splits overlap heavily, so the reported spread is a rough guide, not a formal confidence interval.
- Part of the performance comes from the 25% popularity blend; the blend weights were not tuned.
- Static dataset snapshot: it doesn't include the newest releases.
- No true collaborative filtering: recommendations are based solely on my own ratings and item content, not on patterns across other users.

## What I'd try next

- Vary the similarity/popularity blend (for example 100/0 up to 50/50) to measure how much of the score comes from popularity.
- True multi-user collaborative filtering, using a public multi-user MAL ratings dataset alongside content-based similarity.
- Accept any user's uploaded MAL export (currently the script is set up for my own list) via a simple frontend, and test whether the results hold on other people's lists.
- Try a larger or more expressive embedding model and compare retrieval quality.
- Re-run against a live, current dataset rather than a static snapshot.

## How to run this yourself

**1. Clone the repo and install dependencies**

```
git clone https://github.com/eveline248/anime-recommendation-system.git
cd anime-recommendation-system
pip install pandas numpy scikit-learn sentence-transformers
```

**2. Get the anime metadata dataset**

- Download the dataset from Kaggle: [Anime Dataset 2026: 5,000 Titles from MyAnimeList](https://www.kaggle.com/datasets/soulforge/anime-dataset-2026-5000-titles-from-myanimelist)
- Unzip it and place `anime_dataset.csv` in the project's root folder.

**3. Export your own MyAnimeList list**

- Log in to MyAnimeList and go to `myanimelist.net/panel.php?go=export`.
- Export your anime list (it downloads as a `.xml.gz` file).
- Unzip it, then **rename the resulting file to `mal_export.xml`**.
- Place it in the project's root folder, alongside `anime_dataset.csv`.

**4. Run the recommender**

```
python recommender.py
```

- The first run takes a few minutes to generate embeddings (it downloads the MiniLM model automatically, then embeds ~4,900 synopses). They are cached to `anime_embeddings.npy`, so later runs are near-instant.
- The script prints your top 10 unified recommendations and the precision@10 of a single train/test split.

**5. Reproduce the evaluation table**

```
python evaluate_splits.py
```

- Runs 50 random train/test splits and prints the mean and spread of precision@10 for the popularity baseline, the single taste vector, and clustering with n=2, 3 and 4. Per-split results are saved to `split_results.csv`.

**Note**: this currently only works with *your own* exported MAL list (`my_score` and `mal_id` matching is done against your specific ratings). Supporting arbitrary uploaded lists is a planned extension (see "What I'd try next" above).

## Files

- `recommender.py`: builds embeddings, clusters your taste, prints recommendations, runs one hold-out evaluation.
- `evaluate_splits.py`: repeated-split evaluation of all methods.

## Stack

Python, pandas, numpy, scikit-learn (KMeans, cosine similarity, train_test_split), sentence-transformers (`all-MiniLM-L6-v2`).