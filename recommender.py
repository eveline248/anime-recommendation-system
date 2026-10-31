import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

#cleaning dataset
df = pd.read_csv("anime_dataset.csv")
df = df[df["is_adult"] == False]
df = df.dropna(subset=["description"]) 
df["genres_list"] = df["genres"].str.split("|")
df["title"] = df["title_english"].fillna(df["title_romaji"])

df = df.reset_index(drop=True) #resets the index in order

#generate or load embeddings
EMBEDDINGS_PATH = "anime_embeddings.npy"

if os.path.exists(EMBEDDINGS_PATH):
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)


#parse MAL XML
def parse_mal_export(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for anime in root.findall("anime"):
        records.append({
            "mal_id": anime.findtext("series_animedb_id"),
            "title": anime.findtext("series_title"),
            "my_score": anime.findtext("my_score"),
            "my_status": anime.findtext("my_status"),
        })

    my_list = pd.DataFrame(records)
    my_list["mal_id"] = pd.to_numeric(my_list["mal_id"], errors="coerce")
    my_list["my_score"] = pd.to_numeric(my_list["my_score"], errors="coerce")
    return my_list

my_list = parse_mal_export("mal_export.xml")

#filter only scored anime
my_list_rated = my_list[my_list["my_score"] > 0]
print("my rated anime:", my_list_rated.shape)

#combine both dataframes
rated_df = df.merge(my_list_rated, on="mal_id", how="inner")
rated_indices = df.index[df["mal_id"].isin(rated_df["mal_id"])].tolist()
rated_embeddings = embeddings[rated_indices]

#take account of popularity
df["popularity_score"] = np.log1p(df["members"]) #using log to not have a big gap which affects the similarity index
df["popularity_norm"] = (df["popularity_score"] - df["popularity_score"].min()) / (df["popularity_score"].max() - df["popularity_score"].min())
already_rated_ids = set(rated_df["mal_id"])

#clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(rated_embeddings)

rated_df["cluster"] = cluster_labels

#combine cluster into one ranking
def get_unified_recommendation(top_n=10, per_cluster_pool=5):
    all_candidates = []

    for c in range(n_clusters):
        cluster_anime = rated_df[rated_df["cluster"] == c]
        cluster_size = len(cluster_anime)

        cluster_indices = df.index[df["mal_id"].isin(cluster_anime["mal_id"])].tolist()
        cluster_embeddings =  embeddings[cluster_indices]
        cluster_weights = cluster_anime["my_score"].values

        cluster_taste_vector = np.average(cluster_embeddings, axis=0, weights=cluster_weights)
        cluster_similarities = cosine_similarity(cluster_taste_vector.reshape(1, -1), embeddings)[0]

        temp_df = df.copy()
        temp_df["cluster_similarity"] = cluster_similarities
        temp_df["cluster_similarity_norm"] = (
            (temp_df["cluster_similarity"] - temp_df["cluster_similarity"].min())
            / (temp_df["cluster_similarity"].max() - temp_df["cluster_similarity"].min())
        )

        temp_df["blended_score"] = 0.75 * temp_df["cluster_similarity_norm"] + 0.25 * temp_df["popularity_norm"]
        temp_df["cluster"] = c

        candidates = temp_df[~temp_df["mal_id"].isin(already_rated_ids)].sort_values(
            "blended_score", ascending=False
        ).head(per_cluster_pool)

        all_candidates.append(candidates)

    combined = pd.concat(all_candidates)
    combined = combined.sort_values("blended_score", ascending=False).drop_duplicates(subset="title")

    return combined[["title", "blended_score", "cluster", "genres"]].head(top_n)

print(get_unified_recommendation())

#evaluation for clustering
def evaluate_clustered(top_k=10, per_cluster_pool=5):
    strong_likes = rated_df[rated_df["my_score"] >= 8]
    train_ids, test_ids = train_test_split(
        strong_likes["mal_id"], test_size=0.3, random_state=42
    )

    train_rated_df = rated_df[rated_df["mal_id"].isin(train_ids)].copy()
    train_rated_ids = set(train_rated_df["mal_id"])

    train_indices = df.index[df["mal_id"].isin(train_rated_df["mal_id"])].tolist()
    train_embeddings = embeddings[train_indices]

    train_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_rated_df["cluster"] = train_kmeans.fit_predict(train_embeddings)

    all_candidates = []

    for c in range(n_clusters):
        cluster_anime = train_rated_df[train_rated_df["cluster"] == c]

        c_indices = df.index[df["mal_id"].isin(cluster_anime["mal_id"])].tolist()
        c_embeddings = embeddings[c_indices]
        c_weights = cluster_anime["my_score"].values

        c_taste_vector = np.average(c_embeddings, axis=0, weights=c_weights)
        c_similarities = cosine_similarity(c_taste_vector.reshape(1, -1), embeddings)[0]

        temp_df = df.copy()
        temp_df["cluster_similarity"] = c_similarities
        temp_df["cluster_similarity_norm"] = (
            (temp_df["cluster_similarity"] - temp_df["cluster_similarity"].min())
            / (temp_df["cluster_similarity"].max() - temp_df["cluster_similarity"].min())
        )
        temp_df["blended_score"] = 0.75 * temp_df["cluster_similarity_norm"] + 0.25 * temp_df["popularity_norm"]

        candidates = temp_df[~temp_df["mal_id"].isin(train_rated_ids)].sort_values(
            "blended_score", ascending=False
        ).head(per_cluster_pool)

        all_candidates.append(candidates)

    combined = pd.concat(all_candidates)
    combined = combined.sort_values("blended_score", ascending=False).drop_duplicates(subset="title")
    top_recs = combined.head(top_k)

    hits = top_recs["mal_id"].isin(test_ids).sum()
    clustered_precision = hits / top_k

    print(f"Clustered (unified) - hits in top {top_k}: {hits}")
    print(f"Clustered (unified) precision@{top_k}: {clustered_precision:.2f}")

    return clustered_precision

clustered_precision = evaluate_clustered()

#DEBUG
# def get_cluster_recommendations(cluster_id, top_n=10):
#     cluster_anime = rated_df[rated_df["cluster"] == cluster_id]

#     cluster_indices = df.index[df["mal_id"].isin(cluster_anime["mal_id"])].tolist()
#     cluster_embeddings = embeddings[cluster_indices]
#     cluster_weights = cluster_anime["my_score"].values

#     cluster_taste_vector = np.average(cluster_embeddings, axis=0, weights=cluster_weights)
#     cluster_similarities = cosine_similarity(cluster_taste_vector.reshape(1, -1), embeddings)[0]

#     temp_df = df.copy()
#     temp_df["cluster_similarity"] = cluster_similarities

#     recs = temp_df[~temp_df["mal_id"].isin(already_rated_ids)].sort_values(
#         "cluster_similarity", ascending=False
#     ).head(top_n)
    
#     return recs[["title", "cluster_similarity", "genres"]]

# for c in range(n_clusters):
#     print(f"\n=== Recommendations based on Cluster {c} ===")
#     print(get_cluster_recommendations(c))