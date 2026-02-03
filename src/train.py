from __future__ import annotations

import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from features import build_training_frame, model_features


def assign_bundle_names(cluster_profiles: pd.DataFrame) -> dict[int, str]:
    bundle_names = [
        "Starter Lite",
        "Voice Essentials",
        "Balanced Connect",
        "Talk & Surf Plus",
        "Unlimited Voice + Data",
    ]

    profiles = cluster_profiles.copy()
    intl_cluster = profiles["Intl usage ratio"].idxmax()

    remaining = profiles.drop(index=intl_cluster).sort_values("Total minutes")
    mapping = {}

    mapping[intl_cluster] = "International Saver"

    for idx, cluster_id in enumerate(remaining.index):
        name = bundle_names[min(idx, len(bundle_names) - 1)]
        mapping[cluster_id] = name

    return mapping


def train(data_path: str, model_path: str) -> None:
    df = pd.read_csv(data_path)
    df = build_training_frame(df)

    X = model_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = {}
    for k in range(3, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)

    # Force six clusters to align with six bundle tiers for UI clarity
    best_k = 6
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(X_scaled)

    df["Cluster"] = final_labels

    cluster_profiles = df.groupby("Cluster")[[
        "Total minutes",
        "Data proxy",
        "Intl usage ratio",
        "Total intl minutes",
        "Total calls",
    ]].mean()

    bundle_map = assign_bundle_names(cluster_profiles)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    projection = pd.DataFrame({
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "cluster": final_labels,
    })

    metadata = {
        "k": best_k,
        "silhouette_scores": scores,
        "bundle_map": bundle_map,
        "cluster_profiles": cluster_profiles.round(2).to_dict(orient="index"),
        "projection": projection.to_dict(orient="records"),
    }

    model_bundle = {
        "scaler": scaler,
        "kmeans": final_kmeans,
        "pca": pca,
        "metadata": metadata,
    }

    joblib.dump(model_bundle, model_path)

    stats = {
        "total_minutes": df["Total minutes"].quantile([0.25, 0.5, 0.75, 0.9]).to_dict(),
        "data_proxy": df["Data proxy"].quantile([0.25, 0.5, 0.75, 0.9]).to_dict(),
        "intl_minutes": df["Total intl minutes"].quantile([0.5, 0.75, 0.9]).to_dict(),
        "intl_ratio": df["Intl usage ratio"].quantile([0.5, 0.75, 0.9]).to_dict(),
        "total_calls": df["Total calls"].quantile([0.25, 0.5, 0.75, 0.9]).to_dict(),
    }

    stats_path = Path(model_path).with_name("training_stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to telecom churn CSV")
    parser.add_argument("--out", default="models/bundle_model.pkl")
    args = parser.parse_args()

    train(args.data, args.out)
