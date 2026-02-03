from __future__ import annotations

import joblib
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from features import add_features, model_features

MODEL_PATH = BASE_DIR / "models" / "bundle_model.pkl"
STATS_PATH = BASE_DIR / "models" / "training_stats.json"
DATA_PATH = BASE_DIR / "data" / "telecom_churn.csv"

BUNDLE_CATALOG = {
    "Starter Lite": {
        "voice": "100 mins",
        "data": "1 GB",
        "message": "Best for light usage and budget-friendly plans.",
    },
    "Voice Essentials": {
        "voice": "250 mins",
        "data": "2 GB",
        "message": "Great for callers who still want basic data access.",
    },
    "Balanced Connect": {
        "voice": "400 mins",
        "data": "4 GB",
        "message": "Balanced voice and data for everyday users.",
    },
    "Talk & Surf Plus": {
        "voice": "700 mins",
        "data": "8 GB",
        "message": "For active users who need more streaming and calls.",
    },
    "Unlimited Voice + Data": {
        "voice": "Unlimited",
        "data": "20 GB",
        "message": "Power users who need always-on connectivity.",
    },
    "International Saver": {
        "voice": "300 mins + intl discount",
        "data": "3 GB",
        "message": "Ideal for frequent international callers.",
    },
}

st.set_page_config(page_title="Smart Bundle Recommendation", layout="centered")

st.title("Smart Bundle Recommendation")
st.write("Enter a customer's recent usage to get bundle suggestions.")

if not MODEL_PATH.exists():
    st.warning(
        "Model not found. Train it first by running `python src/train.py --data data/telecom_churn.csv`."
    )

st.subheader("Lookup Mode")
st.caption("Search for an existing customer by phone number, or switch to manual input.")

mode = st.radio("Choose input method", ["Phone number lookup", "Manual entry"], horizontal=True)

prefill = None
if mode == "Phone number lookup":
    phone_number = st.text_input("Phone number from dataset", placeholder="e.g., 415-555-1234")
    if phone_number and DATA_PATH.exists():
        df_data = pd.read_csv(DATA_PATH)
        df_data.columns = [c.strip() for c in df_data.columns]
        phone_col = "phone number" if "phone number" in df_data.columns else "Phone number"
        if phone_col in df_data.columns:
            match = df_data[df_data[phone_col].astype(str).str.strip() == phone_number.strip()]
            if match.empty:
                st.info("Phone number not found. You can use manual entry below.")
            else:
                prefill = match.iloc[0]
                st.success("Phone number found. Usage fields auto-filled.")
        else:
            st.warning("Phone number column not found in dataset.")

st.subheader("What-if Sliders")
st.caption("Adjust the sliders to see how recommendations shift in real time.")

col1, col2 = st.columns(2)

with col1:
    account_length = st.slider(
        "Account length (days)", 1, 365, int(prefill["account length"]) if prefill is not None else 120, 1
    )
    day_minutes = st.slider(
        "Total day minutes", 0.0, 350.0, float(prefill["total day minutes"]) if prefill is not None else 160.0, 1.0
    )
    day_calls = st.slider(
        "Total day calls", 0, 200, int(prefill["total day calls"]) if prefill is not None else 100, 1
    )
    eve_minutes = st.slider(
        "Total eve minutes", 0.0, 350.0, float(prefill["total eve minutes"]) if prefill is not None else 180.0, 1.0
    )
    eve_calls = st.slider(
        "Total eve calls", 0, 200, int(prefill["total eve calls"]) if prefill is not None else 80, 1
    )
    night_minutes = st.slider(
        "Total night minutes", 0.0, 350.0, float(prefill["total night minutes"]) if prefill is not None else 200.0, 1.0
    )

with col2:
    night_calls = st.slider(
        "Total night calls", 0, 200, int(prefill["total night calls"]) if prefill is not None else 90, 1
    )
    intl_minutes = st.slider(
        "Total intl minutes", 0.0, 20.0, float(prefill["total intl minutes"]) if prefill is not None else 8.0, 0.5
    )
    intl_calls = st.slider(
        "Total intl calls", 0, 20, int(prefill["total intl calls"]) if prefill is not None else 4, 1
    )
    vmail_messages = st.slider(
        "Voice mail messages", 0, 50, int(prefill["number vmail messages"]) if prefill is not None else 15, 1
    )
    service_calls = st.slider(
        "Customer service calls", 0, 10, int(prefill["customer service calls"]) if prefill is not None else 1, 1
    )
    intl_plan = st.selectbox(
        "International plan", ["No", "Yes"], index=1 if prefill is not None and str(prefill["international plan"]).lower() == "yes" else 0
    )
    vmail_plan = st.selectbox(
        "Voice mail plan", ["No", "Yes"], index=1 if prefill is not None and str(prefill["voice mail plan"]).lower() == "yes" else 0
    )

submitted = True

def format_percentile(value: float, ref: dict[str, float]) -> str:
    if value >= ref.get("0.9", float("inf")):
        return "top 10%"
    if value >= ref.get("0.75", float("inf")):
        return "top 25%"
    if value >= ref.get("0.5", float("inf")):
        return "above median"
    if value >= ref.get("0.25", float("inf")):
        return "below median"
    return "bottom 25%"


def build_explanation(row: pd.Series, stats: dict) -> list[str]:
    reasons = []
    if stats:
        reasons.append(
            f"Total minutes is {format_percentile(row['Total minutes'], stats['total_minutes'])} "
            f"({row['Total minutes']:.0f} mins)."
        )
        reasons.append(
            f"Data proxy is {format_percentile(row['Data proxy'], stats['data_proxy'])} "
            f"({row['Data proxy']:.0f} mins)."
        )
        reasons.append(
            f"International minutes are {format_percentile(row['Total intl minutes'], stats['intl_minutes'])} "
            f"({row['Total intl minutes']:.1f} mins)."
        )
        reasons.append(
            f"International ratio is {format_percentile(row['Intl usage ratio'], stats['intl_ratio'])} "
            f"({row['Intl usage ratio']:.2%})."
        )
    else:
        reasons.append("Model explanation uses usage totals and international activity.")
    return reasons


if submitted and MODEL_PATH.exists():
    row = {
        "Account length": account_length,
        "Total day minutes": day_minutes,
        "Total day calls": day_calls,
        "Total eve minutes": eve_minutes,
        "Total eve calls": eve_calls,
        "Total night minutes": night_minutes,
        "Total night calls": night_calls,
        "Total intl minutes": intl_minutes,
        "Total intl calls": intl_calls,
        "Number vmail messages": vmail_messages,
        "Customer service calls": service_calls,
        "International plan": 1 if intl_plan == "Yes" else 0,
        "Voice mail plan": 1 if vmail_plan == "Yes" else 0,
    }

    df = pd.DataFrame([row])
    df = add_features(df)
    features = model_features(df)

    model_bundle = joblib.load(MODEL_PATH)
    scaler = model_bundle["scaler"]
    kmeans = model_bundle["kmeans"]
    pca = model_bundle.get("pca")
    metadata = model_bundle.get("metadata", {})
    bundle_map = {int(k): v for k, v in metadata.get("bundle_map", {}).items()}
    cluster_profiles = metadata.get("cluster_profiles", {})

    X_scaled = scaler.transform(features)
    distances = kmeans.transform(X_scaled)[0]
    ranked_clusters = sorted(
        [(idx, dist) for idx, dist in enumerate(distances)],
        key=lambda x: x[1],
    )

    def similarity(dist: float) -> float:
        return 1 / (1 + dist)

    def bundle_explanation(profile: dict, row: pd.Series) -> list[str]:
        reasons = []
        if not profile:
            return ["Based on similarity to users with comparable usage patterns."]

        comparisons = [
            ("Total minutes", "overall calling"),
            ("Data proxy", "data-like usage"),
            ("Intl usage ratio", "international calling"),
            ("Total intl minutes", "international minutes"),
            ("Total calls", "number of calls"),
        ]

        for key, label in comparisons:
            user_val = row.get(key, 0)
            cluster_val = profile.get(key, 0)
            if cluster_val == 0:
                continue
            diff = (user_val - cluster_val) / cluster_val
            if diff >= 0.25:
                reasons.append(f"Your {label} is higher than this bundle’s average.")
            elif diff <= -0.25:
                reasons.append(f"Your {label} is lower than this bundle’s average.")

        if not reasons:
            reasons.append("Your usage is close to this bundle’s average profile.")

        return reasons[:3]

    top_clusters = ranked_clusters[:3]
    other_clusters = ranked_clusters[3:]

    st.subheader("Top 3 Recommendations")
    for cluster_id, dist in top_clusters:
        label = bundle_map.get(cluster_id, f"Cluster {cluster_id}")
        bundle = BUNDLE_CATALOG.get(label, {})
        score = similarity(dist)
        st.markdown(f"**{label}** — {score:.0%} match")
        st.write(f"Voice: {bundle.get('voice', '—')} | Data: {bundle.get('data', '—')}")
        st.caption(bundle.get("message", ""))
        profile = cluster_profiles.get(str(cluster_id)) or cluster_profiles.get(cluster_id)
        for reason in bundle_explanation(profile, df.iloc[0]):
            st.write(f"- {reason}")

    with st.expander("Other Bundles (ranked)"):
        for cluster_id, dist in other_clusters:
            label = bundle_map.get(cluster_id, f"Cluster {cluster_id}")
            bundle = BUNDLE_CATALOG.get(label, {})
            score = similarity(dist)
            st.markdown(f"**{label}** — {score:.0%} match")
            st.write(f"Voice: {bundle.get('voice', '—')} | Data: {bundle.get('data', '—')}")
            profile = cluster_profiles.get(str(cluster_id)) or cluster_profiles.get(cluster_id)
            for reason in bundle_explanation(profile, df.iloc[0]):
                st.write(f"- {reason}")

    st.divider()
    st.subheader("Why this recommendation?")
    stats = None
    if STATS_PATH.exists():
        with STATS_PATH.open("r", encoding="utf-8") as f:
            stats = json.load(f)

    explanation = build_explanation(df.iloc[0], stats or {})
    st.write(
        "The model compares your usage against historical profiles and picks bundles that "
        "match similar voice, data-proxy, and international usage patterns."
    )
    for line in explanation:
        st.write(f"- {line}")

    if ranked_clusters:
        best_cluster = ranked_clusters[0][0]
        profile = cluster_profiles.get(str(best_cluster)) or cluster_profiles.get(best_cluster)
        if profile:
            st.subheader("Closest Cluster Profile")
            st.write(
                f"Average Total Minutes: {profile['Total minutes']:.0f} | "
                f"Data Proxy: {profile['Data proxy']:.0f} | "
                f"Intl Ratio: {profile['Intl usage ratio']:.2%} | "
                f"Intl Minutes: {profile['Total intl minutes']:.1f}"
            )

    projection = metadata.get("projection", [])
    if projection and pca is not None:
        st.subheader("Cluster Visualization (PCA)")
        proj_df = pd.DataFrame(projection)
        user_coords = pca.transform(X_scaled)[0]
        user_point = pd.DataFrame({
            "pc1": [user_coords[0]],
            "pc2": [user_coords[1]],
            "cluster": ["You"],
        })
        proj_df["cluster"] = proj_df["cluster"].astype(str)
        fig = px.scatter(
            proj_df,
            x="pc1",
            y="pc2",
            color="cluster",
            opacity=0.5,
            title="Customer Clusters (PCA)",
        )
        fig.add_scatter(
            x=user_point["pc1"],
            y=user_point["pc2"],
            mode="markers",
            marker=dict(size=14, color="#ffc857", line=dict(width=2, color="#111111")),
            name="You",
        )
        st.plotly_chart(fig, use_container_width=True)
