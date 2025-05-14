import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import os

st.set_page_config(page_title="GSA Music Playlist Recommender", layout="centered")
st.title("\ud83c\udfb6 Spring Vibes Playlist Recommender")

# === Styling with Spring Colors === #
st.markdown("""
    <style>
    body { background-color: #FDF6F0; }
    .stButton>button {
        background-color: #8BC34A;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stSlider > div > div > div > div {
        background: #FF8A65;
    }
    </style>
""", unsafe_allow_html=True)

# === Load Dataset === #
@st.cache_data

def load_data():
    df = pd.read_csv("data.csv")  # Pre-downloaded Spotify dataset
    feature_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
                    'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']
    df = df.dropna(subset=feature_cols + ['name'])
    df = df.reset_index(drop=True)
    return df, feature_cols

df, feature_cols = load_data()

# === Feature Matrix === #
song_features = df[feature_cols].values
scaler = StandardScaler()
song_features = scaler.fit_transform(song_features)

# === Initialize Recommenders === #
if "recommenders" not in st.session_state:
    st.session_state.recommenders = [np.random.rand(len(feature_cols)) for _ in range(5)]  # 5 recommenders

# === Recommend Songs === #
recs = []
for rec in st.session_state.recommenders:
    dists = euclidean_distances([rec], song_features).flatten()
    idx = np.argmin(dists)
    recs.append((idx, df.iloc[idx]))

st.subheader("Your Spring Playlist Picks")
rating_inputs = []
for i, (idx, song) in enumerate(recs):
    st.markdown(f"**Recommender {i+1}:** {song['name']} ({song['artists'] if 'artists' in song else 'N/A'})")
    rating = st.slider(f"Rate this song (Recommender {i+1})", 0, 10, 5, key=f"rating_{i}")
    rating_inputs.append((rating, idx))

# === Update Recommenders Using Ratings === #
if st.button("Generate Next Suggestions"):
    fitness = np.array([r for r, _ in rating_inputs])
    rated_vectors = np.array([song_features[i] for _, i in rating_inputs])

    masses = fitness / (np.sum(fitness) + 1e-6)

    # Replace 0-rated recommenders
    mean_point = np.average(rated_vectors[fitness > 0], axis=0, weights=fitness[fitness > 0])

    new_recs = []
    for i, (rate, _) in enumerate(rating_inputs):
        if rate == 0:
            new_recs.append(mean_point + 0.1 * np.random.randn(len(feature_cols)))
        else:
            force = np.zeros(len(feature_cols))
            for j, m in enumerate(masses):
                if i != j:
                    dist = np.linalg.norm(st.session_state.recommenders[i] - rated_vectors[j]) + 1e-6
                    force += m * (rated_vectors[j] - st.session_state.recommenders[i]) / dist
            accel = force / (masses[i] + 1e-6)
            velocity = np.random.rand(len(feature_cols)) * accel
            new_pos = st.session_state.recommenders[i] + velocity
            new_recs.append(new_pos)

    st.session_state.recommenders = new_recs
    st.experimental_rerun()

# === Playlist Button === #
if st.button("Finalize Playlist"):
    playlist = [df.iloc[i]["name"] for _, i in rating_inputs if _ > 0]
    st.success("\ud83c\udfb6 Your Playlist:")
    for track in playlist:
        st.markdown(f"- {track}")
