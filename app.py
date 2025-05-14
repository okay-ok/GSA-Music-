import os
import kaggle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(page_title="GSA Music Playlist Recommender", layout="centered")
st.title("GSA-based Spotify Playlist Generator")


KAGGLE_USERNAME = st.secrets["KAGGLE_USERNAME"]
KAGGLE_KEY = st.secrets["KAGGLE_KEY"]

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")

with open(kaggle_json, "w") as f:
    f.write(f"{{\"username\": \"{KAGGLE_USERNAME}\", \"key\": \"{KAGGLE_KEY}\"}}")

os.chmod(kaggle_json, 0o600)

if not os.path.exists("data.csv"):
    st.write("Downloading dataset from Kaggle...")
    try:
        kaggle.api.dataset_download_files(
            'zaheenhamidani/ultimate-spotify-tracks-db', 
            path='.', 
            unzip=True
        )
        st.success("✅ Spotify dataset downloaded and extracted!")
    except Exception as e:
        st.error(f"❌ Failed to download dataset: {str(e)}")

# === Load and Prepare Data ===
@st.cache_data

def load_data():
    st.write("Files in working directory:", os.listdir("."))

    df = pd.read_csv("spotify_dataset.csv")
    feature_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
                    'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']
    df = df.dropna(subset=feature_cols + ['name'])
    df = df.reset_index(drop=True)
    return df, feature_cols

df, feature_cols = load_data()
song_features = df[feature_cols].values
scaler = StandardScaler()
song_features = scaler.fit_transform(song_features)


if "recommenders" not in st.session_state:
    st.session_state.recommenders = [np.random.rand(len(feature_cols)) for _ in range(5)]

# === Recommend Songs ===
recs = []
for rec in st.session_state.recommenders:
    dists = euclidean_distances([rec], song_features).flatten()
    idx = np.argmin(dists)
    recs.append((idx, df.iloc[idx]))

st.subheader("Rate These Songs to Tune Your Playlist")
rating_inputs = []
for i, (idx, song) in enumerate(recs):
    st.markdown(f"**Recommender {i+1}:** {song['name']} ({song['artists'] if 'artists' in song else 'N/A'})")
    rating = st.slider(f"Rate this song (Recommender {i+1})", 0, 10, 5, key=f"rating_{i}")
    rating_inputs.append((rating, idx))

# === Update Recommenders Using GSA ===
if st.button("Generate Next Suggestions"):
    fitness = np.array([r for r, _ in rating_inputs])
    rated_vectors = np.array([song_features[i] for _, i in rating_inputs])

    masses = fitness / (np.sum(fitness) + 1e-6)
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


if st.button("Finalize Playlist"):
    playlist = [df.iloc[i]["name"] for _, i in rating_inputs if _ > 0]
    st.success("Your Final Playlist:")
    for track in playlist:
        st.markdown(f"- {track}")
