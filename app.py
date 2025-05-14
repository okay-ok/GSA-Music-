import os
import kaggle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card

st.set_page_config(page_title="GSA Music Playlist Recommender", layout="centered")
st.markdown("""
    <style>
    .stSlider > div[data-baseweb="slider"] > div {
        background: linear-gradient(90deg, #DD0000, #75E6DA, #189AB4);
        border-radius: 0.1rem;
        padding: 0.1rem;
    }
    .duplicate-warning {
        background-color: #FFF3CD;
        border-left: 6px solid #FFA500;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

colored_header("🎶 GSA-based Spotify Playlist Generator", description="Rate songs to build your perfect playlist!", color_name="green-70")

# === Download and Extract Dataset from Kaggle ===
KAGGLE_USERNAME = st.secrets["KAGGLE_USERNAME"]
KAGGLE_KEY = st.secrets["KAGGLE_KEY"]

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")

with open(kaggle_json, "w") as f:
    f.write(f"{{\"username\": \"{KAGGLE_USERNAME}\", \"key\": \"{KAGGLE_KEY}\"}}")

os.chmod(kaggle_json, 0o600)

if not os.path.exists("spotify_dataset.csv"):
    st.write("Downloading dataset from Kaggle...")
    try:
        kaggle.api.dataset_download_files(
            'zaheenhamidani/ultimate-spotify-tracks-db', 
            path='.', 
            unzip=True
        )
        st.success("Spotify dataset downloaded and extracted!")
    except Exception as e:
        st.error(f"Failed to download dataset: {str(e)}")

# === Load and Prepare Data ===
@st.cache_data
def load_data():
    file_candidates = [f for f in os.listdir(".") if f.endswith(".csv")]
    if not file_candidates:
        st.error("No CSV file found after download!")
        return None, []
    df = pd.read_csv(file_candidates[0])
    feature_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
                    'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']
    df = df.dropna(subset=feature_cols + ['track_name'])
    df = df.reset_index(drop=True)
    return df, feature_cols

df, feature_cols = load_data()
if df is None:
    st.stop()

song_features = df[feature_cols].values
scaler = StandardScaler()
song_features = scaler.fit_transform(song_features)

# === Initialize Recommenders ===
if "recommenders" not in st.session_state:
    st.session_state.recommenders = [np.random.rand(len(feature_cols)) for _ in range(5)]

# === Recommend Songs ===
recs = []
for rec in st.session_state.recommenders:
    dists = euclidean_distances([rec], song_features).flatten()
    idx = np.argmin(dists)
    recs.append((idx, df.iloc[idx]))

# Highlight duplicate songs if any
song_indices = [idx for idx, _ in recs]
if len(song_indices) != len(set(song_indices)):
    st.markdown('<div class="duplicate-warning">⚠️ Duplicate entry found in recommendations!</div>', unsafe_allow_html=True)

st.subheader("✨ Rate These Songs to Tune Your Playlist")
rating_inputs = []

for i, (idx, song) in enumerate(recs):
    with st.container():
        st.markdown(f"**🎵 Recommender {i+1}:** {song['track_name']} — *{song['artist_name'] if 'artist_name' in song else 'N/A'}*")
        rating = st.slider("Rating (0-10)", 0, 10, 5, key=f"rating_{i}")
        rating_inputs.append((rating, idx))

# === Update Recommenders Using GSA ===
if st.button("🚀 Generate Next Suggestions"):
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
    st.rerun()

# === Finalize Playlist ===
if st.button("✅ Finalize Playlist"):
    playlist = [df.iloc[i]["track_name"] for _, i in rating_inputs if _ > 0]
    st.success("🎵 Your Final Playlist:")
    for track in playlist:
        st.markdown(f"- {track}")
