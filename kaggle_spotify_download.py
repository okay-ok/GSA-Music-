import os
import kaggle

# Set Kaggle credentials from environment or input
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Save credentials
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")

with open(kaggle_json, "w") as f:
    f.write(f"{{\"username\": \"{KAGGLE_USERNAME}\", \"key\": \"{KAGGLE_KEY}\"}}")

os.chmod(kaggle_json, 0o600)

# Download the Spotify Tracks Dataset
kaggle.api.dataset_download_files(
    'zaheenhamidani/ultimate-spotify-tracks-db', 
    path='.', 
    unzip=True
)

print("✅ Spotify dataset downloaded and extracted!")
