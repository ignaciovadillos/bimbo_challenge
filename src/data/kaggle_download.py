import kagglehub
import os
import zipfile
from pathlib import Path

# Set download path to data/raw
download_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')

# Create directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Download latest version
path = kagglehub.competition_download('grupo-bimbo-inventory-demand', path=download_path)

print("Path to competition files:", path)

# Find and unpack zip files
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if file.endswith('.zip') and os.path.isfile(file_path):
        print(f"Unpacking {file}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        # Optionally remove the zip file after extraction
        os.remove(file_path)
        print(f"Removed {file}")

print("Download and extraction complete!")