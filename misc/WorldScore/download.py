import os
import zipfile

import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access environment variables
data_path = os.getenv("DATA_PATH")

# Print or use them
print("Dataset Path:", data_path)

# Define URL from Hugging Face repo
zip_url = "https://huggingface.co/datasets/Howieeeee/WorldScore/resolve/main/WorldScore-Dataset.zip"

# Download ZIP file
zip_path = "WorldScore-Dataset.zip"
with requests.get(zip_url, stream=True) as r:
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print("Dataset ZIP file downloaded successfully.")

# Extract the ZIP file
extract_path = data_path
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

# Delete the ZIP file
os.remove(zip_path)
print("Dataset extracted successfully.")
