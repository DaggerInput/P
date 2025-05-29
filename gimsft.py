import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# Get local Documents/pics folder
documents_path = Path.home() / "Documents"
pics_folder = documents_path / "pics"
pics_folder.mkdir(parents=True, exist_ok=True)

# Image URL (example)
image_url = "https://static.ycharts.com/charts/images/upsell/savedFundChart.a7ac8d23cb54.png"

# Extract filename and extension from URL
parsed_url = urlparse(image_url)
filename = os.path.basename(parsed_url.path)  # e.g., "800px-Hopetoun_falls.jpg"

# Define output path using original filename
output_path = pics_folder / filename

# Download the image and save with original extension
try:
    urllib.request.urlretrieve(image_url, output_path)
    print(f"Image downloaded to: {output_path}")
except Exception as e:
    print(f"failed to download image: {e}")
