import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

documents_path = Path.home() / "Documents"
pics_folder = documents_path / "pics"
pics_folder.mkdir(parents=True, exist_ok=True)

image_url = "https://photos5.appleinsider.com/gallery/54910-111271-stocks-xl.jpg"

parsed_url = urlparse(image_url)
filename = os.path.basename(parsed_url.path)
output_path = pics_folder / filename

try:
    req = urllib.request.Request(
        image_url, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print(f"Image downloaded to: {output_path}")
except Exception as e:
    print(f"Failed to download image: {e}")
