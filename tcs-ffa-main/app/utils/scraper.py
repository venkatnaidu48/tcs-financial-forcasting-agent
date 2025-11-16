# app/utils/scraper.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

def fetch_screener_docs(ticker_url="https://www.screener.in/company/TCS/consolidated/#documents"):
    
    r = requests.get(ticker_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for a in soup.select("table.documents a"):
        href = a.get("href")
        title = a.text.strip()
        if href:
            url = urljoin(ticker_url, href)
            results.append({"title": title, "url": url})
    return results

def download_file(url, dest_folder="data"):
    os.makedirs(dest_folder, exist_ok=True)
    local_filename = url.split("/")[-1].split("?")[0]
    path = os.path.join(dest_folder, local_filename)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return path
