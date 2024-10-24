import requests
from bs4 import BeautifulSoup
import os
import urllib.parse

def download_file(url, folder):
    response = requests.get(url)
    if response.status_code == 200:
        filename = os.path.join(folder, os.path.basename(url))
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")

def scrape_upwork_login():
    url = "https://www.upwork.com/ab/account-security/login"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Create folders
    os.makedirs("upwork_login", exist_ok=True)
    os.makedirs("upwork_login/css", exist_ok=True)
    os.makedirs("upwork_login/js", exist_ok=True)
    os.makedirs("upwork_login/images", exist_ok=True)

    # Save HTML
    with open("upwork_login/index.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())

    # Download CSS files
    for link in soup.find_all("link", rel="stylesheet"):
        href = link.get("href")
        if href:
            full_url = urllib.parse.urljoin(url, href)
            download_file(full_url, "upwork_login/css")

    # Download JavaScript files
    for script in soup.find_all("script", src=True):
        src = script.get("src")
        if src:
            full_url = urllib.parse.urljoin(url, src)
            download_file(full_url, "upwork_login/js")

    # Download images
    for img in soup.find_all("img", src=True):
        src = img.get("src")
        if src:
            full_url = urllib.parse.urljoin(url, src)
            download_file(full_url, "upwork_login/images")

if __name__ == "__main__":
    scrape_upwork_login()