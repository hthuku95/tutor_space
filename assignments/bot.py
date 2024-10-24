import requests
import pdfkit
from bs4 import BeautifulSoup
import re

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def download_page_to_pdf(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        title = soup.title.string if soup.title else 'webpage'
        title = sanitize_filename(title.strip())
        
        output_filename = f"{title}.pdf"
        
        pdfkit.from_string(response.text, output_filename)
        
        print(f"PDF saved successfully as {output_filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
    except Exception as e:
        print(f"Error creating the PDF: {e}")

url = "https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html"
download_page_to_pdf(url)
