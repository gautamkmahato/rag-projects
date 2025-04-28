import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
import time


def load_sitemap(file_path: str):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Handle the XML namespace
    ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    urls = []
    for url in root.findall("ns:url", ns):
        loc = url.find("ns:loc", ns)
        if loc is not None:
            urls.append(loc.text)

    return urls

# Example usage:
sitemap_urls = load_sitemap("C:\\Users\\Gautam Kumar Mahato\\Desktop\\apps\\chai\\RAG\\chai-docs\\sitemap.xml")
site_urls = []
for url in sitemap_urls:
    site_urls.append(url)
    # print(url)

def extract_visible_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Optional: remove script/style content
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        return soup.get_text(separator="\n", strip=True)

    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return ""


def data_utils(site_urls):
    print("Data extraction started...")
    content_list = []

    for url in site_urls:
        site_data = {}
        visible_text = extract_visible_text(url)  # get text content
        site_data["data"] = visible_text
        site_data["url"] = url
        content_list.append(site_data)  # ✅ append to list
        print(f"[info] data extraction completed: {url}")

    print("Successfully completed data extraction")
    return content_list


def create_txt_file(content_list):
    file_name = "chai_docs.txt"

    with open(file_name, "w", encoding="utf-8") as f:
        for item in content_list:
            f.write(f"[URL]: {item['url']}\n")
            f.write(item["data"])
            f.write("\n" + "="*80 + "\n\n")  # separator between pages

    print(f"✅ Text content written to {file_name}")


def run():
    content = data_utils(site_urls)
    create_txt_file(content)

run()
