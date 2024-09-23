import sys
import re
import os
import aiohttp
import asyncio
import pickle
from tqdm.asyncio import tqdm
import csv
import numpy as np
from bs4 import BeautifulSoup
from onnxruntime import InferenceSession

import constants

BUCKET_SIZE = 100

def main():
    urls_input = sys.stdin.read()
    urls = preprocess_urls(urls_input)
    output_path = constants.CLASSIFICATION_RESULTS_PATH

    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["URL", "is_dealership", "error"])
        writer.writeheader()
        asyncio.run(process_urls(urls, writer))

    print(f"Results saved to {output_path}")

def preprocess_urls(urls_string):
    # Split the input string based on commas, newlines, or spaces
    urls = [url.strip() for url in re.split(r'[,\n\s]+', urls_string) if url.strip()]
    return urls

async def process_urls(urls, writer):
    # Split the URLs into buckets
    buckets = [urls[i:i + BUCKET_SIZE] for i in range(0, len(urls), BUCKET_SIZE)]

    results = []

    with tqdm(desc="Processing URLs", total=len(urls)) as progress_bar:
        for bucket in buckets:
            bucket_results = await process_bucket(bucket, writer, progress_bar)
            results.append(bucket_results)

    return results

async def process_bucket(urls, writer, progress_bar):
    async with aiohttp.ClientSession() as session:
        tasks = [process_url(url, session, writer, progress_bar) for url in urls]
        return await asyncio.gather(*tasks)

async def process_url(url, session, writer, progress_bar):
    html_content = await download_html(url, session)

    if not html_content['ok']:
        writer.writerow({ 'URL': url, 'is_dealership': 0, 'error': html_content['error'] })
        return;

    parsed_content = parse_html(url, html_content['content'])
    is_dealership = test_model(parsed_content)

    progress_bar.update(1)
    progress_bar.write(f"Processed \"{url}\"")

    writer.writerow({ 'URL': url, 'is_dealership': is_dealership })

async def download_html(url, session):
    cache_filename = os.path.join(constants.WEBSITES_DIR, url.replace('/', '\\'))

    if os.path.exists(cache_filename):
        with open(cache_filename, 'r', encoding='utf-8') as file:
            content = file.read()
            return { "ok": True, "content": content }

    try:
        async with session.get(url, timeout=10) as response:
            if response.status != 200:
                return { "ok": False, "error": "Non 200 status" }

            content = await response.text()

            with open(cache_filename, 'w', encoding='utf-8') as file:
                file.write(content)

            if not os.path.exists(cache_filename):
                raise Exception("Failed to download HTML")

            return { "ok": True, "content": content }

    except Exception as e:
        return { "ok": False, "error": str(e) }

def parse_html(url, html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    page_title = soup.find('title') != None and soup.find('title').get_text()
    text = soup.get_text(separator=" ", strip=True)
    return f"URL: {url} PAGE TITLE: {page_title} CONTENT: {text}"

def load_model():
    return InferenceSession(constants.MODEL_PATH)

# Load ONNX model
model = load_model()

# Load the pre-fitted vectorizer
with open(constants.VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

def test_model(content):
    # Transform the content to TF-IDF features
    features = vectorizer.transform([content])

    # Convert the sparse matrix to a dense numpy array
    features = features.toarray().astype(np.float32)

    # Run inference using ONNX runtime
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    prediction = model.run([output_name], {input_name: features.astype(np.float32)})[0]

    return prediction[0]

if __name__ == '__main__':
    main()
