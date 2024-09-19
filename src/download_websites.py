import os
import pandas
import aiohttp
import asyncio
from tqdm.asyncio import tqdm

URL_BUCKET_SIZE = 1000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITES_CSV_PATH = os.path.join(SCRIPT_DIR, '../data/car-dealership-websites.csv')
WEBSITES_DIR = os.path.join(SCRIPT_DIR, '../output/websites')
NON_CAR_DEALERSHIP_WEBSITES_PATH = os.path.join(SCRIPT_DIR, '../data/non-car-dealership-websites.txt')

def main():
    asyncio.run(get_website_urls())

async def get_website_urls():
    websites_csv = pandas.read_csv(WEBSITES_CSV_PATH)
    non_car_dealership_website_urls = get_non_car_dealership_website_urls()
    urls = websites_csv['URL'].tolist() + non_car_dealership_website_urls
    bucketed_urls = [urls[i:i+URL_BUCKET_SIZE] for i in range(0, len(urls), URL_BUCKET_SIZE)]

    with tqdm(desc="Downloading pages", total=len(urls)) as progress_bar:
        for index, bucket in enumerate(bucketed_urls):
            await download_page_bucket(bucket, progress_bar)

def get_non_car_dealership_website_urls():
    with open(NON_CAR_DEALERSHIP_WEBSITES_PATH, 'r') as file:
        return [url.strip() for url in file.readlines() if url.strip()]

async def download_page_bucket(urls, progress_bar):
    async with aiohttp.ClientSession() as session:
        tasks = [download_page(session, url) for url in urls]

        for (task, url) in zip(asyncio.as_completed(tasks), urls):
            await task
            progress_bar.update(1)
            progress_bar.write(f"Downloaded {truncate(url)}")

async def download_page(session, url):
    normalized_url = normalize_url(url)
    escaped_url = url.replace("/", '\\')
    file_name = f'{WEBSITES_DIR}/{escaped_url}.html'

    # print(f"| Downloading: {normalized_url}")

    try:
        async with session.get(normalized_url, timeout=5) as response:
            if response.status != 200:
                # print(f"Failed to download page. Status code: {response.status}")
                return

            content = await response.text()

            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(content)

            # print(f">> Page saved to {file_name}")

    except Exception as e:
        # print(f">! Failed to download page {normalized_url}");
        return

def normalize_url(url):
    normalized_url = url.startswith('http') and url or f'https://{url}'
    return normalized_url

def truncate(s):
    if (len(s) < 100):
        return s

    return s[:95] + ' ...'

if __name__ == '__main__':
    main()