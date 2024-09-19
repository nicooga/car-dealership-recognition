import pandas
import aiohttp
import asyncio

URL_BUCKET_SIZE = 100
WEBSITES_DIR = 'output/websites'

def main():
    asyncio.run(get_website_urls())

async def get_website_urls():
    websites_csv = pandas.read_csv('data/websites.csv')
    urls = websites_csv['URL']
    bucketed_urls = [urls[i:i+URL_BUCKET_SIZE] for i in range(0, len(urls), URL_BUCKET_SIZE)]

    for index, bucket in enumerate(bucketed_urls):
        print(f"Bucket: {index}")
        await download_page_bucket(bucket)

async def download_page_bucket(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [download_page(session, url) for url in urls]
        await asyncio.gather(*tasks)

async def download_page(session, url):
    normalized_url = normalize_url(url)
    escaped_url = url.replace("/", '\\')
    file_name = f'{WEBSITES_DIR}/{escaped_url}.html'

    print(f"| Downloading: {normalized_url}")

    try:
        async with session.get(normalized_url, timeout=5) as response:
            if response.status != 200:
                print(f"Failed to download page. Status code: {response.status}")
                return

            content = await response.text()

            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(content)

            print(f">> Page saved to {file_name}")

    except Exception as e:
        print(f">! Failed to download page {normalized_url}");

def normalize_url(url):
    normalized_url = url.startswith('http') and url or f'https://{url}'
    return normalized_url

if __name__ == '__main__':
    main()