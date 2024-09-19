import os
import pandas
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm.asyncio import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITES_DIR = os.path.join(SCRIPT_DIR, '../output/websites')
WEBSITES_CSV_PATH = os.path.join(SCRIPT_DIR, '../data/car-dealership-websites.csv')
NON_CAR_DEALERSHIP_WEBSITES_PATH = os.path.join(SCRIPT_DIR, '../data/non-car-dealership-websites.txt')

DEALER_WEBSITES_SOURCES = ['DealerInspire', 'DealerCom', 'LiveWebsites', 'TeamVelocity']

def main():
    df = get_training_data()

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['combined_text'],
        df['label'],
        test_size=0.2,
        random_state=42
    )

    # Convert combined text data (URL + content) into TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a simple Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    accuracy = model.score(X_test_tfidf, y_test)
    print(f"Model accuracy: {accuracy:.4f}")

def get_training_data():
    dealer_websites_data = get_dealer_websites_data()
    non_dealer_websites_data = get_non_dealership_website_data()
    return pandas.DataFrame([*dealer_websites_data, *non_dealer_websites_data])

def get_dealer_websites_data():
    websites_csv = pandas.read_csv(WEBSITES_CSV_PATH)
    dealer_website_urls = websites_csv[websites_csv['Source'].isin(DEALER_WEBSITES_SOURCES)]['URL']

    websites_data = []

    with tqdm(desc="Buildling Dealer Websites Training Data", total=len(dealer_website_urls)) as progress_bar:
        for url in dealer_website_urls:
            data = get_website_data(url, True)
            if data is not None: websites_data.append(data)

            progress_bar.update(1)

    progress_bar.close()

    return websites_data

def get_non_dealership_website_data():
    urls = get_non_car_dealership_website_urls()

    website_data = []

    with tqdm(desc="Buildling Nont-Dealer Websites Training Data", total=len(urls)) as progress_bar:
        for url in urls:
            data = get_website_data(url, False)
            if data is not None: website_data.append(data)

            progress_bar.update(1)

    progress_bar.close()

    return website_data


def get_non_car_dealership_website_urls():
    with open(NON_CAR_DEALERSHIP_WEBSITES_PATH, 'r') as file:
        return [url.strip() for url in file.readlines() if url.strip()]

def get_website_data(url, is_dealership_website):
    file_name = url.replace('/', '\\')
    file_path = os.path.join(WEBSITES_DIR, f'{file_name}.html')

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            page_title = soup.find('title') != None and soup.find('title').get_text()
            text = soup.get_text(separator=" ", strip=True)
            combined_content = f"URL: {url} PAGE TITLE: {page_title} CONTENT: {text}"

            return {
                'url': url,
                'combined_text': combined_content,
                'label': 1 if is_dealership_website else 0
            }

    except FileNotFoundError:
        return None

    except OSError:
        return None


if __name__ == '__main__':
    main()