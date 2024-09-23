import os

script_dir = os.path.dirname(os.path.abspath(__file__))

WEBSITES_DIR = os.path.join(script_dir, '../output/websites')

CAR_DEALERSHIP_WEBSITES_PATH = os.path.join(script_dir, '../data/car-dealership-websites.csv')
NON_CAR_DEALERSHIP_WEBSITES_PATH = os.path.join(script_dir, '../data/non-car-dealership-websites.txt')
MODEL_PATH = os.path.join(script_dir, '../dist/model.onnx')
VECTORIZER_PATH = os.path.join(script_dir, '../dist/vectorizer.pkl')

CLASSIFICATION_RESULTS_PATH = os.path.join(script_dir, '../output/classification-results.csv')