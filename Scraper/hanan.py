import os
import sys
import torch
import requests
from bs4 import BeautifulSoup
import re
import string
import nltk
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertModel
from transformers import logging
from tabulate import tabulate

# Suppress specific warnings from transformers
logging.set_verbosity_error()

nltk.download('punkt')
nltk.download('stopwords')

# Download and process website
def download_website(url, output_dir='downloaded_html'):
    try:
        filename = os.path.join(output_dir, f"{url.replace('https://', '').replace('/', '_')}.html")

        if os.path.exists(filename):
            print(f"File for {url} already exists. Reading from the local file.")
            return filename
        else:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                os.makedirs(output_dir, exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(response.text)
                return filename
            else:
                print(f"Failed to download {url}")
                return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

# Process the HTML file
def process_html_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text()
            return text.strip()
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# Tokenize and preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    sentence_embedding = torch.mean(last_hidden_states, dim=1)
    return sentence_embedding

# Manually labeled URLs
# Manually labeled URLs
urls_with_labels = [
    ('https://www.kompas.com/', "News"),
    ('https://www.liputan6.com/', "News"),
    ('https://www.cnnindonesia.com/', "News"),
    ('https://www.detik.com/', "News"),
    ('https://www.tempo.co/', "News"),
    ('https://kumparan.com/', "News"),
    ('https://www.tirto.id/', "News"),

    ('https://www.zenius.net/', "Education"),
    ('https://www.ruangguru.com/', "Education"),
    ('https://www.coursera.org/', "Education"),
    ('https://www.udemy.com/', "Education"),
    ('https://www.edx.org/', "Education"),
    ('https://www.khanacademy.org/', "Education"),

    ('https://www.tokopedia.com/', "E-commerce"),
    ('https://www.shopee.co.id/', "E-commerce"),
    ('https://www.bukalapak.com/', "E-commerce"),
    ('https://www.amazon.com/', "E-commerce"),
    ('https://www.ebay.com/', "E-commerce"),

    ('https://www.netflix.com/', "Entertainment"),
    ('https://www.spotify.com/', "Entertainment"),
    ('https://www.hulu.com/', "Entertainment"),
    ('https://www.twitch.tv/', "Entertainment"),
    ('https://www.crunchyroll.com/', "Entertainment"),

    ('https://www.facebook.com/', "Social Media"),
    ('https://www.instagram.com/', "Social Media"),
    ('https://www.twitter.com/', "Social Media"),
    ('https://www.snapchat.com/', "Social Media"),
    ('https://www.tiktok.com/', "Social Media"),

    ('https://www.kaskus.co.id/', "Forums & Communities"),
    ('https://www.quora.com/', "Forums & Communities"),
    ('https://www.stackoverflow.com/', "Forums & Communities"),
    ('https://www.reddit.com/', "Forums & Communities"),

    ('https://www.github.com/', "Technology"),
    ('https://www.techcrunch.com/', "Technology"),
    ('https://www.wired.com/', "Technology"),
    ('https://www.theverge.com/', "Technology"),
    ('https://www.hackerrank.com/', "Technology"),

    ('https://www.linkedin.com/', "Professional & Career"),
    ('https://www.glassdoor.com/', "Professional & Career"),
    ('https://www.indeed.com/', "Professional & Career"),
    ('https://www.monster.com/', "Professional & Career"),

    ('https://www.webmd.com/', "Health & Wellness"),
    ('https://www.healthline.com/', "Health & Wellness"),
    ('https://www.mayoclinic.org/', "Health & Wellness"),
    ('https://www.fitbit.com/', "Health & Wellness"),

    ('https://www.traveloka.com/', "Travel & Tourism"),
    ('https://www.agoda.com/', "Travel & Tourism"),
    ('https://www.tripadvisor.com/', "Travel & Tourism"),
    ('https://www.expedia.com/', "Travel & Tourism"),
    ('https://www.booking.com/', "Travel & Tourism"),

    ('https://www.bloomberg.com/', "Finance & Investment"),
    ('https://www.yahoo.com/finance', "Finance & Investment"),
    ('https://www.forbes.com/', "Finance & Investment"),
    ('https://www.cnbc.com/', "Finance & Investment"),
    ('https://www.investing.com/', "Finance & Investment"),

    ('https://www.bet365.com/', "Gambling"),
    ('https://www.pokerstars.com/', "Gambling"),
    ('https://www.888casino.com/', "Gambling"),
    ('https://www.draftkings.com/', "Gambling"),

    ('https://www.pornhub.com/', "Adult (Pornography)"),
    ('https://www.xvideos.com/', "Adult (Pornography)"),
    ('https://www.redtube.com/', "Adult (Pornography)"),
    ('https://www.youporn.com/', "Adult (Pornography)"),

    ('https://www.medium.com/', "Blogs & Personal Sites"),
    ('https://www.wordpress.com/', "Blogs & Personal Sites"),
    ('https://www.blogger.com/', "Blogs & Personal Sites"),

    ('https://www.usa.gov/', "Government & Organizations"),
    ('https://www.who.int/', "Government & Organizations"),
    ('https://www.un.org/', "Government & Organizations"),
    ('https://www.whitehouse.gov/', "Government & Organizations"),

    ('https://www.consumerreports.org/', "Shopping Comparison & Review"),
    ('https://www.pricegrabber.com/', "Shopping Comparison & Review"),
    ('https://www.cnet.com/', "Shopping Comparison & Review"),
    ('https://www.tomsguide.com/', "Shopping Comparison & Review"),
]


scraped_data = {}
labels = []

# Process the data
for url, label in urls_with_labels:
    html_file = download_website(url)
    if html_file:
        content = process_html_file(html_file)
        if content:
            cleaned_text = clean_text(content)
            preprocessed_text = preprocess_text(cleaned_text)
            scraped_data[url] = preprocessed_text
            labels.append(label)

print("Labels:", labels)
sys.stdout.flush()  # Ensure the labels are printed

# Prepare BERT embeddings
bert_embeddings = []
for text in scraped_data.values():
    embedding = get_bert_embedding(text)
    bert_embeddings.append(embedding.squeeze().numpy())

if len(set(labels)) > 1:
    X = np.array(bert_embeddings)
    y = np.array(labels)

    # Use n_splits=2 since we have a small dataset
    skf = StratifiedKFold(n_splits=2)

    # Train or load Random Forest Model
    if os.path.exists('rf_model.pkl'):
        rf_model = joblib.load('rf_model.pkl')
        print("Loaded Random Forest model from file.")
    else:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        joblib.dump(rf_model, 'rf_model.pkl')
        print("Trained and saved Random Forest model.")

    # Train or load SVM Model
    if os.path.exists('svm_model.pkl'):
        svm_model = joblib.load('svm_model.pkl')
        print("Loaded SVM model from file.")
    else:
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X, y)
        joblib.dump(svm_model, 'svm_model.pkl')
        print("Trained and saved SVM model.")

    sys.stdout.flush()  # Ensure the messages are printed

    # Use both models for predictions
    svm_predictions = svm_model.predict(X)
    rf_predictions = rf_model.predict(X)

    # Display results as a table with predicted categories
    table_data_svm = []
    table_data_rf = []
    for url, svm_pred, rf_pred in zip(scraped_data.keys(), svm_predictions, rf_predictions):
        table_data_svm.append([url, svm_pred])
        table_data_rf.append([url, rf_pred])

    print("\nWebsite Categorization Table Based on SVM Model Predictions:")
    print(tabulate(table_data_svm, headers=["Website", "Predicted Category (SVM)"], tablefmt="grid"))

    print("\nWebsite Categorization Table Based on Random Forest Model Predictions:")
    print(tabulate(table_data_rf, headers=["Website", "Predicted Category (RF)"], tablefmt="grid"))

    sys.stdout.flush()  # Ensure the tables are printed

    # Optionally, print cross-validation scores as well
    svm_scores = cross_val_score(svm_model, X, y, cv=skf, scoring='accuracy')
    rf_scores = cross_val_score(rf_model, X, y, cv=skf, scoring='accuracy')

    print(f"\nSVM Accuracy with Cross-Validation: {np.mean(svm_scores):.2f}")
    print(f"Random Forest Accuracy with Cross-Validation: {np.mean(rf_scores):.2f}")
    sys.stdout.flush()  # Ensure accuracy scores are printed

else:
    print("Not enough class variation for model 