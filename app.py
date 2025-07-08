import os
import streamlit as st
import joblib
import nltk

# 1. Setup NLTK data path
NLTK_DATA_DIR = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.insert(0, NLTK_DATA_DIR)

# 2. Ensure corpora are present (cached)
@st.cache_resource
def ensure_nltk_data():
    targets = [
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('corpora/stopwords', 'stopwords'),
    ]
    for resource_path, pkg in targets:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
    return True

ensure_nltk_data()

# 3. Imports now safe
from nltk.corpus import wordnet, stopwords

# 4. Load spaCy model (cached)
@st.cache_resource
def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# 5. Load ML artifacts
vectorizer = joblib.load('vectorizer.pkl')
clf = joblib.load('classifier.pkl')

# 6. Streamlit UI
st.title('Disneyland Review Sentiment Classifier')
user_input = st.text_area('Enter a review:')

def preprocess(text):
    doc = nlp(text)
    stop_words = set(stopwords.words('english'))
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and token.text.lower() not in stop_words
    ]
    return ' '.join(tokens)

if st.button('Classify'):
    clean = preprocess(user_input)
    vect = vectorizer.transform([clean])
    pred = clf.predict(vect)[0]
    st.write(f'Sentiment: {pred}')
