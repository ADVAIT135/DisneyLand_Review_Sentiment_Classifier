import os
import streamlit as st
import joblib

import os
import streamlit as st

import os
os.environ['NLTK_DATA'] = os.path.join(os.path.dirname(__file__), 'nltk_data')



import nltk

try:
    nltk.data.find('')
except LookupError:
    nltk.download('')
    
# 3. Download corpora once per Streamlit session
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords',    download_dir=NLTK_DATA_DIR, quiet=True)
    nltk.download('wordnet',      download_dir=NLTK_DATA_DIR, quiet=True)
    nltk.download('omw-1.4',      download_dir=NLTK_DATA_DIR, quiet=True)
    return

download_nltk_data()

# 4. Now safe to import from nltk.corpus
from nltk.corpus import stopwords, wordnet


# Ensure spaCy model is available once
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

# Load your artifacts
vectorizer = joblib.load('vectorizer.pkl')
clf        = joblib.load('classifier.pkl')

st.title('Disneyland Review Sentiment Classifier')
user_input = st.text_area('Enter a review:')

# Preprocessing function
def preprocess(text):
    doc = nlp(text)
    stop_words = set(stopwords.words('english'))
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if token.is_alpha and token.text.lower() not in stop_words
    ]
    return ' '.join(tokens)

# Classification
if st.button('Classify'):
    clean = preprocess(user_input)
    vect  = vectorizer.transform([clean])
    pred  = clf.predict(vect)[0]
    st.write(f'Sentiment: {pred}')
