import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class NLPProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

    def preprocess_text(self, text):
        """Basic text preprocessing - clean the text"""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def fit_vectorizer(self, texts):
        """Fit TF-IDF vectorizer on training texts"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.vectorizer.fit(processed_texts)

    def get_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0

        processed1 = self.preprocess_text(text1)
        processed2 = self.preprocess_text(text2)

        # Handle empty strings after preprocessing
        if not processed1 or not processed2:
            return 0.0

        try:
            vectors = self.vectorizer.transform([processed1, processed2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])
            return similarity[0][0]
        except:
            return 0.0
