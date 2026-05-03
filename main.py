import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import re
import warnings

warnings.filterwarnings('ignore')

class StylometryAnalyzer:
    def __init__(self):
        self.models = {}
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.feature_names = None
        self.authors = []
        self.scaler = StandardScaler()
        self.combined_features_unscaled = None 

    def extract_stylistic_features(self, text):
        if not text or len(text.strip()) == 0:
            return np.zeros(5)

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            return np.zeros(5)

        words = text.lower().split()
        if len(words) == 0:
            return np.zeros(5)

        avg_sentence_length = len(words) / len(sentences)
        punctuation_count = len(re.findall(r'[.!?,;:\-\'"()]', text))
        punctuation_density = punctuation_count / len(words)

        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are'}
        stop_word_count = sum(1 for word in words if word.lower() in stop_words)
        stop_word_freq = stop_word_count / len(words)

        unique_ratio = len(set(words)) / len(words)
        avg_word_length = np.mean([len(word) for word in words])

        return np.array([avg_sentence_length, punctuation_density, stop_word_freq, unique_ratio, avg_word_length])

    def prepare_training_data(self, texts, labels):
        tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out()
        stylistic_features = np.array([self.extract_stylistic_features(text) for text in texts])
        self.combined_features_unscaled = np.hstack([tfidf_features, stylistic_features])
        combined_features_scaled = self.scaler.fit_transform(self.combined_features_unscaled)
        self.authors = list(set(labels))
        return combined_features_scaled, labels

    def train(self, texts, labels, test_size=0.2):
        X_scaled, y = self.prepare_training_data(texts, labels)
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train_unscaled, X_test_unscaled, _, _ = train_test_split(
            self.combined_features_unscaled, y, test_size=test_size, random_state=42, stratify=y
        )

        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Naive Bayes': MultinomialNB()
        }

        results = {}
        for name, clf in classifiers.items():
            if name == 'Naive Bayes':
                clf.fit(X_train_unscaled, y_train)
                y_pred = clf.predict(X_test_unscaled)
                cv_scores = cross_val_score(clf, X_train_unscaled, y_train, cv=2)
            else:
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=2)

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            self.models[name] = clf
        return pd.DataFrame(results).T

    def predict_author(self, text, model_name='Random Forest'):
        tfidf_features = self.vectorizer.transform([text]).toarray()
        stylistic_features = self.extract_stylistic_features(text).reshape(1, -1)
        combined_features_raw = np.hstack([tfidf_features, stylistic_features])
        if model_name == 'Naive Bayes':
            final_features = combined_features_raw
        else:
            final_features = self.scaler.transform(combined_features_raw)
        clf = self.models[model_name]
        prediction = clf.predict(final_features)[0]
        probabilities = clf.predict_proba(final_features)[0]
        return prediction, dict(zip(clf.classes_, probabilities))

training_texts = [
    "It was a good day. The sun was bright. The water was cold. He swam well.",
    "The old man sat on the bench. He looked at the sea. The sea was vast.",
    "The bar was warm. The whiskey was good. He drank it slowly.",
    "The ancient and venerable city of London presented itself as a testament to centuries of achievement.",
    "In those peculiar times when society seemed to bend, one could scarcely imagine the transformations.",
    "The great cathedral, adorned with exquisite features, stood majestically as a beacon of hope."
]
training_labels = ['Hemingway', 'Hemingway', 'Hemingway', 'Dickens', 'Dickens', 'Dickens']

analyzer = StylometryAnalyzer()
analyzer.train(training_texts, training_labels, test_size=0.3)

test_text = "The room was quiet. Very quiet. She entered slowly. Her footsteps were soft."
pred, probs = analyzer.predict_author(test_text, model_name='Random Forest')

print(f"Predicted Author: {pred}")
for author, prob in probs.items():
    print(f"{author}: {prob:.2%}")