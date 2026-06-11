import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class HarassmentDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'naive_bayes': MultinomialNB()
        }
        self.best_model = None
        self.best_score = 0
        
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and select the best one"""
        # Vectorize text data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Update best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
        
        return results, X_test_vec, y_test
    
    def evaluate_models(self, results, X_test, y_test):
        """Evaluate and compare all models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(classification_report(y_test, result['predictions']))
    
    def save_model(self, model_path='models/harassment_model.pkl', 
                   vectorizer_path='models/vectorizer.pkl'):
        """Save the trained model and vectorizer"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"✅ Best model saved to {model_path}")
        print(f"✅ Vectorizer saved to {vectorizer_path}")