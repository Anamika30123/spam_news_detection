

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import flask
from flask import Flask, request, render_template, jsonify

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Create a Flask app
app = Flask(__name__)

class SpamNewsDetector:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = nltk.word_tokenize(text)
            
            # Remove stopwords and stem words
            processed_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
            
            return ' '.join(processed_tokens)
        else:
            return ''
        
    def load_and_preprocess_data(self, filepath):
        """Load dataset and preprocess text columns."""
        # Load data
        df = pd.read_csv(filepath)
        
        print(f"Dataset loaded with shape: {df.shape}")
        print(f"Column names: {df.columns}")
        
        # Assuming dataset has 'text' column with news content and 'label' column (1 for real, 0 for fake)
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        return df
    
    def train(self, df, text_column='processed_text', label_column='label'):
        """Train the spam news detection model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df[text_column], df[label_column], test_size=0.2, random_state=42
        )
        
        # Feature extraction
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_features = self.vectorizer.fit_transform(X_train)
        X_test_features = self.vectorizer.transform(X_test)
        
        # Train multiple models and pick the best one
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_features, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_features)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Save best model
            if f1 > best_score:
                best_score = f1
                best_model = model
                
            print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save the best model
        self.model = best_model
        
        # Generate detailed evaluation for the best model
        best_name = max(results, key=lambda k: results[k]['f1'])
        print(f"\nBest model: {best_name}")
        y_pred = self.model.predict(X_test_features)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Return test data for visualization
        return X_test, y_test, y_pred, results
    
    def predict(self, text):
        """Predict if a news article is real or fake."""
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        processed_text = self.preprocess_text(text)
        features = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(features)[0]
        prediction_proba = self.model.predict_proba(features)[0]
        
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        return {
            'prediction': 'Real News' if prediction == 1 else 'Fake News',
            'confidence': float(confidence),
            'prediction_code': int(prediction)
        }
    
    def save_model(self, vectorizer_path='vectorizer.pkl', model_path='model.pkl'):
        """Save trained model and vectorizer."""
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        with open(vectorizer_path, 'wb') as file:
            pickle.dump(self.vectorizer, file)
        
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)
            
        print(f"Model saved to {model_path} and vectorizer to {vectorizer_path}")
    
    def load_model(self, vectorizer_path='vectorizer.pkl', model_path='model.pkl'):
        """Load saved model and vectorizer."""
        with open(vectorizer_path, 'rb') as file:
            self.vectorizer = pickle.load(file)
        
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
            
        print(f"Model loaded from {model_path} and vectorizer from {vectorizer_path}")

def visualize_results(X_test, y_test, y_pred, results):
    """Generate visualizations for model performance."""
    # Create figure for subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 0])
    axs[0, 0].set_title('Confusion Matrix')
    axs[0, 0].set_xlabel('Predicted')
    axs[0, 0].set_ylabel('Actual')
    
    # 2. Model Comparison - Accuracy
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    axs[0, 1].bar(models, accuracies, color='skyblue')
    axs[0, 1].set_title('Model Accuracy Comparison')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_ylim(0, 1)
    
    # 3. Model Comparison - F1 Score
    f1_scores = [results[model]['f1'] for model in models]
    
    axs[1, 0].bar(models, f1_scores, color='lightgreen')
    axs[1, 0].set_title('Model F1 Score Comparison')
    axs[1, 0].set_ylabel('F1 Score')
    axs[1, 0].set_ylim(0, 1)
    
    # 4. Precision-Recall by Model
    width = 0.35
    x = np.arange(len(models))
    
    precision_scores = [results[model]['precision'] for model in models]
    recall_scores = [results[model]['recall'] for model in models]
    
    axs[1, 1].bar(x - width/2, precision_scores, width, label='Precision', color='coral')
    axs[1, 1].bar(x + width/2, recall_scores, width, label='Recall', color='lightblue')
    axs[1, 1].set_title('Precision vs Recall by Model')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(models)
    axs[1, 1].legend()
    axs[1, 1].set_ylim(0, 1)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data['text']
    
    # Load detector if not already loaded
    global detector
    if 'detector' not in globals():
        detector = SpamNewsDetector()
        detector.load_model()
    
    # Make prediction
    result = detector.predict(news_text)
    return jsonify(result)

def main():
    # Initialize detector
    detector = SpamNewsDetector()
    
    # Check if saved model exists, otherwise train new one
    try:
        detector.load_model()
        print("Loaded pre-trained model.")
    except:
        print("Training new model...")
        # You would need to provide the path to your dataset here
        df = detector.load_and_preprocess_data('news_dataset.csv')
        X_test, y_test, y_pred, results = detector.train(df)
        visualize_results(X_test, y_test, y_pred, results)
        detector.save_model()
    
    # Start the Flask app
    app.run(debug=True)

if __name__ == "__main__":
    main()