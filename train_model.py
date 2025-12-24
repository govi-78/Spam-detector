"""
Machine Learning Model Training Script
Trains a spam classifier using NLP techniques
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class SpamDetectorTrainer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Preprocess text data:
        - Convert to lowercase
        - Remove special characters and numbers
        - Remove stopwords
        - Apply stemming
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def create_sample_dataset(self):
        """
        Create a sample dataset for training
        In production, replace this with actual dataset
        """
        spam_samples = [
            "Congratulations! You've won a $1000 gift card. Click here to claim now!",
            "URGENT: Your account has been compromised. Verify your password immediately.",
            "Get rich quick! Make $5000 from home. Limited time offer!",
            "Free iPhone! Click this link now to claim your prize!",
            "You have won the lottery! Send your bank details to claim.",
            "Hot singles in your area want to meet you now!",
            "Buy cheap medications online. No prescription needed!",
            "Congratulations! You are our lucky winner. Claim your prize money.",
            "Work from home and earn $10000 per month guaranteed!",
            "Your package is waiting. Click to track your delivery now.",
            "Limited offer! Lose weight fast with this miracle pill!",
            "Dear valued customer, confirm your account details immediately.",
            "You have been selected for a special promotion. Act now!",
            "Increase your followers instantly! Click here for free trial.",
            "WINNER! You've been chosen to receive a brand new car!",
            # Social media specific spam
            "Follow me and I'll follow you back! Instant followers!",
            "Like this comment and win a $500 gift card!",
            "Free Instagram followers! Click the link in my profile!",
            "Check your DMs urgently! You've won a prize!",
            "Limited time offer! Get 1000 followers in 5 minutes!",
            "Click the link in my bio to claim your free gift card!",
            "Tag a friend who needs this! Win amazing prizes now!",
            "Double your followers today! No fake accounts guaranteed!",
            "Like and share to win! Real cash prizes every hour!",
            "Get verified badge now! Special promotion for influencers!"
        ]
        
        ham_samples = [
            "Hey, are we still meeting for lunch tomorrow at 12pm?",
            "The project deadline has been extended to next Friday.",
            "Thanks for your email. I'll review the document and get back to you.",
            "Reminder: Team meeting scheduled for 3pm today in conference room.",
            "Can you send me the report when you get a chance?",
            "Happy birthday! Hope you have a wonderful day!",
            "The weather is beautiful today. Perfect for a walk in the park.",
            "I've attached the presentation slides for tomorrow's meeting.",
            "Let me know if you need any help with the assignment.",
            "Great job on the presentation! Very well done.",
            "The restaurant reservation is confirmed for Saturday at 7pm.",
            "Please review the attached proposal and share your feedback.",
            "Looking forward to seeing you at the conference next week.",
            "Thank you for your support and assistance on this project.",
            "The shipment has been delivered successfully to your address.",
            # Social media normal comments
            "This is such a great post! Really enjoyed reading it.",
            "Thanks for sharing this useful information!",
            "I totally agree with your point of view.",
            "Can you explain this part a bit more?",
            "Looking forward to more content like this!",
            "This made my day! So inspiring!",
            "Great insights! Learned something new today.",
            "Your content is always so valuable!",
            "I've been looking for this information. Thank you!",
            "This is exactly what I needed. Appreciate it!"
        ]
        
        # Create DataFrame
        data = []
        for text in spam_samples:
            data.append({'text': text, 'label': 'spam'})
        for text in ham_samples:
            data.append({'text': text, 'label': 'ham'})
            
        df = pd.DataFrame(data)
        return df
    
    def train(self, df, model_type='naive_bayes'):
        """
        Train the spam detection model
        Args:
            df: DataFrame with 'text' and 'label' columns
            model_type: 'naive_bayes', 'logistic_regression', or 'svm'
        """
        print("Preprocessing text data...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Create TF-IDF vectorizer
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train model
        print(f"Training {model_type} model...")
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True, random_state=42)
        else:
            raise ValueError("Invalid model type")
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"Model: {model_type}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"{'='*50}\n")
        
        return accuracy
    
    def save_model(self, model_dir='models'):
        """Save trained model and vectorizer"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'spam_classifier.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def predict(self, text):
        """Predict if text is spam or not"""
        processed_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        return {
            'prediction': 'Spam' if prediction == 'spam' else 'Not Spam',
            'confidence': max(probability)
        }


def main():
    """Main training function"""
    print("Starting Spam Detection Model Training...")
    
    trainer = SpamDetectorTrainer()
    
    # Create sample dataset
    print("Creating sample dataset...")
    df = trainer.create_sample_dataset()
    print(f"Dataset size: {len(df)} samples")
    print(f"Spam samples: {len(df[df['label'] == 'spam'])}")
    print(f"Ham samples: {len(df[df['label'] == 'ham'])}\n")
    
    # Train model (you can try different models)
    trainer.train(df, model_type='naive_bayes')
    
    # Save model
    trainer.save_model()
    
    # Test predictions
    print("\nTesting model with sample inputs:")
    test_texts = [
        "You have won a million dollars! Click here now!",
        "Let's meet for coffee tomorrow afternoon.",
        "URGENT: Your account will be closed. Verify now!",
        "The meeting has been rescheduled to 3pm."
    ]
    
    for text in test_texts:
        result = trainer.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
    
    print("\n✓ Model training completed successfully!")


if __name__ == "__main__":
    main()
