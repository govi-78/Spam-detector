"""
Spam Detector Model - Prediction Interface
"""
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from config import Config

class SpamDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            with open(Config.MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(Config.VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("Model loaded successfully")
        except FileNotFoundError:
            print("Warning: Model files not found. Please train the model first.")
    
    def preprocess_text(self, text):
        """Preprocess text data"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def predict(self, text):
        """Predict if text is spam or not"""
        if not self.model or not self.vectorizer:
            return {
                'result': 'Error',
                'confidence': 0.0,
                'message': 'Model not loaded'
            }
        
        processed_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        result = 'Spam' if prediction == 'spam' else 'Not Spam'
        confidence = max(probability)
        
        return {
            'result': result,
            'confidence': float(confidence)
        }
    
    def predict_url(self, url):
        """Detect if URL is spam (can be enhanced with URL-specific features)"""
        # Basic URL spam detection based on common patterns
        spam_patterns = [
            r'win.*prize', r'click.*here.*now', r'free.*money',
            r'verify.*account', r'urgent.*action', r'claim.*reward'
        ]
        
        # Suspicious URL shorteners (more likely to be spam)
        # Make patterns more specific to avoid false positives
        suspicious_shorteners = [
            r'\bbit\.ly\b', r'\btinyurl\b', r'\bgoo\.gl\b', r'\bt\.co\b', 
            r'\bow\.ly\b', r'\bbuff\.ly\b'
        ]
        
        url_lower = url.lower()
        
        # Check for spam patterns in URL
        spam_score = sum(1 for pattern in spam_patterns if re.search(pattern, url_lower))
        
        # Check for suspicious URL shorteners
        shortener_score = sum(1 for pattern in suspicious_shorteners if re.search(pattern, url_lower))
        
        # If we have strong spam indicators, classify as spam
        if spam_score >= 2 or (spam_score >= 1 and shortener_score >= 1):
            return {'result': 'Spam', 'confidence': 0.90}
        elif spam_score >= 1:
            return {'result': 'Spam', 'confidence': 0.75}
        elif shortener_score >= 1:
            # Suspicious shorteners but no spam patterns - moderate confidence
            return {'result': 'Spam', 'confidence': 0.65}
        else:
            # For legitimate URLs, don't automatically classify as spam
            # Just analyze the domain name part
            try:
                # Extract domain from URL
                if '//' in url_lower:
                    domain = url_lower.split('//')[1].split('/')[0]
                else:
                    domain = url_lower.split('/')[0]
                
                # Check if it's a known legitimate domain
                legitimate_domains = [
                    'google.com', 'github.com', 'stackoverflow.com', 
                    'amazon.com', 'microsoft.com', 'facebook.com',
                    'twitter.com', 'linkedin.com', 'youtube.com',
                    'reddit.com', 'wikipedia.org', 'paypal.com'
                ]
                
                # Check for exact or partial matches
                for legit_domain in legitimate_domains:
                    if legit_domain in domain:
                        return {'result': 'Not Spam', 'confidence': 0.80}
            except:
                pass
            
            # For other URLs, use standard text analysis but with lower confidence
            text_prediction = self.predict(url)
            
            # If standard analysis says spam, trust it
            if text_prediction['result'] == 'Spam':
                return text_prediction
            else:
                # If standard analysis says not spam, slightly increase confidence
                # since we've already ruled out obvious spam patterns
                return {
                    'result': 'Not Spam', 
                    'confidence': min(0.85, text_prediction['confidence'] + 0.1)
                }
    
    def predict_email(self, subject, body):
        """Detect if email is spam"""
        # Combine subject and body
        email_text = f"{subject} {body}"
        return self.predict(email_text)
    
    def predict_sms(self, text):
        """Detect if SMS is spam"""
        return self.predict(text)
    
    def predict_comment(self, text):
        """Detect if social media comment is spam"""
        # For social media comments, we might want to be more sensitive to spam
        # Let's add some specific patterns for social media spam
        social_media_spam_patterns = [
            r'follow.*back', r'like.*comment.*win', r'free.*gift.*card',
            r'instant.*followers', r'click.*link.*profile', r'win.*prize.*now',
            r'check.*dm.*urgent', r'limited.*time.*offer', r'double.*followers',
            r'tag.*friend.*win', r'verified.*badge', r'cash.*prize'
        ]
        
        text_lower = text.lower()
        spam_score = sum(1 for pattern in social_media_spam_patterns if re.search(pattern, text_lower))
        
        # Get standard prediction
        standard_prediction = self.predict(text)
        
        # If we detect social media spam patterns, increase spam likelihood
        if spam_score >= 1:
            # Boost confidence for detected patterns but cap it
            boosted_confidence = min(0.95, standard_prediction['confidence'] + 0.2)
            return {'result': 'Spam', 'confidence': boosted_confidence}
        else:
            return standard_prediction