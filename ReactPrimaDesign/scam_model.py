"""
Advanced QR Scam Detection Model for SafePay
Uses scikit-learn's RandomForestClassifier for UPI scam detection
"""

import re
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.base import BaseEstimator, TransformerMixin

# Security: Allow only UPI-like patterns
VALID_UPI_REGEX = r'^[a-zA-Z0-9.\-]{3,256}@[a-zA-Z]{3,64}$'

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering pipeline"""
    def __init__(self):
        self.url_keywords = ["login", "verify", "phish", "scam"]
        self.hasher = FeatureHasher(n_features=10, input_type='string')

    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        features = []
        for vpa, amt, raw_text in zip(X['bene_vpa'], X['amount'], X['raw_text']):
            # Domain analysis
            domain = vpa.split('@')[-1] if '@' in vpa else ''
            hashed = self.hasher.transform([[domain]]).toarray()[0]

            # URL heuristics
            is_shortened = 1 if re.search(r"(bit\.ly|goo\.gl)", raw_text) else 0
            has_suspicious_keyword = 1 if any(kw in raw_text.lower() for kw in self.url_keywords) else 0

            # UPI syntax checks
            syntax_valid = 1 if re.match(VALID_UPI_REGEX, vpa) else 0
            special_chars = len(re.findall(r'[%&#=]', vpa))

            # Temporal features (mock - integrate real data)
            recent_frequency = 0 # Would come from user's transaction history

            feature_vec = np.concatenate([
                [amt, is_shortened, has_suspicious_keyword, syntax_valid, special_chars, recent_frequency],
                hashed
            ])
            features.append(feature_vec)

        return np.array(features)

class QRScamModel:
    """QR Scam Detection Model Manager"""
    def __init__(self):
        # Initialize model and feature engineer
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        
    def train(self, X, y):
        """Train the model with provided data"""
        # Transform features
        X_features = self.feature_engineer.transform(X)
        
        # Train the model
        self.model.fit(X_features, y)
        self.is_trained = True
        
    def predict(self, bene_vpa, amount, raw_text):
        """Make a prediction for a single QR code"""
        # Format input
        X = {
            'bene_vpa': [bene_vpa],
            'amount': [float(amount) if amount else 0],
            'raw_text': [raw_text]
        }
        
        # Transform features
        X_features = self.feature_engineer.transform(X)
        
        # Make prediction
        if not self.is_trained:
            # If model not yet trained, use rule-based approach
            return self._rule_based_check(bene_vpa, raw_text)
        
        proba = self.model.predict_proba(X_features)[0][1]
        
        return {
            'probability': float(proba),
            'label': 'Scam' if proba > 0.65 else 'Safe',
            'reason': 'ML model detected suspicious patterns' if proba > 0.65 else 'QR code appears legitimate'
        }
    
    def _rule_based_check(self, bene_vpa, raw_text):
        """Fallback rule-based detection if model not trained"""
        # Check for valid UPI syntax
        if not re.match(VALID_UPI_REGEX, bene_vpa):
            return {
                'probability': 0.9,
                'label': 'Scam',
                'reason': 'Invalid UPI syntax'
            }
        
        # Check for suspicious keywords
        suspicious_keywords = ["urgent", "verify", "login", "kyc", "block", "expired"]
        if any(kw in raw_text.lower() for kw in suspicious_keywords):
            return {
                'probability': 0.8,
                'label': 'Scam',
                'reason': 'Contains suspicious keywords'
            }
        
        # Default to safe with moderate confidence
        return {
            'probability': 0.2,
            'label': 'Safe',
            'reason': 'No suspicious patterns detected'
        }
    
    def save(self, filename='scam_model.pkl'):
        """Save model to file"""
        with open(filename, 'wb') as f:
            pickle.dump((self.model, self.feature_engineer), f)
    
    def load(self, filename='scam_model.pkl'):
        """Load model from file"""
        try:
            with open(filename, 'rb') as f:
                self.model, self.feature_engineer = pickle.load(f)
            self.is_trained = True
            return True
        except:
            return False

# Create and initialize model with synthetic data if run directly
if __name__ == '__main__':
    import pandas as pd
    
    # Create synthetic data
    data = {
        'bene_vpa': [
            'merchant@okaxis',
            'payment@okicici',
            'phishing.scam@okhdfc',
            'verify.kyc@oksbi',
            'login.verify@okaxis'
        ],
        'amount': [100, 200, 50, 500, 1000],
        'raw_text': [
            'upi://pay?pa=merchant@okaxis&pn=Legitimate%20Store&am=100',
            'upi://pay?pa=payment@okicici&pn=Online%20Store',
            'upi://pay?pa=phishing.scam@okhdfc&pn=Verify%20Account%20Now%20Urgent',
            'Your KYC is expiring verify now: upi://pay?pa=verify.kyc@oksbi',
            'Login to verify account: upi://pay?pa=login.verify@okaxis'
        ],
        'is_scam': [0, 0, 1, 1, 1]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Train model
    model = QRScamModel()
    model.train(df, df['is_scam'])
    
    # Save model
    model.save()
    
    print("Model trained and saved successfully!")