"""
Voice and Text Scam Detection Model
Provides ML-based detection of voice and text scams using scikit-learn
"""

import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import re

# Path constants
MODEL_DIR = "models"
VOICE_MODEL_PATH = os.path.join(MODEL_DIR, "voice_scam_model.joblib")
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_scam_model.joblib")
VOICE_VECTORIZER_PATH = os.path.join(MODEL_DIR, "voice_vectorizer.joblib")
TEXT_VECTORIZER_PATH = os.path.join(MODEL_DIR, "text_vectorizer.joblib")
DATA_DIR = "data"
SCAM_KEYWORDS_PATH = os.path.join("attached_assets", "scam_keywords_dataset.json")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor for text content"""
    
    def __init__(self):
        self.scam_keywords = self._load_scam_keywords()
        self.urgency_words = [
            'urgent', 'immediately', 'action required', 'alert', 'warning',
            'limited time', 'act now', 'hurry', 'expires', 'deadline',
            'important', 'critical', 'emergency', 'attention'
        ]
        self.financial_words = [
            'money', 'bank', 'account', 'credit', 'debit', 'card', 'transfer',
            'payment', 'transaction', 'cash', 'fund', 'deposit', 'withdraw',
            'wallet', 'upi', 'offer', 'discount', 'free', 'prize', 'reward',
            'lottery', 'loan', 'tax', 'refund', 'kyc', 'update', 'verify'
        ]
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+')
        self.phone_pattern = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        
    def _load_scam_keywords(self):
        """Load scam keywords from JSON file"""
        try:
            if os.path.exists(SCAM_KEYWORDS_PATH):
                with open(SCAM_KEYWORDS_PATH, 'r') as f:
                    data = json.load(f)
                return data.get('keywords', [])
            else:
                # Fallback keywords if file not found
                return [
                    "verify", "account suspended", "kyc", "suspicious activity", 
                    "bank account", "card blocked", "prize", "lottery", "won", 
                    "claim", "update", "password", "information", "link", "click",
                    "offer", "limited", "urgent", "attention", "verify", "validate"
                ]
        except Exception as e:
            print(f"Error loading scam keywords: {e}")
            return []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract features from text content"""
        features = []
        
        for text in X:
            text = str(text).lower()
            
            # Common scam indicators
            keyword_count = sum(1 for keyword in self.scam_keywords if keyword.lower() in text)
            urgency_count = sum(1 for word in self.urgency_words if word in text)
            financial_count = sum(1 for word in self.financial_words if word in text)
            url_count = len(self.url_pattern.findall(text))
            phone_count = len(self.phone_pattern.findall(text))
            
            # Text characteristics
            text_length = len(text)
            avg_word_length = sum(len(word) for word in text.split()) / max(1, len(text.split()))
            capital_ratio = sum(1 for char in text if char.isupper()) / max(1, len(text))
            exclamation_count = text.count('!')
            question_count = text.count('?')
            special_char_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
            special_char_ratio = special_char_count / max(1, len(text))
            
            # Combined features
            feature_dict = {
                'keyword_count': keyword_count,
                'keyword_density': keyword_count / max(1, len(text.split())),
                'urgency_count': urgency_count,
                'urgency_density': urgency_count / max(1, len(text.split())),
                'financial_count': financial_count,
                'url_count': url_count, 
                'phone_count': phone_count,
                'text_length': text_length,
                'avg_word_length': avg_word_length,
                'capital_ratio': capital_ratio,
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'special_char_ratio': special_char_ratio
            }
            
            features.append(feature_dict)
            
        return features

class VoiceTextScamDetector:
    """Class for detecting scams in voice transcripts and text messages"""
    
    def __init__(self):
        self.voice_vectorizer = None
        self.voice_model = None
        self.text_vectorizer = None
        self.text_model = None
        self.feature_extractor = TextFeatureExtractor()
        self.load_models()
        
    def load_models(self):
        """Load or initialize models"""
        try:
            # Try to load existing voice model
            if os.path.exists(VOICE_MODEL_PATH) and os.path.exists(VOICE_VECTORIZER_PATH):
                self.voice_model = joblib.load(VOICE_MODEL_PATH)
                self.voice_vectorizer = joblib.load(VOICE_VECTORIZER_PATH)
                print("Voice model loaded successfully")
            else:
                print("Voice model not found, will be trained on first use")
                
            # Try to load existing text model
            if os.path.exists(TEXT_MODEL_PATH) and os.path.exists(TEXT_VECTORIZER_PATH):
                self.text_model = joblib.load(TEXT_MODEL_PATH)
                self.text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
                print("Text model loaded successfully")
            else:
                print("Text model not found, will be trained on first use")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def train_voice_model(self, voice_data=None):
        """Train voice scam detection model"""
        # Sample data if none provided
        if not voice_data:
            voice_data = [
                {"transcript": "Your bank account has been suspended due to suspicious activity. Please verify your identity by providing your account number and password.", "is_scam": True},
                {"transcript": "I'm calling from the bank to inform you that your account has been blocked. Press 1 to speak with our verification team.", "is_scam": True},
                {"transcript": "Congratulations! You've won a lottery prize of 10 lakh rupees. To claim your prize, send us your bank details immediately.", "is_scam": True},
                {"transcript": "Your KYC verification is pending. Your account will be blocked if not updated within 24 hours. Press 1 to update now.", "is_scam": True},
                {"transcript": "This is regarding your recent transaction. Please confirm if you made this purchase.", "is_scam": False},
                {"transcript": "Hello, I'm calling to schedule the delivery of your order. Would tomorrow afternoon work for you?", "is_scam": False},
                {"transcript": "This is a reminder about your appointment scheduled for tomorrow at 3 PM. Please confirm your attendance.", "is_scam": False},
                {"transcript": "Your OTP for the transaction is 456789. Do not share this OTP with anyone, including bank officials.", "is_scam": False},
                {"transcript": "I'm calling to inform you that the tickets you booked are confirmed. You'll receive the details by SMS shortly.", "is_scam": False},
                {"transcript": "This is a system update notification. Please verify your kyc today to avoid service interruption. Please share UPI immediately.", "is_scam": True},
                {"transcript": "We noticed unauthorized access to your account. Please verify your identity by sharing your secure PIN for verification.", "is_scam": True},
                {"transcript": "Government stimulus payment of Rs 5000 has been approved. For immediate deposit, please share your UPI ID or bank account information.", "is_scam": True},
                {"transcript": "This is an important call regarding your auto warranty. Your coverage is about to expire. Press 1 to speak to a representative.", "is_scam": True},
                {"transcript": "Hi, this is Ravi from PayTM. We just need to confirm your recent transaction. Can you tell me if you purchased something for Rs 2999?", "is_scam": False},
                {"transcript": "Thank you for your recent payment. Your transaction ID is TRX45678. If you have any questions, please contact customer service.", "is_scam": False}
            ]
        
        # Extract features and labels
        texts = [item["transcript"] for item in voice_data]
        labels = [1 if item["is_scam"] else 0 for item in voice_data]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        # Create and train vectorizer
        self.voice_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_train_vec = self.voice_vectorizer.fit_transform(X_train)
        
        # Create and train model
        self.voice_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.voice_model.fit(X_train_vec, y_train)
        
        # Evaluate model
        X_test_vec = self.voice_vectorizer.transform(X_test)
        y_pred = self.voice_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Voice model trained with accuracy: {accuracy:.2f}")
        
        # Save model
        os.makedirs(os.path.dirname(VOICE_MODEL_PATH), exist_ok=True)
        joblib.dump(self.voice_model, VOICE_MODEL_PATH)
        joblib.dump(self.voice_vectorizer, VOICE_VECTORIZER_PATH)
        
        return accuracy
    
    def train_text_model(self, text_data=None):
        """Train text scam detection model"""
        # Sample data if none provided
        if not text_data:
            text_data = [
                {"text": "ATTENTION: Your account has been suspended. Click here to verify: http://bit.ly/suspicious-link", "is_scam": True},
                {"text": "Congratulations! You've won Rs 10,00,000 in lottery! Send your bank details to claim your prize now!", "is_scam": True},
                {"text": "URGENT: Your KYC verification is pending. Account will be blocked in 24 hours. Update now: www.fakebank.com", "is_scam": True},
                {"text": "Your card has been blocked due to suspicious activity. Call +91-9834567890 immediately to unblock.", "is_scam": True},
                {"text": "Your Flipkart order #45678 has been shipped. Track here: https://flipkart.com/tracking", "is_scam": False},
                {"text": "Your OTP for transaction is 456789. Valid for 5 minutes. Do not share with anyone.", "is_scam": False},
                {"text": "Hi, this is a reminder for your doctor appointment tomorrow at 3 PM. Reply YES to confirm.", "is_scam": False},
                {"text": "Thank you for your payment of Rs 500. Transaction ID: TXN123456", "is_scam": False},
                {"text": "Your electricity bill of Rs 1,500 is due on 25/03. Pay online at www.electricity.com", "is_scam": False},
                {"text": "ALERT! Unauthorized transaction of Rs 25,000 detected. Call immediately or click: bit.ly/fraudlink", "is_scam": True},
                {"text": "Dear customer, your account will be suspended due to non-completion of KYC. Please click here: tinyurl.com/fakekyc", "is_scam": True},
                {"text": "Congratulations! You've been selected for a free iPhone 15. Claim now: www.freeiphone-scam.com", "is_scam": True},
                {"text": "Your FedEx package is held due to incomplete address. Update details: bit.ly/scam-fedex", "is_scam": True},
                {"text": "Dear customer, your Swiggy order has been delivered. Rate your experience here: swiggy.in/feedback", "is_scam": False},
                {"text": "Hi, your Uber ride with Rajesh has been confirmed. Driver will arrive in 3 minutes.", "is_scam": False},
                {"text": "BHIM UPI: You have received Rs 500 from Amit S. UPI Ref: 123456789.", "is_scam": False},
                {"text": "URGENT: Your phone will be disconnected in 24 hours due to a security issue. Call +91-1234567890 now.", "is_scam": True},
                {"text": "ICICI Bank: Deposit Rs 10,000 to extend your loan terms. Pay here - upi.scam.co.in", "is_scam": True}
            ]
        
        # Extract features and labels
        texts = [item["text"] for item in text_data]
        labels = [1 if item["is_scam"] else 0 for item in text_data]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        # Create and train pipeline with custom feature extractor and model
        self.text_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        X_train_vec = self.text_vectorizer.fit_transform(X_train)
        
        # Extract additional features
        X_train_features = self.feature_extractor.transform(X_train)
        
        # Convert to numpy arrays
        X_train_features = np.array([[v for k, v in item.items()] for item in X_train_features])
        
        # Create combined feature matrix
        X_train_combined = np.hstack((X_train_vec.toarray(), X_train_features))
        
        # Train model
        self.text_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.text_model.fit(X_train_combined, y_train)
        
        # Evaluate model
        X_test_vec = self.text_vectorizer.transform(X_test)
        X_test_features = self.feature_extractor.transform(X_test)
        X_test_features = np.array([[v for k, v in item.items()] for item in X_test_features])
        X_test_combined = np.hstack((X_test_vec.toarray(), X_test_features))
        
        y_pred = self.text_model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Text model trained with accuracy: {accuracy:.2f}")
        
        # Save model
        os.makedirs(os.path.dirname(TEXT_MODEL_PATH), exist_ok=True)
        joblib.dump(self.text_model, TEXT_MODEL_PATH)
        joblib.dump(self.text_vectorizer, TEXT_VECTORIZER_PATH)
        
        return accuracy
    
    def analyze_voice(self, transcript, audio_features=None):
        """Analyze voice transcript for scam detection"""
        # Train model if necessary
        if self.voice_model is None or self.voice_vectorizer is None:
            print("Voice model not found, training...")
            self.train_voice_model()
        
        # Prepare features
        try:
            # Vectorize transcript
            transcript_vec = self.voice_vectorizer.transform([transcript])
            
            # Get probability prediction
            proba = self.voice_model.predict_proba(transcript_vec)[0]
            prediction = self.voice_model.predict(transcript_vec)[0]
            
            # Determine scam type and risk indicators
            scam_type = None
            scam_indicators = []
            
            if prediction == 1:
                # Basic scam type detection based on keywords
                if re.search(r'\b(bank|account|suspend|block|verify)\b', transcript, re.I):
                    scam_type = "Banking Scam"
                    scam_indicators.append("Banking or account verification requests")
                elif re.search(r'\b(kyc|verification|document|identity)\b', transcript, re.I):
                    scam_type = "KYC Scam"
                    scam_indicators.append("KYC or identity verification requests")
                elif re.search(r'\b(lottery|prize|win|reward|gift)\b', transcript, re.I):
                    scam_type = "Lottery Scam"
                    scam_indicators.append("Unexpected prize or lottery winnings")
                elif re.search(r'\b(refund|return|cashback)\b', transcript, re.I):
                    scam_type = "Refund Scam"
                    scam_indicators.append("Unsolicited refund offers")
                else:
                    scam_type = "Unknown Scam"
                
                # Add common scam indicators
                if re.search(r'\b(urgent|immediately|emergency|alert|warning)\b', transcript, re.I):
                    scam_indicators.append("Creates false urgency or panic")
                if re.search(r'\b(password|pin|otp|cvv|secure code)\b', transcript, re.I):
                    scam_indicators.append("Asks for sensitive information like passwords or PINs")
                if re.search(r'\b(click|link|website|download)\b', transcript, re.I):
                    scam_indicators.append("Directs to suspicious links or websites")
                if re.search(r'\b(money|payment|fee|deposit|transfer)\b', transcript, re.I):
                    scam_indicators.append("Requests money transfers or payments")
            
            return {
                "is_scam": bool(prediction),
                "confidence": float(proba[1]),  # probability of being scam
                "risk_score": float(proba[1] * 100),  # convert to percentage
                "scam_type": scam_type,
                "scam_indicators": scam_indicators,
                "analysis_method": "voice_ml_model"
            }
        except Exception as e:
            print(f"Error analyzing voice: {e}")
            # Fallback to rule-based detection
            return self._rule_based_voice_analysis(transcript)
    
    def _rule_based_voice_analysis(self, transcript):
        """Rule-based voice analysis as fallback"""
        text = transcript.lower()
        
        # Scam indicator keywords
        banking_keywords = ['bank', 'account', 'suspend', 'blocked', 'verify', 'verify kyc', 'update kyc', 'card block']
        urgent_keywords = ['urgent', 'immediately', 'emergency', 'warning', 'alert', 'action required']
        sensitive_keywords = ['password', 'pin', 'otp', 'card number', 'cvv', 'account number', 'verification code']
        incentive_keywords = ['lottery', 'prize', 'won', 'reward', 'gift', 'offer', 'free']
        threat_keywords = ['suspended', 'blocked', 'legal action', 'police', 'arrest', 'terminate', 'fine']
        
        # Calculate matches for each category
        banking_score = sum(1 for kw in banking_keywords if kw in text)
        urgent_score = sum(1 for kw in urgent_keywords if kw in text)
        sensitive_score = sum(1 for kw in sensitive_keywords if kw in text)
        incentive_score = sum(1 for kw in incentive_keywords if kw in text)
        threat_score = sum(1 for kw in threat_keywords if kw in text)
        
        # Calculate combined score (weighted)
        total_score = (
            banking_score * 1 + 
            urgent_score * 1.5 + 
            sensitive_score * 2 + 
            incentive_score * 1.2 + 
            threat_score * 1.5
        )
        
        # Normalize to probability
        max_possible_score = (
            len(banking_keywords) * 1 + 
            len(urgent_keywords) * 1.5 + 
            len(sensitive_keywords) * 2 + 
            len(incentive_keywords) * 1.2 + 
            len(threat_keywords) * 1.5
        )
        
        probability = min(0.95, total_score / (max_possible_score * 0.3))  # Cap at 0.95, adjust threshold
        
        # Identify scam indicators
        scam_indicators = []
        if banking_score > 0:
            scam_indicators.append("Banking or financial terms used to create legitimacy")
        if urgent_score > 0:
            scam_indicators.append("Creates false urgency or panic")
        if sensitive_score > 0:
            scam_indicators.append("Requests sensitive information")
        if incentive_score > 0:
            scam_indicators.append("Offers suspicious rewards or incentives")
        if threat_score > 0:
            scam_indicators.append("Uses threats or intimidation")
        
        # Determine scam type
        scam_type = None
        if banking_score > 0 and sensitive_score > 0:
            scam_type = "Banking Scam"
        elif incentive_score > 0:
            scam_type = "Lottery/Prize Scam"
        elif "kyc" in text:
            scam_type = "KYC Scam"
        elif "refund" in text:
            scam_type = "Refund Scam"
        else:
            scam_type = "Unknown Scam"
        
        # Final decision
        is_scam = probability > 0.5
        
        return {
            "is_scam": is_scam,
            "confidence": float(probability),
            "risk_score": float(probability * 100),
            "scam_type": scam_type if is_scam else None,
            "scam_indicators": scam_indicators if is_scam else [],
            "analysis_method": "voice_rule_based"
        }
    
    def analyze_text(self, text, message_type='SMS', context=None):
        """Analyze text message for scam detection"""
        # Train model if necessary
        if self.text_model is None or self.text_vectorizer is None:
            print("Text model not found, training...")
            self.train_text_model()
        
        # Prepare features
        try:
            # Vectorize text
            text_vec = self.text_vectorizer.transform([text])
            
            # Extract additional features
            text_features = self.feature_extractor.transform([text])
            text_features = np.array([[v for k, v in item.items()] for item in text_features])
            
            # Combine features
            combined_features = np.hstack((text_vec.toarray(), text_features))
            
            # Get probability prediction
            proba = self.text_model.predict_proba(combined_features)[0]
            prediction = self.text_model.predict(combined_features)[0]
            
            # Determine scam type and risk indicators
            scam_type = None
            scam_indicators = []
            
            if prediction == 1:
                # Basic scam type detection based on keywords
                if re.search(r'\b(bank|account|suspend|block|verify)\b', text, re.I):
                    scam_type = "Banking Scam"
                    scam_indicators.append("Banking or account verification requests")
                elif re.search(r'\b(kyc|verification|document|identity)\b', text, re.I):
                    scam_type = "KYC Scam"
                    scam_indicators.append("KYC or identity verification requests")
                elif re.search(r'\b(lottery|prize|win|reward|gift)\b', text, re.I):
                    scam_type = "Lottery Scam"
                    scam_indicators.append("Unexpected prize or lottery winnings")
                elif re.search(r'\b(refund|return|cashback)\b', text, re.I):
                    scam_type = "Refund Scam"
                    scam_indicators.append("Unsolicited refund offers")
                else:
                    scam_type = "Unknown Scam"
                
                # Check for URLs
                urls = self.feature_extractor.url_pattern.findall(text)
                if urls:
                    scam_indicators.append(f"Contains suspicious URL: {urls[0]}")
                
                # Check for phone numbers
                phone_numbers = self.feature_extractor.phone_pattern.findall(text)
                if phone_numbers:
                    scam_indicators.append(f"Contains phone number: {phone_numbers[0]}")
                
                # Add common scam indicators
                if re.search(r'\b(urgent|immediately|emergency|alert|warning)\b', text, re.I):
                    scam_indicators.append("Creates false urgency or panic")
                if re.search(r'\b(password|pin|otp|cvv|secure code)\b', text, re.I):
                    scam_indicators.append("Asks for sensitive information like passwords or PINs")
                if re.search(r'\b(money|payment|fee|deposit|transfer|rs\.?|₹)\b', text, re.I):
                    scam_indicators.append("Mentions money transfers or payments")
            
            return {
                "is_scam": bool(prediction),
                "confidence": float(proba[1]),  # probability of being scam
                "risk_score": float(proba[1] * 100),  # convert to percentage
                "scam_type": scam_type,
                "scam_indicators": scam_indicators,
                "analysis_method": "text_ml_model"
            }
        except Exception as e:
            print(f"Error analyzing text: {e}")
            # Fallback to rule-based detection
            return self._rule_based_text_analysis(text, message_type)
    
    def _rule_based_text_analysis(self, text, message_type='SMS'):
        """Rule-based text analysis as fallback"""
        text = text.lower()
        
        # Scam indicator keywords and patterns
        patterns = {
            'urls': re.compile(r'https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+'),
            'suspicious_domains': re.compile(r'(?:bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly|tiny\.cc|lnkd\.in|tr\.im)\b'),
            'phone_numbers': re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            'urgent': re.compile(r'\b(?:urgent|immediately|alert|warning|attention|action required)\b'),
            'financial': re.compile(r'\b(?:bank|account|credit|debit|card|verify|suspended|blocked|security|unauthorized|transaction)\b'),
            'sensitive': re.compile(r'\b(?:password|pin|otp|cvv|verify|login|click|access)\b'),
            'rewards': re.compile(r'\b(?:congrat|won|winner|prize|reward|gift|offer|free|discount)\b'),
            'threats': re.compile(r'\b(?:suspend|terminat|block|cancel|legal|polic|lawsuit|fine|penalt|arrest)\b')
        }
        
        # Calculate scores for each category
        scores = {
            'url_score': 2.0 if patterns['urls'].search(text) else 0.0,
            'suspicious_url_score': 3.0 if patterns['suspicious_domains'].search(text) else 0.0,
            'phone_score': 1.0 if patterns['phone_numbers'].search(text) else 0.0,
            'urgent_score': 1.5 * len(patterns['urgent'].findall(text)),
            'financial_score': 1.0 * len(patterns['financial'].findall(text)),
            'sensitive_score': 2.0 * len(patterns['sensitive'].findall(text)),
            'rewards_score': 1.2 * len(patterns['rewards'].findall(text)),
            'threats_score': 1.5 * len(patterns['threats'].findall(text))
        }
        
        # Additional penalties for WhatsApp messages (they tend to have more scams)
        if message_type.lower() == 'whatsapp':
            scores['base_penalty'] = 0.5
        else:
            scores['base_penalty'] = 0.0
        
        # Calculate total score
        total_score = sum(scores.values())
        
        # Normalize to probability (adjusted threshold)
        max_expected_score = 18.0  # Empirical max score
        probability = min(0.95, total_score / max_expected_score)
        
        # Identify scam indicators
        scam_indicators = []
        if patterns['urls'].search(text):
            urls = patterns['urls'].findall(text)
            scam_indicators.append(f"Contains URL: {urls[0]}")
            if patterns['suspicious_domains'].search(text):
                scam_indicators.append("Uses suspicious shortened URL")
        
        if patterns['phone_numbers'].search(text):
            phones = patterns['phone_numbers'].findall(text)
            scam_indicators.append(f"Contains phone number: {phones[0]}")
        
        if patterns['urgent'].search(text):
            scam_indicators.append("Creates false urgency")
        
        if patterns['financial'].search(text) and (patterns['sensitive'].search(text) or patterns['urls'].search(text)):
            scam_indicators.append("Combines financial terms with requests for sensitive information or links")
        
        if patterns['rewards'].search(text):
            scam_indicators.append("Offers suspicious rewards or incentives")
        
        if patterns['threats'].search(text):
            scam_indicators.append("Uses threats or intimidation")
        
        # Determine scam type
        scam_type = None
        if patterns['financial'].search(text):
            if 'kyc' in text:
                scam_type = "KYC Scam"
            else:
                scam_type = "Banking Scam"
        elif patterns['rewards'].search(text):
            scam_type = "Lottery/Prize Scam"
        elif 'refund' in text:
            scam_type = "Refund Scam"
        else:
            scam_type = "Phishing Scam"
        
        # Final decision
        is_scam = probability > 0.5
        
        return {
            "is_scam": is_scam,
            "confidence": float(probability),
            "risk_score": float(probability * 100),
            "scam_type": scam_type if is_scam else None,
            "scam_indicators": scam_indicators if is_scam else [],
            "analysis_method": "text_rule_based"
        }
    
    def batch_analyze_text(self, texts, message_types=None):
        """Analyze multiple text messages in batch"""
        if message_types is None:
            message_types = ['SMS'] * len(texts)
        
        results = []
        for i, text in enumerate(texts):
            message_type = message_types[i] if i < len(message_types) else 'SMS'
            result = self.analyze_text(text, message_type)
            results.append(result)
        
        return results


# Singleton instance
detector = VoiceTextScamDetector()

def analyze_voice(transcript, audio_features=None):
    """Analyze voice transcript for scams"""
    return detector.analyze_voice(transcript, audio_features)

def analyze_text(text, message_type='SMS', context=None):
    """Analyze text message for scams"""
    return detector.analyze_text(text, message_type, context)

def batch_analyze_text(texts, message_types=None):
    """Analyze multiple text messages in batch"""
    return detector.batch_analyze_text(texts, message_types)

def train_models():
    """Train or retrain all models"""
    voice_accuracy = detector.train_voice_model()
    text_accuracy = detector.train_text_model()
    return {
        "voice_model_accuracy": voice_accuracy,
        "text_model_accuracy": text_accuracy
    }

# If this script is run directly, train the models
if __name__ == "__main__":
    print("Training voice and text scam detection models...")
    results = train_models()
    print(f"Training complete. Voice model accuracy: {results['voice_model_accuracy']:.2f}, "
          f"Text model accuracy: {results['text_model_accuracy']:.2f}")