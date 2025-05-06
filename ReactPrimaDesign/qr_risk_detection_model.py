"""
QR Code Risk Detection Model
Uses scikit-learn to train a model for detecting risky QR codes
"""
import os
import re
import gdown
import pandas as pd
import numpy as np
import joblib
from urllib.parse import urlparse, parse_qs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# File paths
MODEL_FILE = 'qr_risk_model.joblib'
DATASET_FILE = 'qr_dataset.csv'
DATASET_URL = 'https://drive.google.com/uc?id=16Hc2TRGOBCmqB5lfTJdVAjvWlVRbaJET'

def download_dataset():
    """Download dataset from Google Drive if not present"""
    if not os.path.exists(DATASET_FILE):
        print(f"Downloading QR dataset from {DATASET_URL}...")
        gdown.download(DATASET_URL, DATASET_FILE, quiet=False)
        print("Dataset downloaded successfully.")
    else:
        print(f"Dataset already exists at {DATASET_FILE}")
    
    return pd.read_csv(DATASET_FILE)

def extract_upi_id(qr_content):
    """Extract UPI ID from QR content with improved detection"""
    try:
        # Standard UPI QR format
        match = re.search(r'pa=([\w%@._-]+)', qr_content)
        if match:
            upi_id = match.group(1)
            
            # If it's URL encoded, decode it
            if '%' in upi_id:
                try:
                    from urllib.parse import unquote
                    upi_id = unquote(upi_id)
                except:
                    pass
                    
            return upi_id
        
        # Look for direct UPI ID pattern (name@provider)
        possible_upis = re.findall(r'([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+)', qr_content)
        if possible_upis:
            return possible_upis[0]
            
        return None
    except:
        return None

def is_shortened_url(url):
    """Check if URL is shortened"""
    shortened_domains = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly']
    try:
        domain = urlparse(url).netloc
        return any(short_domain in domain for short_domain in shortened_domains)
    except:
        return False

def contains_urgency_keywords(text):
    """Check if text contains urgency keywords and return severity"""
    urgency_words = [
        # High severity keywords
        ('kyc', 3), ('verify', 3), ('urgent', 3), ('blocked', 3), ('suspend', 3),
        # Medium severity keywords
        ('alert', 2), ('warning', 2), ('expire', 2), ('immediately', 2), ('limited', 2),
        # Lower severity keywords
        ('confirm', 1), ('action', 1), ('required', 1), ('attention', 1)
    ]
    lower_text = text.lower()
    
    # Calculate severity score based on matching keywords
    severity = 0
    for word, score in urgency_words:
        if word in lower_text:
            severity += score
    
    return min(severity, 5)  # Cap at 5 for normalization

def extract_features(df):
    """Extract features from dataset"""
    print("Extracting features from UPI dataset...")
    
    # Create a copy of the dataframe for modifications
    df_features = df.copy()
    
    # Set the 'is_scam' column based on IS_FRAUD
    df_features['is_scam'] = df_features['IS_FRAUD'].astype(int)
    
    # Extract UPI IDs from BENEFICIARY_VPA
    df_features['upi_id'] = df_features['BENEFICIARY_VPA']
    
    # Create synthetic QR content from UPI data
    df_features['qr_content'] = df_features.apply(
        lambda row: f"upi://pay?pa={row['BENEFICIARY_VPA']}&pn=Merchant&am={row['AMOUNT']}", 
        axis=1
    )
    
    # Synthetic redirect URL (empty for most entries)
    df_features['redirect_url'] = ""
    
    # For some fraud entries, add shortened URLs to test the model
    fraud_mask = df_features['IS_FRAUD'] == 1
    df_features.loc[fraud_mask, 'redirect_url'] = df_features.loc[fraud_mask].apply(
        lambda _: 'http://bit.ly/abc123' if np.random.random() < 0.5 else '', 
        axis=1
    )
    
    # URL features
    df_features['is_shortened'] = df_features['redirect_url'].apply(
        lambda x: 1 if x and is_shortened_url(x) else 0
    )
    
    # Add account age - use transaction timestamp as proxy
    df_features['creation_date'] = pd.to_datetime(df_features['TXN_TIMESTAMP'], format='%d/%m/%Y %H:%M', errors='coerce')
    current_date = pd.to_datetime('today')
    df_features['account_age_days'] = (current_date - df_features['creation_date']).dt.days.fillna(30)
    
    # UPI ID features
    df_features['upi_id_length'] = df_features['upi_id'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
    df_features['upi_has_number'] = df_features['upi_id'].apply(
        lambda x: 1 if pd.notnull(x) and any(c.isdigit() for c in str(x)) else 0
    )
    
    # Text pattern features - check if urgency keywords in UPI ID or QR content
    df_features['urgency_keywords'] = df_features['qr_content'].apply(contains_urgency_keywords)
    
    # QR content complexity
    df_features['qr_content_length'] = df_features['qr_content'].apply(len)
    df_features['param_count'] = df_features['qr_content'].apply(lambda x: len(x.split('&')) if '&' in x else 1)
    
    # Use count of transactions per BENEFICIARY_VPA as report_count
    vpa_counts = df_features.groupby('BENEFICIARY_VPA').size().reset_index(name='vpa_count')
    df_features = df_features.merge(vpa_counts, on='BENEFICIARY_VPA', how='left')
    df_features['report_count'] = df_features.apply(
        lambda row: int(row['vpa_count'] * 0.1) if row['IS_FRAUD'] == 1 else 0, 
        axis=1
    )
    
    print(f"Extracted features for {len(df_features)} transactions")
    
    # Final feature set
    features = [
        'is_shortened', 'account_age_days', 'urgency_keywords', 'report_count',
        'upi_id_length', 'upi_has_number', 'qr_content_length', 'param_count'
    ]
    
    return df_features[features], df_features['is_scam']

def train_model():
    """Train QR risk detection model"""
    print("Training QR risk detection model...")
    
    # Download and load dataset
    df = download_dataset()
    
    # Extract features
    X, y = extract_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    
    # Feature importance
    features = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature importance:")
    for i in range(len(features)):
        print(f"  {features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return model

def prepare_input_features(qr_content, redirect_url=None, report_count=0):
    """Prepare input features for prediction from QR content"""
    upi_id = extract_upi_id(qr_content)
    
    features = {
        'is_shortened': 1 if redirect_url and is_shortened_url(redirect_url) else 0,
        'account_age_days': 0,  # Default to 0 days (new account) for maximum caution
        'urgency_keywords': contains_urgency_keywords(qr_content),
        'report_count': report_count,
        'upi_id_length': len(upi_id) if upi_id else 0,
        'upi_has_number': 1 if upi_id and any(c.isdigit() for c in upi_id) else 0,
        'qr_content_length': len(qr_content),
        'param_count': len(qr_content.split('&')) if '&' in qr_content else 1
    }
    
    # Convert to DataFrame with proper column order
    columns = [
        'is_shortened', 'account_age_days', 'urgency_keywords', 'report_count',
        'upi_id_length', 'upi_has_number', 'qr_content_length', 'param_count'
    ]
    
    input_df = pd.DataFrame([features], columns=columns)
    return input_df

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def analyze_qr_risk(qr_content, redirect_url=None, report_count=0):
    """Analyze risk of a QR code"""
    # Load model if exists, otherwise train new one
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        model = train_model()
    
    # Prepare features
    features = prepare_input_features(qr_content, redirect_url, report_count)
    
    # Make prediction
    is_scam = model.predict(features)[0]
    risk_score = model.predict_proba(features)[0][1]  # Probability of is_scam==1
    
    # Get feature importance for explanation
    feature_importance = {}
    for i, col in enumerate(features.columns):
        value = features.iloc[0, i]
        feature_importance[col] = {
            'value': convert_to_serializable(value),
            'importance': convert_to_serializable(model.feature_importances_[i])
        }
    
    # Adjust risk score based on urgency keywords
    urgency_score = features['urgency_keywords'].iloc[0]
    if urgency_score >= 3:
        # High urgency words found - boost risk score
        risk_score = max(risk_score, 0.75) 
    elif urgency_score > 0:
        # Some urgency words found - ensure minimum risk
        risk_score = max(risk_score, 0.5)
    
    # Risk level based on score
    risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"
    
    # Generate explanation
    explanation = []
    
    # Add urgency keyword explanation with appropriate severity
    if urgency_score >= 3:
        explanation.append("Contains high-risk keywords commonly used in scams (KYC, urgent, verify)")
    elif urgency_score > 0:
        explanation.append("Contains suspicious urgency-related keywords")
    
    # Add other risk explanations
    if features['is_shortened'].iloc[0] > 0:
        explanation.append("Uses shortened URLs which can hide real destinations")
    if report_count > 0:
        explanation.append(f"Has been reported {report_count} times by users")
    if features['param_count'].iloc[0] > 5:
        explanation.append("Contains an unusually high number of parameters")
    if features['account_age_days'].iloc[0] < 7:
        explanation.append("Account is less than 7 days old")
    
    # Default explanation if none found
    if not explanation:
        explanation.append("No specific risk factors identified" if risk_score < 0.3 else 
                          "Multiple minor risk factors combined")
    
    # Build result dict and ensure all values are JSON-serializable
    result = {
        'is_scam': bool(is_scam),
        'risk_score': int(risk_score * 100),
        'risk_level': risk_level,
        'explanation': explanation,
        'features': {k: convert_to_serializable(v['value']) for k, v in feature_importance.items()},
        'feature_importance': {k: convert_to_serializable(float(v['importance'])) for k, v in feature_importance.items()}
    }
    
    return result

def main():
    """Main function for standalone execution"""
    # Train model
    model = train_model()
    
    # Test prediction
    test_qr = "upi://pay?pa=test123@okaxis&pn=Test%20Merchant&am=500.00"
    result = analyze_qr_risk(test_qr)
    
    print("\nTest prediction:")
    print(f"QR content: {test_qr}")
    print(f"Risk assessment: {result['risk_level']} risk ({result['risk_score']}%)")
    print(f"Explanation: {', '.join(result['explanation'])}")

if __name__ == "__main__":
    main()