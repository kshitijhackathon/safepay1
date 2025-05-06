"""
UPI Fraud Detection Model
Uses scikit-learn to train a model for detecting fraudulent UPI transactions
Based on the actual dataset structure
"""
import os
import re
import gdown
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# File paths
MODEL_FILE = 'upi_fraud_model.joblib'
DATASET_FILE = 'qr_dataset.csv'
DATASET_URL = 'https://drive.google.com/uc?id=16Hc2TRGOBCmqB5lfTJdVAjvWlVRbaJET'
SCALER_FILE = 'upi_scaler.joblib'

def download_dataset():
    """Download dataset from Google Drive if not present"""
    if not os.path.exists(DATASET_FILE):
        print(f"Downloading UPI dataset from {DATASET_URL}...")
        gdown.download(DATASET_URL, DATASET_FILE, quiet=False)
        print("Dataset downloaded successfully.")
    else:
        print(f"Dataset already exists at {DATASET_FILE}")
    
    return pd.read_csv(DATASET_FILE)

def extract_features(df):
    """Extract features from UPI transaction dataset"""
    print("Extracting features from UPI transaction dataset...")
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Dataset shape: {df.shape}")
    
    # Create features dataframe
    features_df = pd.DataFrame()
    
    # Extract fraud label
    if 'IS_FRAUD' in df.columns:
        y = df['IS_FRAUD'].astype(int)
        print(f"Fraud distribution: {y.value_counts().to_dict()}")
    else:
        # Default all transactions as safe if no fraud indicator
        y = pd.Series([0] * len(df))
        print("WARNING: No fraud indicator column found. Assuming all transactions are safe.")
    
    # Transaction amount features
    if 'AMOUNT' in df.columns:
        # Basic amount
        features_df['amount'] = df['AMOUNT'].astype(float)
        # Amount thresholds
        features_df['amount_high'] = df['AMOUNT'].apply(lambda x: 1 if float(x) > 5000 else 0)
        features_df['amount_very_high'] = df['AMOUNT'].apply(lambda x: 1 if float(x) > 10000 else 0)
        features_df['amount_round'] = df['AMOUNT'].apply(lambda x: 1 if float(x) % 1000 == 0 else 0)
        # Amount log transform for better distribution
        features_df['amount_log'] = np.log1p(df['AMOUNT'].astype(float))
    
    # UPI ID features
    vpa_columns = ['PAYER_VPA', 'BENEFICIARY_VPA']
    for col in vpa_columns:
        if col in df.columns:
            # UPI ID length
            features_df[f'{col.lower()}_length'] = df[col].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
            
            # Extract domain from UPI ID
            df[f'{col}_DOMAIN'] = df[col].apply(lambda x: str(x).split('@')[-1] if '@' in str(x) else '')
            
            # Check for standard UPI domains
            standard_domains = ['oksbi', 'okaxis', 'okicici', 'ybl', 'paytm', 'okhdfcbank', 'okbizaxis']
            features_df[f'{col.lower()}_std_domain'] = df[f'{col}_DOMAIN'].apply(
                lambda x: 1 if any(domain in x.lower() for domain in standard_domains) else 0
            )
            
            # Complexity features
            features_df[f'{col.lower()}_has_numbers'] = df[col].apply(
                lambda x: 1 if any(c.isdigit() for c in str(x)) else 0
            )
            features_df[f'{col.lower()}_has_special'] = df[col].apply(
                lambda x: 1 if any(c in '._-+&' for c in str(x)) else 0
            )
            
            # Username length
            df[f'{col}_USERNAME'] = df[col].apply(lambda x: str(x).split('@')[0] if '@' in str(x) else str(x))
            features_df[f'{col.lower()}_username_len'] = df[f'{col}_USERNAME'].apply(len)
    
    # Transaction type features
    if 'TRANSACTION_TYPE' in df.columns:
        # One-hot encode transaction type
        tx_types = pd.get_dummies(df['TRANSACTION_TYPE'], prefix='tx_type')
        features_df = pd.concat([features_df, tx_types], axis=1)
    
    # Device features
    if 'DEVICE_ID' in df.columns:
        # Hash device ID for anonymity
        features_df['device_id_hash'] = df['DEVICE_ID'].apply(lambda x: hash(str(x)) % 1000)
    
    # Time-based features
    if 'TXN_TIMESTAMP' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['TXN_TIMESTAMP'])
            
            # Time components
            features_df['hour'] = df['datetime'].dt.hour
            features_df['day'] = df['datetime'].dt.day
            features_df['month'] = df['datetime'].dt.month
            features_df['day_of_week'] = df['datetime'].dt.dayofweek
            
            # Special time flags
            features_df['is_weekend'] = df['datetime'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
            features_df['is_night'] = df['datetime'].dt.hour.apply(lambda x: 1 if (x < 6 or x >= 22) else 0)
            features_df['is_evening'] = df['datetime'].dt.hour.apply(lambda x: 1 if (x >= 18 and x < 22) else 0)
        except Exception as e:
            print(f"Error processing timestamp: {e}")
    
    # Transaction status
    if 'TRN_STATUS' in df.columns:
        status_dummies = pd.get_dummies(df['TRN_STATUS'], prefix='status')
        features_df = pd.concat([features_df, status_dummies], axis=1)
    
    # Handle any missing values
    features_df = features_df.fillna(0)
    
    print(f"Created {len(features_df.columns)} features")
    print(f"Feature columns: {features_df.columns.tolist()}")
    
    return features_df, y

def train_model():
    """Train UPI fraud detection model"""
    print("Training UPI fraud detection model...")
    
    # Download and load dataset
    df = download_dataset()
    
    # Extract features
    X, y = extract_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for future use
    joblib.dump(scaler, SCALER_FILE)
    
    # Train model
    # Use class_weight='balanced' to handle imbalanced data
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Model performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    
    # Feature importance
    features = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 most important features:")
    for i in range(min(10, len(features))):
        print(f"  {features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return model, scaler

def process_upi_qr(qr_content):
    """Extract UPI data from QR content for fraud prediction"""
    upi_data = {}
    
    # Extract UPI ID
    upi_id_match = re.search(r'pa=([^&]+)', qr_content)
    if upi_id_match:
        upi_id = upi_id_match.group(1)
        upi_data['beneficiary_vpa'] = upi_id
        
        # Split UPI ID
        parts = upi_id.split('@')
        if len(parts) == 2:
            upi_data['username'] = parts[0]
            upi_data['domain'] = parts[1]
    
    # Extract amount
    amount_match = re.search(r'am=([^&]+)', qr_content)
    if amount_match:
        try:
            upi_data['amount'] = float(amount_match.group(1))
        except ValueError:
            upi_data['amount'] = 0
    
    # Extract payee name
    name_match = re.search(r'pn=([^&]+)', qr_content)
    if name_match:
        upi_data['payee_name'] = name_match.group(1)
    
    return upi_data

def prepare_features_from_qr(qr_content):
    """Prepare features from QR code for prediction"""
    # Extract UPI data
    upi_data = process_upi_qr(qr_content)
    
    # Create feature dictionary
    features = {}
    
    # Amount features
    amount = upi_data.get('amount', 0)
    features['amount'] = amount
    features['amount_high'] = 1 if amount > 5000 else 0
    features['amount_very_high'] = 1 if amount > 10000 else 0
    features['amount_round'] = 1 if amount % 1000 == 0 else 0
    features['amount_log'] = np.log1p(amount)
    
    # Beneficiary VPA features
    vpa = upi_data.get('beneficiary_vpa', '')
    features['beneficiary_vpa_length'] = len(vpa)
    
    domain = upi_data.get('domain', '')
    standard_domains = ['oksbi', 'okaxis', 'okicici', 'ybl', 'paytm', 'okhdfcbank', 'okbizaxis']
    features['beneficiary_vpa_std_domain'] = 1 if any(d in domain.lower() for d in standard_domains) else 0
    
    features['beneficiary_vpa_has_numbers'] = 1 if any(c.isdigit() for c in vpa) else 0
    features['beneficiary_vpa_has_special'] = 1 if any(c in '._-+&' for c in vpa) else 0
    
    username = upi_data.get('username', '')
    features['beneficiary_vpa_username_len'] = len(username)
    
    # Add default values for other features
    # Payer VPA features (defaults)
    features['payer_vpa_length'] = 0
    features['payer_vpa_std_domain'] = 1  # Assume sender uses standard domain
    features['payer_vpa_has_numbers'] = 0
    features['payer_vpa_has_special'] = 0
    features['payer_vpa_username_len'] = 0
    
    # Transaction type
    features['tx_type_P2P'] = 0
    features['tx_type_P2M'] = 1  # Assume merchant payment
    
    # Time features (current time)
    import datetime
    now = datetime.datetime.now()
    features['hour'] = now.hour
    features['day'] = now.day
    features['month'] = now.month
    features['day_of_week'] = now.weekday()
    features['is_weekend'] = 1 if now.weekday() >= 5 else 0
    features['is_night'] = 1 if (now.hour < 6 or now.hour >= 22) else 0
    features['is_evening'] = 1 if (now.hour >= 18 and now.hour < 22) else 0
    
    # Transaction status
    features['status_COMPLETED'] = 1  # Assume transaction will complete
    features['status_FAILED'] = 0     # Add the missing feature
    
    # Add device hash
    features['device_id_hash'] = 0    # Add the missing feature
    
    return features

def predict_fraud_risk(qr_content):
    """Predict fraud risk for a QR code"""
    # Load model and scaler
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print("Training new model...")
        model, scaler = train_model()
    else:
        print("Loading existing model...")
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    
    # Extract UPI data for reporting
    upi_data = process_upi_qr(qr_content)
    
    # Prepare features
    features = prepare_features_from_qr(qr_content)
    
    # Create DataFrame with proper column order
    model_features = pd.DataFrame([features])
    
    # Scale features
    model_features_scaled = scaler.transform(model_features)
    
    # Make prediction
    fraud_prob = model.predict_proba(model_features_scaled)[0, 1]
    is_fraud = model.predict(model_features_scaled)[0]
    
    # Determine risk level
    if fraud_prob < 0.3:
        risk_level = "Low"
    elif fraud_prob < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    # Generate explanation
    explanation = []
    
    # Amount-based explanations
    if features['amount_very_high']:
        explanation.append("Transaction amount is very high (>₹10,000)")
    elif features['amount_high']:
        explanation.append("Transaction amount is high (>₹5,000)")
    
    # UPI ID explanations
    if not features['beneficiary_vpa_std_domain']:
        explanation.append("Recipient UPI ID uses non-standard domain")
    
    if features['beneficiary_vpa_username_len'] > 20:
        explanation.append("Unusually long UPI username")
    
    # Time-based explanations
    if features['is_night']:
        explanation.append("Transaction occurs during night hours")
    
    # Add default explanation if none generated
    if not explanation:
        if fraud_prob > 0.5:
            explanation.append("Multiple minor risk factors combined")
        else:
            explanation.append("No specific risk factors identified")
    
    # Feature importance for this prediction
    if hasattr(model, 'feature_importances_'):
        important_features = []
        importances = model.feature_importances_
        feature_names = model_features.columns
        
        # Find the top 3 features that contributed to this prediction
        for i in np.argsort(importances)[-3:]:
            important_features.append({
                'name': feature_names[i],
                'importance': float(importances[i])
            })
    
    # Create result
    result = {
        'upi_data': upi_data,
        'fraud_probability': float(fraud_prob),
        'is_fraud': bool(is_fraud),
        'risk_score': int(fraud_prob * 100),
        'risk_level': risk_level,
        'explanation': explanation,
        'important_features': important_features if 'important_features' in locals() else []
    }
    
    return result

def main():
    """Main function to train and test the model"""
    # Train model
    model, scaler = train_model()
    
    # Test with sample QR codes
    test_qrs = [
        "upi://pay?pa=merchant@oksbi&pn=Trusted%20Store&am=500",
        "upi://pay?pa=urgent-verify@gmail.com&pn=URGENT%20VERIFY&am=9999",
        "upi://pay?pa=random123@randomdomain.com&pn=Test&am=500"
    ]
    
    print("\nTesting model with sample QR codes:")
    for qr in test_qrs:
        print(f"\nAnalyzing QR: {qr}")
        result = predict_fraud_risk(qr)
        print(f"  Fraud probability: {result['fraud_probability']:.2f}")
        print(f"  Risk score: {result['risk_score']}%")
        print(f"  Risk level: {result['risk_level']}")
        print(f"  Fraud prediction: {'Yes' if result['is_fraud'] else 'No'}")
        print(f"  Explanation: {', '.join(result['explanation'])}")

if __name__ == "__main__":
    main()