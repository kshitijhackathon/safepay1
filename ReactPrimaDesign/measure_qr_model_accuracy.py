"""
Measure QR Risk Model Accuracy
Evaluates the QR risk detection model performance across multiple metrics and test cases
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from qr_risk_detection_model import (
    download_dataset, extract_features, train_model, analyze_qr_risk
)

def download_and_prepare_data():
    """Download and prepare dataset for testing"""
    print("Downloading and preparing dataset...")
    df = download_dataset()
    X, y = extract_features(df)
    return X, y, df

def evaluate_model_performance(X, y):
    """Perform detailed evaluation of model performance"""
    print("\nEvaluating model performance...")
    
    # Load or train model
    if os.path.exists('qr_risk_model.joblib'):
        model = joblib.load('qr_risk_model.joblib')
        print("Loaded existing model for evaluation")
    else:
        model = train_model()
        print("Trained new model for evaluation")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"\nCross-validation results:")
    print(f"  Mean accuracy: {cv_scores.mean():.4f}")
    print(f"  Std deviation: {cv_scores.std():.4f}")
    
    # Split data for detailed metrics
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train on train set if needed
    if hasattr(model, 'fit'):
        model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nDetailed metrics on test set:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (True positive rate)")
    print(f"  Recall:    {recall:.4f} (Detection rate)")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    total = cm.sum()
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {tn} ({tn/total:.1%}) - Correctly identified safe QR codes")
    print(f"  False Positives: {fp} ({fp/total:.1%}) - Safe QR codes misclassified as scams")
    print(f"  False Negatives: {fn} ({fn/total:.1%}) - Scam QR codes missed")
    print(f"  True Positives:  {tp} ({tp/total:.1%}) - Correctly identified scam QR codes")
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X.columns
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature importance ranking:")
        for i in range(min(10, len(features))):
            print(f"  {i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Optimal threshold analysis
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_proba)
    # Find threshold that maximizes F1
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    
    print(f"\nOptimal threshold for maximum F1 score: {optimal_threshold:.4f}")
    print(f"  At this threshold, precision: {precision_curve[optimal_idx]:.4f}, recall: {recall_curve[optimal_idx]:.4f}")
    
    return model, X_test, y_test, y_proba, fpr, tpr, roc_auc

def test_with_custom_examples(model=None):
    """Test with a range of custom examples to demonstrate model behavior"""
    print("\nTesting model with custom examples...")
    
    test_cases = [
        # Clear safe cases
        "upi://pay?pa=merchant@oksbi&pn=Trusted%20Store&am=500",
        "upi://pay?pa=shop@okaxis&pn=Retail%20Store&am=1000",
        
        # Clear scam cases
        "upi://pay?pa=urgent@verify.com&pn=URGENT%20VERIFY%20ACCOUNT&am=9999&tn=Refund%20Verification",
        "http://bit.ly/3xScam?pa=scam@domain&redirect=true&am=5000",
        
        # Ambiguous cases
        "upi://pay?pa=service@gmail.com&pn=Service%20Provider&am=2500",
        "upi://pay?pa=help@support.net&pn=Tech%20Support&am=750&tn=Support%20Fee",
        
        # Edge cases
        "upi://pay?pa=really.long.username.with.numbers.123456@okicici&pn=Long%20Name%20With%20Numbers&am=1",
        "upi://pay?pa=merchant@paytm&pn=&am=&tn=&cu=INR&mc=&tid=",  # Minimal parameters
    ]
    
    results = []
    for i, qr in enumerate(test_cases):
        print(f"\nTest case {i+1}: {qr}")
        # Analyze using our standalone function
        result = analyze_qr_risk(qr)
        
        risk_score = result["risk_score"]
        risk_level = result["risk_level"]
        is_scam = result["is_scam"]
        explanation = result["explanation"]
        
        print(f"  Risk Score: {risk_score}%")
        print(f"  Risk Level: {risk_level}")
        print(f"  Classified as: {'Scam' if is_scam else 'Safe'}")
        print(f"  Explanation: {explanation}")
        
        results.append({
            'qr_content': qr,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'is_scam': is_scam,
            'explanation': explanation
        })
    
    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    
    print("\nSummary of test cases:")
    print(f"  Average risk score: {results_df['risk_score'].mean():.1f}%")
    print(f"  Safe cases: {(~results_df['is_scam']).sum()}")
    print(f"  Scam cases: {results_df['is_scam'].sum()}")
    
    return results_df

def main():
    """Main function to measure model accuracy"""
    print("Measuring QR Risk Detection Model Accuracy\n" + "="*40)
    
    try:
        # Download and prepare data
        X, y, raw_df = download_and_prepare_data()
        
        # Evaluate model performance
        model, X_test, y_test, y_proba, fpr, tpr, roc_auc = evaluate_model_performance(X, y)
        
        # Test with custom examples
        results_df = test_with_custom_examples()
        
        print("\nModel accuracy measurement complete.")
        
    except Exception as e:
        print(f"Error during accuracy measurement: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()