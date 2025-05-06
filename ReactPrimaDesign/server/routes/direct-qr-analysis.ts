import { Router } from 'express';
import { log } from '../vite';

/**
 * Direct QR analysis implementation in TypeScript
 * This serves as a fallback when the Python ML service is not available
 */

// Create a dedicated router for direct QR analysis
export function createDirectQRRouter() {
  const router = Router();

  // Analyze a QR code text with ML model if available
  router.post('/predict', async (req, res) => {
    try {
      const { qr_text, redirect_url, report_count } = req.body;
      
      if (!qr_text) {
        return res.status(400).json({ 
          error: 'Missing QR text',
          message: 'Please provide qr_text in the request body' 
        });
      }
      
      // Try to use the ML model directly via Python
      try {
        // Use child_process for ES modules compatibility
        const childProcess = await import('child_process');
        
        // Properly escape inputs for shell execution
        const escapedQrText = qr_text.replace(/"/g, '\\"').replace(/'/g, "\\'");
        const escapedRedirectUrl = redirect_url ? redirect_url.replace(/"/g, '\\"').replace(/'/g, "\\'") : '';
        const reportCountValue = report_count || 0;
        
        const pythonCommand = `python -c "
import sys
import json
from qr_risk_detection_model import analyze_qr_risk

try:
    result = analyze_qr_risk(
        qr_content='${escapedQrText}',
        redirect_url='${escapedRedirectUrl}',
        report_count=${reportCountValue}
    )
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"`;
        
        const output = childProcess.execSync(pythonCommand, { timeout: 5000 }).toString().trim();
        const mlResult = JSON.parse(output);
        
        if (mlResult.error) {
          throw new Error(mlResult.error);
        }
        
        log(`ML-based QR analysis: ${qr_text} → ${mlResult.risk_score}%`, 'qrscan');
        
        // Add ML source flag
        mlResult.analysis_type = 'ml_model';
        
        return res.status(200).json(mlResult);
      } catch (mlError) {
        // Fall back to TypeScript implementation
        log(`ML model failed, using TS fallback: ${mlError.message}`, 'qrscan');
        const fallbackResult = analyzeQRText(qr_text);
        
        // Add fallback info
        fallbackResult.analysis_type = 'typescript_fallback';
        fallbackResult.fallback_reason = mlError.message;
        
        log(`Direct QR analysis: ${qr_text} → ${fallbackResult.risk_score}%`, 'qrscan');
        
        return res.status(200).json(fallbackResult);
      }
    } catch (error) {
      console.error('Error in direct QR analysis:', error);
      return res.status(500).json({ 
        error: 'Analysis failed',
        message: 'Failed to analyze QR code',
        details: error.message
      });
    }
  });

  // Process user feedback (for learning)
  router.post('/feedback', (req, res) => {
    try {
      const { qr_text, is_scam } = req.body;
      
      if (!qr_text) {
        return res.status(400).json({ error: 'Missing qr_text parameter' });
      }
      
      if (is_scam === undefined) {
        return res.status(400).json({ error: 'Missing is_scam parameter' });
      }
      
      // Store feedback for future model improvements
      // In a real system, this would update our model
      log(`Received direct QR feedback: ${qr_text} → ${is_scam ? 'scam' : 'safe'}`, 'qrscan');
      
      return res.status(200).json({ status: 'feedback_recorded' });
    } catch (error) {
      console.error('Error processing QR feedback:', error);
      return res.status(500).json({ error: 'Failed to process feedback' });
    }
  });

  return router;
}

/**
 * Calculate string entropy (information density)
 * Higher entropy often correlates with potentially suspicious content
 */
function calculateStringEntropy(str: string): number {
  if (!str || str.length === 0) return 0;
  
  // Calculate character frequency
  const charFreq: Record<string, number> = {};
  for (let i = 0; i < str.length; i++) {
    const char = str[i];
    charFreq[char] = (charFreq[char] || 0) + 1;
  }
  
  // Calculate entropy
  let entropy = 0;
  Object.values(charFreq).forEach(freq => {
    const p = freq / str.length;
    entropy -= p * Math.log2(p);
  });
  
  // Normalize to 0-1 range (typical English text has entropy around 4-5)
  return Math.min(entropy / 6, 1);
}

/**
 * Analyze QR code text for potential fraud/risk
 * This serves as a fallback when the ML model fails
 */
function analyzeQRText(qrText: string): { 
  risk_score: number; 
  latency_ms: number;
  features?: Record<string, number>;
  analysis_type?: string;
  fallback_reason?: string;
} {
  const startTime = Date.now();
  
  // Basic risk features
  const isUpi = qrText.toLowerCase().startsWith('upi://');
  const hasUpi = qrText.includes('@');
  const entropy = calculateStringEntropy(qrText);
  const numParams = (qrText.match(/&/g) || []).length + (qrText.match(/\?/g) || []).length;
  const length = qrText.length;
  
  // Extract UPI ID for additional checks
  let upiId = '';
  const upiMatch = qrText.match(/pa=([^&]+)/);
  if (upiMatch) {
    upiId = upiMatch[1];
  } else {
    // Try direct format
    const directMatch = qrText.match(/([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+)/);
    if (directMatch) {
      upiId = directMatch[0];
    }
  }

  // Check UPI domain trustworthiness
  let isDomainSafe = false;
  let isUnknownDomain = false;
  
  if (upiId && upiId.includes('@')) {
    const domain = upiId.split('@')[1].toLowerCase();
    const safeUpiDomains = [
      'oksbi', 'okaxis', 'okicici', 'okhdfcbank', 'ybl', 'upi', 
      'paytm', 'apl', 'ibl', 'gpay', 'fbl', 'yapl'
    ];
    isDomainSafe = safeUpiDomains.includes(domain);
    isUnknownDomain = !isDomainSafe;
  }
  
  // Regular patterns (UPI format)
  const upiRegex = /upi:\/\/pay\?[a-zA-Z0-9=&@\.-]+/;
  const isValidUpiFormat = upiRegex.test(qrText);
  
  // Suspicious keywords with weighted severity
  const highRiskKeywords = ['kyc', 'verify', 'urgent', 'blocked', 'suspend'];
  const mediumRiskKeywords = ['alert', 'warning', 'expire', 'immediately', 'limited'];
  const lowRiskKeywords = ['confirm', 'action', 'required', 'attention'];
  
  const qrLower = qrText.toLowerCase();
  
  // Count keywords by severity
  const highRiskCount = highRiskKeywords.reduce((count, keyword) => {
    return count + (qrLower.includes(keyword) ? 1 : 0);
  }, 0);
  
  const mediumRiskCount = mediumRiskKeywords.reduce((count, keyword) => {
    return count + (qrLower.includes(keyword) ? 1 : 0);
  }, 0);
  
  const lowRiskCount = lowRiskKeywords.reduce((count, keyword) => {
    return count + (qrLower.includes(keyword) ? 1 : 0);
  }, 0);
  
  // Check for shortened URLs
  const hasShortUrl = /bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly|is\.gd/.test(qrLower);
  
  // Calculate base risk score
  let riskScore = 5; // Start with minimal risk
  
  // Legitimate UPI QR codes usually have lower risk
  if (isUpi && isValidUpiFormat) {
    // Valid UPI format
    riskScore += 5;
  } else if (isUpi) {
    // Invalid UPI format but claims to be UPI
    riskScore += 40;
  }
  
  // Add risk based on suspicious patterns with weighted severity
  riskScore += highRiskCount * 15;   // High-risk keywords have greater impact
  riskScore += mediumRiskCount * 10; // Medium-risk keywords
  riskScore += lowRiskCount * 5;     // Low-risk keywords
  
  // Shortened URLs are a significant risk
  if (hasShortUrl) {
    riskScore += 30;
  }
  
  // UPI domain safety checks
  if (isDomainSafe) {
    riskScore -= 25; // Known safe domains lower risk
  } else if (isUnknownDomain) {
    riskScore += 15; // Unknown domains increase risk
  }
  
  // Additional risk factors
  riskScore += numParams > 5 ? 15 : 0;  // Too many parameters is suspicious 
  riskScore += entropy * 10;            // Higher entropy adds up to 10%
  
  // Very long QR codes might be suspicious
  if (length > 200) {
    riskScore += 15;
  }
  
  // Adjustment based on high-risk keywords in UPI ID specifically
  if (upiId) {
    const upiLower = upiId.toLowerCase();
    const hasHighRiskInUpi = highRiskKeywords.some(keyword => upiLower.includes(keyword));
    
    if (hasHighRiskInUpi) {
      riskScore += 25; // Major risk increase for high-risk words in UPI ID itself
    }
  }
  
  // Ensure risk score is within bounds
  riskScore = Math.max(0, Math.min(100, riskScore));
  
  // Features for debugging and model comparison
  const features = {
    length,
    has_upi: hasUpi ? 1 : 0,
    is_upi: isUpi ? 1 : 0,
    num_params: numParams,
    entropy: Math.round(entropy * 100) / 100,
    high_risk_keywords: highRiskCount,
    medium_risk_keywords: mediumRiskCount,
    low_risk_keywords: lowRiskCount,
    has_short_url: hasShortUrl ? 1 : 0,
    is_domain_safe: isDomainSafe ? 1 : 0,
    is_unknown_domain: isUnknownDomain ? 1 : 0,
    valid_upi_format: isValidUpiFormat ? 1 : 0,
    upi_id_length: upiId ? upiId.length : 0
  };
  
  const latencyMs = Date.now() - startTime;
  
  return {
    risk_score: Math.round(riskScore),
    latency_ms: latencyMs,
    features,
    analysis_type: 'typescript_fallback'
  };
}