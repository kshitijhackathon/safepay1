/**
 * ML-powered QR Code Scanner Routes
 * Proxies requests to the Python ML service
 */

import { Express, Request, Response } from 'express';

// ML service URL 
// Use consistent port configuration
const QR_ML_SERVICE_PORT = process.env.ML_QR_SERVICE_PORT || 8081;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || `http://localhost:${QR_ML_SERVICE_PORT}`;

// Fallback ML analysis for when service is unreachable
function getFallbackAnalysis(qrText: string) {
  console.log('Using fallback ML analysis for:', qrText);
  
  // Basic pattern matching for UPI ID safety
  const hasUpiPattern = qrText.includes('upi://') || qrText.includes('@');
  const suspiciousPatterns = [
    /scam/i, /fraud/i, /urgent/i, /emergency/i, /payment/i, 
    /\d{10,}/, // Long number sequences
    /[A-Za-z0-9]{20,}/ // Very long alphanumeric strings
  ];
  
  const detectedPatterns = suspiciousPatterns
    .filter(pattern => pattern.test(qrText))
    .map(pattern => pattern.toString());
  
  // Analyze string entropy as a basic measure
  const entropy = calculateStringEntropy(qrText);
  
  // Determine risk level based on patterns and entropy
  let riskScore = 0.2; // Start with low risk
  
  if (detectedPatterns.length > 0) {
    riskScore += detectedPatterns.length * 0.15; // Add for each suspicious pattern
  }
  
  if (entropy > 4.5) {
    riskScore += 0.2; // Higher entropy often means more randomness
  }
  
  // Cap at 0.95 for fallback
  riskScore = Math.min(0.95, riskScore);
  
  return {
    risk_score: riskScore,
    confidence: 0.7,
    features: {
      pattern_match: detectedPatterns.length > 0 ? 0.7 : 0.2,
      domain_check: hasUpiPattern ? 0.3 : 0.6,
      syntax_validation: hasUpiPattern ? 0.2 : 0.7,
      entropy: entropy / 5, // Normalize to 0-1 range
      length_score: qrText.length > 50 ? 0.7 : 0.3
    },
    scan_time_ms: 5,
    detected_patterns: detectedPatterns.length > 0 ? detectedPatterns : ['NONE'],
    message: "Fallback analysis (ML service unavailable)"
  };
}

// Helper to calculate string entropy
function calculateStringEntropy(str: string): number {
  const len = str.length;
  const frequencies: Record<string, number> = {};
  
  // Count character frequencies
  for (let i = 0; i < len; i++) {
    const char = str[i];
    frequencies[char] = (frequencies[char] || 0) + 1;
  }
  
  // Calculate entropy
  return Object.values(frequencies).reduce((entropy, count) => {
    const p = count / len;
    return entropy - p * Math.log2(p);
  }, 0);
}

export function registerMLQRScanRoutes(app: Express): void {
  /**
   * Predict risk for a QR code
   * POST /api/ml/qr-scan
   */
  app.post('/api/ml/qr-scan', async (req: Request, res: Response) => {
    try {
      const { qr_text } = req.body;
      
      if (!qr_text) {
        return res.status(400).json({ error: 'QR text is required' });
      }
      
      try {
        // Try to forward the request to the ML service
        const response = await fetch(`${ML_SERVICE_URL}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ qr_text }),
          // Set a timeout to avoid long waiting times
          signal: AbortSignal.timeout(2000)
        });
        
        if (response.ok) {
          const data = await response.json();
          return res.json(data);
        }
        
        console.warn(`ML service error: ${response.status} ${response.statusText}. Using fallback analysis.`);
        // Fall through to fallback analysis
      } catch (mlError: any) {
        console.warn(`ML service unavailable: ${mlError.message || mlError}. Using fallback analysis.`);
        // Fall through to fallback analysis
      }
      
      // Use fallback analysis when ML service is unavailable or responds with an error
      const fallbackAnalysis = getFallbackAnalysis(qr_text);
      res.json(fallbackAnalysis);
    } catch (error: any) {
      console.error('Error processing QR scan request:', error.message || error);
      res.status(500).json({ error: 'Internal server error processing QR code' });
    }
  });

  /**
   * Submit feedback about a QR code
   * POST /api/ml/qr-scan/feedback
   */
  app.post('/api/ml/qr-scan/feedback', async (req: Request, res: Response) => {
    try {
      const { qr_text, is_scam } = req.body;
      
      if (!qr_text || typeof is_scam !== 'boolean') {
        return res.status(400).json({ error: 'QR text and is_scam boolean are required' });
      }
      
      // Forward the feedback to the ML service
      const response = await fetch(`${ML_SERVICE_URL}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ qr_text, is_scam }),
      });
      
      if (!response.ok) {
        console.error(`ML service feedback error: ${response.status} ${response.statusText}`);
        return res.status(response.status).json({ 
          error: 'ML service error',
          message: response.statusText
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error: any) {
      console.error('Error submitting QR scan feedback:', error.message || error);
      res.status(500).json({ error: 'Internal server error processing feedback' });
    }
  });

  /**
   * Batch analyze multiple QR codes
   * POST /api/ml/qr-scan/batch
   */
  app.post('/api/ml/qr-scan/batch', async (req: Request, res: Response) => {
    try {
      const { qr_texts } = req.body;
      
      if (!Array.isArray(qr_texts) || qr_texts.length === 0) {
        return res.status(400).json({ error: 'Array of QR texts is required' });
      }
      
      // Forward the batch request to the ML service
      const response = await fetch(`${ML_SERVICE_URL}/batch-predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ qr_texts }),
      });
      
      if (!response.ok) {
        console.error(`ML service batch error: ${response.status} ${response.statusText}`);
        return res.status(response.status).json({ 
          error: 'ML service error',
          message: response.statusText
        });
      }
      
      const data = await response.json();
      res.json(data);
    } catch (error: any) {
      console.error('Error processing batch QR scan:', error.message || error);
      res.status(500).json({ error: 'Internal server error processing batch' });
    }
  });

  /**
   * Health check for the ML service
   * GET /api/ml/qr-scan/health
   */
  app.get('/api/ml/qr-scan/health', async (req: Request, res: Response) => {
    try {
      // Check if the ML service is running
      const response = await fetch(`${ML_SERVICE_URL}/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        console.error(`ML service health check failed: ${response.status} ${response.statusText}`);
        return res.status(503).json({ 
          status: 'unavailable',
          message: 'ML service is not responding'
        });
      }
      
      const data = await response.json();
      res.json({
        status: 'healthy',
        ml_service: data
      });
    } catch (error: any) {
      console.error('Error checking ML service health:', error.message || error);
      res.status(503).json({ 
        status: 'unavailable',
        message: 'ML service is not responding'
      });
    }
  });
}