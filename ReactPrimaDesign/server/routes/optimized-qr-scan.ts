import { Router, Request, Response } from 'express';
import { log } from '../vite';
import fetch from 'node-fetch';

// Create router
const router = Router();

// QR ML Service URL
// Use environment variable for port or fall back to a default
const QR_ML_SERVICE_PORT = process.env.ML_QR_SERVICE_PORT || 8081;
const QR_ML_SERVICE_URL = `http://localhost:${QR_ML_SERVICE_PORT}`;

/**
 * Proxies requests to the QR ML service
 * @param path - API path
 * @param req - Express request
 * @param res - Express response
 */
async function proxyToMLService(path: string, req: Request, res: Response) {
  try {
    // Forward the request to the ML service
    const mlResponse = await fetch(`${QR_ML_SERVICE_URL}${path}`, {
      method: req.method,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    if (!mlResponse.ok) {
      throw new Error(`ML service responded with status: ${mlResponse.status}`);
    }

    const data = await mlResponse.json();
    res.json(data);
  } catch (error: any) {
    log(`QR ML service error: ${error.message || 'Unknown error'}`, 'qrscan');
    
    // For predict endpoint, we'll return fallback prediction
    if (path === '/predict') {
      // Enhanced fallback risk assessment with UPI safety emphasis
      const qrText = req.body.qr_text || '';
      
      // Extract features for consistent output format
      const features = {
        length: qrText.length,
        has_upi: qrText.toLowerCase().startsWith('upi://') ? 1 : 0,
        num_params: (qrText.match(/&/g) || []).length,
        urgent: /urgent|emergency|immediate|kyc|expired|blocked/i.test(qrText) ? 1 : 0,
        payment: /payment|pay|amount|money|transfer/i.test(qrText) ? 1 : 0,
        currency: /inr|rs|\₹|rupee/i.test(qrText) ? 1 : 0
      };
      
      // Start with a moderate risk score
      let riskScore = 30;
      
      // ** FIXED: Apply significant score reduction for legitimate UPI codes **
      if (features.has_upi === 1) {
        riskScore -= 25; // Major reduction for UPI QR codes
        
        // Look for valid UPI ID pattern (pa=something@something)
        if (qrText.includes('pa=')) {
          const upiIdMatch = qrText.match(/pa=([^&]+)/);
          if (upiIdMatch && upiIdMatch[1].includes('@')) {
            riskScore -= 5; // Additional reduction for valid UPI ID format
          }
        }
      }
      
      // Adjust score based on suspicious patterns
      if (features.urgent === 1) {
        riskScore += 20; // Higher penalty for urgency indicators
      }
      
      // Adjust for excessive parameters
      if (features.num_params > 5) {
        riskScore += 10;
      } else if (features.num_params > 10) {
        riskScore += 20;
      }
      
      // Ensure score stays in valid range
      const finalScore = Math.max(0, Math.min(100, riskScore));
      
      res.json({
        risk_score: finalScore,
        latency_ms: 5,
        fallback: true,
        features: features
      });
    } else {
      // For other endpoints, return an error
      res.status(503).json({
        error: 'QR ML service unavailable',
        message: 'The QR analysis service is currently unavailable.'
      });
    }
  }
}

// QR predict endpoint
router.post('/predict', async (req, res) => {
  await proxyToMLService('/predict', req, res);
});

// QR feedback endpoint
router.post('/feedback', async (req, res) => {
  await proxyToMLService('/feedback', req, res);
});

// Health check endpoint
router.get('/', (req, res) => {
  res.json({ status: 'ok', service: 'optimized-qr-scan' });
});

export default router;