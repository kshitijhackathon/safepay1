/**
 * Advanced QR Scan Service Integration
 * Connects to the Python-based advanced QR scan service for enhanced security
 */

import { Router } from 'express';
import { log } from '../vite';
import fetch from 'node-fetch';
import { spawn } from 'child_process';

// Create router
const router = Router();

// QR ML Service URL
const QR_ML_SERVICE_PORT = process.env.ML_QR_SERVICE_PORT || 8081;
const QR_ML_SERVICE_URL = `http://localhost:${QR_ML_SERVICE_PORT}`;
let serviceProcess: any = null;

// Ensure service is running
async function ensureServiceRunning() {
  // Check if the service is already running
  try {
    const response = await fetch(`${QR_ML_SERVICE_URL}/`, { 
      method: 'GET',
      timeout: 1000  // Only wait 1 second to avoid blocking
    });
    
    if (response.ok) {
      log("✅ Advanced QR service is already running", "qrscan");
      return true;
    }
  } catch (error) {
    log("Starting advanced QR scan service...", "qrscan");
  }

  // Start the service if it's not running
  try {
    // First check if the process is already started
    if (serviceProcess) {
      log("Service process exists, checking status...", "qrscan");
      if (serviceProcess.killed) {
        log("Previous process was killed, restarting", "qrscan");
      } else {
        // Process exists and is not killed
        return true;
      }
    }

    // Start a new process
    serviceProcess = spawn('python', ['start_qr_service.py'], {
      detached: true,
      stdio: 'inherit'
    });

    // Log startup
    log(`Started QR service with PID: ${serviceProcess.pid}`, "qrscan");
    
    // Wait for service to start
    return new Promise((resolve) => {
      setTimeout(async () => {
        try {
          const checkResp = await fetch(`${QR_ML_SERVICE_URL}/`, { 
            method: 'GET',
            timeout: 2000
          });
          if (checkResp.ok) {
            log("✅ Advanced QR service started successfully", "qrscan");
            resolve(true);
          } else {
            log("⚠️ Service started but health check failed", "qrscan");
            resolve(false);
          }
        } catch (error) {
          log("⚠️ Service may still be starting up", "qrscan");
          resolve(false);
        }
      }, 3000); // Give 3 seconds for the service to start
    });
  } catch (error) {
    log(`Error starting QR service: ${error.message}`, "qrscan");
    return false;
  }
}

/**
 * Proxies requests to the QR ML service with automatic service startup
 * @param path - API path
 * @param req - Express request
 * @param res - Express response
 */
async function proxyToAdvancedMLService(path: string, req: any, res: any) {
  // First ensure the service is running
  await ensureServiceRunning();
  
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
  } catch (error) {
    log(`Advanced QR ML service error: ${error.message}`, 'qrscan');
    
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
      let riskScore = 50; // Default to medium risk for fallback
      
      // Apply scoring rules similar to those in the Python model
      if (features.has_upi === 1) {
        riskScore -= 25; // Major reduction for UPI QR codes
      }
      
      if (features.urgent === 1) {
        riskScore += 20; // Higher penalty for urgency indicators
      }
      
      // Return fallback result
      res.json({
        risk_score: Math.max(0, Math.min(100, riskScore)),
        latency_ms: 10,
        fallback: true,
        label: riskScore > 70 ? 'Scam' : riskScore > 30 ? 'Medium' : 'Safe',
        features: features
      });
    } else {
      // For other endpoints, return an error
      res.status(503).json({
        error: 'Advanced QR ML service unavailable',
        message: 'The QR analysis service is currently unavailable.'
      });
    }
  }
}

// QR predict endpoint
router.post('/predict', async (req, res) => {
  await proxyToAdvancedMLService('/predict', req, res);
});

// QR feedback endpoint
router.post('/feedback', async (req, res) => {
  await proxyToAdvancedMLService('/feedback', req, res);
});

// Health check endpoint
router.get('/', async (req, res) => {
  const serviceStatus = await ensureServiceRunning();
  res.json({ 
    status: serviceStatus ? 'ok' : 'starting',
    service: 'advanced-qr-scan'
  });
});

export default router;