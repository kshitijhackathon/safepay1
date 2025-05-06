/**
 * SIM Swap Protection Routes
 * Handles API endpoints for SIM swap detection and Multi-Factor Authentication
 */

import { Express, Request, Response } from 'express';
import { 
  checkSimSwap, 
  initiateMfa, 
  verifyOtp, 
  verifyBiometric, 
  verifyLocation, 
  completeMfaVerification 
} from '../services/sim-swap-detection';

export function registerSimSwapProtectionRoutes(app: Express): void {
  /**
   * Check if a SIM swap has occurred for a phone number
   * GET /api/security/check-sim-swap
   */
  app.get('/api/security/check-sim-swap', async (req: Request, res: Response) => {
    try {
      const { phoneNumber } = req.query;
      
      if (!phoneNumber || typeof phoneNumber !== 'string') {
        return res.status(400).json({ error: 'Phone number is required' });
      }
      
      const result = await checkSimSwap(phoneNumber);
      res.json(result);
    } catch (error) {
      console.error('Error checking SIM swap:', error);
      res.status(500).json({ error: 'Failed to check SIM swap status' });
    }
  });
  
  /**
   * Initiate Multi-Factor Authentication
   * POST /api/security/initiate-mfa
   */
  app.post('/api/security/initiate-mfa', async (req: Request, res: Response) => {
    try {
      const { userId, phoneNumber, reason, channel } = req.body;
      
      if (!userId || !phoneNumber) {
        return res.status(400).json({ error: 'User ID and phone number are required' });
      }
      
      const result = await initiateMfa({
        userId: Number(userId),
        phoneNumber,
        reason,
        channel
      });
      
      res.json(result);
    } catch (error) {
      console.error('Error initiating MFA:', error);
      res.status(500).json({ error: 'Failed to initiate MFA' });
    }
  });
  
  /**
   * Verify OTP for MFA
   * POST /api/security/verify-otp
   */
  app.post('/api/security/verify-otp', async (req: Request, res: Response) => {
    try {
      const { userId, otpCode } = req.body;
      
      if (!userId || !otpCode) {
        return res.status(400).json({ error: 'User ID and OTP code are required' });
      }
      
      const result = await verifyOtp(Number(userId), otpCode);
      res.json(result);
    } catch (error) {
      console.error('Error verifying OTP:', error);
      res.status(500).json({ error: 'Failed to verify OTP' });
    }
  });
  
  /**
   * Verify biometric data for MFA
   * POST /api/security/verify-biometric
   */
  app.post('/api/security/verify-biometric', async (req: Request, res: Response) => {
    try {
      const { userId, biometricData } = req.body;
      
      if (!userId || !biometricData) {
        return res.status(400).json({ error: 'User ID and biometric data are required' });
      }
      
      const result = await verifyBiometric(Number(userId), biometricData);
      res.json(result);
    } catch (error) {
      console.error('Error verifying biometric:', error);
      res.status(500).json({ error: 'Failed to verify biometric data' });
    }
  });
  
  /**
   * Verify location for MFA
   * POST /api/security/verify-location
   */
  app.post('/api/security/verify-location', async (req: Request, res: Response) => {
    try {
      const { userId, location } = req.body;
      
      if (!userId || !location || !location.lat || !location.lng) {
        return res.status(400).json({ error: 'User ID and location data are required' });
      }
      
      const result = await verifyLocation(Number(userId), location);
      res.json(result);
    } catch (error) {
      console.error('Error verifying location:', error);
      res.status(500).json({ error: 'Failed to verify location' });
    }
  });
  
  /**
   * Complete MFA verification process
   * POST /api/security/complete-mfa
   */
  app.post('/api/security/complete-mfa', async (req: Request, res: Response) => {
    try {
      const { 
        userId, 
        phoneNumber, 
        verificationChannel, 
        biometricVerified, 
        geoLocation 
      } = req.body;
      
      if (!userId || !phoneNumber || !verificationChannel) {
        return res.status(400).json({ error: 'User ID, phone number, and verification channel are required' });
      }
      
      const result = await completeMfaVerification({
        userId: Number(userId),
        phoneNumber,
        verificationChannel,
        biometricVerified,
        geoLocation
      });
      
      res.json(result);
    } catch (error) {
      console.error('Error completing MFA:', error);
      res.status(500).json({ error: 'Failed to complete MFA verification' });
    }
  });
  
  /**
   * Get MFA settings for a user
   * GET /api/security/mfa-settings/:userId
   */
  app.get('/api/security/mfa-settings/:userId', async (req: Request, res: Response) => {
    try {
      const { userId } = req.params;
      
      // In a real implementation, this would fetch MFA settings from the database
      // For demo, we'll return mock settings
      
      res.json({
        userId: Number(userId),
        mfaEnabled: true,
        preferredChannel: 'email',
        biometricEnabled: true,
        lastVerified: new Date().toISOString(),
        securityLevel: 'high'
      });
    } catch (error) {
      console.error('Error fetching MFA settings:', error);
      res.status(500).json({ error: 'Failed to fetch MFA settings' });
    }
  });
  
  /**
   * Update MFA settings for a user
   * PUT /api/security/mfa-settings/:userId
   */
  app.put('/api/security/mfa-settings/:userId', async (req: Request, res: Response) => {
    try {
      const { userId } = req.params;
      const { mfaEnabled, preferredChannel, biometricEnabled, securityLevel } = req.body;
      
      // In a real implementation, this would update MFA settings in the database
      // For demo, we'll return success
      
      res.json({
        userId: Number(userId),
        mfaEnabled,
        preferredChannel,
        biometricEnabled,
        securityLevel,
        updated: true,
        updatedAt: new Date().toISOString()
      });
    } catch (error) {
      console.error('Error updating MFA settings:', error);
      res.status(500).json({ error: 'Failed to update MFA settings' });
    }
  });
}