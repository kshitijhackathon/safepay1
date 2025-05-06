/**
 * Risk Analysis Routes
 * Handles transaction risk assessment API endpoints
 */

import { Express, Request, Response } from 'express';
import { analyzeTransactionRisk } from '../services/risk-scoring';
import { storage } from '../storage';

export function registerRiskAnalysisRoutes(app: Express): void {
  /**
   * Analyze transaction risk
   * POST /api/analyze-transaction
   */
  app.post('/api/analyze-transaction', async (req: Request, res: Response) => {
    try {
      const { 
        upiId, 
        amount, 
        userId, 
        note, 
        recipient,
        location,
        deviceInfo 
      } = req.body;
      
      // Basic validation
      if (!upiId || !amount) {
        return res.status(400).json({ 
          error: 'Missing required fields: upiId and amount are required' 
        });
      }
      
      if (typeof amount !== 'number' || amount <= 0) {
        return res.status(400).json({ 
          error: 'Invalid amount: must be a positive number' 
        });
      }
      
      // Perform risk analysis
      const riskAnalysis = await analyzeTransactionRisk({
        upiId,
        amount,
        userId: userId ? Number(userId) : undefined,
        note,
        recipient,
        timestamp: new Date(),
        location,
        deviceInfo
      });
      
      // Store the risk data for future analysis
      // If the UPI risk report doesn't exist, this will initialize it
      try {
        await storage.updateUpiRiskScore(upiId);
      } catch (error) {
        console.error('Error updating UPI risk score:', error);
        // Continue with response even if updating fails
      }
      
      // Add transaction to history if authenticated
      if (userId) {
        try {
          await storage.createTransaction({
            userId: Number(userId),
            amount,
            upiId,
            status: 'analyzed', // Not yet completed
            currency: 'INR',
            transactionType: 'payment',
            description: note || `Payment to ${recipient || upiId}`
          });
        } catch (error) {
          console.error('Error creating transaction record:', error);
          // Continue with response even if record creation fails
        }
      }
      
      // Return risk analysis
      res.json({
        upi_id: upiId,
        amount,
        ...riskAnalysis,
        timestamp: new Date().toISOString()
      });
      
    } catch (error) {
      console.error('Error analyzing transaction risk:', error);
      res.status(500).json({ 
        error: 'Failed to analyze transaction risk',
        message: error.message
      });
    }
  });
  
  /**
   * Get historical risk analysis for a UPI ID
   * GET /api/risk-history/:upiId
   */
  app.get('/api/risk-history/:upiId', async (req: Request, res: Response) => {
    try {
      const { upiId } = req.params;
      
      // Get UPI risk report
      const riskReport = await storage.getUpiRiskByUpiId(upiId);
      
      if (!riskReport) {
        return res.status(404).json({ 
          error: 'No risk data found for this UPI ID' 
        });
      }
      
      // Get scam reports
      const scamReports = await storage.getScamReportsByUpiId(upiId);
      
      // Get most common scam type
      const mostCommonScamType = await storage.getMostCommonScamType(upiId);
      
      res.json({
        upi_id: upiId,
        risk_score: riskReport.riskScore,
        reports_count: scamReports.length,
        last_updated: riskReport.updatedAt,
        most_common_scam_type: mostCommonScamType,
        reports: scamReports.map(report => ({
          id: report.id,
          type: report.scamType,
          description: report.description,
          amount_lost: report.amountLost,
          reported_at: report.createdAt
        }))
      });
      
    } catch (error) {
      console.error('Error getting risk history:', error);
      res.status(500).json({ 
        error: 'Failed to get risk history',
        message: error.message
      });
    }
  });
  
  /**
   * Get transaction risk stats for a user
   * GET /api/user/:userId/risk-stats
   */
  app.get('/api/user/:userId/risk-stats', async (req: Request, res: Response) => {
    try {
      const userId = parseInt(req.params.userId);
      
      if (isNaN(userId)) {
        return res.status(400).json({ error: 'Invalid user ID' });
      }
      
      // Get user's transactions
      const transactions = await storage.getTransactionsByUserId(userId);
      
      // Calculate stats
      const total = transactions.length;
      const highRiskCount = transactions.filter(t => t.riskScore && t.riskScore > 0.7).length;
      const mediumRiskCount = transactions.filter(t => t.riskScore && t.riskScore > 0.3 && t.riskScore <= 0.7).length;
      const lowRiskCount = transactions.filter(t => !t.riskScore || t.riskScore <= 0.3).length;
      
      // Calculate average amount
      const totalAmount = transactions.reduce((sum, t) => sum + (t.amount || 0), 0);
      const averageAmount = total > 0 ? totalAmount / total : 0;
      
      res.json({
        userId,
        total_transactions: total,
        risk_distribution: {
          high_risk: highRiskCount,
          medium_risk: mediumRiskCount,
          low_risk: lowRiskCount,
          high_risk_percentage: total > 0 ? (highRiskCount / total) * 100 : 0,
          medium_risk_percentage: total > 0 ? (mediumRiskCount / total) * 100 : 0,
          low_risk_percentage: total > 0 ? (lowRiskCount / total) * 100 : 0
        },
        average_amount: averageAmount,
        total_amount: totalAmount,
        currency: 'INR'
      });
      
    } catch (error) {
      console.error('Error getting user risk stats:', error);
      res.status(500).json({ 
        error: 'Failed to get user risk statistics',
        message: error.message
      });
    }
  });
}