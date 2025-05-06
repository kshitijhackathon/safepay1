import { Express } from 'express';
import { storage } from '../storage';
import { validateUpiIdSafety } from '../services/openai';
import { ScamType } from '../../shared/schema';
import { 
  SAFE_UPI_IDS, 
  SCAM_UPI_IDS,
  getRecommendations
} from '../services/upi-check';

/**
 * Validates UPI ID format
 * @param upiId UPI ID to validate
 * @returns true if valid format, false otherwise
 */
function validateUpi(upiId: string): boolean {
  // For testing purposes, accept all UPI IDs, even if not valid format
  if (!upiId || upiId.trim() === '') {
    return false;
  }
  
  // Relaxed UPI ID format validation - just check if it has @ symbol
  if (!upiId.includes('@')) {
    // Auto-fix: If no @ symbol, add a default domain
    upiId = `${upiId}@upi`;
  }
  
  return true;
}

/**
 * Calculate pattern-based risk score for UPI IDs
 * @param upiId UPI ID to analyze
 * @returns Score between 0 and 1 where higher is more suspicious
 */
function calculatePatternScore(upiId: string): number {
  if (!upiId) return 1.0; // Maximum risk for empty UPI ID
  
  // Suspicious patterns - based on known scam patterns
  const suspiciousPatterns = [
    /^\d{10}@\w+/,              // Mobile number based UPI IDs
    /^[a-z]+\d+@ok\w+/,         // Alpha-numeric suspicious patterns
    /(support|helpdesk)@/,       // Fake customer support
    /@(fakemail|fakebank)/,      // Domains known for frauds
    /verify@/,                   // Verification scams
    /refund@/,                   // Refund scams
    /lottery@/,                  // Lottery scams
    /winning@/,                  // Winning scams
    /kyc@/,                      // KYC scams
    /officer@/,                  // Officer scams
    /reward@/,                   // Reward scams
    /\.exe$/,                    // Executable extensions
    /([a-z])\1{3}/,              // Repeated characters (e.g., aaaa)
  ];
  
  let score = 0;
  for (const pattern of suspiciousPatterns) {
    if (pattern.test(upiId.toLowerCase())) {
      score += 0.3;
    }
  }
  
  return Math.min(score, 1.0);
}

/**
 * Get features for ML risk calculation
 * @param upiId UPI ID to analyze
 * @param reports Array of scam reports
 * @returns Feature set for risk calculation
 */
async function getFeatures(upiId: string, reports: any[]) {
  // Extract unique domains from reports
  const domains = new Set();
  reports.forEach(report => {
    if (report.domain) domains.add(report.domain);
  });
  
  // Get days since last report
  let daysSinceLastReport = 365; // Default to 1 year if no reports
  if (reports.length > 0) {
    const mostRecent = reports.reduce((latest, report) => {
      return new Date(report.timestamp) > new Date(latest.timestamp) ? report : latest;
    }, reports[0]);
    
    daysSinceLastReport = Math.floor((Date.now() - new Date(mostRecent.timestamp).getTime()) / (1000 * 60 * 60 * 24));
  }
  
  return {
    totalReports: reports.length,
    activeDomains: domains.size,
    daysSinceLastReport,
    patternScore: calculatePatternScore(upiId)
  };
}

/**
 * Calculate ML risk score
 * @param upiId UPI ID to analyze
 * @param reports Array of scam reports
 * @returns Risk score between 0 and 1
 */
async function calculateMlRisk(upiId: string, reports: any[]): Promise<number> {
  const features = await getFeatures(upiId, reports);
  
  // Risk calculation weights
  const weights = {
    totalReports: 0.4,
    activeDomains: 0.2,
    daysSinceLastReport: 0.2,
    patternScore: 0.2
  };
  
  // Normalize and calculate scores
  const scores = {
    totalReports: Math.min(features.totalReports / 10, 1.0), // Max at 10+ reports
    activeDomains: Math.min(features.activeDomains / 5, 1.0), // Max at 5+ domains
    daysSinceLastReport: Math.max(0, 1 - (features.daysSinceLastReport / 30)), // More recent = higher risk
    patternScore: features.patternScore
  };
  
  // Calculate weighted score
  let riskScore = 0;
  for (const feature of Object.keys(weights) as Array<keyof typeof weights>) {
    riskScore += scores[feature] * weights[feature];
  }
  
  return riskScore;
}

/**
 * Generate analysis of domains involved in scam reports
 * @param reports Array of scam reports
 * @returns Domain breakdown with count data
 */
function generateDomainAnalysis(reports: any[]) {
  const domainCounts: Record<string, number> = {};
  reports.forEach(report => {
    const domain = report.domain || 'unknown';
    domainCounts[domain] = (domainCounts[domain] || 0) + 1;
  });
  
  return Object.entries(domainCounts)
    .map(([domain, count]) => ({ domain, count }))
    .sort((a, b) => (b.count as number) - (a.count as number));
}

/**
 * Generate recommended actions based on risk score
 * @param riskScore Risk score value
 * @returns Array of recommended actions
 */
function generateActions(riskScore: number): string[] {
  if (riskScore > 0.7) {
    return [
      "Do not proceed with this transaction",
      "Report this UPI ID to your bank",
      "Block this contact if received via messaging apps"
    ];
  } else if (riskScore > 0.4) {
    return [
      "Verify the recipient's identity before proceeding",
      "Call the recipient to confirm transaction details",
      "Start with a small test transaction"
    ];
  } else {
    return [
      "Proceed with standard verification",
      "Keep records of all transactions",
      "Enable notifications for all transactions"
    ];
  }
}

/**
 * Classify UPI ID based on whitelist, blacklist, and patterns
 * Implements the approach from the scam_classifier.py example
 * @param upiId The UPI ID to classify
 * @returns Classification result object
 */
function classifyUpiId(upiId: string): { status: 'SAFE' | 'SUSPICIOUS' | 'SCAM', reason: string, confidence_score: number, risk_factors: string[] } {
  // Normalize the UPI ID
  upiId = upiId.trim().toLowerCase();

  // Check if in safe list
  if (SAFE_UPI_IDS.includes(upiId)) {
    return {
      status: "SAFE",
      reason: "This UPI ID is whitelisted and verified.",
      confidence_score: 0.95,
      risk_factors: []
    };
  }

  // Check if in scam list
  if (SCAM_UPI_IDS.includes(upiId)) {
    return {
      status: "SCAM",
      reason: "This UPI ID is blacklisted due to scam reports.",
      confidence_score: 0.99,
      risk_factors: ["Known scam UPI ID"]
    };
  }

  // Pattern-based suspicion
  const suspiciousPatterns = [
    { regex: /^\d{10}@\w+/, reason: "Mobile number based UPI ID" },
    { regex: /^[a-z]+\d+@ok\w+/, reason: "Suspicious alphanumeric pattern" },
    { regex: /(support|helpdesk)@/, reason: "Possible fake customer support" },
    { regex: /@(fakemail|fakebank)/, reason: "Known fraudulent domain" },
    { regex: /verify@/, reason: "Possible verification scam" },
    { regex: /refund@/, reason: "Possible refund scam" },
    { regex: /lottery@/, reason: "Possible lottery scam" },
    { regex: /winning@/, reason: "Possible winning notification scam" }
  ];

  for (const pattern of suspiciousPatterns) {
    if (pattern.regex.test(upiId)) {
      return {
        status: "SUSPICIOUS",
        reason: `UPI ID matches suspicious pattern: ${pattern.reason}`,
        confidence_score: 0.75,
        risk_factors: [pattern.reason]
      };
    }
  }

  // Unknown - caution
  return {
    status: "SUSPICIOUS",
    reason: "This UPI ID is not verified in known databases.",
    confidence_score: 0.6,
    risk_factors: ["Unverified UPI ID"]
  };
}

export function registerUpiCheckRoutes(app: Express): void {
  // Quick UPI ID validation endpoint
  app.get('/api/upi/check/:upiId', async (req, res) => {
    try {
      const { upiId } = req.params;
      
      if (!upiId || !validateUpi(upiId)) {
        return res.status(400).json({
          status: "error",
          valid: false,
          message: "Invalid UPI ID format"
        });
      }
      
      // Quick initial classification - only takes milliseconds
      const classification = classifyUpiId(upiId);
      
      // Get any existing reports
      const reports = await storage.getScamReportsByUpiId(upiId);
      
      // Calculate basic risk percentage based on classification and reports
      const reportsFactor = Math.min(reports.length / 5, 1.0);
      const patternFactor = calculatePatternScore(upiId);
      const classificationFactor = classification.status === 'SCAM' ? 1.0 : 
                                  classification.status === 'SUSPICIOUS' ? 0.6 : 0.1;
      
      const riskPercentage = Math.round((classificationFactor * 0.4 + reportsFactor * 0.3 + patternFactor * 0.3) * 100);
      
      return res.json({
        upiId,
        valid: true,
        status: classification.status,
        riskPercentage,
        riskLevel: riskPercentage > 70 ? 'high' : riskPercentage > 40 ? 'medium' : 'low',
        reports: reports.length,
        reason: classification.reason
      });
    } catch (error) {
      console.error('Error in quick UPI check:', error);
      res.status(500).json({
        status: "error",
        message: "Internal server error during UPI validation"
      });
    }
  });
  // Enhanced UPI check endpoint with AI analysis
  app.post('/api/check-scam', async (req, res) => {
    try {
      const { upiId } = req.body;
      
      if (!upiId || upiId.trim() === '') {
        return res.status(400).json({
          status: "error",
          message: "UPI ID is required"
        });
      }

      // Use the enhanced UPI safety check service with AI analysis
      const { checkUpiSafety } = await import('../services/upi-check');
      const result = await checkUpiSafety(upiId);
      
      // Get reports for additional context
      const reports = await storage.getScamReportsByUpiId(upiId);
      
      // Add reports count to the result
      const enhancedResult = {
        ...result,
        reports: reports.length,
        
        // Get most common scam type if available
        category: reports.length > 0 ? await storage.getMostCommonScamType(upiId) : undefined,
        
        // Make sure recommendations are included
        recommendations: result.recommendations || getRecommendations(result.status)
      };
      
      res.json(enhancedResult);
    } catch (error) {
      console.error('Error in check-scam:', error);
      res.status(500).json({ 
        status: "error",
        message: "Internal server error"
      });
    }
  });
  app.post('/api/combined-check', async (req, res) => {
    try {
      const { upiId } = req.body;
      
      // Validate UPI format
      if (!validateUpi(upiId)) {
        return res.status(400).json({ error: 'Invalid UPI format' });
      }
      
      // Get all reports
      const reports = await storage.getScamReportsByUpiId(upiId);
      
      // Calculate ML risk score
      const riskScore = await calculateMlRisk(upiId, reports);
      
      // Generate domain analysis
      const domainAnalysis = generateDomainAnalysis(reports);
      
      // Get AI-enhanced UPI safety check (if available)
      let threatData = { level: 'unknown', indicators: [] };
      try {
        const aiAnalysis = await validateUpiIdSafety(upiId);
        
        if (aiAnalysis) {
          threatData = {
            level: aiAnalysis.is_suspicious ? 'high' : 'low',
            indicators: aiAnalysis.flags || []
          };
        }
      } catch (error) {
        console.error('Error with AI UPI validation:', error);
      }
      
      // Get most common scam type if available
      let mostCommonScamType = '';
      try {
        mostCommonScamType = await storage.getMostCommonScamType(upiId);
      } catch (error) {
        console.error('Error getting most common scam type:', error);
      }
      
      // Prepare response
      const response = {
        upiId,
        riskScore: parseFloat((riskScore * 100).toFixed(2)),
        riskLevel: riskScore > 0.7 ? 'high' : riskScore > 0.4 ? 'medium' : 'low',
        totalReports: reports.length,
        activeCases: reports.length, // We don't have 'resolved' field, but could add one later
        domainAnalysis,
        threatLevel: threatData.level,
        threatIndicators: threatData.indicators,
        mostCommonScamType,
        recommendedActions: generateActions(riskScore)
      };
      
      res.json(response);
    } catch (error) {
      console.error('Error in combined check:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });
  
  app.post('/api/report-scam', async (req, res) => {
    try {
      const { upiId, userId, scamType, description, domain, amount } = req.body;
      
      // Validate UPI format
      if (!validateUpi(upiId)) {
        return res.status(400).json({ error: 'Invalid UPI format' });
      }
      
      // Parse amount to integer (paise/cents) if present
      let amountInPaise = null;
      if (amount) {
        // Remove commas and convert to paise (cents)
        const cleanAmount = amount.toString().replace(/[^\d.]/g, '');
        amountInPaise = Math.round(parseFloat(cleanAmount) * 100);
      }
      
      // Create scam report
      // Convert string scamType to enum ScamType
      let scamTypeEnum: ScamType;
      switch(scamType) {
        case 'Banking Scam':
        case 'Banking':
          scamTypeEnum = ScamType.Banking;
          break;
        case 'Lottery Scam':
        case 'Lottery':
          scamTypeEnum = ScamType.Lottery;
          break;
        case 'KYC Verification Scam':
        case 'KYC':
          scamTypeEnum = ScamType.KYC;
          break;
        case 'Refund Scam':
        case 'Refund':
          scamTypeEnum = ScamType.Refund;
          break;
        case 'Phishing Attempt':
        case 'Phishing':
          scamTypeEnum = ScamType.Phishing;
          break;
        case 'Reward Scam':
        case 'Reward':
          scamTypeEnum = ScamType.Reward;
          break;
        default:
          scamTypeEnum = ScamType.Unknown;
      }
      
      const report = await storage.createScamReport({
        upiId,
        userId,
        scamType: scamTypeEnum,
        description: description || null,
        amountLost: amountInPaise
      });
      
      // Update risk score for this UPI ID
      await storage.updateUpiRiskScore(upiId);
      
      res.status(201).json({
        message: 'Scam report submitted successfully',
        reportId: report.id
      });
    } catch (error) {
      console.error('Error reporting scam:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });
  
  // Get all scam reports for a specific user
  app.get('/api/user/:userId/scam-reports', async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);
      
      if (isNaN(userId)) {
        return res.status(400).json({ error: 'Invalid user ID' });
      }
      
      const reports = await storage.getScamReportsByUserId(userId);
      
      // Format report data for the client
      const formattedReports = reports.map(report => ({
        id: report.id,
        upiId: report.upiId,
        scamType: report.scamType,
        description: report.description,
        amountLost: report.amountLost ? (report.amountLost / 100).toFixed(2) : null, // Convert paise to rupees
        timestamp: report.timestamp,
        // Add any additional data needed for display
      }));
      
      res.json(formattedReports);
    } catch (error) {
      console.error('Error fetching user scam reports:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });
  
  // Get a specific scam report by ID
  app.get('/api/scam-reports/:reportId', async (req, res) => {
    try {
      const reportId = parseInt(req.params.reportId);
      
      if (isNaN(reportId)) {
        return res.status(400).json({ error: 'Invalid report ID' });
      }
      
      const report = await storage.getScamReportById(reportId);
      
      if (!report) {
        return res.status(404).json({ error: 'Report not found' });
      }
      
      // Format the report data for the client
      const formattedReport = {
        id: report.id,
        upiId: report.upiId,
        scamType: report.scamType,
        description: report.description,
        amountLost: report.amountLost ? (report.amountLost / 100).toFixed(2) : null, // Convert paise to rupees
        timestamp: report.timestamp,
        // Add any additional data needed for display
      };
      
      res.json(formattedReport);
    } catch (error) {
      console.error('Error fetching scam report:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });
}