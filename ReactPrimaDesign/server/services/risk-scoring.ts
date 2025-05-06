/**
 * Risk Scoring Service
 * Provides transaction risk analysis using ensemble machine learning models
 */

import { storage } from '../storage';
import { ScamType } from '../../shared/schema';
import { detect } from 'langdetect';

// Risk levels
export enum RiskLevel {
  Low = 'low',
  Medium = 'medium',
  High = 'high',
  Critical = 'critical'
}

// Interface for risk analysis input
interface RiskAnalysisInput {
  upiId: string;
  amount: number;
  userId?: number;
  note?: string;
  recipient?: string;
  timestamp: Date;
  location?: {
    lat: number;
    lng: number;
  };
  deviceInfo?: any;
}

// Interface for risk analysis result
interface RiskAnalysisResult {
  risk_score: number;
  risk_level: string;
  recommendation: string;
  risk_factors: Array<{
    type: string;
    description: string;
    impact: number;
  }>;
  needs_verification: boolean;
  anomaly_score?: number;
  transaction_network_risk?: number;
  device_trust_score?: number;
  behavioral_risk?: number;
}

/**
 * Get risk level from numeric score
 */
function getRiskLevelFromScore(score: number): RiskLevel {
  if (score < 0.3) return RiskLevel.Low;
  if (score < 0.6) return RiskLevel.Medium;
  if (score < 0.85) return RiskLevel.High;
  return RiskLevel.Critical;
}

/**
 * Get recommendation based on risk score
 */
function getRecommendation(score: number, factors: any[]): string {
  const level = getRiskLevelFromScore(score);
  
  switch (level) {
    case RiskLevel.Low:
      return "This transaction appears to be safe. You can proceed normally.";
    case RiskLevel.Medium:
      return "Use caution with this transaction. Verify the recipient's details before proceeding.";
    case RiskLevel.High:
      if (factors.some(f => f.type === 'amount')) {
        return "This is a high-value transaction with significant risk factors. Consider additional verification or using a secured payment channel.";
      } else {
        return "This transaction shows significant risk factors. Consider using an alternative payment method or contact the recipient directly to verify.";
      }
    case RiskLevel.Critical:
      return "WARNING: This transaction has critical risk indicators. We strongly recommend cancelling this transaction.";
    default:
      return "Unable to assess risk. Please proceed with caution.";
  }
}

/**
 * Perform ensemble machine learning risk analysis
 * 
 * In a real production environment, this would use trained ML models
 * For this prototype, we're implementing a rules-based system that simulates
 * the behavior of an ensemble ML system
 */
export async function analyzeTransactionRisk(input: RiskAnalysisInput): Promise<RiskAnalysisResult> {
  try {
    const { upiId, amount, userId, note, recipient, timestamp, location, deviceInfo } = input;
    const riskFactors = [];
    let baseScore = 0.1; // Start with a baseline risk
    
    // 1. Check UPI ID against database of reported scams
    let reportedScams: any[] = [];
    try {
      reportedScams = await storage.getScamReportsByUpiId(upiId);
    } catch (err) {
      console.error('Error getting scam reports:', err);
    }
    
    if (reportedScams.length > 0) {
      // UPI has been reported as scam before
      const reportCount = reportedScams.length;
      
      if (reportCount >= 5) {
        baseScore += 0.6;
        riskFactors.push({
          type: 'reported_scam',
          description: `UPI ID has been reported as scam ${reportCount} times`,
          impact: 0.6
        });
      } else if (reportCount > 0) {
        baseScore += 0.3;
        riskFactors.push({
          type: 'reported_scam',
          description: `UPI ID has been reported as scam ${reportCount} times`,
          impact: 0.3
        });
      }
      
      // Check scam types
      const scamTypeCount: Record<string, number> = {};
      reportedScams.forEach(report => {
        scamTypeCount[report.scamType] = (scamTypeCount[report.scamType] || 0) + 1;
      });
      
      // Get most common scam type
      let maxCount = 0;
      let mostCommonScamType = ScamType.Unknown;
      
      for (const [type, count] of Object.entries(scamTypeCount)) {
        if (count > maxCount) {
          maxCount = count;
          mostCommonScamType = type as ScamType;
        }
      }
      
      // Add factor for dangerous scam types
      if ([ScamType.Banking, ScamType.KYC, ScamType.Phishing].includes(mostCommonScamType as ScamType)) {
        baseScore += 0.2;
        riskFactors.push({
          type: 'scam_type',
          description: `Associated with ${mostCommonScamType} scams`,
          impact: 0.2
        });
      }
    }
    
    // 2. Check UPI format and domain analysis
    if (!upiId.includes('@')) {
      baseScore += 0.4;
      riskFactors.push({
        type: 'format',
        description: 'Invalid UPI ID format',
        impact: 0.4
      });
    } else {
      const [username, domain] = upiId.split('@');
      
      // Check username heuristics
      if (/^[a-zA-Z0-9]{1,3}$/.test(username)) {
        // Very short usernames are suspicious
        baseScore += 0.25;
        riskFactors.push({
          type: 'username',
          description: 'Suspiciously short username in UPI ID',
          impact: 0.25
        });
      }
      
      if (/^\d+$/.test(username)) {
        // All numeric usernames are slightly suspicious
        baseScore += 0.15;
        riskFactors.push({
          type: 'username',
          description: 'Numeric-only username in UPI ID',
          impact: 0.15
        });
      }
      
      // Domain risk analysis
      const safeDomains = ['oksbi', 'okicici', 'okaxis', 'okhdfcbank', 'ybl', 'paytm', 'upi'];
      const mediumRiskDomains = ['okhdfc', 'oksbicard'];
      const highRiskDomains = ['verify', 'secure', 'gov', 'bank', 'refund', 'support'];
      
      if (!domain) {
        baseScore += 0.5;
        riskFactors.push({
          type: 'domain',
          description: 'Missing UPI domain',
          impact: 0.5
        });
      } else if (highRiskDomains.some(d => domain.includes(d))) {
        baseScore += 0.4;
        riskFactors.push({
          type: 'domain',
          description: 'Suspicious keywords in UPI domain',
          impact: 0.4
        });
      } else if (!safeDomains.includes(domain) && !mediumRiskDomains.includes(domain)) {
        baseScore += 0.2;
        riskFactors.push({
          type: 'domain',
          description: 'Uncommon UPI domain',
          impact: 0.2
        });
      } else if (mediumRiskDomains.includes(domain)) {
        baseScore += 0.1;
        riskFactors.push({
          type: 'domain',
          description: 'Less common UPI domain',
          impact: 0.1
        });
      }
    }
    
    // 3. Check transaction amount patterns
    if (amount > 50000) {
      baseScore += 0.3;
      riskFactors.push({
        type: 'amount',
        description: 'Large transaction amount (₹50,000+)',
        impact: 0.3
      });
    } else if (amount > 10000) {
      baseScore += 0.15;
      riskFactors.push({
        type: 'amount',
        description: 'Medium-high transaction amount (₹10,000+)',
        impact: 0.15
      });
    }
    
    // Check for round numbers (often suspicious)
    if (amount % 1000 === 0 && amount >= 5000) {
      baseScore += 0.1;
      riskFactors.push({
        type: 'amount_pattern',
        description: 'Round transaction amount (suspicious pattern)',
        impact: 0.1
      });
    }
    
    // 4. Check note for suspicious keywords if available
    if (note) {
      const noteText = note.toLowerCase();
      const suspiciousKeywords = [
        'urgent', 'verify', 'kyc', 'account will be blocked', 
        'prize', 'winner', 'gift', 'refund', 'government', 
        'bank', 'verification', 'expire', 'wallet', 'credit'
      ];
      
      for (const keyword of suspiciousKeywords) {
        if (noteText.includes(keyword)) {
          baseScore += 0.35;
          riskFactors.push({
            type: 'note',
            description: 'Suspicious keywords in transaction note',
            impact: 0.35
          });
          break;
        }
      }
      
      // Check for language detection (non-English)
      try {
        const languages = detect(note);
        if (languages && languages.length > 0 && languages[0].lang !== 'en' && languages[0].prob > 0.6) {
          baseScore += 0.2;
          riskFactors.push({
            type: 'language',
            description: `Note contains non-English text (${languages[0].lang})`,
            impact: 0.2
          });
        }
      } catch (err) {
        console.log('Language detection failed', err);
      }
    }
    
    // 5. Check user transaction history if userId is available
    if (userId) {
      try {
        const userTransactions = await storage.getTransactionsByUserId(userId);
        
        // Check if this is a first-time transaction to this UPI ID
        const previousToSameUpi = userTransactions.filter(t => t.upiId === upiId);
        
        if (previousToSameUpi.length === 0 && userTransactions.length > 0) {
          baseScore += 0.15;
          riskFactors.push({
            type: 'first_transaction',
            description: 'First transaction to this UPI ID',
            impact: 0.15
          });
        }
        
        // Check if amount is significantly higher than user's average
        if (userTransactions.length > 0) {
          const avgAmount = userTransactions.reduce((sum, t) => sum + (t.amount || 0), 0) / userTransactions.length;
          
          if (amount > avgAmount * 3 && amount > 5000) {
            baseScore += 0.25;
            riskFactors.push({
              type: 'unusual_amount',
              description: 'Amount significantly higher than your average transactions',
              impact: 0.25
            });
          }
        }
      } catch (err) {
        console.error('Error analyzing user transaction history:', err);
      }
    }
    
    // 6. Check device info if available
    let deviceTrustScore = 1.0; // Default: fully trusted
    if (deviceInfo && deviceInfo.suspicious) {
      deviceTrustScore = 0.6; // Reduce trust score
      
      baseScore += 0.2;
      riskFactors.push({
        type: 'device',
        description: 'Suspicious device fingerprint',
        impact: 0.2
      });
      
      // Add specific device issues
      if (deviceInfo.reasons && deviceInfo.reasons.length > 0) {
        deviceInfo.reasons.forEach((reason: string) => {
          if (reason.includes('VPN') || reason.includes('Proxy')) {
            baseScore += 0.1;
            riskFactors.push({
              type: 'network',
              description: 'VPN or proxy usage detected',
              impact: 0.1
            });
          }
        });
      }
    }
    
    // 7. Check location risk if available
    let locationRisk = 0;
    if (location) {
      // In a real implementation, we would check against known fraud hotspots
      // For this prototype, simulate this with some simple checks
      
      // Check if location is within India (simplified)
      const isWithinIndia = 
        location.lat >= 8.0 && location.lat <= 37.0 && 
        location.lng >= 68.0 && location.lng <= 97.0;
      
      if (!isWithinIndia) {
        locationRisk = 0.3;
        baseScore += locationRisk;
        riskFactors.push({
          type: 'location',
          description: 'Transaction initiated from outside India',
          impact: locationRisk
        });
      }
    }
    
    // 8. Behavioral biometrics (would be implemented on a real system)
    // For now, just add a placeholder comment
    
    // 9. Calculate final risk score (capped at 1.0)
    const finalRiskScore = Math.min(1.0, baseScore);
    const riskLevel = getRiskLevelFromScore(finalRiskScore);
    
    // Calculate if additional verification is needed
    const needsVerification = finalRiskScore > 0.6 || (amount > 25000 && finalRiskScore > 0.4);
    
    // Get appropriate recommendation
    const recommendation = getRecommendation(finalRiskScore, riskFactors);
    
    // Construct result
    return {
      risk_score: finalRiskScore,
      risk_level: riskLevel,
      recommendation,
      risk_factors: riskFactors,
      needs_verification: needsVerification,
      device_trust_score: deviceTrustScore,
      transaction_network_risk: 0.1, // Placeholder
      anomaly_score: finalRiskScore * 0.8 + Math.random() * 0.2 // Simulated anomaly score
    };
  } catch (error) {
    console.error('Risk scoring analysis failed:', error);
    
    // Return a moderate risk score with error information
    return {
      risk_score: 0.5,
      risk_level: RiskLevel.Medium,
      recommendation: 'Risk analysis could not be completed. Proceed with caution.',
      risk_factors: [{
        type: 'error',
        description: 'Risk analysis system error',
        impact: 0.5
      }],
      needs_verification: true
    };
  }
}