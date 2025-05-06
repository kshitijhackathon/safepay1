import { storage } from '../storage';
import { Groq } from 'groq-sdk';

// Initialize Groq client
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY
});

// Model to use
const MODEL = 'llama-3.3-70b-versatile';

// Helper function to safely parse JSON, handling markdown code blocks
function safeJsonParse(jsonString: string | null | undefined, defaultValue: any = {}) {
  try {
    if (!jsonString) return defaultValue;
    
    // Handle markdown code blocks (```json ... ```)
    let cleanedJson = jsonString;
    
    // Check for markdown code blocks and extract JSON
    const markdownMatch = cleanedJson.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
    if (markdownMatch && markdownMatch[1]) {
      cleanedJson = markdownMatch[1];
    }
    
    // Additional cleanup for other formatting
    cleanedJson = cleanedJson.trim();
    
    return JSON.parse(cleanedJson);
  } catch (error) {
    console.error('Error parsing JSON:', error);
    // Log the problematic string for debugging
    if (jsonString) {
      console.error('Content that failed to parse:', jsonString.substring(0, 100) + '...');
    }
    return defaultValue;
  }
}

/**
 * Lists of known safe UPI IDs
 */
export const SAFE_UPI_IDS = [
  // Common payment services
  'pay@paytm',
  'payment@paytm',
  'merchant@paytm',
  'merchant@axis',
  'paytm@upi',
  'googlepay@okicici',
  'flipkart@icici',
  'zomato@upi',
  'bookmyshow@hdfcbank',
  'gpay@okicici',
  'payments@phonepe',
  'transaction@phonepe',
  'amazonpay@apl',
  'merchant@ybl',
  // Banks
  'support@icici',
  'support@okicici',
  'care@hdfcbank',
  'help@sbi',
  'support@okaxis',
  'help@yesbank',
  'info@kotak',
  'care@oksbi',
  'official@axl',
  // Test/demo accounts
  'merchant@okhdfcbank',
  'test@okicici',
  'demo@upi'
];

/**
 * Lists of known scam UPI IDs
 */
export const SCAM_UPI_IDS = [
  'fraudpay@upi',
  'scammer123@okhdfc',
  'fakepaytm@upi',
  '1234567890@ybl',
  'helpdesk@fakebank',
  'verify@paytm', // Scammer pretending to be verification service
  'refund@phonepe', // Fake refund scam
  'helpdesk@upi',
  'support@okpaytm', // Fake support
  'security@payment',
  'kyc@paytmbank', // KYC scam
  'lottery@ybl',
  'winning@okicici', // Lottery scam
  'officer@bank',
  'reward@gift'
];

/**
 * Lists of known scam domains in UPI IDs
 */
const SUSPICIOUS_DOMAINS = [
  'gift',
  'win',
  'lucky',
  'prize',
  'verify',
  'refund',
  'paytmxxx',
  'fund',
  'winning',
  'return',
  'reward',
  'cash',
  'lottery',
  'helpdesk'
];

/**
 * Regular expressions for suspicious UPI patterns
 */
const SUSPICIOUS_PATTERNS = [
  /^[0-9]{6,}@/, // UPI IDs that start with many numbers
  /^[0-9]{4,}[a-z]{1,2}@/, // Numbers followed by 1-2 letters
  /^random[0-9]*@/, // Random with numbers
  /^user[0-9]{4,}@/, // Generic usernames with numbers
  /^temp[0-9]*@/, // Temporary accounts
  /^[a-z]{1,2}[0-9]{6,}@/, // 1-2 letters followed by many numbers
  /^verify[a-z0-9]*@/, // Verification scams
  /^refund[a-z0-9]*@/, // Refund scams
  /^pay[a-z0-9]*@(?!paytm)/, // Paytm-like but not paytm
  /^(kyc|KYC)[a-z0-9]*@/, // KYC scams
  /support[0-9]+@/ // Fake support with numbers
];

/**
 * Domain reputation scores (-1 to 1)
 * -1 = highly suspicious, 0 = neutral, 1 = trusted
 */
const DOMAIN_REPUTATION: Record<string, number> = {
  // Trusted domains
  'paytm': 1,
  'okicici': 0.9,
  'oksbi': 0.9,
  'okhdfcbank': 0.9,
  'okaxis': 0.9,
  'ybl': 0.8,
  'upi': 0.5, // Generic UPI is less trusted
  // Suspicious domains
  'xyz': -0.6,
  'money': -0.5,
  'cash': -0.7,
  'win': -0.8,
  'gift': -0.8,
  'verify': -0.7,
  'refund': -0.7,
  'lottery': -0.9
};

export interface UpiCheckResult {
  status: 'SAFE' | 'SUSPICIOUS' | 'SCAM';
  reason: string;
  confidence_score: number;
  risk_factors?: string[];
  category?: string;
  recommendations?: string[];
  safety_score?: number; // Safety score from 0-100
  ai_analysis?: string; // AI analysis explanation
}

/**
 * Validates UPI ID format
 * @param upiId UPI ID to validate
 * @returns true if valid format, false otherwise
 */
export function validateUpiFormat(upiId: string): boolean {
  if (!upiId || upiId.trim() === '') {
    return false;
  }
  
  // Basic UPI ID format validation
  const upiPattern = /^[a-zA-Z0-9.\-_]{3,}@[a-zA-Z0-9]{3,}$/;
  return upiPattern.test(upiId);
}

/**
 * Extract the domain part from a UPI ID
 * @param upiId Full UPI ID
 * @returns Domain part (after @)
 */
export function extractUpiDomain(upiId: string): string {
  const parts = upiId.split('@');
  return parts.length > 1 ? parts[1].toLowerCase() : '';
}

/**
 * Extract the username part from a UPI ID
 * @param upiId Full UPI ID
 * @returns Username part (before @)
 */
export function extractUpiUsername(upiId: string): string {
  const parts = upiId.split('@');
  return parts.length > 0 ? parts[0].toLowerCase() : '';
}

/**
 * Calculate pattern-based risk score for UPI IDs
 * @param upiId UPI ID to analyze
 * @returns Score between 0 and 1 where higher is more suspicious
 */
export function calculatePatternScore(upiId: string): number {
  if (!upiId) return 1.0;
  
  let score = 0;
  const upiLower = upiId.toLowerCase();
  const username = extractUpiUsername(upiLower);
  const domain = extractUpiDomain(upiLower);
  
  // Check against explicit safe list
  if (SAFE_UPI_IDS.includes(upiLower)) {
    return 0;
  }
  
  // Check against explicit scam list
  if (SCAM_UPI_IDS.includes(upiLower)) {
    return 1;
  }
  
  // Check domain reputation
  if (domain in DOMAIN_REPUTATION) {
    const reputation = DOMAIN_REPUTATION[domain];
    // Negative reputation increases suspicion
    if (reputation < 0) {
      score += Math.abs(reputation) * 0.4; // Weight domain reputation at 40%
    } else {
      // Positive reputation decreases suspicion
      score -= reputation * 0.3;
    }
  }
  
  // Check for suspicious domains
  for (const suspiciousDomain of SUSPICIOUS_DOMAINS) {
    if (domain.includes(suspiciousDomain)) {
      score += 0.3;
      break;
    }
  }
  
  // Check username patterns
  for (const pattern of SUSPICIOUS_PATTERNS) {
    if (pattern.test(upiLower)) {
      score += 0.3;
      break;
    }
  }
  
  // Length-based rules
  if (username.length < 3) {
    score += 0.1; // Too short, might be suspicious
  }
  if (username.length > 15) {
    score += 0.1; // Too long, might be randomly generated
  }
  
  // Character repetition (e.g., "aaaaa")
  const repetitionPattern = /(.)\1{3,}/;
  if (repetitionPattern.test(username)) {
    score += 0.2;
  }
  
  // Character entropy - random-looking usernames are suspicious
  const uniqueChars = new Set(username).size;
  const entropy = uniqueChars / username.length;
  if (entropy < 0.3) {
    score += 0.1; // Low entropy, repetitive characters
  }
  
  // Numbers in username
  const digitCount = (username.match(/\d/g) || []).length;
  if (digitCount > 5) {
    score += 0.2; // Many digits are suspicious
  }
  
  // Numeric username
  if (/^\d+$/.test(username)) {
    score += 0.3; // All digits is suspicious
  }
  
  // Adjust to 0-1 range
  return Math.max(0, Math.min(1, score));
}

/**
 * Get detailed risk factors based on UPI ID analysis
 * @param upiId UPI ID to analyze
 * @returns Array of risk factors
 */
export function getRiskFactors(upiId: string): string[] {
  const factors: string[] = [];
  const upiLower = upiId.toLowerCase();
  const username = extractUpiUsername(upiLower);
  const domain = extractUpiDomain(upiLower);
  
  // Domain checks
  if (SUSPICIOUS_DOMAINS.some(d => domain.includes(d))) {
    factors.push(`Domain "${domain}" matches known suspicious patterns`);
  }
  
  // Pattern checks
  for (const pattern of SUSPICIOUS_PATTERNS) {
    if (pattern.test(upiLower)) {
      factors.push("UPI ID matches suspicious pattern format");
      break;
    }
  }
  
  // Length checks
  if (username.length < 3) {
    factors.push("Username is unusually short");
  }
  if (username.length > 15) {
    factors.push("Username is unusually long");
  }
  
  // Character repetition
  const repetitionPattern = /(.)\1{3,}/;
  if (repetitionPattern.test(username)) {
    factors.push("Username contains unusual character repetition");
  }
  
  // Numeric username
  const digitCount = (username.match(/\d/g) || []).length;
  if (digitCount > 5) {
    factors.push("Username contains an unusual number of digits");
  }
  if (/^\d+$/.test(username)) {
    factors.push("Username consists only of numbers");
  }
  
  return factors;
}

/**
 * Generate recommendations based on UPI check result
 * @param status The UPI status (SAFE, SUSPICIOUS, SCAM)
 * @returns Array of recommendation strings
 */
export function getRecommendations(status: 'SAFE' | 'SUSPICIOUS' | 'SCAM'): string[] {
  switch (status) {
    case 'SAFE':
      return [
        "Proceed with standard payment verification",
        "Confirm transaction details before paying",
        "Keep a record of your transaction"
      ];
    case 'SUSPICIOUS':
      return [
        "Verify the recipient's identity before proceeding",
        "Call the recipient to confirm transaction details",
        "Consider using a different payment method",
        "If uncertain, start with a small test payment"
      ];
    case 'SCAM':
      return [
        "Do not proceed with this transaction",
        "Report this UPI ID to your bank",
        "Block this contact if received via messaging apps",
        "Do not share any personal information"
      ];
    default:
      return ["Proceed with caution"];
  }
}

/**
 * Analyze UPI ID using Groq for advanced AI-powered safety assessment
 * @param upiId UPI ID to analyze
 * @returns AI analysis with safety score and explanation
 */
export async function analyzeUpiWithAI(upiId: string): Promise<{
  safety_score: number;
  explanation: string;
  flags: string[];
}> {
  try {
    // Split UPI ID to analyze parts more effectively
    const [username, domain] = upiId.split('@');
    
    // Prepare system prompt
    const systemPrompt = `Analyze this UPI ID and tell if it looks suspicious or genuine. Consider patterns such as:
1. Random numbers in the ID (e.g., long numeric sequences).
2. Mismatch between the ID's domain and the issuer (e.g., Paytm ID with HDFC).
3. Common scam patterns (e.g., too many characters, or odd sequence).
4. Known scam reports for similar IDs.

Valid UPI domains include:
- okicici = ICICI Bank
- okhdfcbank = HDFC Bank  
- oksbi = State Bank of India
- okaxis = Axis Bank
- ybl = PhonePe/Yes Bank
- paytm = Paytm
- upi = National Payments Corporation of India
- ibl = ICICI Bank
- axl = Axis Bank

Return a JSON with:
1. safety_score: Number from 0 to 100 (higher is safer)
2. explanation: 2-3 lines explaining the reasoning
3. flags: Array of specific concerns found (empty if none)`;
    
    // Send request to Groq
    const response = await groq.chat.completions.create({
      model: MODEL, 
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: upiId }
      ],
      temperature: 0.1, // Low temperature for more deterministic responses
      max_tokens: 1024
    });
    
    // Parse response with fallback for safety
    const defaultResponse = {
      safety_score: 50,
      explanation: "Unable to analyze UPI ID with AI. Using standard checks instead.",
      flags: []
    };
    
    return safeJsonParse(response.choices[0].message.content, defaultResponse);
  } catch (error) {
    console.error('Error analyzing UPI with AI:', error);
    return {
      safety_score: 50,
      explanation: "Error during AI analysis. Using standard checks instead.",
      flags: []
    };
  }
}

/**
 * Analyze a UPI ID using ML-based QR code risk assessment
 * @param upiId UPI ID to analyze
 * @param reportCount Number of times this UPI has been reported
 * @returns ML-based risk assessment result
 */
export async function analyzeUpiWithML(upiId: string, reportCount: number = 0): Promise<{
  is_scam: boolean;
  risk_score: number;
  risk_level: string;
  explanation: string[];
  features: Record<string, any>;
} | null> {
  try {
    // Generate a mock QR code content for the UPI ID
    const qrContent = `upi://pay?pa=${upiId}&pn=Merchant`;
    
    // Use the child_process module to call the Python ML model
    const childProcess = await import('child_process');
    
    // Properly escape inputs for shell execution
    const escapedQrText = qrContent.replace(/"/g, '\\"').replace(/'/g, "\\'");
    
    const pythonCommand = `python -c "
import sys
import json
from qr_risk_detection_model import analyze_qr_risk

try:
    result = analyze_qr_risk(
        qr_content='${escapedQrText}',
        redirect_url='',
        report_count=${reportCount}
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
    
    return mlResult;
  } catch (error) {
    console.error('Error analyzing UPI with ML model:', error);
    return null;
  }
}

/**
 * Check if a UPI ID is safe, suspicious, or a scam
 * @param upiId UPI ID to check
 * @returns Result with status, reason, and confidence score
 */
export async function checkUpiSafety(upiId: string): Promise<UpiCheckResult> {
  // Normalize the UPI ID
  upiId = upiId.trim().toLowerCase();
  
  // Validate format
  if (!upiId.includes('@')) {
    return {
      status: 'SUSPICIOUS',
      reason: 'Invalid UPI ID format (missing @ symbol)',
      confidence_score: 0.8
    };
  }
  
  // Direct matches against known safe UPIs
  if (SAFE_UPI_IDS.includes(upiId)) {
    return {
      status: 'SAFE',
      reason: 'Verified legitimate UPI ID',
      confidence_score: 0.95,
      category: 'Verified'
    };
  }
  
  // Direct matches against known scam UPIs
  if (SCAM_UPI_IDS.includes(upiId)) {
    return {
      status: 'SCAM',
      reason: 'Known scam UPI ID',
      confidence_score: 0.98,
      risk_factors: ['Previously reported in scams'],
      category: 'Known Scam'
    };
  }
  
  // Check database for previous reports
  const reports = await storage.getScamReportsByUpiId(upiId);
  const reportCount = reports.length;
  
  if (reportCount > 0) {
    // If multiple reports exist, this is likely a scam
    if (reportCount >= 2) {
      return {
        status: 'SCAM',
        reason: `Reported ${reportCount} times by users`,
        confidence_score: Math.min(0.9 + (reportCount * 0.02), 0.99),
        risk_factors: [`Reported by ${reportCount} users`],
        category: 'User Reported',
        recommendations: getRecommendations('SCAM')
      };
    } else {
      // If only one report, treat as suspicious
      return {
        status: 'SUSPICIOUS',
        reason: 'Previously reported by a user',
        confidence_score: 0.7,
        risk_factors: ['Single user report exists'],
        category: 'User Reported',
        recommendations: getRecommendations('SUSPICIOUS')
      };
    }
  }
  
  // Try ML-based analysis first (our trained model)
  let mlAnalysis = null;
  try {
    mlAnalysis = await analyzeUpiWithML(upiId, reportCount);
  } catch (error) {
    console.error('Error during ML UPI analysis:', error);
    // Continue with standard checks if ML fails
  }
  
  // If ML analysis is available and successful, use it as the primary assessment
  if (mlAnalysis) {
    const { is_scam, risk_score, risk_level, explanation, features } = mlAnalysis;
    
    // Map ML risk level to our status categories
    let status: 'SAFE' | 'SUSPICIOUS' | 'SCAM';
    if (risk_score > 50) {
      status = 'SCAM';
    } else if (risk_score > 20) {
      status = 'SUSPICIOUS';
    } else {
      status = 'SAFE';
    }
    
    const result: UpiCheckResult = {
      status,
      reason: explanation[0] || `ML risk assessment: ${risk_level} risk`,
      confidence_score: risk_score / 100,
      risk_factors: explanation,
      category: 'ML Analysis',
      recommendations: getRecommendations(status),
      safety_score: 100 - risk_score
    };
    
    // Still run AI analysis to enhance the result
    try {
      const aiAnalysis = await analyzeUpiWithAI(upiId);
      if (aiAnalysis) {
        // Blend ML and AI scores with ML given higher priority
        result.safety_score = Math.round((result.safety_score * 0.7) + (aiAnalysis.safety_score * 0.3));
        result.ai_analysis = aiAnalysis.explanation;
        
        // Add AI-detected flags to risk factors
        if (aiAnalysis.flags && aiAnalysis.flags.length > 0) {
          if (!result.risk_factors) {
            result.risk_factors = [];
          }
          result.risk_factors.push(...aiAnalysis.flags);
        }
      }
    } catch (error) {
      console.error('Error during AI UPI analysis:', error);
    }
    
    return result;
  }
  
  // Fallback to traditional analysis if ML fails
  
  // Calculate pattern-based risk score
  const patternScore = calculatePatternScore(upiId);
  const riskFactors = getRiskFactors(upiId);
  
  // Use AI-powered analysis to enhance results
  let aiAnalysis = null;
  try {
    aiAnalysis = await analyzeUpiWithAI(upiId);
  } catch (error) {
    console.error('Error during AI UPI analysis:', error);
    // Continue with standard checks if AI fails
  }
  
  // Determine status based on pattern score
  let result: UpiCheckResult;
  if (patternScore > 0.7) {
    result = {
      status: 'SCAM',
      reason: 'High-risk pattern detected',
      confidence_score: patternScore,
      risk_factors: riskFactors,
      category: 'Pattern Analysis',
      recommendations: getRecommendations('SCAM')
    };
  } else if (patternScore > 0.4) {
    result = {
      status: 'SUSPICIOUS',
      reason: 'Moderate-risk pattern detected',
      confidence_score: patternScore,
      risk_factors: riskFactors,
      category: 'Pattern Analysis',
      recommendations: getRecommendations('SUSPICIOUS')
    };
  } else {
    result = {
      status: 'SAFE',
      reason: 'No known risk factors detected',
      confidence_score: 1 - patternScore,
      recommendations: getRecommendations('SAFE')
    };
  }
  
  // Add AI analysis if available
  if (aiAnalysis) {
    result.safety_score = aiAnalysis.safety_score;
    result.ai_analysis = aiAnalysis.explanation;
    
    // Add AI-detected flags to risk factors
    if (aiAnalysis.flags && aiAnalysis.flags.length > 0) {
      if (!result.risk_factors) {
        result.risk_factors = [];
      }
      result.risk_factors.push(...aiAnalysis.flags);
    }
  }
  
  return result;
}