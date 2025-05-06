/**
 * Scam Keywords Database
 * This file contains keywords and phrases commonly found in scam messages,
 * organized by category for more accurate detection.
 */

export interface ScamKeywordCategory {
  name: string;
  weight: number; // Higher weight means stronger indicator of a scam (0.1 to 1.0)
  keywords: string[];
}

export const scamKeywords: ScamKeywordCategory[] = [
  {
    name: "Banking Fraud",
    weight: 0.7,
    keywords: [
      "account blocked", "KYC update", "bank verification", "ATM blocked", 
      "re-activate your account", "net banking closed", "your card is suspended", 
      "update PAN card", "IFSC update", "unauthorized transaction",
      "account suspended", "verify your account", "banking issue", "debit card blocked",
      "credit card suspended", "urgent KYC", "KYC deadline", "KYC expired",
      "bank policy update", "mandatory verification"
    ]
  },
  {
    name: "Lottery & Fake Rewards",
    weight: 0.8,
    keywords: [
      "you've won", "congratulations", "lottery winner", "lucky draw", 
      "free car", "claim your prize", "gift card offer", "you are selected", 
      "100% cash back", "bonus credited", "lucky customer", "you are a winner",
      "millions of rupees", "cash prize", "selected randomly", "without any registration",
      "promotional offer", "reward points", "free gift", "unclaimed prize"
    ]
  },
  {
    name: "Phishing & Links",
    weight: 0.75,
    keywords: [
      "click here", "verify now", "urgent link", "shortlink", 
      "bit.ly", "tinyurl", "login to secure", "malicious link", 
      "suspicious URL", "secure your account", "click to update", "click to recover",
      "open this link", "important link", "click below", "verify with link",
      "complete verification", "login now", "update info", "restricted access"
    ]
  },
  {
    name: "Payment Fraud",
    weight: 0.85,
    keywords: [
      "UPI request", "Google Pay transfer", "send ₹1 to confirm", "money stuck", 
      "QR code payment", "recharge offer", "transaction failed", "amount deducted", 
      "wallet balance low", "pay now to unlock", "payment failed", "payment pending",
      "refund processing", "cashback offer", "payment link", "complete payment",
      "transaction error", "payment verification", "scan code to receive", "online transfer"
    ]
  },
  {
    name: "Identity & Impersonation",
    weight: 0.8,
    keywords: [
      "friend in trouble", "help me urgently", "I lost my phone", "posing as relative", 
      "this is my new number", "acting like police", "acting like bank", 
      "pretending to be support", "fraud agent", "fake customer care",
      "government official", "tax officer", "police investigation", "legal proceeding",
      "RBI official", "income tax department", "cybercrime branch", "customer service"
    ]
  },
  {
    name: "Pressure or Urgency Tactics",
    weight: 0.9,
    keywords: [
      "last chance", "within 5 minutes", "your account will be closed", "urgent reply", 
      "immediate action needed", "final warning", "or else", "do it now", 
      "today only", "24 hours left", "time running out", "immediate attention required",
      "urgent action", "critical alert", "warning alert", "immediately contact",
      "deadline approaching", "limited time", "act now", "respond immediately"
    ]
  },
  {
    name: "Technical Support Scams",
    weight: 0.75,
    keywords: [
      "your computer has virus", "detected malware", "security breach", 
      "account compromised", "suspicious activity", "technical department",
      "remote assistance", "security alert", "microsoft support", "google support",
      "apple support", "amazon support", "device infected", "system warning",
      "fix errors", "security issues", "tech support", "service expiring"
    ]
  },
  {
    name: "Investment & Crypto Scams",
    weight: 0.85,
    keywords: [
      "double your money", "investment opportunity", "guaranteed returns", 
      "high ROI", "bitcoin investment", "crypto profits", "best trading platform",
      "exclusive investment", "stock market tips", "inside information",
      "day trading", "quick profits", "minimum investment", "secret strategy",
      "risk-free returns", "passive income", "financial freedom", "market prediction"
    ]
  },
  {
    name: "Job & Income Scams",
    weight: 0.65,
    keywords: [
      "work from home", "earn daily", "part time job", "online earning", 
      "easy money", "make money online", "typing job", "data entry work",
      "no experience needed", "earn while learning", "high salary",
      "registration fee", "hiring immediately", "job guarantee", "earn per task",
      "payment proof", "weekly payment", "online jobs", "simple tasks"
    ]
  },
  {
    name: "Hindi-English Mixed Phrases",
    weight: 0.6,
    keywords: [
      "paisa bhejo urgent", "bank band ho gaya", "account update karo", 
      "jaldi karo", "abhi bhejo", "turant action lo", "KYC jaldi karo",
      "paise wapas chahiye", "account block ho jayega", "verification jaruri hai",
      "lottery jeeta hai", "gift claim karo", "mobile number change ho gaya",
      "OTP batao", "password batao", "card details do"
    ]
  }
];

/**
 * Get scam score based on text analysis against keywords
 * @param text Text to analyze for scam keywords
 * @returns Object with score and matched keywords
 */
export function analyzeTextForScamKeywords(text: string): {
  score: number;
  matches: Array<{category: string, keyword: string, weight: number}>;
  categories: {[key: string]: number};
} {
  // Normalize text for case-insensitive matching
  const normalizedText = text.toLowerCase();
  let totalScore = 0;
  let maxPossibleScore = 0;
  let matches: Array<{category: string, keyword: string, weight: number}> = [];
  let categoryCounts: {[key: string]: number} = {};
  
  // Check each category and its keywords
  scamKeywords.forEach(category => {
    let categoryHits = 0;
    
    category.keywords.forEach(keyword => {
      if (normalizedText.includes(keyword.toLowerCase())) {
        matches.push({
          category: category.name,
          keyword: keyword,
          weight: category.weight
        });
        
        // Only count the category weight once, but track multiple hits
        categoryHits++;
        
        // Update category statistics
        if (!categoryCounts[category.name]) {
          categoryCounts[category.name] = 0;
        }
        categoryCounts[category.name]++;
      }
    });
    
    // Add category weight to the score if there were any matches
    if (categoryHits > 0) {
      // Apply diminishing returns for multiple hits in same category
      const categoryScore = category.weight * Math.min(1, Math.log(categoryHits + 1) / Math.log(5));
      totalScore += categoryScore;
    }
    
    // Add to max possible score (if all categories matched)
    maxPossibleScore += category.weight;
  });
  
  // Normalize the score from 0 to 1
  const normalizedScore = maxPossibleScore > 0 ? totalScore / maxPossibleScore : 0;
  
  // Apply a sigmoid function to make mid-range scores more differentiating
  // This will make scores cluster less at extremes
  const sigmoidScore = 1 / (1 + Math.exp(-5 * (normalizedScore - 0.5)));
  
  return {
    score: sigmoidScore,
    matches: matches,
    categories: categoryCounts
  };
}