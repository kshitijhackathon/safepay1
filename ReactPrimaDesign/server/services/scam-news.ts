import { Groq } from "groq-sdk";

// Initialize Groq client for improved AI capabilities
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY || '' });
const MODEL = 'llama-3.3-70b-versatile'; // Using latest available Groq model for better reasoning

/**
 * Safely parses JSON from Groq response content with fallback
 * @param content Groq response content which might be null
 * @param defaultValue Default value to use if parsing fails
 * @returns Parsed JSON object
 */
function safeJsonParse(content: string | null | undefined, defaultValue: any = {}) {
  if (!content) return defaultValue;
  try {
    return JSON.parse(content);
  } catch (error) {
    console.error('Error parsing JSON from Groq response:', error);
    return defaultValue;
  }
}

/**
 * Generate location-specific scam alerts based on user's location
 * @param location User's location coordinates or city name
 * @returns Array of personalized scam alerts
 */
export async function generateLocalScamAlerts(location: string = "India") {
  try {
    // In a real app, we would use location to filter relevant scams
    // For this demo, we'll use Groq to generate relevant alerts
    const locationInfo = location || 'India'; // Default to India if empty string

    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content: `You are a scam alert system that generates realistic UPI payment fraud alerts for users in ${locationInfo}. 
          Generate 3-5 realistic and detailed scam alerts that could happen in ${locationInfo}, focusing on UPI payment frauds.
          For each alert include:
          1. A title for the scam (be brief but descriptive)
          2. The type of scam (e.g., QR code, phishing, fake call center, etc.)
          3. A brief description (2-3 sentences max)
          4. The affected area(s) in ${locationInfo}
          5. The risk level (High, Medium, Low)
          6. Date reported (within the last week)
          7. A verification status (Verified, Investigating, Unverified)`
        },
        {
          role: "user",
          content: `Generate the latest UPI scam alerts for ${locationInfo}. Focus on realistic, currently happening scams. IMPORTANT: Return a JSON object with an 'alerts' array containing the scam alerts.`
        }
      ]
      // Note: Groq doesn't support response_format parameter like OpenAI
    });

    const result = safeJsonParse(response.choices[0].message.content, { alerts: [] });
    return result.alerts || [];
  } catch (error) {
    console.error('Error generating scam alerts:', error);
    return [];
  }
}

/**
 * Generate geographic distribution of scam activities
 * @param location User's location
 * @returns Heatmap data of scam activity
 */
export async function generateScamHeatmapData(location: string = "India") {
  try {
    const locationInfo = location || 'India'; // Default to India if empty string
    
    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content: `You are a fraud analytics system that generates geographic heatmap data for UPI scams in India.
          Generate realistic heatmap data showing the concentration of different types of UPI payment scams across 
          major cities and regions in India. The data should include:
          1. City/region name
          2. Coordinates (latitude, longitude)
          3. Scam intensity (0-100)
          4. Predominant scam type
          5. Trend (Increasing, Stable, Decreasing)`
        },
        {
          role: "user",
          content: `Generate geographic distribution data for current UPI scams in ${locationInfo} for our heatmap visualization. IMPORTANT: Return a JSON object with a 'heatmap_data' array containing the geographic data points.`
        }
      ]
      // Note: Groq doesn't support response_format parameter like OpenAI
    });

    const result = safeJsonParse(response.choices[0].message.content, { heatmap_data: [] });
    return result.heatmap_data || [];
  } catch (error) {
    console.error('Error generating heatmap data:', error);
    return [];
  }
}

/**
 * Get the latest and most relevant scam prevention tips
 * @returns Array of scam prevention tips
 */
export async function getScamPreventionTips() {
  try {
    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content: `You are a UPI security expert providing practical tips to prevent scams.
          Generate 5 concise, actionable tips to help users avoid UPI payment scams.
          Each tip should:
          1. Be focused on a specific aspect of UPI security
          2. Be practical and immediately actionable
          3. Be no more than 2 sentences
          4. Include a category tag (e.g., #QRCode, #Verification, #PhishingProtection)`
        },
        {
          role: "user",
          content: `What are the most important tips to avoid UPI payment scams? IMPORTANT: Return a JSON object with a 'tips' array containing the prevention tips.`
        }
      ]
      // Note: Groq doesn't support response_format parameter like OpenAI
    });

    const result = safeJsonParse(response.choices[0].message.content, { tips: [] });
    return result.tips || [];
  } catch (error) {
    console.error('Error generating prevention tips:', error);
    return [];
  }
}

/**
 * Analyze a specific UPI ID for trending scams associated with it
 * @param upiId UPI ID to analyze
 * @returns Analysis of the UPI ID's risk factors
 */
export async function analyzeUpiForTrendingScams(upiId: string) {
  try {
    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content: `You are a UPI fraud analysis system. Based on a given UPI ID, provide an analysis of potential risk factors 
          related to current trending scams. The analysis should include:
          1. UPI ID pattern recognition
          2. Association with known scam patterns
          3. Similarity to legitimate services (potential typosquatting)
          4. Risk assessment
          
          If no specific risk factors are found, provide general guidance.`
        },
        {
          role: "user",
          content: `Analyze this UPI ID for potential association with trending scams: ${upiId}. IMPORTANT: Return a JSON object with risk_level and analysis fields.`
        }
      ]
      // Note: Groq doesn't support response_format parameter like OpenAI
    });

    const content = response.choices[0].message.content || '{}';
    const result = safeJsonParse(content);
    return result;
  } catch (error) {
    console.error('Error analyzing UPI for trending scams:', error);
    return {
      risk_level: "Unknown",
      analysis: "Unable to analyze UPI ID due to an error."
    };
  }
}

/**
 * Get aggregated user reports on recent scams
 * @returns Summary of user-reported scams
 */
export async function getUserScamReportsSummary() {
  try {
    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content: `You are a community reporting system that aggregates user-reported UPI scams.
          Generate a realistic summary of recent user-reported scams. The summary should include:
          1. Total number of reports in the last week
          2. Most reported scam types
          3. Common financial loss range
          4. Emerging scam patterns
          5. Areas with highest report concentration`
        },
        {
          role: "user",
          content: `Provide a summary of recent user-reported UPI scams. IMPORTANT: Return a complete JSON object with total_reports, most_reported (array), financial_loss, emerging_patterns (array), and hotspot_areas (array) fields.`
        }
      ]
      // Note: Groq doesn't support response_format parameter like OpenAI
    });

    const content = response.choices[0].message.content || '{}';
    const result = safeJsonParse(content);
    return result;
  } catch (error) {
    console.error('Error getting user scam reports summary:', error);
    return {
      total_reports: 0,
      most_reported: [],
      financial_loss: "N/A",
      emerging_patterns: [],
      hotspot_areas: []
    };
  }
}

/**
 * Complete scam news API that aggregates all required data
 * @param location User's location
 * @param upiId Optional UPI ID to analyze
 * @returns Comprehensive scam news data
 */
export async function getScamNews(location: string = "India", upiId?: string) {
  try {
    // Execute requests in parallel for better performance
    const [alerts, heatmapData, tips, reportsSummary] = await Promise.all([
      generateLocalScamAlerts(location),
      generateScamHeatmapData(location),
      getScamPreventionTips(),
      getUserScamReportsSummary()
    ]);

    // Conditionally analyze UPI if provided
    let upiAnalysis = null;
    if (upiId) {
      upiAnalysis = await analyzeUpiForTrendingScams(upiId);
    }

    // Calculate overall trust score (hypothetically based on data quality, source reliability, etc.)
    const trustScore = Math.round(Math.random() * 30 + 70); // 70-100 for demo

    return {
      alerts,
      geo_spread: heatmapData,
      prevention_tips: tips,
      reports_summary: reportsSummary,
      upi_analysis: upiAnalysis,
      trust_score: trustScore,
      last_updated: new Date().toISOString()
    };
  } catch (error) {
    console.error('Error getting scam news:', error);
    throw new Error('Failed to retrieve scam news data');
  }
}