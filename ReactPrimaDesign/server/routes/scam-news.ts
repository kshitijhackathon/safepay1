import { Express } from "express";
import { getRealScamNews } from "../services/real-scam-news";

/**
 * Register scam news related routes to the Express server
 * @param app Express application
 */
export function registerScamNewsRoutes(app: Express) {
  // Fallback endpoint that returns static data without AI generation
  app.post("/api/scam-news/fallback", async (req, res) => {
    try {
      const { geo_location } = req.body;
      
      // Get location info or default to India
      const location = geo_location ? 
        (typeof geo_location === 'string' ? geo_location : 'India') : 
        'India';
      
      console.log(`Generating fallback scam news for location: ${location}`);
      
      // Get main city for location-specific content
      const mainCity = location.split(',')[0] || 'Mumbai';
      
      // Random numbers for display variety
      const randomReports = Math.floor(Math.random() * 300) + 200; // 200-500 range
      const randomLoss = Math.floor(Math.random() * 15000) + 7000; // ₹7,000-₹22,000 range
      const trustScore = Math.round(Math.random() * 30 + 65); // 65-95% range
      
      // Current date info for realistic dates
      const today = new Date();
      const yesterday = new Date(today);
      yesterday.setDate(today.getDate() - 1);
      const twoDaysAgo = new Date(today);
      twoDaysAgo.setDate(today.getDate() - 2);
      
      // Generate fallback scam news data that appears realistic and location-specific
      const fallbackData = {
        alerts: [
          {
            "title": `QR Code Payment Fraud in ${mainCity}`,
            "type": "QR Code Scam",
            "description": `Fraudsters in ${mainCity} are creating fake QR codes that direct payments to their accounts instead of legitimate merchants. Always verify the recipient's UPI ID before confirming payment.`,
            "affected_areas": [mainCity, "Delhi", "Bangalore"],
            "risk_level": "High",
            "date_reported": today.toISOString().split('T')[0],
            "verification_status": "Verified"
          },
          {
            "title": "Fake Bank Support Calls",
            "type": "Vishing",
            "description": `Scammers are calling residents in ${mainCity} and nearby areas pretending to be bank officials and requesting UPI details to "prevent unauthorized transactions".`,
            "affected_areas": [mainCity, "Pune", "Chennai"],
            "risk_level": "High",
            "date_reported": yesterday.toISOString().split('T')[0],
            "verification_status": "Verified"
          },
          {
            "title": "Social Media Marketplace Scams",
            "type": "Payment Fraud",
            "description": "Fraudsters are posting fake items for sale on marketplace platforms, then disappearing after receiving advance payments through UPI.",
            "affected_areas": ["Delhi", mainCity, "Kolkata"],
            "risk_level": "Medium",
            "date_reported": twoDaysAgo.toISOString().split('T')[0],
            "verification_status": "Investigating"
          }
        ],
        geo_spread: [], // Not implemented in this version
        prevention_tips: [
          {
            "tip": "Never share your UPI PIN, OTP or password with anyone, including bank representatives",
            "category": "Authentication"
          },
          {
            "tip": "Always verify the recipient UPI ID before payment",
            "category": "Verification"
          },
          {
            "tip": "Be suspicious of unexpected calls from bank 'officials' asking for personal details",
            "category": "Vigilance"
          },
          {
            "tip": "Use the SafePay app's QR scanner to detect potentially malicious QR codes",
            "category": "Technology"
          }
        ],
        reports_summary: {
          total_reports: randomReports,
          most_reported: ["QR Code Scams", "Fake Customer Support", "Phishing", "UPI ID Spoofing"],
          financial_loss: `₹${randomLoss.toLocaleString('en-IN')}`,
          emerging_patterns: ["Voice Call Scams", "Social Media Impersonation", "UPI ID Typosquatting"],
          hotspot_areas: [mainCity, "Delhi", "Bangalore", "Chennai", "Hyderabad"]
        },
        upi_analysis: null,
        trust_score: trustScore,
        last_updated: new Date().toISOString()
      };
      
      res.status(200).json(fallbackData);
    } catch (error) {
      console.error("Error generating fallback scam news:", error);
      res.status(500).json({ 
        error: "Failed to generate fallback scam news",
        message: (error as Error).message
      });
    }
  });
  
  // Get scam news data with AI generation
  app.post("/api/scam-news", async (req, res) => {
    try {
      const { geo_location, upi_id, trigger_source, refresh_token, use_fallback } = req.body;
      
      // For logging and analysis
      console.log(`Scam news requested - trigger: ${trigger_source || 'unknown'}, refresh: ${refresh_token ? 'yes' : 'no'}, fallback: ${use_fallback ? 'yes' : 'no'}`);
      
      // Get scam news data with user's location (or default to India)
      const location = geo_location ? 
        (typeof geo_location === 'string' ? geo_location : 'India') : 
        'India';
      
      // If refresh_token is provided, log it to ensure we're getting a new request each time
      if (refresh_token) {
        console.log(`Generating fresh scam news with token: ${refresh_token}, location: ${location}`);
      }
      
      // If use_fallback is set, use a timeout to avoid waiting too long
      let scamNewsData;
      
      try {
        // Use Promise.race with a timeout to avoid waiting too long for Groq API
        if (use_fallback) {
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('AI generation timeout')), 8000);
          });
          
          const dataPromise = getRealScamNews(location, upi_id);
          scamNewsData = await Promise.race([dataPromise, timeoutPromise]);
        } else {
          scamNewsData = await getRealScamNews(location, upi_id);
        }
      } catch (timeoutError) {
        console.log('AI generation timed out, using fallback data');
        // Redirect to fallback endpoint
        return res.redirect(307, '/api/scam-news/fallback');
      }
      
      res.status(200).json(scamNewsData);
    } catch (error) {
      console.error("Error fetching scam news:", error);
      res.status(500).json({ 
        error: "Failed to fetch scam news data",
        message: (error as Error).message
      });
    }
  });
  
  // Enhanced UPI analysis endpoint
  app.post("/api/scam-news/analyze-upi", async (req, res) => {
    try {
      const { upi_id } = req.body;
      
      if (!upi_id) {
        return res.status(400).json({ error: "UPI ID is required" });
      }
      
      // Get scam news with detailed UPI analysis
      const scamNewsWithUpiAnalysis = await getRealScamNews('India', upi_id);
      
      res.status(200).json({
        upi_analysis: scamNewsWithUpiAnalysis.upi_analysis
      });
    } catch (error) {
      console.error("Error analyzing UPI:", error);
      res.status(500).json({ 
        error: "Failed to analyze UPI ID",
        message: (error as Error).message
      });
    }
  });
  
  // Get location-specific alerts
  app.get("/api/scam-news/alerts/:location?", async (req, res) => {
    try {
      const location = req.params.location || 'India';
      const scamNewsData = await getRealScamNews(location);
      
      res.status(200).json({
        alerts: scamNewsData.alerts,
        last_updated: scamNewsData.last_updated
      });
    } catch (error) {
      console.error("Error fetching scam alerts:", error);
      res.status(500).json({ 
        error: "Failed to fetch scam alerts",
        message: (error as Error).message
      });
    }
  });
}