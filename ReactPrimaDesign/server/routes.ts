import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import {
  insertScamReportSchema,
  insertUserSchema,
  insertPaymentMethodSchema,
  insertChatMessageSchema,
  insertChatFeedbackSchema,
  ScamType,
} from "@shared/schema";
import { generateOtp, verifyOtp } from "./services/otp";
import { getChatResponse, generateQuickReplies } from "./services/chat";
import { getScamNews } from "./services/scam-news-fixed";
import { registerScamNewsRoutes } from "./routes/scam-news";
import { registerTestOpenAIRoute as registerTestGroqRoute } from "./routes/test-openai";
import { registerUpiCheckRoutes } from "./routes/upi-check";
import { registerVoiceCheckRoutes } from "./routes/voice-check";
import { registerEnhancedVoiceTextCheckRoutes } from "./routes/voice-text-ml";
import { registerWhatsAppCheckRoutes } from "./routes/whatsapp-check";
import { registerStreamlitRoutes } from "./routes/streamlit-routes";
import { registerRiskAnalysisRoutes } from "./routes/risk-analysis";
import { registerSimSwapProtectionRoutes } from "./routes/sim-swap-protection";
import { registerMLQRScanRoutes } from "./routes/ml-qr-scan";
import { registerOptimizedQRScanRoutes } from "./routes/register-optimized-qr-scan";
import { registerAdvancedQRScanRoutes } from "./routes/register-advanced-qr-scan";
import { registerPoliceComplaintRoutes } from "./routes/police-complaint";
import videoDetectionRouter, { setupVideoDetectionWebsocket } from "./routes/video_detection";
import { WebSocketServer } from "ws";
import { scrypt, randomBytes, timingSafeEqual } from "crypto";
import { promisify } from "util";
import Stripe from "stripe";

const scryptAsync = promisify(scrypt);

async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16).toString("hex");
  const buf = (await scryptAsync(password, salt, 64)) as Buffer;
  return `${buf.toString("hex")}.${salt}`;
}

async function comparePasswords(supplied: string, stored: string): Promise<boolean> {
  const [hashed, salt] = stored.split(".");
  const hashedBuf = Buffer.from(hashed, "hex");
  const suppliedBuf = (await scryptAsync(supplied, salt, 64)) as Buffer;
  return timingSafeEqual(hashedBuf, suppliedBuf);
}

// Import for nodemailer to handle sending emails
import nodemailer from "nodemailer";

// Groq client for generating formal email content (using mixtral-8x7b-32768 model)
import { Groq } from "groq-sdk";
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY || "" });
const MODEL = "mixtral-8x7b-32768";

export async function registerRoutes(app: Express): Promise<Server> {
  // OTP Authentication routes
  app.post("/api/auth/request-otp", async (req, res) => {
    try {
      const { phoneNumber } = req.body;

      if (!phoneNumber) {
        return res.status(400).json({ message: "Phone number is required" });
      }

      // Always return 123456 as the OTP
      res.status(200).json({
        message: "OTP sent successfully",
        otp: "123456",
      });
    } catch (error) {
      console.error("Error in OTP request:", error);
      res.status(500).json({ message: "Failed to process OTP request" });
    }
  });

  app.post("/api/auth/verify-otp", async (req, res) => {
    try {
      const { phoneNumber, otp } = req.body;

      if (!phoneNumber || !otp) {
        return res.status(400).json({
          success: false,
          message: "Phone number and OTP are required",
        });
      }

      // Only accept 123456 as the OTP
      if (otp !== "123456") {
        return res.status(401).json({
          success: false,
          message: "Invalid OTP. Please enter 123456",
        });
      }

      try {
        // Check if user exists with this phone number
        let user = await storage.getUserByPhoneNumber(phoneNumber);

        if (!user) {
          // Register new user with a random password (secure as it's hashed)
          const randomPassword = await hashPassword(randomBytes(16).toString("hex"));

          user = await storage.createUser({
            username: `user_${Date.now()}`, // Generate a unique username
            password: randomPassword,
            phoneNumber: phoneNumber,
            name: "", // Empty name indicates new user
          });
        }

        // Return success response with user ID and new user status
        return res.status(200).json({
          success: true,
          message: "Authentication successful",
          userId: user.id,
          isNewUser: !user.name, // If name is empty, it's a new user
        });
      } catch (dbError) {
        console.error("Database error during OTP verification:", dbError);
        return res.status(500).json({
          success: false,
          message: "Error accessing user data",
        });
      }
    } catch (error) {
      console.error("Error in OTP verification:", error);
      return res.status(500).json({
        success: false,
        message: "Failed to verify OTP",
      });
    }
  });

  app.post("/api/auth/setup-pin", async (req, res) => {
    try {
      const { userId, pin } = req.body;

      // For demo, PIN is required but userId is optional
      if (!pin) {
        return res.status(400).json({ message: "PIN is required" });
      }

      // Validate PIN format (should be numeric and 4-6 digits)
      if (!/^\d{4,6}$/.test(pin)) {
        return res.status(400).json({ message: "PIN must be 4-6 digits" });
      }

      // Demo mode - if userId is provided, update that user
      if (userId) {
        try {
          // Get the user
          const user = await storage.getUser(parseInt(userId));

          if (user) {
            // Hash the PIN for security
            const hashedPin = await hashPassword(pin);

            // Update user's PIN
            const updatedUser = await storage.updateUser(parseInt(userId), {
              pin: hashedPin,
              usePin: true,
            });

            if (!updatedUser) {
              console.log("Warning: Failed to update user with ID:", userId);
            }
          } else {
            console.log("Warning: User not found with ID:", userId);
          }
        } catch (error) {
          console.error("Error updating user PIN:", error);
          // Continue with demo mode
        }
      }

      // For demo purposes, always return success
      res.status(200).json({
        message: "PIN setup successful",
      });
    } catch (error) {
      console.error("Error setting up PIN:", error);
      res.status(500).json({ message: "Failed to set up PIN" });
    }
  });

  app.post("/api/auth/biometric", async (req, res) => {
    try {
      const { userId, enable, deviceId } = req.body;

      // For demo, not requiring user ID
      if (userId) {
        try {
          // Get the user
          const user = await storage.getUser(parseInt(userId));

          if (user) {
            // Update biometric settings
            const updatedUser = await storage.updateUser(parseInt(userId), {
              useBiometric: enable === true,
              deviceId: deviceId || user.deviceId,
            });

            if (!updatedUser) {
              console.log("Warning: Failed to update user biometric settings with ID:", userId);
            }
          } else {
            console.log("Warning: User not found with ID:", userId);
          }
        } catch (error) {
          console.error("Error updating biometric settings:", error);
          // Continue with demo mode
        }
      }

      // For demo purposes, always return success
      res.status(200).json({
        message: enable ? "Biometric authentication enabled" : "Biometric authentication disabled",
      });
    } catch (error) {
      console.error("Error updating biometric settings:", error);
      res.status(500).json({ message: "Failed to update biometric settings" });
    }
  });

  // Existing User routes
  app.post("/api/users/signup", async (req, res) => {
    try {
      const userData = insertUserSchema.parse(req.body);
      const user = await storage.createUser(userData);
      res.status(201).json({ message: "User created successfully", userId: user.id });
    } catch (error) {
      res.status(400).json({ message: "Invalid user data", error });
    }
  });

  app.post("/api/users/login", async (req, res) => {
    try {
      const { username, password } = req.body;
      const user = await storage.getUserByUsername(username);

      if (!user || user.password !== password) {
        return res.status(401).json({ message: "Invalid credentials" });
      }

      res.status(200).json({ message: "Login successful", userId: user.id });
    } catch (error) {
      res.status(500).json({ message: "Error logging in", error });
    }
  });

  // UPI risk check routes
  app.get("/api/upi/check/:upiId", async (req, res) => {
    try {
      const { upiId } = req.params;

      // Check UPI safety using improved algorithm
      const { checkUpiSafety } = await import("./services/upi-check");
      const safetyCheck = await checkUpiSafety(upiId);

      // Get additional risk data from the database
      let riskReport;
      try {
        riskReport = await storage.getUpiRiskByUpiId(upiId);
      } catch (dbError) {
        console.error("Error checking UPI risk:", dbError);
        // Continue with null riskReport
      }

      // Convert status to risk level
      let riskLevel: "Low" | "Medium" | "High" = "Low";

      switch (safetyCheck.status) {
        case "SCAM":
          riskLevel = "High";
          break;
        case "SUSPICIOUS":
          riskLevel = "Medium";
          break;
        case "SAFE":
          riskLevel = "Low";
          break;
      }

      // Use random calculation for QR code detection per user's request
      let riskPercentage: number;

      // Follow the original risk percentage rules
      if (safetyCheck.status === "SAFE") {
        // For SAFE, risk score should be low (higher number = lower risk)
        riskPercentage = Math.floor(Math.random() * 20) + 70; // 70-89%
      } else if (safetyCheck.status === "SUSPICIOUS") {
        // For SUSPICIOUS, risk score should be medium
        riskPercentage = Math.floor(Math.random() * 20) + 40; // 40-59%
      } else {
        // For SCAM, risk score should be high (lower number = higher risk)
        riskPercentage = Math.floor(Math.random() * 30) + 10; // 10-39%
      }

      // Combine data and send response
      res.status(200).json({
        upiId,
        status: safetyCheck.status,
        riskPercentage,
        riskLevel,
        reports: riskReport ? riskReport.reportCount : 0,
        reason: safetyCheck.reason,
        confidence_score: safetyCheck.confidence_score,
        risk_factors: safetyCheck.risk_factors || [],
        recommendations: safetyCheck.recommendations || [],
        age: riskReport ? calculateUpiAge(riskReport.firstReportDate) : "Unknown",
        reportedFor: riskReport ? await storage.getMostCommonScamType(upiId) : "N/A",
      });
    } catch (error) {
      console.error("Error checking UPI risk:", error);
      res.status(500).json({ message: "Error checking UPI risk", error });
    }
  });

  // Simple check-scam API
  app.post("/api/check-scam", async (req, res) => {
    try {
      const { upiId } = req.body;

      if (!upiId) {
        return res.status(400).json({
          error: "Missing UPI ID",
          message: "Please provide a UPI ID to check",
        });
      }

      // Check UPI safety using improved algorithm
      const { checkUpiSafety } = await import("./services/upi-check");
      const result = await checkUpiSafety(upiId);

      res.status(200).json(result);
    } catch (error) {
      console.error("Error in /check-scam:", error);
      res.status(500).json({
        error: "Server error",
        message: "An error occurred while checking the UPI ID",
      });
    }
  });

  // Scam report routes
  app.post("/api/report/scam", async (req, res) => {
    try {
      const reportData = insertScamReportSchema.parse(req.body);
      await storage.createScamReport(reportData);

      // Update UPI risk score
      await storage.updateUpiRiskScore(reportData.upiId);

      res.status(201).json({ message: "Scam report submitted successfully" });
    } catch (error) {
      res.status(400).json({ message: "Invalid report data", error });
    }
  });

  // New API endpoint for client compatibility
  app.post("/api/scam-reports", async (req, res) => {
    try {
      const { upiId, userId, scamType, description, amountLost } = req.body;

      // Data validation
      if (!upiId) {
        return res.status(400).json({ error: "UPI ID is required" });
      }

      if (!userId) {
        return res.status(400).json({ error: "User ID is required" });
      }

      // Convert amount to number if needed
      let numericAmount = null;
      if (amountLost) {
        const cleanAmount = amountLost.toString().replace(/[^\d.]/g, "");
        numericAmount = parseFloat(cleanAmount);
      }

      // Map string scam type to enum if needed
      let finalScamType = scamType;
      if (typeof scamType === "string") {
        switch (scamType.toLowerCase()) {
          case "banking":
          case "banking scam":
            finalScamType = ScamType.Banking;
            break;
          case "kyc":
          case "kyc verification scam":
            finalScamType = ScamType.KYC;
            break;
          case "lottery":
          case "lottery scam":
            finalScamType = ScamType.Lottery;
            break;
          case "refund":
          case "refund scam":
            finalScamType = ScamType.Refund;
            break;
          case "phishing":
          case "phishing attempt":
            finalScamType = ScamType.Phishing;
            break;
          case "reward":
          case "reward scam":
            finalScamType = ScamType.Reward;
            break;
          default:
            finalScamType = ScamType.Unknown;
        }
      }

      const report = await storage.createScamReport({
        upiId,
        userId: parseInt(userId.toString()),
        scamType: finalScamType,
        description: description || null,
        amountLost: numericAmount,
      });

      // Update UPI risk score
      await storage.updateUpiRiskScore(upiId);

      res.status(201).json({
        message: "Scam report submitted successfully",
        reportId: report.id,
      });
    } catch (error) {
      console.error("Error creating scam report:", error);
      res.status(500).json({ error: "Failed to submit scam report" });
    }
  });

  // Get scam reports for a specific user
  app.get("/api/user/:userId/scam-reports", async (req, res) => {
    try {
      const userId = parseInt(req.params.userId);

      if (isNaN(userId)) {
        return res.status(400).json({ error: "Invalid user ID" });
      }

      const reports = await storage.getScamReportsByUserId(userId);

      // Format the report data for the client
      const formattedReports = reports.map((report) => ({
        id: report.id,
        upiId: report.upiId,
        scamType: report.scamType,
        description: report.description,
        amountLost: report.amountLost ? report.amountLost.toFixed(2) : null,
        timestamp: report.timestamp,
      }));

      res.json(formattedReports);
    } catch (error) {
      console.error("Error fetching user scam reports:", error);
      res.status(500).json({ error: "Failed to retrieve scam reports" });
    }
  });

  // Police complaint routes
  app.post("/api/report/police-complaint", async (req, res) => {
    try {
      const {
        upiId,
        name,
        amount,
        policeStation = "Cyber Crime Police Station",
        city = "New Delhi",
        userDetails = {},
        transactionDate,
        description,
      } = req.body;

      // Basic validation
      if (!upiId || !name) {
        return res.status(400).json({ message: "UPI ID and recipient name are required" });
      }

      // Create a scam report entry in our system
      try {
        const reportData = {
          upiId,
          userId: userDetails.userId || 1, // Use default user if not provided
          scamType: ScamType.Banking, // Using Banking scam type for fraud
          description: description || `Police complaint filed against ${upiId}`,
          amountLost: parseFloat(amount.replace(/,/g, "")) || null,
        };

        await storage.createScamReport(reportData);
        await storage.updateUpiRiskScore(upiId);
      } catch (error) {
        console.error("Error creating scam report:", error);
        // Continue with police complaint even if our internal reporting fails
      }

      // In a real system, this would communicate with police API
      // For now, we'll simulate successful submission

      res.status(201).json({
        message: "Police complaint submitted successfully",
        complaintId: "PC" + Date.now().toString().slice(-8),
        submissionTime: new Date().toISOString(),
      });
    } catch (error) {
      console.error("Error submitting police complaint:", error);
      res.status(500).json({ message: "Error submitting police complaint", error });
    }
  });

  // Transactions routes
  app.get("/api/transactions/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const transactions = await storage.getTransactionsByUserId(parseInt(userId));
      res.status(200).json(transactions);
    } catch (error) {
      res.status(500).json({ message: "Error fetching transactions", error });
    }
  });

  // User profile routes
  app.get("/api/users/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const user = await storage.getUser(parseInt(userId));

      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }

      // Exclude sensitive information
      const { password, pin, ...safeUser } = user;

      res.status(200).json(safeUser);
    } catch (error) {
      res.status(500).json({ message: "Error fetching user data", error });
    }
  });

  app.patch("/api/users/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const updates = req.body;

      // Never allow password updates through this endpoint for security
      if (updates.password) {
        delete updates.password;
      }

      const updatedUser = await storage.updateUser(parseInt(userId), updates);

      if (!updatedUser) {
        return res.status(404).json({ message: "User not found" });
      }

      // Exclude sensitive information
      const { password, pin, ...safeUser } = updatedUser;

      res.status(200).json(safeUser);
    } catch (error) {
      res.status(500).json({ message: "Error updating user data", error });
    }
  });

  // Payment method routes
  app.get("/api/payment-methods/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const methods = await storage.getPaymentMethodsByUserId(parseInt(userId));

      // Mask sensitive data for extra security
      const maskedMethods = methods.map((method) => {
        // Only return last 4 digits for card numbers and account numbers
        return {
          ...method,
          cardNumber: method.cardNumber ? `xxxx-xxxx-xxxx-${method.cardNumber}` : null,
          accountNumber: method.accountNumber ? `xxxx-xxxx-${method.accountNumber}` : null,
        };
      });

      res.status(200).json(maskedMethods);
    } catch (error) {
      res.status(500).json({ message: "Error fetching payment methods", error });
    }
  });

  app.post("/api/payment-methods", async (req, res) => {
    try {
      const methodData = insertPaymentMethodSchema.parse(req.body);
      const method = await storage.createPaymentMethod(methodData);

      // Mask sensitive data for response
      const { cardNumber, accountNumber, ...safeMethod } = method;
      const maskedMethod = {
        ...safeMethod,
        cardNumber: cardNumber ? `xxxx-xxxx-xxxx-${cardNumber}` : null,
        accountNumber: accountNumber ? `xxxx-xxxx-${accountNumber}` : null,
      };

      res.status(201).json(maskedMethod);
    } catch (error) {
      res.status(400).json({ message: "Invalid payment method data", error });
    }
  });

  app.patch("/api/payment-methods/:methodId", async (req, res) => {
    try {
      const { methodId } = req.params;
      const updates = req.body;

      const updatedMethod = await storage.updatePaymentMethod(parseInt(methodId), updates);

      if (!updatedMethod) {
        return res.status(404).json({ message: "Payment method not found" });
      }

      // Mask sensitive data for response
      const { cardNumber, accountNumber, ...safeMethod } = updatedMethod;
      const maskedMethod = {
        ...safeMethod,
        cardNumber: cardNumber ? `xxxx-xxxx-xxxx-${cardNumber}` : null,
        accountNumber: accountNumber ? `xxxx-xxxx-${accountNumber}` : null,
      };

      res.status(200).json(maskedMethod);
    } catch (error) {
      res.status(500).json({ message: "Error updating payment method", error });
    }
  });

  app.delete("/api/payment-methods/:methodId", async (req, res) => {
    try {
      const { methodId } = req.params;
      const success = await storage.deletePaymentMethod(parseInt(methodId));

      if (!success) {
        return res.status(404).json({ message: "Payment method not found" });
      }

      res.status(200).json({ message: "Payment method deleted successfully" });
    } catch (error) {
      res.status(500).json({ message: "Error deleting payment method", error });
    }
  });

  app.post("/api/payment-methods/:userId/set-default/:methodId", async (req, res) => {
    try {
      const { userId, methodId } = req.params;
      const success = await storage.setDefaultPaymentMethod(parseInt(userId), parseInt(methodId));

      if (!success) {
        return res.status(404).json({ message: "User or payment method not found" });
      }

      res.status(200).json({ message: "Default payment method updated successfully" });
    } catch (error) {
      res.status(500).json({ message: "Error updating default payment method", error });
    }
  });

  // Chat routes
  app.get("/api/chat/:userId/history", async (req, res) => {
    try {
      const { userId } = req.params;
      const messages = await storage.getChatHistoryByUserId(parseInt(userId));
      res.status(200).json(messages);
    } catch (error) {
      console.error("Error fetching chat history:", error);
      res.status(500).json({ message: "Error fetching chat history", error });
    }
  });

  app.post("/api/chat/:userId/message", async (req, res) => {
    try {
      const { userId } = req.params;
      const { content } = req.body;

      if (!content) {
        return res.status(400).json({ message: "Message content is required" });
      }

      // Save user message
      const userMessage = await storage.saveChatMessage(parseInt(userId), {
        role: "user",
        content,
      });

      // Get chat history for context
      const chatHistory = await storage.getChatHistoryByUserId(parseInt(userId));

      // Convert DB messages to service-compatible format
      const serviceMessages = chatHistory.slice(-10).map((msg) => ({
        role: msg.role as "user" | "assistant" | "system",
        content: msg.content,
      }));

      // Process with AI service
      const assistantResponse = await getChatResponse(serviceMessages, content);

      // Save assistant response
      const assistantMessage = await storage.saveChatMessage(parseInt(userId), {
        role: "assistant",
        content: assistantResponse,
      });

      // Generate quick replies
      const updatedServiceMessages = [
        ...serviceMessages,
        {
          role: "assistant" as const,
          content: assistantResponse,
        },
        {
          role: "user" as const,
          content,
        },
      ];

      const quickReplies = await generateQuickReplies(updatedServiceMessages);

      res.status(200).json({
        userMessage,
        assistantMessage,
        quickReplies,
      });
    } catch (error) {
      console.error("Error processing chat message:", error);
      res.status(500).json({
        message: "Error processing chat message",
        error,
        fallbackResponse:
          "I apologize, but I'm having trouble connecting to my knowledge base. Please try again later.",
      });
    }
  });

  app.post("/api/chat/:userId/feedback", async (req, res) => {
    try {
      const { userId } = req.params;
      const { messageId, rating, feedback } = req.body;

      if (!messageId) {
        return res.status(400).json({ message: "Message ID is required" });
      }

      // At least one of rating or feedback should be provided
      if (rating === undefined && !feedback) {
        return res.status(400).json({ message: "Rating or feedback is required" });
      }

      const savedFeedback = await storage.saveChatFeedback(parseInt(userId), parseInt(messageId), {
        rating: rating !== undefined ? parseInt(rating) : undefined,
        feedback,
      });

      res.status(200).json({
        message: "Feedback saved successfully",
        feedback: savedFeedback,
      });
    } catch (error) {
      console.error("Error saving chat feedback:", error);
      res.status(500).json({ message: "Error saving chat feedback", error });
    }
  });

  // Advanced Fraud Detection API (Mock Service)
  app.post("/api/fraud-check", (req, res) => {
    try {
      const { upiId, amount, deviceInfo } = req.body;

      if (!upiId) {
        return res.status(400).json({ message: "UPI ID is required" });
      }

      // Simulate response from Python fraud detection service
      const mockResponse = {
        prediction: Math.random() > 0.8, // 20% chance of fraud detection
        confidence: parseFloat((Math.random() * 0.8 + 0.1).toFixed(2)), // Value between 0.1 and 0.9
        features: {
          hourly_reports: Math.floor(Math.random() * 5),
          tx_frequency: Math.floor(Math.random() * 50),
          amount_deviation: parseFloat((Math.random() * 2 - 1).toFixed(2)),
          device_risk: Math.floor(Math.random() * 3),
          platform_reports: Math.floor(Math.random() * 10),
        },
        live_data: {
          tx_frequency: Math.floor(Math.random() * 30),
          avg_amount: parseFloat((Math.random() * 5000 + 500).toFixed(2)),
          device_mismatches: Math.floor(Math.random() * 3),
          recent_reports: Math.floor(Math.random() * 5),
        },
        message: "Fraud risk assessment completed successfully",
        meta: {
          service: "mock-ml-service",
          version: "1.0.0",
          latency_ms: Math.floor(Math.random() * 200 + 100),
        },
      };

      // Add a 500ms delay to simulate ML processing time
      setTimeout(() => {
        res.status(200).json(mockResponse);
      }, 500);
    } catch (error: any) {
      console.error("Fraud detection error:", error);
      res.status(500).json({
        message: "Fraud detection service error",
        error: error?.message || "Unknown error",
      });
    }
  });

  // Register Scam News routes
  registerScamNewsRoutes(app);

  // Register Test Groq AI route
  registerTestGroqRoute(app);

  // Register UPI Check routes
  registerUpiCheckRoutes(app);

  // Register Voice Check routes
  registerVoiceCheckRoutes(app);

  // Register Enhanced ML-Based Voice/Text Check routes
  registerEnhancedVoiceTextCheckRoutes(app);

  // Register WhatsApp Check routes
  registerWhatsAppCheckRoutes(app);

  // Register Streamlit routes for Fraud Map
  registerStreamlitRoutes(app);

  // Register Risk Analysis routes
  registerRiskAnalysisRoutes(app);

  // Register SIM swap protection routes
  registerSimSwapProtectionRoutes(app);

  // Register ML-powered QR scanner routes
  registerMLQRScanRoutes(app);

  // Register Optimized QR scanner routes
  registerOptimizedQRScanRoutes(app);

  // Register Advanced ML-based QR scan routes
  registerAdvancedQRScanRoutes(app);

  registerPoliceComplaintRoutes(app);

  // Register video detection routes
  console.log("[routes] Registering Video Detection Routes...");
  app.use("/api/video-detection", videoDetectionRouter);
  console.log("[routes] ✅ Video detection routes registered");

  // No test pages anymore

  // Initialize Stripe with error handling
  let stripe: Stripe | null = null;
  try {
    if (process.env.STRIPE_SECRET_KEY) {
      stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
    } else {
      console.warn("STRIPE_SECRET_KEY not found in environment variables");
    }
  } catch (error) {
    console.error("Failed to initialize Stripe:", error);
  }

  // Stripe payment routes
  app.post("/api/create-payment-intent", async (req, res) => {
    try {
      const { amount, currency = "inr", upiId, description } = req.body;

      if (!amount || amount <= 0) {
        return res.status(400).json({ error: "Valid amount is required" });
      }

      console.log("Creating payment intent for:", { amount, currency, upiId, description });

      if (!stripe) {
        return res.status(503).json({ error: "Payment service is not available" });
      }

      // Create payment intent with Stripe
      const paymentIntent = await stripe.paymentIntents.create({
        amount: Math.round(amount * 100), // Convert to smallest currency unit (paise/cents)
        currency: currency,
        payment_method_types: ["card"],
        metadata: {
          upiId: upiId || "",
          description: description || "UPI Payment",
        },
      });

      // Return client secret for frontend to complete payment
      res.json({
        clientSecret: paymentIntent.client_secret,
        paymentIntentId: paymentIntent.id,
      });
    } catch (error: any) {
      console.error("Error creating payment intent:", error);
      res.status(500).json({ error: error.message || "Failed to create payment" });
    }
  });

  // Route to confirm a payment was successful
  app.get("/api/payment/:paymentIntentId", async (req, res) => {
    try {
      const { paymentIntentId } = req.params;

      console.log("Checking payment status for:", paymentIntentId);

      const paymentIntent = await stripe.paymentIntents.retrieve(paymentIntentId);

      if (paymentIntent.status === "succeeded") {
        // Record the successful transaction in our database
        const transaction = await storage.createTransaction({
          userId: req.query.userId ? Number(req.query.userId) : 1, // Default to user 1 if no user ID
          amount: paymentIntent.amount / 100, // Convert back from smallest currency unit
          currency: paymentIntent.currency,
          upiId: paymentIntent.metadata.upiId || "",
          status: "completed",
          transactionType: "payment",
          paymentMethod: "card",
          paymentIntentId: paymentIntentId,
          description: paymentIntent.metadata.description || "Card Payment",
        });

        res.json({
          success: true,
          status: paymentIntent.status,
          transaction,
        });
      } else {
        res.json({
          success: false,
          status: paymentIntent.status,
        });
      }
    } catch (error: any) {
      console.error("Error checking payment status:", error);
      res.status(500).json({ error: error.message || "Failed to check payment status" });
    }
  });

  // Create HTTP server and setup WebSocket server
  const httpServer = createServer(app);

  // Initialize WebSocket server on the same HTTP server instance
  // This ensures the WebSocket server uses the same dynamic port as the HTTP server
  const wss = new WebSocketServer({
    server: httpServer,
    path: "/ws", // Base WebSocket path
  });

  // Setup WebSocket handlers for video detection
  console.log("[routes] Setting up WebSocket handlers for video detection...");
  setupVideoDetectionWebsocket(wss);
  console.log("[routes] ✅ WebSocket handlers for video detection configured");

  return httpServer;
}

// Helper function to calculate UPI age
function calculateUpiAge(date: Date): string {
  const now = new Date();
  const diffTime = Math.abs(now.getTime() - date.getTime());
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

  if (diffDays < 30) return `${diffDays} days`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months`;
  return `${Math.floor(diffDays / 365)} years`;
}
