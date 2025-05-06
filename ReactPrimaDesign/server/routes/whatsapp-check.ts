import { Express, Request, Response } from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { ParamsDictionary } from "express-serve-static-core";
import { ParsedQs } from "qs";
import { analyzeWhatsAppMessage } from "../services/openai";
import { analyzeTextForScamKeywords } from "../data/scam-keywords";
import { extractTextFromImage } from "../services/ocr";

// Configure multer for image upload
const storage = multer.diskStorage({
  destination: (req: any, file: any, cb: any) => {
    const uploadDir = path.join(import.meta.dirname, "..", "..", "uploads");
    
    // Create directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    
    cb(null, uploadDir);
  },
  filename: (req: any, file: any, cb: any) => {
    // Create unique filename with timestamp
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    const ext = path.extname(file.originalname);
    cb(null, `whatsapp-${uniqueSuffix}${ext}`);
  },
});

// Filter to accept only image files
const fileFilter = (req: Request, file: any, cb: any) => {
  const allowedTypes = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error("Only image files are allowed"));
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB max size
  },
});

// Type definition for multer file, but we'll use 'any' for the route handler
// to avoid TypeScript errors with the Express File interface
type MulterFile = {
  fieldname: string;
  originalname: string;
  encoding: string;
  mimetype: string;
  size: number;
  destination: string;
  filename: string;
  path: string;
  buffer: Buffer;
};

/**
 * Register WhatsApp check routes
 * @param app Express application
 */
export function registerWhatsAppCheckRoutes(app: Express): void {
  // Endpoint to analyze WhatsApp messages
  app.post("/api/analyze-whatsapp", upload.single("image"), async (req: any, res: Response) => {
    try {
      const description = req.body.description || "";
      let imageBase64 = null;
      let extractedText = "";
      
      // If image is uploaded, read it and convert to base64
      if (req.file) {
        const imagePath = req.file.path;
        const imageBuffer = fs.readFileSync(imagePath);
        imageBase64 = imageBuffer.toString("base64");
        
        try {
          // Extract text from the image using OCR
          console.log("Performing OCR on WhatsApp screenshot...");
          extractedText = await extractTextFromImage(imageBase64);
          console.log(`OCR extracted ${extractedText.length} characters`);
        } catch (ocrError) {
          console.error("OCR extraction failed:", ocrError);
          // Continue with the process even if OCR fails
        }
        
        // Clean up the uploaded file after processing
        fs.unlinkSync(imagePath);
      }
      
      // If neither image nor description is provided
      if (!imageBase64 && !description) {
        return res.status(400).json({
          status: "error",
          message: "Please provide either an image or a description",
        });
      }
      
      // Combine extracted text with manual description for keyword analysis
      const combinedText = [extractedText, description].filter(Boolean).join("\n");
      
      // Perform keyword-based analysis
      console.log("Performing keyword analysis...");
      const keywordAnalysis = analyzeTextForScamKeywords(combinedText);
      
      // Set initial status based on keyword analysis score threshold
      let keywordBasedStatus = "Safe Message";
      if (keywordAnalysis.score > 0.7) {
        keywordBasedStatus = "Scam Likely";
      } else if (keywordAnalysis.score > 0.4) {
        keywordBasedStatus = "Suspicious Message";
      }
      
      console.log(`Keyword analysis score: ${keywordAnalysis.score}, status: ${keywordBasedStatus}`);
      
      // Analyze the message using OpenAI
      console.log("Performing AI analysis...");
      const aiAnalysis = await analyzeWhatsAppMessage(imageBase64, combinedText);
      
      // Combine both analyses for a more comprehensive result
      // Trust the AI analysis but boost confidence if keywords strongly suggest scam
      const isSuspicious = 
        aiAnalysis.is_scam || 
        keywordAnalysis.score > 0.7 || 
        (keywordAnalysis.score > 0.5 && aiAnalysis.confidence < 0.7);
      
      // Adjust confidence based on keyword matches
      let adjustedConfidence = aiAnalysis.confidence;
      if (keywordAnalysis.matches.length > 0) {
        // Boost confidence if keywords found, but avoid extremes
        adjustedConfidence = Math.min(
          0.95, 
          Math.max(
            0.3,
            // Weight AI more heavily (70%) but consider keywords (30%)
            (aiAnalysis.confidence * 0.7) + (keywordAnalysis.score * 0.3)
          )
        );
      }
      
      // Prepare the final analysis result
      const result = {
        is_scam: isSuspicious,
        confidence: adjustedConfidence,
        status: isSuspicious ? 
          (adjustedConfidence > 0.7 ? "Scam Likely" : "Suspicious Message") : 
          "Safe Message",
        scam_type: isSuspicious ? (aiAnalysis.scam_type || "Potential Fraud") : null,
        scam_indicators: [
          ...(aiAnalysis.scam_indicators || []),
          ...keywordAnalysis.matches.map(match => 
            `Contains suspicious phrase: "${match.keyword}" (${match.category})`
          )
        ],
        ocr_text: extractedText || null,
        keyword_analysis: {
          score: keywordAnalysis.score,
          match_count: keywordAnalysis.matches.length,
          categories: keywordAnalysis.categories
        },
        ai_analysis: {
          is_scam: aiAnalysis.is_scam,
          confidence: aiAnalysis.confidence,
          scam_type: aiAnalysis.scam_type
        }
      };
      
      console.log("WhatsApp analysis complete");
      res.json(result);
    } catch (error: any) {
      console.error("Error analyzing WhatsApp message:", error);
      res.status(500).json({
        status: "error",
        message: "Failed to analyze message",
        error: error.message,
      });
    }
  });
}