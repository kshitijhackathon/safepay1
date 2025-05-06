/**
 * OCR Service
 * Uses Groq's capabilities to extract text from images via base64 conversion
 * Note: Groq doesn't directly support vision models yet, so we'll implement a text-based 
 * approach or fallback to OpenAI if needed for production
 */

import { Groq } from "groq-sdk";

// Initialize Groq client
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY || ''
});

/**
 * Extract text from an image using available models
 * @param imageBase64 Base64 encoded image
 * @returns Extracted text from the image
 */
export async function extractTextFromImage(imageBase64: string): Promise<string> {
  try {
    // NOTE: Groq doesn't have direct vision capabilities yet
    // So we're implementing a fallback that indicates the limitation
    
    // In a production app, we might:
    // 1. Use a different OCR service (like Google Vision API)
    // 2. Continue using OpenAI for this specific feature
    // 3. Use base64 encoding in the prompt and ask Groq to extract what it can
    
    console.log("Image OCR requested - Groq doesn't support vision features directly");
    
    // Since we can't process the image directly with Groq, return a message
    // In a real application, you'd implement a proper OCR alternative here
    return "OCR with Groq is not yet supported. Please try a different approach for image text extraction.";
  } catch (error) {
    console.error("Groq OCR error:", error);
    return "";
  }
}

export default {
  extractTextFromImage
};