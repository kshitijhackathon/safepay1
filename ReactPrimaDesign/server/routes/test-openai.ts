import { Express } from "express";
import { testOpenAI } from "../services/test-openai";

export function registerTestOpenAIRoute(app: Express) {
  // Keep the old route for backward compatibility
  app.post("/api/test-openai", async (req, res) => {
    try {
      const result = await testOpenAI();
      res.status(200).json({ result });
    } catch (error) {
      console.error("Error in AI test route:", error);
      res.status(500).json({ 
        error: "Failed to test AI service", 
        message: (error as Error).message 
      });
    }
  });
  
  // Add a new route with more accurate name
  app.post("/api/test-groq", async (req, res) => {
    try {
      const result = await testOpenAI();
      res.status(200).json({ result });
    } catch (error) {
      console.error("Error in test Groq route:", error);
      res.status(500).json({ 
        error: "Failed to test Groq service", 
        message: (error as Error).message 
      });
    }
  });
}