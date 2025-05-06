// This file provides a compatibility layer for OpenAI API services
// It now uses Groq behind the scenes for improved performance and lower latency

import { Groq } from 'groq-sdk';
import { analyzeUpiId, analyzeMessageForScams, analyzeChatSentiment, generateSecurityTip } from './groq';

// Initialize Groq client
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY || '',
});

// Model to use
const MODEL = 'llama-3.3-70b-versatile';

// Just reexport the Groq implementations directly to maintain compatibility
export { analyzeUpiId, analyzeMessageForScams, analyzeChatSentiment, generateSecurityTip };

/**
 * Validate UPI ID safety using advanced AI
 * @param upiId The UPI ID to validate
 * @returns Analysis of UPI ID safety
 */
export async function validateUpiIdSafety(upiId: string) {
  return await analyzeUpiId(upiId);
}

/**
 * Analyze WhatsApp message for scams
 * @param message The WhatsApp message to analyze
 * @returns Analysis of message safety
 */
export async function analyzeWhatsAppMessage(message: string) {
  return await analyzeMessageForScams(message, 'WhatsApp');
}

// For backward compatibility
export default {
  chat: {
    completions: {
      create: async (params: any) => {
        try {
          const {
            messages,
            max_tokens = 1024,
            temperature = 0.7,
            response_format,
            ...otherParams
          } = params;

          // Call Groq API with compatible parameters
          const response = await groq.chat.completions.create({
            model: MODEL,
            messages,
            max_tokens,
            temperature,
            // Ignore incompatible params
          });

          // If JSON format was requested, try to ensure the response is valid JSON
          if (response_format?.type === 'json_object') {
            try {
              const content = response.choices[0].message.content || '{}';
              JSON.parse(content);
              // If it parses successfully, return the response as is
            } catch (e) {
              // If it's not valid JSON, wrap it in a simple object
              const originalContent = response.choices[0].message.content;
              response.choices[0].message.content = `{"content": ${JSON.stringify(originalContent)}}`;
            }
          }

          return response;
        } catch (error) {
          console.error('Error in Groq adapter:', error);
          throw error;
        }
      }
    }
  },
  // Audio transcription is not yet supported by Groq, provide a stub
  audio: {
    transcriptions: {
      create: async (params: any) => {
        console.warn("Audio transcription not available through Groq - using fallback method");
        return {
          text: "Audio transcription via Groq not implemented. Please use a dedicated transcription service."
        };
      }
    }
  }
};