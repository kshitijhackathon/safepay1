// This file provides an adapter layer to make Groq API compatible with the OpenAI API interface
// So that we can easily migrate from OpenAI to Groq without changing all the client code

import { Groq } from 'groq-sdk';

// Initialize Groq client
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY
});

// Model to use
const MODEL = 'llama-3.3-70b-versatile';

// Create a class that mimics the OpenAI interface
class GroqAdapter {
  chat = {
    completions: {
      create: async (params: any) => {
        try {
          const {
            messages,
            max_tokens = 1024,
            temperature = 0.7,
            response_format,
            // Ignore OpenAI-specific params that don't exist in Groq
            ...otherParams
          } = params;

          // Call Groq API with compatible parameters
          const response = await groq.chat.completions.create({
            model: MODEL,
            messages,
            max_tokens,
            temperature,
            // Ignore other incompatible params
          });

          // If JSON format was requested, try to ensure the response is valid JSON
          if (response_format?.type === 'json_object') {
            // Check if the response contains valid JSON
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
  }
}

// Export an instance of the adapter
export default new GroqAdapter();