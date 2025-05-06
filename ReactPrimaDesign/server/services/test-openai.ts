import { Groq } from "groq-sdk";

// Function kept as testOpenAI for backward compatibility, but now uses Groq
async function testOpenAI() {
  try {
    const groq = new Groq({
      apiKey: process.env.GROQ_API_KEY || '',
    });
    
    const MODEL = 'llama-3.3-70b-versatile';
    
    console.log("Testing Groq connection with llama-3.3-70b-versatile model...");
    
    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        { role: "system", content: "You are a helpful assistant. Please respond with valid JSON." },
        { role: "user", content: "Generate a JSON object with 3 recent UPI scam alerts in India with title, description, and risk_level fields." }
      ],
      temperature: 0.3,
      max_tokens: 1024
    });
    
    console.log("Groq Response:", response.choices[0].message.content);
    return response.choices[0].message.content;
  } catch (error) {
    console.error("Groq Test Error:", error);
    throw error;
  }
}

export { testOpenAI };