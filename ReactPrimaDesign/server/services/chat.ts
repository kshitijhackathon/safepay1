import { analyzeChatSentiment, generateSecurityTip } from './groq';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const systemPrompt: ChatMessage = {
  role: 'system',
  content: `You are an expert AI assistant for a UPI payment security app in India. Help users with:
    1. Identifying and avoiding UPI payment scams
    2. Answering questions about secure payment practices
    3. Guidance on what to do if they suspect fraud
    4. Information about UPI payment system features and security
    
    Keep responses concise, practical, and relevant to Indian UPI payment users.
    If a user appears to be in an emergency situation involving fraud, escalate the issue.
    Never provide advice that could be used for malicious purposes.
    
    For all financial fraud issues, include the contact for India's National Cybercrime Reporting Portal: https://cybercrime.gov.in/ 
    or the dedicated helpline 1930.`
};

/**
 * Safely parse JSON with fallback for null response
 */
function safeJsonParse(jsonString: string | null | undefined): any {
  if (!jsonString) return {};
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    console.error('Error parsing JSON:', error);
    return {};
  }
}

/**
 * Process a user message and generate an assistant response using OpenAI
 * @param messages Previous chat history
 * @param userMessage New message from the user
 * @returns Assistant's response
 */
export async function getChatResponse(messages: ChatMessage[], userMessage: string): Promise<string> {
  // Check for emergency/distress signals in the message
  const sentiment = await analyzeChatSentiment(userMessage);
  
  // Create the updated message history including the user's new message
  const chatHistory: ChatMessage[] = [
    systemPrompt,
    ...messages,
    { role: 'user', content: userMessage }
  ];
  
  // If high distress is detected, modify the system prompt to acknowledge the emergency
  if (sentiment.urgency === 'high' || (sentiment.distress_indicators && sentiment.distress_indicators.length > 2)) {
    chatHistory.unshift({
      role: 'system',
      content: 'The user appears to be in distress. Provide immediate, actionable guidance. Include emergency contact information for cybercrime reporting in India.'
    });
  }
  
  try {
    // Use our Groq adapter that provides OpenAI-compatible interface
    const groqAdapter = (await import('./openai-adapter')).default;
    
    const response = await groqAdapter.chat.completions.create({
      messages: chatHistory as any[], // Type casting for API compatibility
      max_tokens: 500,
      temperature: 0.7
    });
    
    return response.choices[0].message.content || "I'm sorry, I couldn't generate a response.";
  } catch (error) {
    console.error('Error getting chat response:', error);
    return "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.";
  }
}

/**
 * Detect the emergency level of a user message
 * @param message User message
 * @returns Emergency level (high, medium, low)
 */
export async function detectEmergencyLevel(message: string): Promise<'high' | 'medium' | 'low'> {
  try {
    const sentiment = await analyzeChatSentiment(message);
    
    if (sentiment.urgency === 'high' || (sentiment.distress_indicators && sentiment.distress_indicators.length > 2)) {
      return 'high';
    } else if (sentiment.urgency === 'medium' || (sentiment.distress_indicators && sentiment.distress_indicators.length > 0)) {
      return 'medium';
    } else {
      return 'low';
    }
  } catch (error) {
    console.error('Error detecting emergency level:', error);
    return 'low'; // Default to low if analysis fails
  }
}

/**
 * Generate quick reply suggestions based on the conversation context
 * @param messages Chat history
 * @returns List of suggested quick replies
 */
export async function generateQuickReplies(messages: ChatMessage[]): Promise<string[]> {
  if (messages.length === 0) {
    return [
      "How do I verify if a UPI ID is safe?",
      "I think I sent money to a scammer",
      "How do I report a UPI scam?",
      "What are common UPI scams I should know about?"
    ];
  }
  
  try {
    // Use our Groq adapter
    const groqAdapter = (await import('./openai-adapter')).default;
    
    const response = await groqAdapter.chat.completions.create({
      messages: [
        {
          role: "system",
          content: "Based on the conversation history, generate 3-4 short, helpful quick reply options that the user might want to send next. Each reply should be under 60 characters. Provide only the text of each reply with no additional explanations or numbering."
        },
        ...messages
      ],
      temperature: 0.3, // Lower temperature for more consistent output
      max_tokens: 150
    });
    
    const result = safeJsonParse(response.choices[0].message.content);
    return result.quick_replies || result.replies || [];
  } catch (error) {
    console.error('Error generating quick replies:', error);
    
    // Fallback suggestions
    return [
      "Tell me more about this",
      "How can I stay safe?",
      "What should I do next?",
      "Thank you for the help"
    ];
  }
}