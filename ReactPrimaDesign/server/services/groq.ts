import { Groq } from 'groq-sdk';

// Initialize the Groq client with the API key
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY || '',
});

// Model parameters
const MODEL = 'llama-3.3-70b-versatile';
const MAX_TOKENS = 1024;

// Helper function to safely parse JSON
function safeJsonParse(jsonString: string | null | undefined, defaultValue: any = {}) {
  try {
    if (!jsonString) return defaultValue;
    return JSON.parse(jsonString);
  } catch (error) {
    console.error('Error parsing JSON:', error);
    return defaultValue;
  }
}

/**
 * Analyze a transaction for potential fraud using Groq LLM
 * @param transactionDetails Transaction details to analyze
 * @returns Analysis results with risk score and reasoning
 */
export async function analyzeTransaction(transactionDetails: any) {
  try {
    const systemPrompt = `
      You are a UPI payment security expert specializing in detecting fraudulent transactions in India.
      Analyze the given transaction and determine if it's potentially fraudulent.
      Consider:
      1. Unusual recipient/sender patterns
      2. Irregular transaction amounts or times
      3. Known scam patterns in the Indian UPI ecosystem
      4. Transaction description and context
      
      Return a structured JSON with the following fields:
      {
        "risk_score": <number between 0-100>,
        "risk_level": <"low"|"medium"|"high">,
        "reasoning": <explanation for assessment>,
        "scam_indicators": <array of identified scam patterns>,
        "recommendation": <action recommendation for user>
      }
    `;

    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: JSON.stringify(transactionDetails) }
      ],
      max_tokens: MAX_TOKENS,
      temperature: 0.2, // Lower temperature for more consistent analysis
    });

    // Parse the response as JSON
    try {
      const content = response.choices[0].message.content || '{}';
      const result = JSON.parse(content);
      return result;
    } catch (e) {
      console.error('Error parsing Groq response:', e);
      return {
        risk_score: 50,
        risk_level: 'medium',
        reasoning: 'Error processing analysis. Treating as medium risk as a precaution.',
        scam_indicators: ['Unable to process full analysis'],
        recommendation: 'Proceed with caution and verify recipient details.'
      };
    }
  } catch (error) {
    console.error('Error using Groq API for transaction analysis:', error);
    // Return a fallback response
    return {
      risk_score: 50,
      risk_level: 'medium',
      reasoning: 'Unable to complete fraud check. Treating as medium risk as a precaution.',
      scam_indicators: ['Service interruption'],
      recommendation: 'Verify recipient details and consider trying again later.'
    };
  }
}

/**
 * Analyze a UPI ID for potential typosquatting or suspicious patterns
 * @param upiId UPI ID to analyze
 * @param context Additional context (transaction history, etc.)
 * @returns Analysis of UPI ID
 */
export async function analyzeUpiId(upiId: string, context: any = {}) {
  try {
    const systemPrompt = `
      You are an expert in detecting fraudulent UPI IDs in India.
      Analyze the provided UPI ID and determine if it might be:
      1. A typosquatting attempt of a legitimate service
      2. Mimicking a well-known bank or payment service
      3. Using suspicious patterns or random characters
      4. Known to be associated with scams based on context
      
      Return a structured JSON with the following fields:
      {
        "is_suspicious": <boolean>,
        "risk_score": <number between 0-100>,
        "suspicious_patterns": <array of identified suspicious patterns>,
        "possible_legitimate_alternative": <string or null>,
        "recommendation": <string>
      }
    `;

    const userContent = JSON.stringify({
      upi_id: upiId,
      context: context
    });

    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userContent }
      ],
      max_tokens: MAX_TOKENS,
      temperature: 0.1, // Very low temperature for consistency
    });

    try {
      const content = response.choices[0].message.content || '{}';
      const result = JSON.parse(content);
      return result;
    } catch (e) {
      console.error('Error parsing Groq response for UPI ID analysis:', e);
      return {
        is_suspicious: false,
        risk_score: 40,
        suspicious_patterns: ['Unable to complete full analysis'],
        possible_legitimate_alternative: null,
        recommendation: 'Verify UPI ID with the intended recipient.'
      };
    }
  } catch (error) {
    console.error('Error using Groq API for UPI ID analysis:', error);
    return {
      is_suspicious: false,
      risk_score: 40,
      suspicious_patterns: ['Service interruption'],
      possible_legitimate_alternative: null,
      recommendation: 'Verify UPI ID with the intended recipient before proceeding.'
    };
  }
}

/**
 * Analyze message content for scam indicators
 * @param messageContent Content of the message to analyze
 * @param messageType Type of message (SMS, WhatsApp, etc.)
 * @returns Analysis of message for scam indicators
 */
export async function analyzeMessageForScams(messageContent: string, messageType: string = 'SMS') {
  try {
    const systemPrompt = `
      You are an expert in detecting scam messages in the Indian context, particularly UPI payment scams.
      Analyze the provided message and determine if it contains indicators of a scam.
      Consider:
      1. Urgency or pressure tactics
      2. Requests for personal information
      3. Suspicious URLs or UPI IDs
      4. Impersonation of banks, government, or payment services
      5. Grammatical errors or inconsistencies
      6. Known scam patterns in India
      
      Return a structured JSON with the following fields:
      {
        "is_scam": <boolean>,
        "confidence": <number between 0-100>,
        "scam_type": <string or null>,
        "scam_indicators": <array of identified scam elements>,
        "unsafe_elements": <array of unsafe elements like URLs or UPI IDs>,
        "recommendation": <string advice for user>
      }
    `;

    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Message (${messageType}): ${messageContent}` }
      ],
      max_tokens: MAX_TOKENS,
      temperature: 0.2,
    });

    try {
      const content = response.choices[0].message.content || '{}';
      const result = JSON.parse(content);
      return result;
    } catch (e) {
      console.error('Error parsing Groq response for message analysis:', e);
      return {
        is_scam: false,
        confidence: 30,
        scam_type: null,
        scam_indicators: ['Unable to complete full analysis'],
        unsafe_elements: [],
        recommendation: 'Be cautious and avoid sharing sensitive information.'
      };
    }
  } catch (error) {
    console.error('Error using Groq API for message analysis:', error);
    return {
      is_scam: false,
      confidence: 30,
      scam_type: null,
      scam_indicators: ['Service interruption'],
      unsafe_elements: [],
      recommendation: 'Be cautious and avoid sharing sensitive information.'
    };
  }
}

/**
 * Analyze voice transcript for scam indicators
 * @param transcript Transcript of voice content
 * @returns Analysis of voice content for scam indicators
 */
export async function analyzeVoiceTranscript(transcript: string) {
  try {
    const systemPrompt = `
      You are an expert in detecting voice scams in India, particularly related to UPI payments.
      Analyze the provided voice transcript and determine if it contains indicators of a scam.
      Consider:
      1. Urgency or pressure tactics
      2. Requests for personal information or OTPs
      3. Impersonation of bank officials, government, or payment services
      4. Threats or warnings of account closure
      5. Offers that seem too good to be true
      6. Instructions to download unknown apps or visit suspicious websites
      
      Return a structured JSON with the following fields:
      {
        "is_scam": <boolean>,
        "risk_score": <number between 0-100>,
        "confidence": <number between 0-100>,
        "scam_type": <string or null>,
        "scam_indicators": <array of identified scam elements>,
        "key_phrases": <array of suspicious phrases>,
        "recommendation": <string advice for user>
      }
    `;

    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Voice Transcript: ${transcript}` }
      ],
      max_tokens: MAX_TOKENS,
      temperature: 0.3,
    });

    try {
      const content = response.choices[0].message.content || '{}';
      const result = JSON.parse(content);
      return result;
    } catch (e) {
      console.error('Error parsing Groq response for voice analysis:', e);
      return {
        is_scam: false,
        risk_score: 50,
        confidence: 40,
        scam_type: null,
        scam_indicators: ['Unable to complete full analysis'],
        key_phrases: [],
        recommendation: 'Be cautious about any requests for personal information or money transfers.'
      };
    }
  } catch (error) {
    console.error('Error using Groq API for voice analysis:', error);
    return {
      is_scam: false,
      risk_score: 50,
      confidence: 40,
      scam_type: null,
      scam_indicators: ['Service interruption'],
      key_phrases: [],
      recommendation: 'Be cautious about any requests for personal information or money transfers.'
    };
  }
}

/**
 * Advanced voice analysis with multiple detection features
 * @param transcript Voice transcript to analyze
 * @param audioFeatures Optional audio features extracted from voice
 * @returns Comprehensive analysis with multiple detection approaches
 */
export async function analyzeVoiceAdvanced(transcript: string, audioFeatures: any = {}) {
  try {
    const systemPrompt = `
      You are an expert in detecting voice-based UPI payment scams in India.
      Analyze the provided voice transcript and any audio features to identify scam patterns.
      
      Consider:
      1. Linguistic patterns common in Indian scammers
      2. Social engineering tactics specific to UPI/banking
      3. Urgency markers and pressure tactics
      4. Request patterns for sensitive information
      5. Cultural context and regional scam variations
      
      Return a structured JSON with the following fields:
      {
        "is_scam": <boolean>,
        "risk_score": <number between 0-100>,
        "confidence": <number between 0-100>,
        "scam_type": <string or null>,
        "scam_indicators": <array of concerning elements>,
        "recommendation": <detailed advice for user>
      }
    `;

    const userContent = JSON.stringify({
      transcript: transcript,
      audio_features: audioFeatures
    });

    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userContent }
      ],
      max_tokens: MAX_TOKENS,
      temperature: 0.3,
    });

    try {
      const content = response.choices[0].message.content || '{}';
      const result = JSON.parse(content);
      return result;
    } catch (e) {
      console.error('Error parsing Groq response for advanced voice analysis:', e);
      return {
        is_scam: false,
        risk_score: 50,
        confidence: 40,
        scam_type: null,
        scam_indicators: ['Unable to complete full analysis'],
        recommendation: 'Exercise caution with this communication.'
      };
    }
  } catch (error) {
    console.error('Error using Groq API for advanced voice analysis:', error);
    return {
      is_scam: false,
      risk_score: 50,
      confidence: 40,
      scam_type: null,
      scam_indicators: ['Service interruption'],
      recommendation: 'Exercise caution with this communication.'
    };
  }
}

/**
 * Analyze chat message sentiment and detect distress/urgency
 * @param message User message to analyze
 * @returns Analysis with urgency level and distress indicators
 */
export async function analyzeChatSentiment(message: string): Promise<{
  urgency: 'low' | 'medium' | 'high';
  emotional_state: string;
  distress_indicators?: string[];
}> {
  try {
    const systemPrompt = `
      You are an AI assistant specializing in detecting distress and urgency in messages related to
      financial fraud, particularly in the context of UPI payments in India.
      
      Analyze the given message and determine:
      1. How urgent the situation appears to be
      2. The emotional state of the user
      3. Any indicators of distress or fraud victimization
      
      Return a structured JSON with these fields:
      {
        "urgency": "low" | "medium" | "high",
        "emotional_state": <brief description of emotional state>,
        "distress_indicators": <array of phrases indicating distress or fraud>
      }
    `;

    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: message }
      ],
      max_tokens: 256,
      temperature: 0.1,
    });

    try {
      const content = response.choices[0].message.content || '{}';
      const result = JSON.parse(content);
      return result;
    } catch (e) {
      console.error('Error parsing Groq response for sentiment analysis:', e);
      return {
        urgency: 'low',
        emotional_state: 'uncertain',
        distress_indicators: []
      };
    }
  } catch (error) {
    console.error('Error using Groq API for sentiment analysis:', error);
    // Return a safe default
    return {
      urgency: 'low',
      emotional_state: 'uncertain',
      distress_indicators: []
    };
  }
}

/**
 * Generate a security tip related to UPI payments
 * @param context Optional context to tailor the tip
 * @returns Security tip text
 */
export async function generateSecurityTip(context: string = 'general'): Promise<string> {
  try {
    const systemPrompt = `
      You are an expert in UPI payment security in India. Generate a short, helpful security tip
      related to safe UPI usage. The tip should be practical, concise (max 200 characters),
      and relevant to everyday users in India.
      
      Focus on: Authentication safety, transaction verification, or scam awareness.
      
      Generate only the tip text without any prefixes or formatting.
    `;

    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Generate a UPI security tip related to: ${context}` }
      ],
      max_tokens: 100,
      temperature: 0.7,
    });

    const tip = response.choices[0].message.content?.trim() || 
      "Always verify the UPI ID before making a payment, and never share your UPI PIN or OTP with anyone, even if they claim to be from your bank.";
    
    return tip;
  } catch (error) {
    console.error('Error generating security tip:', error);
    // Return a fallback tip
    return "Always verify the UPI ID before making a payment, and never share your UPI PIN or OTP with anyone, even if they claim to be from your bank.";
  }
}

/**
 * Generate security recommendations based on user behavior
 * @param userBehavior User behavior data
 * @returns Personalized security recommendations
 */
export async function generateSecurityRecommendations(userBehavior: any) {
  try {
    const systemPrompt = `
      You are a UPI payment security advisor specializing in the Indian financial ecosystem.
      Based on the provided user behavior, generate personalized security recommendations.
      Consider:
      1. Transaction patterns
      2. Common vulnerabilities
      3. Regional scam trends
      4. Best practices for UPI security
      
      Return a structured JSON with the following fields:
      {
        "risk_areas": <array of identified risk areas>,
        "recommendations": <array of specific, actionable recommendations>,
        "priority_actions": <array of high-priority security actions>,
        "education_topics": <array of topics the user should learn about>
      }
    `;

    const response = await groq.chat.completions.create({
      model: MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: JSON.stringify(userBehavior) }
      ],
      max_tokens: MAX_TOKENS,
      temperature: 0.4,
    });

    try {
      const content = response.choices[0].message.content || '{}';
      const result = JSON.parse(content);
      return result;
    } catch (e) {
      console.error('Error parsing Groq response for security recommendations:', e);
      return {
        risk_areas: ['General UPI security'],
        recommendations: [
          'Verify recipient details before sending money',
          'Never share OTPs or passwords',
          'Keep your UPI app updated'
        ],
        priority_actions: ['Enable additional security features in your UPI app'],
        education_topics: ['Common UPI scams in India']
      };
    }
  } catch (error) {
    console.error('Error using Groq API for security recommendations:', error);
    return {
      risk_areas: ['General UPI security'],
      recommendations: [
        'Verify recipient details before sending money',
        'Never share OTPs or passwords',
        'Keep your UPI app updated'
      ],
      priority_actions: ['Enable additional security features in your UPI app'],
      education_topics: ['Common UPI scams in India']
    };
  }
}

/**
 * Transcribe audio content to text (fallback implementation)
 * @param audioBuffer Audio buffer to transcribe
 * @returns Transcribed text
 */
export async function transcribeAudio(audioBuffer: Buffer): Promise<string> {
  try {
    // Since Groq doesn't have direct audio transcription, we use OpenAI Whisper API for this specific task
    console.log("Using OpenAI Whisper API for audio transcription");
    
    // Create a new OpenAI instance for audio transcription only
    const OpenAI = require('openai').default;
    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
    
    // Check if OpenAI API key is available
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OpenAI API key not configured for audio transcription');
    }
    
    // Create a temporary file
    const fs = require('fs');
    const path = require('path');
    const tmpFilePath = path.join(__dirname, '../../tmp', `audio-${Date.now()}.webm`);
    
    // Ensure tmp directory exists
    if (!fs.existsSync(path.join(__dirname, '../../tmp'))) {
      fs.mkdirSync(path.join(__dirname, '../../tmp'), { recursive: true });
    }
    
    // Write the buffer to a temporary file
    fs.writeFileSync(tmpFilePath, audioBuffer);
    
    // Use OpenAI's Whisper model for speech-to-text
    const response = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tmpFilePath),
      model: "whisper-1",
      language: "en"
    });
    
    // Clean up the temporary file
    fs.unlinkSync(tmpFilePath);
    
    // Return the transcribed text
    return response.text;
    
  } catch (error) {
    console.error('Error in audio transcription:', error);
    return "Error transcribing audio content. Please try again later.";
  }
}