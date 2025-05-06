import { Express, Request } from 'express';
import { storage } from '../storage';
import { analyzeVoiceTranscript, analyzeVoiceAdvanced, transcribeAudio } from '../services/groq';
import { ScamType } from '../../shared/schema';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';

// Since langdetect is challenging to type, create a simple mock for now
// In a production app, we would fully implement the langdetect functionality
const mockLanguageDetect = (text: string): { lang: string; prob: number }[] => {
  // Very simple language detection mock
  const hindiPattern = /[ा-ू]/;
  const bengaliPattern = /[অ-ৰ]/;
  
  if (hindiPattern.test(text)) {
    return [{ lang: 'hi', prob: 0.85 }];
  } else if (bengaliPattern.test(text)) {
    return [{ lang: 'bn', prob: 0.82 }];
  }
  
  return [{ lang: 'en', prob: 0.95 }];
};

// Mock the langdetect library
const detect = mockLanguageDetect;

// Setup multer for audio file uploads
const memStorage = multer.memoryStorage();
const upload = multer({
  storage: memStorage,
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB limit for audio files
  },
  fileFilter: (_req, file, cb) => {
    // Accept only audio files
    if (file.mimetype.startsWith('audio/')) {
      cb(null, true);
    } else {
      cb(new Error('Only audio files are allowed'));
    }
  }
});

/**
 * Helper function to map a string scam type to the ScamType enum
 * @param scamTypeStr The string representation of scam type
 * @returns The corresponding ScamType enum value
 */
function mapToScamType(scamTypeStr?: string): ScamType {
  if (!scamTypeStr) return ScamType.Unknown;
  
  const lowerType = scamTypeStr.toLowerCase();

  if (lowerType.includes('bank')) return ScamType.Banking;
  if (lowerType.includes('lottery')) return ScamType.Lottery;
  if (lowerType.includes('kyc') || lowerType.includes('verification')) return ScamType.KYC;
  if (lowerType.includes('refund')) return ScamType.Refund;
  if (lowerType.includes('phish')) return ScamType.Phishing;
  if (lowerType.includes('reward') || lowerType.includes('prize')) return ScamType.Reward;
  
  return ScamType.Unknown;
}

// Define the type for analyzeVoiceTranscript response
interface VoiceAnalysisResult {
  is_scam: boolean;
  confidence: number;
  scam_type?: string;
  scam_indicators?: string[];
}

// Define the type for our analysis object
interface ScamAnalysis {
  is_scam: boolean;
  confidence: number;
  scam_type: ScamType;
  indicators: string[]; // Using string[] for indicators
}

/**
 * Register voice check related routes
 * @param app Express application
 */
export function registerVoiceCheckRoutes(app: Express): void {
  /**
   * Process audio file for advanced analysis
   */
  app.post('/api/process-audio', upload.single('audio'), async (req: Request, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          status: 'error',
          message: 'Audio file is required'
        });
      }
      
      // Get the audio buffer from the uploaded file
      const audioBuffer = req.file.buffer;
      
      // Transcribe the audio using OpenAI's Whisper
      console.log('Transcribing audio...');
      const transcript = await transcribeAudio(audioBuffer);
      
      if (!transcript) {
        return res.status(400).json({
          status: 'error',
          message: 'Failed to transcribe audio'
        });
      }
      
      // Detect language (mock implementation since we didn't install langdetect)
      let detectedLanguage = 'en';
      try {
        // In a real implementation, we would use a proper language detection library
        // detectedLanguage = detect(transcript)[0].lang;
        // This is just a mock implementation
        if (transcript.match(/[ा-ू]/)) {
          detectedLanguage = 'hi'; // Hindi
        } else if (transcript.match(/[অ-ৰ]/)) {
          detectedLanguage = 'bn'; // Bengali
        }
      } catch (error) {
        console.error('Language detection failed:', error);
      }
      
      // Simulate noise level detection (in a real implementation, we would analyze the audio)
      const noiseLevel = 0.2; // Mock value between 0-1
      
      // Perform advanced analysis using Groq
      console.log('Performing advanced analysis...');
      const audioFeatures = { 
        language: detectedLanguage, 
        noise_level: noiseLevel 
      };
      const advancedAnalysis = await analyzeVoiceAdvanced(transcript, audioFeatures);
      
      // Map scam type to our enum
      const scamType = mapToScamType(advancedAnalysis.scam_type);
      
      // Log the analysis for scam detection
      const userId = req.body.userId ? parseInt(req.body.userId) : undefined;
      if (userId) {
        try {
          await storage.saveChatMessage(userId, {
            role: 'user',
            content: `Audio analysis: "${transcript}"`
          });
          
          const responseContent = `Analysis: ${advancedAnalysis.is_scam ? 'SCAM DETECTED' : 'No scam detected'}
            Confidence: ${Math.round((advancedAnalysis.confidence || 0) * 100)}%
            ${advancedAnalysis.scam_indicators?.length ? 'Warning signs: ' + advancedAnalysis.scam_indicators.join(', ') : ''}
            ${advancedAnalysis.recommendation ? 'Recommendation: ' + advancedAnalysis.recommendation : ''}`;
            
          await storage.saveChatMessage(userId, {
            role: 'assistant',
            content: responseContent
          });
        } catch (err) {
          console.error('Error saving audio analysis to chat:', err);
        }
      }
      
      // Return the analysis results
      res.json({
        status: 'success',
        transcript,
        language: detectedLanguage,
        noise_level: noiseLevel,
        analysis: {
          ...advancedAnalysis,
          scam_type: scamType
        }
      });
      
    } catch (error) {
      console.error('Error processing audio:', error);
      res.status(500).json({
        status: 'error',
        message: 'Failed to process audio file'
      });
    }
  });
  
  /**
   * Process voice command for fraud detection
   */
  app.post('/api/process-voice', async (req: Request, res) => {
    try {
      const { command, language = 'en-US', session } = req.body;
      
      if (!command) {
        return res.status(400).json({
          status: 'error',
          message: 'Voice command is required'
        });
      }

      // Basic action detection
      const action = determineAction(command);
      
      // Try to use OpenAI for enhanced analysis if available
      let analysis: ScamAnalysis = { 
        is_scam: false, 
        confidence: 0.5, 
        scam_type: ScamType.Unknown,
        indicators: []
      };
      
      try {
        const aiAnalysis = await analyzeVoiceTranscript(command);
        if (aiAnalysis) {
          analysis = {
            is_scam: aiAnalysis.is_scam,
            confidence: aiAnalysis.confidence,
            scam_type: mapToScamType(aiAnalysis.scam_type),
            indicators: aiAnalysis.scam_indicators || []
          };
        }
      } catch (error) {
        console.error('Error with AI voice analysis:', error);
        // Fall back to basic detection if AI fails
        analysis = performBasicScamDetection(command) as ScamAnalysis;
      }
      
      // We'll save the voice check if we have a userId in the request body
      if (req.body.userId) {
        try {
          await storage.saveChatMessage(parseInt(req.body.userId), {
            role: 'user',
            content: `Voice check: "${command}"`
          });
          
          await storage.saveChatMessage(parseInt(req.body.userId), {
            role: 'assistant',
            content: `Analysis: ${analysis.is_scam ? 'Potential scam detected' : 'No scam detected'} 
                      (Confidence: ${Math.round(analysis.confidence * 100)}%)
                      ${analysis.indicators.length > 0 ? 'Indicators: ' + analysis.indicators.join(', ') : ''}`
          });
        } catch (err) {
          console.error('Error saving voice check to chat:', err);
        }
      }
      
      res.json({
        status: 'success',
        command,
        action,
        risk_score: analysis.confidence,
        is_scam: analysis.is_scam,
        scam_type: analysis.scam_type,
        indicators: analysis.indicators
      });
    } catch (error) {
      console.error('Error in process-voice:', error);
      res.status(500).json({
        status: 'error',
        message: 'Internal server error'
      });
    }
  });
}

/**
 * Determine action from voice command
 * @param command Voice command text
 * @returns Inferred action
 */
function determineAction(command: string): string {
  const lowerCommand = command.toLowerCase();
  
  if (lowerCommand.includes('send') || lowerCommand.includes('pay') || lowerCommand.includes('transfer')) {
    return 'payment';
  }
  
  if (lowerCommand.includes('check') || lowerCommand.includes('verify')) {
    return 'verification';
  }
  
  if (lowerCommand.includes('history') || lowerCommand.includes('transactions')) {
    return 'history';
  }
  
  if (lowerCommand.includes('help') || lowerCommand.includes('support')) {
    return 'help';
  }
  
  return 'unknown';
}

/**
 * Perform basic scam detection on a voice command
 * @param command Voice command text
 * @returns Basic analysis result
 */
function performBasicScamDetection(command: string): { 
  is_scam: boolean; 
  confidence: number; 
  scam_type: ScamType;
  indicators: string[];
} {
  const lowerCommand = command.toLowerCase();
  const indicators: string[] = [];
  
  // Common scam phrases
  const scamPhrases = [
    { phrase: 'verify your account', indicator: 'Account verification request' },
    { phrase: 'suspicious activity', indicator: 'Mentioning suspicious activity' },
    { phrase: 'password', indicator: 'Password request' },
    { phrase: 'pin', indicator: 'PIN request' },
    { phrase: 'otp', indicator: 'OTP request' },
    { phrase: 'urgent', indicator: 'Creating urgency' },
    { phrase: 'immediately', indicator: 'Creating urgency' },
    { phrase: 'your account will be', indicator: 'Account threat' },
    { phrase: 'bank', indicator: 'Bank reference' },
    { phrase: 'offer', indicator: 'Promotion or offer' },
    { phrase: 'prize', indicator: 'Prize mention' }
  ];
  
  // Check for scam indicators
  scamPhrases.forEach(item => {
    if (lowerCommand.includes(item.phrase)) {
      indicators.push(item.indicator);
    }
  });
  
  // Determine scam type and confidence
  let scamType = ScamType.Unknown;
  
  if (indicators.length >= 3) {
    if (lowerCommand.includes('bank') || lowerCommand.includes('account')) {
      scamType = ScamType.Banking;
    } else if (lowerCommand.includes('offer') || lowerCommand.includes('prize')) {
      scamType = ScamType.Reward;
    } else {
      scamType = ScamType.Phishing;
    }
  }
  
  return {
    is_scam: indicators.length >= 3,
    confidence: Math.min(0.4 + (indicators.length * 0.1), 0.95),
    scam_type: scamType,
    indicators
  };
}