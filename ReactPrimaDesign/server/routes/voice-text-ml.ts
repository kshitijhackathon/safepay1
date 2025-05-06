import { Express, Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { ParamsDictionary } from 'express-serve-static-core';
import { ParsedQs } from 'qs';
import { storage } from '../storage';
import { analyzeVoiceTranscript, analyzeVoiceAdvanced, transcribeAudio, analyzeMessageForScams } from '../services/groq';
import { ScamType } from '../../shared/schema';
import { spawn } from 'child_process';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

// Interface for ML voice analysis response
interface MLVoiceAnalysisResponse {
  is_scam: boolean;
  confidence: number;
  risk_score: number;
  scam_type: string | null;
  scam_indicators: string[] | null;
  processing_time_ms: number;
}

// Interface for ML text analysis response
interface MLTextAnalysisResponse {
  is_scam: boolean;
  confidence: number;
  risk_score: number;
  scam_type: string | null;
  scam_indicators: string[] | null;
  processing_time_ms: number;
}

// Service configuration
// Use 8082 as the default - this avoids conflicts with common server ports (3000, 5000, 8000, 8080)
const VOICE_TEXT_ML_SERVICE_PORT = process.env.ML_VOICE_TEXT_SERVICE_PORT || 8082;
const ML_SERVICE_URL = `http://localhost:${VOICE_TEXT_ML_SERVICE_PORT}`;

// Configure multer for audio file uploads
const storage_config = multer.diskStorage({
  destination: (req: any, file: any, cb: any) => {
    const uploadDir = path.join(process.cwd(), 'uploads');
    
    // Create directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    
    cb(null, uploadDir);
  },
  filename: (req: any, file: any, cb: any) => {
    // Create unique filename with timestamp
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    const ext = path.extname(file.originalname);
    cb(null, `audio-${uniqueSuffix}${ext}`);
  },
});

// Filter to accept only audio files
const fileFilter = (req: Request, file: any, cb: any) => {
  const allowedTypes = ['audio/webm', 'audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg'];
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Only audio files are allowed'));
  }
};

const upload = multer({
  storage: storage_config,
  fileFilter: fileFilter,
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB max size
  },
});

// Service management
let mlServiceProcess: any = null;

/**
 * Start the ML service if it's not already running
 */
function ensureMLServiceRunning(): Promise<boolean> {
  return new Promise(async (resolve) => {
    // Check if service is already running
    try {
      const response = await axios.get(`${ML_SERVICE_URL}/status`, { timeout: 1000 });
      if (response.status === 200) {
        console.log('ML voice/text service already running');
        return resolve(true);
      }
    } catch (error) {
      // Service not running, start it
      console.log('Starting ML voice/text service...');
    }

    // Kill any existing process
    if (mlServiceProcess) {
      try {
        mlServiceProcess.kill();
      } catch (e) {
        console.error('Error killing existing ML service process:', e);
      }
    }

    // Start the ML service
    mlServiceProcess = spawn('python3', ['voice_text_scam_service.py'], {
      detached: true,
      stdio: 'pipe',
    });

    // Log stdout
    mlServiceProcess.stdout.on('data', (data: Buffer) => {
      console.log(`ML Voice/Text Service: ${data.toString().trim()}`);
    });

    // Log stderr
    mlServiceProcess.stderr.on('data', (data: Buffer) => {
      console.error(`ML Voice/Text Service Error: ${data.toString().trim()}`);
    });

    // Handle process exit
    mlServiceProcess.on('exit', (code: number) => {
      console.log(`ML Voice/Text Service exited with code ${code}`);
      mlServiceProcess = null;
    });

    // Wait for service to start with retries
    let retries = 0;
    const maxRetries = 5;
    const retryInterval = 2000; // 2 seconds between retries
    
    const checkService = async () => {
      try {
        const response = await axios.get(`${ML_SERVICE_URL}/status`, { timeout: 2000 });
        if (response.status === 200) {
          console.log('ML voice/text service started successfully');
          resolve(true);
          return;
        } else {
          console.warn(`ML voice/text service returned status ${response.status}, retrying...`);
        }
      } catch (error: any) {
        console.warn(`Waiting for ML voice/text service to be ready... (attempt ${retries + 1}/${maxRetries})`);
        console.error(`Error: ${error.message || 'Unknown error'}`);
      }
      
      retries++;
      if (retries < maxRetries) {
        // Try again after interval
        setTimeout(checkService, retryInterval);
      } else {
        console.error(`ML voice/text service failed to start after ${maxRetries} attempts`);
        console.log('Application will continue without ML voice/text analysis capability');
        resolve(false);
      }
    };
    
    // Give initial time for process to start before first check
    setTimeout(checkService, 3000);
  });
}

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
  if (lowerType.includes('job')) return ScamType.JobScam;
  if (lowerType.includes('invest')) return ScamType.Investment;
  
  return ScamType.Unknown;
}

/**
 * Register enhanced voice and text check routes with ML support
 * @param app Express application
 */
export function registerEnhancedVoiceTextCheckRoutes(app: Express): void {
  // Start ML service
  ensureMLServiceRunning().then((success) => {
    console.log(`ML Voice/Text service initialization ${success ? 'successful' : 'failed'}`);
  });

  /**
   * Process voice for fraud detection with ML
   */
  app.post('/api/ml-process-voice', async (req: Request, res: Response) => {
    try {
      const { transcript, language = 'en-US', audio_features } = req.body;
      
      if (!transcript) {
        return res.status(400).json({
          status: 'error',
          message: 'Voice transcript is required'
        });
      }

      // Try to use ML service first for enhanced analysis
      try {
        const serviceResponse = await axios.post(`${ML_SERVICE_URL}/analyze-voice`, {
          transcript,
          audio_features
        }, { timeout: 5000 });
        
        const mlAnalysis: MLVoiceAnalysisResponse = serviceResponse.data;
        
        // Use ML analysis result
        const result = {
          status: 'success',
          is_scam: mlAnalysis.is_scam,
          confidence: mlAnalysis.confidence,
          risk_score: mlAnalysis.risk_score,
          scam_type: mapToScamType(mlAnalysis.scam_type || undefined),
          indicators: mlAnalysis.scam_indicators || [],
          analysis_method: 'machine_learning',
          processing_time_ms: mlAnalysis.processing_time_ms
        };
        
        // Save to chat if userId is provided
        if (req.body.userId) {
          try {
            await storage.saveChatMessage(parseInt(req.body.userId), {
              role: 'user',
              content: `Voice analysis: "${transcript}"`
            });
            
            await storage.saveChatMessage(parseInt(req.body.userId), {
              role: 'assistant',
              content: `ML Analysis: ${result.is_scam ? 'Potential scam detected' : 'No scam detected'} 
                      (Confidence: ${Math.round(result.confidence * 100)}%)
                      ${result.indicators.length > 0 ? 'Indicators: ' + result.indicators.join(', ') : ''}`
            });
          } catch (err) {
            console.error('Error saving voice check to chat:', err);
          }
        }
        
        return res.json(result);
      } catch (mlError) {
        console.error('ML service error, falling back to LLM analysis:', mlError);
        // Fall back to LLM-based analysis
      }

      // Fallback to Groq LLM analysis
      const llmAnalysis = await analyzeVoiceTranscript(transcript);
      
      const result = {
        status: 'success',
        is_scam: llmAnalysis.is_scam,
        confidence: llmAnalysis.confidence, 
        risk_score: llmAnalysis.risk_score || (llmAnalysis.confidence * 100),
        scam_type: mapToScamType(llmAnalysis.scam_type),
        indicators: llmAnalysis.scam_indicators || [],
        analysis_method: 'llm_fallback'
      };
      
      // Save to chat if userId is provided
      if (req.body.userId) {
        try {
          await storage.saveChatMessage(parseInt(req.body.userId), {
            role: 'user',
            content: `Voice check: "${transcript}"`
          });
          
          await storage.saveChatMessage(parseInt(req.body.userId), {
            role: 'assistant',
            content: `LLM Analysis: ${result.is_scam ? 'Potential scam detected' : 'No scam detected'} 
                    (Confidence: ${Math.round(result.confidence * 100)}%)
                    ${result.indicators.length > 0 ? 'Indicators: ' + result.indicators.join(', ') : ''}`
          });
        } catch (err) {
          console.error('Error saving voice check to chat:', err);
        }
      }
      
      res.json(result);
    } catch (error: any) {
      console.error('Error in ml-process-voice:', error);
      res.status(500).json({
        status: 'error',
        message: error.message || 'Internal server error'
      });
    }
  });

  /**
   * Process audio file for advanced ML analysis
   */
  app.post('/api/ml-process-audio', upload.single('audio'), async (req: any, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          status: 'error',
          message: 'Audio file is required'
        });
      }
      
      // Get the audio file path
      const audioPath = req.file.path;
      
      // Transcribe the audio using OpenAI's Whisper
      console.log('Transcribing audio...');
      const audioBuffer = fs.readFileSync(audioPath);
      const transcript = await transcribeAudio(audioBuffer);
      
      if (!transcript) {
        // Clean up file
        try { fs.unlinkSync(audioPath); } catch (e) {}
        
        return res.status(400).json({
          status: 'error',
          message: 'Failed to transcribe audio'
        });
      }
      
      // Detect language (simplified implementation)
      let detectedLanguage = 'en';
      try {
        if (transcript.match(/[ा-ू]/)) {
          detectedLanguage = 'hi'; // Hindi
        } else if (transcript.match(/[অ-ৰ]/)) {
          detectedLanguage = 'bn'; // Bengali
        }
      } catch (error) {
        console.error('Language detection failed:', error);
      }
      
      // First try ML analysis
      let mlAnalysis = null;
      try {
        const serviceResponse = await axios.post(`${ML_SERVICE_URL}/analyze-voice`, {
          transcript,
          audio_features: { language: detectedLanguage }
        }, { timeout: 5000 });
        
        mlAnalysis = serviceResponse.data;
      } catch (mlError) {
        console.error('ML analysis failed, falling back to LLM:', mlError);
      }
      
      // Perform advanced analysis using LLM as fallback
      const audioFeatures = { 
        language: detectedLanguage
      };
      
      let analysis;
      if (mlAnalysis) {
        // Use ML analysis
        analysis = {
          is_scam: mlAnalysis.is_scam,
          confidence: mlAnalysis.confidence,
          risk_score: mlAnalysis.risk_score,
          scam_type: mlAnalysis.scam_type,
          scam_indicators: mlAnalysis.scam_indicators || [],
          recommendation: mlAnalysis.is_scam 
            ? "This appears to contain scam elements. Exercise extreme caution."
            : "No obvious scam elements detected, but always verify directly with trusted sources.",
          analysis_method: 'machine_learning'
        };
      } else {
        // Fallback to LLM
        const llmAnalysis = await analyzeVoiceAdvanced(transcript, audioFeatures);
        
        analysis = {
          ...llmAnalysis,
          analysis_method: 'llm_fallback'
        };
      }
      
      // Log the analysis for scam detection
      const userId = req.body.userId ? parseInt(req.body.userId) : undefined;
      if (userId) {
        try {
          await storage.saveChatMessage(userId, {
            role: 'user',
            content: `Audio analysis: "${transcript}"`
          });
          
          const responseContent = `Analysis: ${analysis.is_scam ? 'SCAM DETECTED' : 'No scam detected'}
            Confidence: ${Math.round((analysis.confidence || 0) * 100)}%
            ${analysis.scam_indicators?.length ? 'Warning signs: ' + analysis.scam_indicators.join(', ') : ''}
            ${analysis.recommendation ? 'Recommendation: ' + analysis.recommendation : ''}`;
            
          await storage.saveChatMessage(userId, {
            role: 'assistant',
            content: responseContent
          });
        } catch (err) {
          console.error('Error saving audio analysis to chat:', err);
        }
      }
      
      // Clean up the uploaded file after processing
      try {
        fs.unlinkSync(audioPath);
      } catch (e) {
        console.error('Error removing temporary audio file:', e);
      }
      
      // Return the analysis results
      res.json({
        status: 'success',
        transcript,
        language: detectedLanguage,
        analysis
      });
      
    } catch (error: any) {
      console.error('Error processing audio:', error);
      res.status(500).json({
        status: 'error',
        message: error.message || 'Failed to process audio file'
      });
    }
  });

  /**
   * Analyze text message for scams using ML
   */
  app.post('/api/ml-analyze-text', async (req: Request, res: Response) => {
    try {
      const { text, message_type = 'SMS', context } = req.body;
      
      if (!text) {
        return res.status(400).json({
          status: 'error',
          message: 'Text message is required'
        });
      }

      // Try to use ML service first
      try {
        const serviceResponse = await axios.post(`${ML_SERVICE_URL}/analyze-text`, {
          text,
          message_type,
          context
        }, { timeout: 5000 });
        
        const mlAnalysis: MLTextAnalysisResponse = serviceResponse.data;
        
        // Use ML analysis result
        const result = {
          status: 'success',
          is_scam: mlAnalysis.is_scam,
          confidence: mlAnalysis.confidence,
          risk_score: mlAnalysis.risk_score,
          scam_type: mapToScamType(mlAnalysis.scam_type || undefined),
          scam_indicators: mlAnalysis.scam_indicators || [],
          analysis_method: 'machine_learning',
          processing_time_ms: mlAnalysis.processing_time_ms
        };
        
        // Save to chat if userId is provided
        if (req.body.userId) {
          try {
            await storage.saveChatMessage(parseInt(req.body.userId), {
              role: 'user',
              content: `Text analysis: "${text}"`
            });
            
            await storage.saveChatMessage(parseInt(req.body.userId), {
              role: 'assistant',
              content: `ML Analysis: ${result.is_scam ? 'Potential scam detected' : 'No scam detected'} 
                      (Confidence: ${Math.round(result.confidence * 100)}%)
                      ${result.scam_indicators.length > 0 ? 'Indicators: ' + result.scam_indicators.join(', ') : ''}`
            });
          } catch (err) {
            console.error('Error saving text analysis to chat:', err);
          }
        }
        
        return res.json(result);
      } catch (mlError) {
        console.error('ML text service error, falling back to LLM analysis:', mlError);
        // Fall back to LLM-based analysis
      }

      // Fallback to Groq LLM analysis
      const llmAnalysis = await analyzeMessageForScams(text, message_type);
      
      // Map the response to our standard format
      const result = {
        status: 'success',
        is_scam: llmAnalysis.is_scam,
        confidence: llmAnalysis.confidence,
        risk_score: llmAnalysis.confidence * 100,
        scam_type: mapToScamType(llmAnalysis.scam_type),
        scam_indicators: llmAnalysis.scam_indicators || [],
        unsafe_elements: llmAnalysis.unsafe_elements || [],
        recommendation: llmAnalysis.recommendation,
        analysis_method: 'llm_fallback'
      };
      
      // Save to chat if userId is provided
      if (req.body.userId) {
        try {
          await storage.saveChatMessage(parseInt(req.body.userId), {
            role: 'user',
            content: `Text analysis: "${text}"`
          });
          
          await storage.saveChatMessage(parseInt(req.body.userId), {
            role: 'assistant',
            content: `LLM Analysis: ${result.is_scam ? 'Potential scam detected' : 'No scam detected'} 
                    (Confidence: ${Math.round(result.confidence * 100)}%)
                    ${result.scam_indicators.length > 0 ? 'Indicators: ' + result.scam_indicators.join(', ') : ''}`
          });
        } catch (err) {
          console.error('Error saving text analysis to chat:', err);
        }
      }
      
      res.json(result);
    } catch (error: any) {
      console.error('Error in ml-analyze-text:', error);
      res.status(500).json({
        status: 'error',
        message: error.message || 'Internal server error'
      });
    }
  });

  /**
   * Batch analyze multiple text messages
   */
  app.post('/api/ml-batch-analyze-text', async (req: Request, res: Response) => {
    try {
      const { messages } = req.body;
      
      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        return res.status(400).json({
          status: 'error',
          message: 'Array of messages is required'
        });
      }

      // Try to use ML service first
      try {
        const serviceResponse = await axios.post(`${ML_SERVICE_URL}/batch-analyze-text`, {
          messages: messages.map(msg => ({
            text: msg.text,
            message_type: msg.message_type || 'SMS',
            context: msg.context || {}
          }))
        }, { timeout: 10000 });
        
        return res.json({
          status: 'success',
          results: serviceResponse.data.results,
          count: serviceResponse.data.count,
          analysis_method: 'machine_learning',
          processing_time_ms: serviceResponse.data.processing_time_ms
        });
      } catch (mlError) {
        console.error('ML batch service error, falling back to LLM analysis:', mlError);
        // Fall back to processing each message individually with LLM
      }

      // Process each message individually with LLM as fallback
      const results = await Promise.all(messages.map(async (msg) => {
        try {
          const llmAnalysis = await analyzeMessageForScams(msg.text, msg.message_type || 'SMS');
          
          return {
            text: msg.text,
            is_scam: llmAnalysis.is_scam,
            confidence: llmAnalysis.confidence,
            risk_score: llmAnalysis.confidence * 100,
            scam_type: llmAnalysis.scam_type,
            scam_indicators: llmAnalysis.scam_indicators || []
          };
        } catch (error) {
          console.error('Error analyzing message:', error);
          return {
            text: msg.text,
            is_scam: false,
            confidence: 0.5,
            risk_score: 50,
            scam_type: 'Error',
            scam_indicators: ['Processing error']
          };
        }
      }));
      
      res.json({
        status: 'success',
        results,
        count: results.length,
        analysis_method: 'llm_fallback'
      });
    } catch (error: any) {
      console.error('Error in ml-batch-analyze-text:', error);
      res.status(500).json({
        status: 'error',
        message: error.message || 'Internal server error'
      });
    }
  });
}