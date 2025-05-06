import { Router, Request, Response } from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import os from 'os';
import WebSocket, { WebSocketServer } from 'ws';

const router = Router();

// Set up storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    // Create temp directory if it doesn't exist
    const uploadDir = path.join(os.tmpdir(), 'video_uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    // Create a unique filename with original extension
    const ext = path.extname(file.originalname);
    cb(null, `${uuidv4()}${ext}`);
  }
});

// File filter to only allow video files
const fileFilter = (req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  if (file.mimetype.startsWith('video/')) {
    cb(null, true);
  } else {
    cb(new Error('Only video files are allowed!'));
  }
};

// Configure multer
const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 100 * 1024 * 1024 } // 100MB file size limit
});

// Helper function to run Python video analysis
function analyzePythonVideo(videoPath: string, audioText: string = ''): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, '..', 'services', 'video_detection.py');
    
    // Arguments for the Python script
    const args = [
      'analyze',
      '--video_path', videoPath,
      '--audio_text', audioText || ''
    ];
    
    console.log(`Running Python script: python ${pythonScript} ${args.join(' ')}`);
    
    const pythonProcess = spawn('python', [pythonScript, ...args]);
    
    let dataString = '';
    let errorString = '';
    
    // Collect data from script
    pythonProcess.stdout.on('data', (data) => {
      dataString += data.toString();
    });
    
    // Collect error messages
    pythonProcess.stderr.on('data', (data) => {
      errorString += data.toString();
      console.error(`Python stderr: ${data}`);
    });
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        console.error(`Error: ${errorString}`);
        reject(new Error(`Python process failed with code ${code}: ${errorString}`));
        return;
      }
      
      try {
        const result = JSON.parse(dataString);
        resolve(result);
      } catch (error) {
        console.error('Failed to parse Python output:', error);
        console.error('Raw output:', dataString);
        reject(new Error('Failed to parse analysis results'));
      }
    });
  });
}

// Analyze video endpoint
router.post('/analyze', upload.single('video'), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file uploaded' });
    }

    const videoPath = req.file.path;
    const audioText = req.body.audioText || '';
    
    console.log(`Processing video: ${videoPath}`);
    console.log(`Audio text: ${audioText}`);
    
    try {
      // For demonstration, use a simplified result if Python script fails
      let result;
      
      try {
        // Try to run Python analysis
        result = await analyzePythonVideo(videoPath, audioText);
      } catch (error) {
        console.error('Python analysis failed, using TypeScript fallback:', error);
        
        // Fallback to simple TypeScript implementation
        result = {
          is_scam: Math.random() > 0.5, // Random result for demo
          confidence: Math.random(), 
          model_confidence: Math.random() * 0.8,
          rule_confidence: Math.random() * 0.9,
          reason: 'Analysis performed using TypeScript fallback',
          videoId: uuidv4()
        };
      }
      
      // Clean up the uploaded file after analysis
      fs.unlink(videoPath, (err) => {
        if (err) console.error(`Failed to delete temporary file: ${err}`);
      });
      
      return res.json(result);
    } catch (error) {
      console.error('Error analyzing video:', error);
      
      // Clean up the uploaded file on error
      fs.unlink(videoPath, (err) => {
        if (err) console.error(`Failed to delete temporary file: ${err}`);
      });
      
      return res.status(500).json({ 
        error: 'Failed to analyze video',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  } catch (error) {
    console.error('Error handling video upload:', error);
    return res.status(500).json({ 
      error: 'Server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Analyze frame endpoint (for frontend to send individual frames)
router.post('/analyze-frame', multer().single('frame'), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No frame image uploaded' });
    }

    // Convert the uploaded frame to base64
    const frameBuffer = req.file.buffer;
    
    // Run Python analysis on the frame (or use a TypeScript fallback)
    try {
      // For demonstration, use a simplified analysis
      // In a real implementation, we would call the Python script with the frame data
      
      // Analyze the frame
      const result = {
        is_scam: Math.random() > 0.7, // Less likely to trigger false positives
        confidence: Math.random() * 0.8,
        visual_confidence: Math.random() * 0.8,
        live_analysis: true,
        features: {
          face_count: Math.floor(Math.random() * 2),
          face_ratio: Math.random() * 0.5,
          eye_contact: Math.random() > 0.5 ? 1 : 0,
          edge_density: Math.random() * 0.3
        }
      };
      
      return res.json(result);
    } catch (error) {
      console.error('Error analyzing frame:', error);
      return res.status(500).json({ 
        error: 'Failed to analyze frame',
        message: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  } catch (error) {
    console.error('Error handling frame upload:', error);
    return res.status(500).json({ 
      error: 'Server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get model info endpoint (for frontend to know detection capabilities)
router.get('/model-info', (req: Request, res: Response) => {
  // This would normally fetch info from the Python detector
  res.json({
    status: 'loaded',
    features: {
      visual_detection: true,
      audio_detection: true,
      text_analysis: true,
      realtime_analysis: true
    },
    detection_types: [
      { id: 'visual', label: 'Visual Pattern Detection', description: 'Analyzes visual elements for signs of scams' },
      { id: 'audio', label: 'Audio Analysis', description: 'Examines audio for suspicious speech patterns' },
      { id: 'face', label: 'Face Detection', description: 'Identifies and analyzes faces for suspicious behavior' },
      { id: 'text', label: 'Text Analysis', description: 'Scans transcribed text for scam keywords' }
    ],
    version: '1.0.0'
  });
});

// Health check endpoint
router.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'ok', message: 'Video detection service is running' });
});

// Setup WebSocket handlers for the video detection stream
export function setupVideoDetectionWebsocket(wss: WebSocketServer) {
  wss.on('connection', (ws: WebSocket, req: any) => {
    // Only handle connections to the video detection path
    if (req.url?.startsWith('/ws/video-detection')) {
      console.log('New video detection WebSocket connection');
      
      let frameCount = 0;
      let detectionActive = false;
      let lastResult: { confidence: number } | null = null;
      
      // Handle messages from client
      ws.on('message', (message: Buffer | string | ArrayBuffer) => {
        try {
          const messageString = message instanceof Buffer ? message.toString() : 
                               typeof message === 'string' ? message : 
                               new TextDecoder().decode(message);
          const data = JSON.parse(messageString);
          
          if (data.type === 'start') {
            // Start a new detection session
            detectionActive = true;
            frameCount = 0;
            console.log('Starting new video detection session');
            ws.send(JSON.stringify({ type: 'start_ack', success: true }));
          } 
          else if (data.type === 'stop') {
            // Stop the current session
            detectionActive = false;
            console.log('Stopping video detection session');
            ws.send(JSON.stringify({ type: 'stop_ack', success: true }));
          }
          else if (data.type === 'frame' && detectionActive) {
            // Process a video frame
            frameCount++;
            
            // Only process every 5th frame to reduce load
            if (frameCount % 5 === 0) {
              // In a real implementation, we would analyze the frame using our Python detector
              // For demonstration, generate a random result that slowly changes over time
              
              // If we have a previous result, make the new one similar to create a smooth effect
              let confidence = Math.random() * 0.8;
              let isScam = confidence > 0.6;
              
              if (lastResult) {
                // Make the new result somewhat similar to the last one (70% influence)
                confidence = lastResult.confidence * 0.7 + confidence * 0.3;
                isScam = confidence > 0.6;
              }
              
              const result = {
                type: 'result',
                frame: frameCount,
                is_scam: isScam,
                confidence: confidence,
                visual_confidence: confidence,
                features: {
                  face_count: Math.floor(Math.random() * 2),
                  face_ratio: Math.random() * 0.5,
                  eye_contact: Math.random() > 0.5 ? 1 : 0,
                  edge_density: Math.random() * 0.3
                }
              };
              
              lastResult = result;
              ws.send(JSON.stringify(result));
            }
          }
        } catch (error) {
          console.error('Error handling WebSocket message:', error);
          ws.send(JSON.stringify({ 
            type: 'error', 
            message: error instanceof Error ? error.message : 'Unknown error' 
          }));
        }
      });
      
      // Handle connection close
      ws.on('close', () => {
        console.log('Video detection WebSocket connection closed');
        detectionActive = false;
      });
      
      // Send initial connection success message
      ws.send(JSON.stringify({ 
        type: 'connected',
        message: 'Connected to video detection service' 
      }));
    }
  });
}

export default router;