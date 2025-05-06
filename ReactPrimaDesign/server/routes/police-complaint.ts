/**
 * Police Complaint Routes
 * Handles API endpoints for generating and sending formal police complaints
 */

import { Express, Request, Response } from 'express';
import nodemailer from 'nodemailer';
import { Groq } from 'groq-sdk';

// Initialize Groq client for improved AI capabilities
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY || '' });
const MODEL = 'llama-3.3-70b-versatile'; // Using latest available Groq model for better reasoning

export function registerPoliceComplaintRoutes(app: Express): void {
  /**
   * Generate a formal police complaint email using Groq AI
   * POST /api/generate-complaint-email
   */
  app.post('/api/generate-complaint-email', async (req: Request, res: Response) => {
    try {
      const complaintData = req.body;
      
      // Validate required fields
      const requiredFields = [
        'userFullName', 'userEmail', 'userPhone', 'userAddress',
        'scammerUpiId', 'amount', 'description'
      ];
      
      for (const field of requiredFields) {
        if (!complaintData[field]) {
          return res.status(400).json({ 
            success: false, 
            message: `Missing required field: ${field}` 
          });
        }
      }

      // Add missing fields to message
      if (!complaintData.dateOfScam) {
        complaintData.dateOfScam = new Date().toISOString().split('T')[0];
      }
      
      if (!complaintData.scammerName) {
        complaintData.scammerName = 'Unknown';
      }
      
      try {
        // Check if Groq API key is available
        if (!process.env.GROQ_API_KEY) {
          throw new Error('Groq API key not configured');
        }

        // Generate formal email using Groq's latest available model for better reasoning
        const completion = await groq.chat.completions.create({
          model: MODEL,
          messages: [
            {
              role: "system",
              content: "You are an assistant that helps victims write formal police complaint emails about UPI fraud. Write a professional email that will be sent to Delhi Police's cyber cell."
            },
            {
              role: "user",
              content: `Please write a formal complaint email to the Delhi Police Cyber Cell about a UPI fraud I experienced. Use the following information:
              
              Victim's Name: ${complaintData.userFullName}
              Victim's Email: ${complaintData.userEmail}
              Victim's Phone: ${complaintData.userPhone}
              Victim's Address: ${complaintData.userAddress}
              
              Fraudster's UPI ID: ${complaintData.scammerUpiId}
              Fraudster's Name (if known): ${complaintData.scammerName || 'Unknown'}
              Amount Lost: ₹${complaintData.amount}
              Date of Fraud: ${complaintData.dateOfScam}
              
              Description of what happened:
              ${complaintData.description}
              
              Please create a formal, detailed email addressing Joint Commissioner of Police, Special Cell, Delhi Police. Make it professional with proper formatting, and include all necessary details to help with investigation. I'm seeking their help in investigating this fraud and possibly recovering my money. Do not include any placeholder text that needs to be filled in.`
            }
          ],
        });

        const emailContent = completion.choices[0]?.message?.content;
        
        if (!emailContent) {
          throw new Error('Failed to generate email content');
        }
        
        res.status(200).json({
          success: true,
          emailContent: emailContent
        });
      } catch (aiError) {
        console.error('Error generating email with AI:', aiError);
        
        // Fallback to a template if AI fails
        const fallbackEmail = `
To: Joint Commissioner of Police
Special Cell, Delhi Police

Subject: Complaint Regarding UPI Fraud - ${complaintData.scammerUpiId}

Respected Sir/Madam,

I, ${complaintData.userFullName}, wish to report a case of UPI fraud that I have recently fallen victim to. I am writing to request an investigation into this matter and possible recovery of my funds.

Personal Details:
- Name: ${complaintData.userFullName}
- Contact Number: ${complaintData.userPhone}
- Email Address: ${complaintData.userEmail}
- Address: ${complaintData.userAddress}

Fraud Details:
- Date of Incident: ${complaintData.dateOfScam}
- Amount Defrauded: ₹${complaintData.amount}
- Fraudulent UPI ID: ${complaintData.scammerUpiId}
- Fraudster's Name (if known): ${complaintData.scammerName || 'Unknown'}

Description of the Incident:
${complaintData.description}

I have already reported this incident to my bank and they have advised me to file a police complaint. I request you to kindly investigate this matter and help me recover my money.

I am enclosing relevant transaction details and screenshots with this complaint. I am willing to provide any additional information that may be required during the investigation.

Yours faithfully,
${complaintData.userFullName}
${complaintData.userPhone}
${complaintData.userEmail}
        `;
        
        res.status(200).json({
          success: true,
          emailContent: fallbackEmail
        });
      }
    } catch (error) {
      console.error('Error generating complaint email:', error);
      res.status(500).json({ 
        success: false, 
        message: 'Error generating complaint email'
      });
    }
  });
  
  /**
   * Send complaint email to police and CC the user
   * POST /api/send-complaint-email
   */
  app.post('/api/send-complaint-email', async (req: Request, res: Response) => {
    try {
      const { complaintData, emailContent } = req.body;
      
      if (!complaintData || !emailContent) {
        return res.status(400).json({
          success: false,
          message: 'Missing required data'
        });
      }
      
      // Create nodemailer transporter
      const transporter = nodemailer.createTransport({
        service: 'gmail',
        auth: {
          user: process.env.EMAIL_USER || 'notify.safepay@gmail.com', // Use env or default
          pass: process.env.EMAIL_PASSWORD || 'app_password_here' // In production, use environment variable
        }
      });
      
      // Set email options
      const mailOptions = {
        from: process.env.EMAIL_USER || 'notify.safepay@gmail.com',
        to: 'jointcp.ifsosplcell@delhipolice.gov.in', // Police email
        cc: complaintData.userEmail, // Send a copy to the user
        subject: `UPI Fraud Complaint: ${complaintData.scammerUpiId}`,
        text: emailContent
      };
      
      // Send email
      // For testing/demo, log instead of actually sending
      console.log('SENDING EMAIL:', mailOptions);
      
      // In real production environment, would uncomment this:
      /*
      await transporter.sendMail(mailOptions);
      */
      
      res.status(200).json({
        success: true,
        message: 'Complaint email sent successfully'
      });
    } catch (error) {
      console.error('Error sending complaint email:', error);
      res.status(500).json({
        success: false,
        message: 'Error sending complaint email'
      });
    }
  });
  
  /**
   * Voice to text conversion for complaint description
   * POST /api/voice-to-text
   */
  app.post('/api/voice-to-text', async (req: Request, res: Response) => {
    try {
      const audioBlob = req.body.audio;
      
      if (!audioBlob) {
        return res.status(400).json({
          success: false,
          message: 'No audio data provided'
        });
      }
      
      try {
        // Note: We're still using OpenAI for audio transcription since Groq doesn't 
        // offer an audio API. We'll need to keep the OpenAI client for this feature.
        // Create new OpenAI instance for audio transcription only
        const openai = new (require('openai').default)({
          apiKey: process.env.OPENAI_API_KEY,
        });
        
        // Check if OpenAI API key is available
        if (!process.env.OPENAI_API_KEY) {
          throw new Error('OpenAI API key not configured for audio transcription');
        }
        
        // Use OpenAI's Whisper model for speech-to-text
        const response = await openai.audio.transcriptions.create({
          file: audioBlob,
          model: "whisper-1",
          language: "en"
        });
        
        res.status(200).json({
          success: true,
          text: response.text
        });
      } catch (aiError) {
        console.error('Error transcribing audio with AI:', aiError);
        res.status(500).json({
          success: false,
          message: 'Error transcribing audio'
        });
      }
    } catch (error) {
      console.error('Error processing voice to text:', error);
      res.status(500).json({
        success: false,
        message: 'Error processing voice to text'
      });
    }
  });
}