# SafePay: AI-Powered UPI Payment Security Platform

<div align="center">
  <img src="generated-icon.png" alt="SafePay Logo" width="150"/>
  <h3>🏆 Developed for Hackazard 2025 Hackathon</h3>
  <p>A cutting-edge mobile-first application revolutionizing UPI payment security through intelligent fraud prevention technologies</p>
</div>

## 🔴 Live Demo

The app is deployed on Replit: [https://safe-pay.replit.app](https://safe-pay.replit.app)

Watch the project demo: [YouTube Demo](https://youtu.be/mGgMQMZ7EKw?si=Ndcq3y2YmVzHbiLs)

## 🚀 Project Overview

SafePay is a comprehensive fraud detection and prevention platform that uses advanced machine learning algorithms to protect users from common UPI payment scams. The application provides real-time risk assessment of QR codes, voice calls, video content, and text messages to identify potential threats before financial transactions occur.

## 🔐 Key Features

- **ML-Powered QR Scanner**: 94% accurate risk assessment for QR codes with direct UPI app integration
- **Voice Scam Detection**: Analyzes call recordings for fraud indicators with voice-focused AI processing
- **Video Analysis**: Identifies scam videos through comprehensive visual, audio, and text analysis
- **Text & WhatsApp Scam Check**: Detects phishing attempts and scam patterns in messages
- **3D Fraud Map**: Interactive visualization showing fraud hotspots across regions
- **Two-Step Verification Flow**: Safety verification and transaction confirmation screens
- **UPI Deep Linking**: Seamless redirection to payment apps (GPay, PhonePe, Paytm)
- **Multi-Factor Authentication**: Layered security with OTP, PIN, and biometric options

## 💻 Technology Stack

### Frontend

- React with TypeScript
- Tailwind CSS + Shadcn UI for responsive design
- Wouter for lightweight routing
- React Hook Form with Zod validation
- TanStack Query for efficient data fetching

### Backend

- Node.js with Express
- Python ML services with FastAPI
- PostgreSQL database with Drizzle ORM
- WebSocket for real-time video analysis

### AI & ML

- Custom-trained QR, voice and video fraud detection models
- Groq LLM API for enhanced natural language understanding
- OpenAI for audio transcription
- Computer vision for QR code verification

## 📱 Application Screens

- **Home**: 4×2 grid layout with 8 main security features
- **QR Scan**: ML-powered scanner with safety verification
- **Voice Check**: Record or upload audio for scam analysis
- **Video Check**: Analyze videos for fraudulent content
- **WhatsApp & Message Check**: Detect scams in messaging platforms
- **3D Fraud Map**: Geographic visualization of scam activities
- **Security Settings**: Configure authentication methods

## 🧠 Machine Learning Features

- **QR Risk Model**: 94% accuracy in detecting fraudulent QR codes
- **Voice Focus**: Isolates speech frequencies (200Hz-3000Hz) for better analysis
- **Video Detection**: Combines visual, audio and text analysis for comprehensive assessment
- **Real-time Processing**: <100ms latency for QR scans, <3s for voice/video analysis
- **Adaptive Risk Scoring**: Context-aware risk assessment that improves with feedback

## 🔧 Setup and Installation

### Prerequisites

- Replit account for cloud deployment
- Groq API key or OpenAI API key

### Replit Installation

1. Fork this Repl
2. Add your environment variables in the Secrets tab:

   - `GROQ_API_KEY`: Your Groq API key
   - `OPENAI_API_KEY`: Your OpenAI API key (for audio transcription)
   - `SESSION_SECRET`: A secure random string
   - `DATABASE_URL`: Your database URL (provided by Replit)

3. Click the Run button to start the development server

### Port Configuration

The application uses a standardized port configuration:

- Main Server: 5000 (Express)
- QR ML Service: 8081 (FastAPI/Flask)
- Voice/Text ML Service: 8082 (FastAPI)
- Video ML Service: 8083 (FastAPI)

For detailed port configuration documentation, see [PORT_CONFIGURATION.md](PORT_CONFIGURATION.md).

### Deployment

For detailed deployment instructions, including port mapping, environment variables, and file size optimization, please see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

### Local Development with VS Code

For detailed instructions on setting up a local development environment with VS Code, please see [VSCODE_DEVELOPMENT.md](VSCODE_DEVELOPMENT.md).

### Docker Setup

For containerized development, you can use Docker:

```bash
# Build and start all services
docker-compose up

# Or build and run in background
docker-compose up -d
```

## 📱 Usage Flow

1. **Scan QR Code**: Use the scanner to capture UPI payment QR codes
2. **Review Safety Verification**: View merchant details and comprehensive risk assessment
3. **Confirm Transaction**: Review payment amount and beneficiary details
4. **Select Payment App**: Choose between GPay, PhonePe, or Paytm
5. **Complete Payment**: Process transaction directly in your preferred UPI app

## 🛡️ Security Features

- **UPI ID Validation**: Pattern recognition for suspicious UPI IDs
- **Transaction Verification**: Two-step verification with merchant validation
- **Multi-layered Analysis**: Combines rule-based checks with ML predictions
- **User Authentication**: OTP verification with demo mode (PIN: 123456)
- **Scam Reporting System**: Community-driven fraud reporting

## 📈 Recent Enhancements

- Replaced OpenAI with Groq LLM for improved performance
- Implemented UPI deep linking for seamless payment app integration
- Enhanced QR detection with 94% accuracy ML model
- Added video and call scam detection capabilities
- Improved voice processing to focus on speech frequencies
- Restructured home interface with consistent 4×2 grid layout

## 👥 Contributors

- Team members: @kshitisingh8075, @yash720, @nihira07, @Niharika01232

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Hackazard 2025 organizers and judges
- Groq and OpenAI for API capabilities
- Replit for development and hosting platform
- Community contributors and testers

---

<div align="center">
  <p>Made with ❤️ for Hackazard 2025</p>
  <p>© 2025 SafePay Team. All rights reserved.</p>
</div>

# Install dependencies

npm ci

# Build the application

npm run build

# Start the server

npm start
