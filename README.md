# Scam Detection and Prevention System

A comprehensive system for detecting and preventing scams through QR codes, voice, text, and UPI transactions. Built with React and Python.

## Features

- QR Code Risk Detection
- Voice & Text Scam Analysis
- UPI Fraud Detection
- Video-based Verification
- Real-time Risk Assessment

## Tech Stack

- Frontend: React with TypeScript
- Backend: Python
- ML Models: scikit-learn, TensorFlow
- API: FastAPI
- Database: As configured in drizzle.config.ts

## Prerequisites

- Node.js (v14+)
- Python 3.8+
- pip
- npm or yarn

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd ReactPrimaDesign
```

2. Install Python dependencies:
```bash
pip install -r requirements_qr.txt
```

3. Install Node.js dependencies:
```bash
npm install
# or
yarn install
```

4. Initialize ML models:
```bash
python initialize_models.py
```

## Running the Application

1. Start the Python backend services:
```bash
python start_qr_service.py
python voice_text_scam_service.py
python upi_fraud_detection_service.py
```

2. Start the React frontend:
```bash
npm run dev
# or
yarn dev
```

## Development

- See VSCODE_DEVELOPMENT.md for VS Code setup
- See CONTRIBUTING.md for contribution guidelines
- See DEPLOYMENT_GUIDE.md for deployment instructions

## Security

For security concerns, please see SECURITY.md

## License

This project is licensed under the terms specified in LICENSE file.

## Contributors

[Your team information here]