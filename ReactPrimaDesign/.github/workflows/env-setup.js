/**
 * GitHub Actions Environment Setup Script
 * 
 * This script sets up environment variables for GitHub Actions workflows.
 * It's used in both CI and deployment workflows to ensure consistent configuration.
 */
const fs = require('fs');

// Create a .env file for GitHub Actions
const envContent = `
# Database Configuration
DATABASE_URL=${process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/upi_secure'}

# API Keys
OPENAI_API_KEY=${process.env.OPENAI_API_KEY || 'sk-placeholder-for-ci-workflow'}

# Authentication
SESSION_SECRET=${process.env.SESSION_SECRET || 'github-actions-test-session-secret'}

# App Configuration
PORT=5001
NODE_ENV=${process.env.NODE_ENV || 'test'}
`;

// Write the environment file
fs.writeFileSync('.env', envContent.trim());
console.log('Environment file created for GitHub Actions');