# Contributing to SafePay

Thank you for your interest in contributing to SafePay! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. **Fork the repository** and clone it to your local machine.
2. **Set up your development environment** by following the instructions in the README.md file.
3. **Create a new branch** for your feature or bug fix:
   ```
   git checkout -b feature/your-feature-name
   ```

## Environment Setup

1. Copy `.env.example` to `.env` and fill in the required variables:
   ```
   cp .env.example .env
   ```
2. Make sure you have the following environment variables set:
   - `DATABASE_URL`: PostgreSQL connection string
   - `GROQ_API_KEY`: Your Groq API key
   - `OPENAI_API_KEY`: Your OpenAI API key (for audio transcription)
   - `SESSION_SECRET`: A secure random string
   - `PORT`: 5000 (default)

## Project Architecture

The project follows a microservice architecture:

1. **Frontend (React + TypeScript)**: Client-side application with UI components
2. **Backend (Node.js + Express)**: Main API server and frontend serving
3. **ML Services (Python + FastAPI)**: 
   - QR Code risk assessment service
   - Voice/Text scam detection service
   - Video analysis service

## Development Workflow

1. **Make your changes** in your feature branch.
2. **Test your changes** manually to ensure they work as expected.
3. **Format your code** with proper indentation and consistent style.
4. **Commit your changes** with a clear and descriptive commit message:
   ```
   git commit -m "Add feature: your feature description"
   ```

## Pull Request Process

1. **Push your changes** to your fork:
   ```
   git push origin feature/your-feature-name
   ```
2. **Create a pull request** against the main repository's `main` branch.
3. **Describe your changes** in the pull request description, including:
   - The purpose of the changes
   - How to test the changes
   - Any relevant screenshots or demos
4. **Address any review comments** and make the necessary changes.
5. **Wait for approval** from the maintainers before merging.

## Code Style Guidelines

- Follow the existing code style used in the project.
- Use meaningful variable and function names.
- Add comments for complex logic, especially in ML components.
- Keep functions small and focused on a single responsibility.
- Use TypeScript types for better code quality.

## Working with ML Services

- Python services use FastAPI and should expose consistent API endpoints.
- ML models should be saved in the `models/` directory.
- Include proper error handling and fallback mechanisms.
- Document any new ML features thoroughly.
- Test ML components with representative data samples.

## Working with TypeScript

- Maintain type safety throughout the codebase.
- Define interfaces for complex data structures in `shared/schema.ts`.
- Use React hooks for state management.
- Follow component organization patterns in the existing codebase.

## Working with the Database

- Use Drizzle ORM for database operations.
- Add new schema definitions in `shared/schema.ts`.
- Run `npm run db:push` to apply schema changes to the database.
- Ensure proper indexing for frequently queried fields.

## Performance Considerations

- Optimize large ML operations with caching when possible.
- Use WebSockets for real-time features like video analysis.
- Implement proper loading states in the UI for async operations.
- Consider mobile performance for all features.

## Thank You!

Your contributions are greatly appreciated and help make SafePay better for everyone!
Together, we can build a safer UPI payment ecosystem for all users.