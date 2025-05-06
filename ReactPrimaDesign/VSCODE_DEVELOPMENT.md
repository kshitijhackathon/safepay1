# SafePay: Local Development with VS Code

This guide provides instructions for setting up and developing the SafePay application using Visual Studio Code.

## Prerequisites

1. [Node.js](https://nodejs.org/) (v18 or higher)
2. [Python](https://www.python.org/) (v3.11 or higher)
3. [PostgreSQL](https://www.postgresql.org/) database
4. [Visual Studio Code](https://code.visualstudio.com/)
5. [Git](https://git-scm.com/)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/safepay.git
   cd safepay
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in the required variables (database connection, API keys, etc.)

## VS Code Setup

The repository includes pre-configured VS Code settings and extensions. When you open the project in VS Code, you should be prompted to install the recommended extensions.

### Recommended Extensions

- ESLint
- Prettier
- Tailwind CSS IntelliSense
- Python
- Pylance
- GitLens
- Live Server
- Docker
- Jupyter

## Running the Application

The project includes VS Code tasks and launch configurations for easy development.

### Launch Configurations

1. **Launch Frontend**: Start the Vite development server
2. **Launch Backend**: Start the Node.js backend server
3. **Python: FastAPI**: Start the FastAPI ML service
4. **Python: Current File**: Run the currently open Python file
5. **Full Stack: Frontend + Backend**: Start both frontend and backend servers

### Starting with Tasks

Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and select "Tasks: Run Task", then choose one of:

1. **Start Frontend**: Start the React frontend
2. **Start Backend**: Start the Express backend
3. **Start QR ML Service**: Start the QR code ML service
4. **Start Voice ML Service**: Start the voice/text ML service
5. **Start All Services**: Start all services at once

## Project Structure

- `client/`: React frontend application
  - `src/`: Source code
    - `components/`: Reusable UI components
    - `pages/`: Page components for each route
    - `hooks/`: Custom React hooks
    - `lib/`: Utility functions
    - `styles/`: CSS styles
- `server/`: Express backend application
  - `routes.ts`: API route definitions
  - `storage.ts`: Database storage interface
  - `auth.ts`: Authentication logic
- `shared/`: Shared code between frontend and backend
  - `schema.ts`: Database schema definitions
- `models/`: ML model files
- `data/`: Training and test data
- Python ML services:
  - `qr_scan_ml_service.py`: QR code risk assessment service
  - `voice_text_scam_service.py`: Voice and text scam detection service
  - `video_detection.py`: Video scam detection service

## Development Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the project's code style and conventions

3. Test your changes:
   - Use the VS Code debugging tools
   - Check browser console for errors
   - Verify API endpoints using the integrated terminal or REST client

4. Push your changes:
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   git push origin feature/your-feature-name
   ```

5. Open a pull request

## Debugging

1. Frontend: Use the browser's developer tools and React DevTools
2. Backend: Use the VS Code debugger with the "Launch Backend" configuration
3. Python services: Use the "Python: FastAPI" or "Python: Current File" launch configurations

## Additional Resources

- [React Documentation](https://reactjs.org/docs)
- [Express Documentation](https://expressjs.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TanStack Query Documentation](https://tanstack.com/query)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Drizzle ORM Documentation](https://github.com/drizzle-team/drizzle-orm)

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure no other applications are using ports 5000 (main server), 8000 (QR service), or 8100 (voice service)
2. **Database connection issues**: Verify your PostgreSQL connection string in `.env`
3. **API key errors**: Make sure you've set the required API keys in `.env`
4. **Python dependency issues**: If you encounter errors with Python dependencies, try creating a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check existing GitHub Issues
2. Ask in the project's Discord channel
3. Open a new Issue with detailed information about the problem