# SafePay Deployment Guide

This document provides instructions for deploying the SafePay UPI Scam Detection application with proper port configurations and optimized deployment size.

## Prerequisites

- Replit account
- Git access to the repository
- Required API keys (GROQ_API_KEY, OPENAI_API_KEY)

## Port Configuration

### Required Port Mappings

The application uses a standardized port configuration:

| Service                | Local Port | External Port | Notes                            |
|------------------------|------------|--------------|----------------------------------|
| Main Express Server    | 5000       | 80           | For Replit workflow compatibility |
| ML QR Service          | 8081       | 8081         | FastAPI/Flask service            |
| ML Voice/Text Service  | 8082       | 8082         | FastAPI service                  |
| ML Video Service       | 8083       | 8083         | For video analysis               |

**Important**: All other port mappings should be removed to reduce deployment image size.

## Deployment Steps

1. **Update .replit file**

   Navigate to the Replit configuration and update the port mappings to match the following:

   ```
   [[ports]]
   localPort = 5000
   externalPort = 80

   [[ports]]
   localPort = 8081
   externalPort = 8081

   [[ports]]
   localPort = 8082
   externalPort = 8082

   [[ports]]
   localPort = 8083
   externalPort = 8083
   ```

   Remove all other port mappings to reduce deployment image size.

2. **Configure .gitignore**

   Ensure the `.gitignore` file contains all necessary exclusions to reduce repository and deployment size:

   ```
   # dependencies
   node_modules/

   # build outputs
   dist/
   build/

   # cache
   .cache/
   .npm/
   cache/
   prediction_cache/

   # logs
   logs/
   *.log

   # media and datasets
   *.mp4
   *.mov
   *.mp3
   *.wav
   *.csv
   *.json.gz

   # ML model files
   *.h5
   *.pkl
   *.pt
   *.pth
   *.onnx

   # Python
   __pycache__/
   *.py[cod]
   *.so
   .Python
   env/
   venv/
   ENV/
   env.bak/
   venv.bak/
   .pythonlibs/

   # Development files
   .vscode/
   .github/

   # Large files
   mydata.zip
   data/
   ```

3. **Configure Environment Variables**

   Ensure the following environment variables are set in the Replit Secrets tab:

   - `PORT=5000`
   - `NODE_ENV=production`
   - `ML_QR_SERVICE_PORT=8081`
   - `ML_VOICE_TEXT_SERVICE_PORT=8082`
   - `ML_VIDEO_SERVICE_PORT=8083`
   - `GROQ_API_KEY=your-groq-api-key`
   - `OPENAI_API_KEY=your-openai-api-key`
   - `DATABASE_URL=postgres://...`
   - `SESSION_SECRET=secure-random-string`

4. **Update .replit.deploy**

   Ensure the `.replit.deploy` file contains the correct port configuration:

   ```
   [env]
   PORT = "5000"
   NODE_ENV = "production"
   ML_QR_SERVICE_PORT = "8081"
   ML_VOICE_TEXT_SERVICE_PORT = "8082"
   ML_VIDEO_SERVICE_PORT = "8083"

   [workflow]
   defaultPort = 5000
   ```

5. **Deploy the Application**

   Click the "Deploy" button in Replit's interface to start the deployment process. The application will be available at your `.replit.app` domain.

## Troubleshooting

If you encounter deployment issues:

1. **Port Configuration Issues**:
   - Verify that the port mappings in `.replit` match the recommended configuration
   - Check that the server is listening on port 5000 in `server/index.ts`
   - Ensure that external ports are correctly mapped

2. **Deployment Size Issues**:
   - The deployment size must be under 8GB (current size with large directories is approximately 5.9GB)
   - Key large directories to exclude:
     - `.pythonlibs/` (5.1GB)
     - `node_modules/` (740MB)
     - `data/` (31MB)
   - Important: You must manually edit the `.replit` file using the Replit UI and add the following lines:
     ```
     [packaging]
     ignoredPaths = [
       ".pythonlibs/",
       "node_modules/",
       "data/",
       "mydata.zip",
       "attached_assets/*.mp4",
       "attached_assets/*.wav",
       "attached_assets/*.mp3",
       "cache/",
       "prediction_cache/"
     ]
     ```
   - This configuration is critical for deployment and cannot be added via automation

3. **Application Not Starting**:
   - Check the deployment logs for error messages
   - Verify that all required environment variables are correctly set
   - Ensure that PORT=5000 is used for the main server

## Reference

For detailed port configuration documentation, see [PORT_CONFIGURATION.md](PORT_CONFIGURATION.md).