import { Express, Request, Response } from 'express';
import { spawn, ChildProcess } from 'child_process';
import { log } from '../vite';
import { createProxyMiddleware } from 'http-proxy-middleware';

// Store reference to the Streamlit process
let streamlitProcess: ChildProcess | null = null;
let streamlitPort = 8501; // Default Streamlit port

/**
 * Register routes for managing Streamlit application
 * @param app Express application instance
 */
export function registerStreamlitRoutes(app: Express) {
  // Setup proxy for Streamlit
  app.use(
    '/proxy/streamlit',
    createProxyMiddleware({
      target: `http://localhost:${streamlitPort}`,
      changeOrigin: true,
      ws: true,
      pathRewrite: {
        '^/proxy/streamlit': '',
      },
      onProxyReq: (proxyReq, req, res) => {
        // Add debug logs for proxy requests
        log(`Proxying request to Streamlit: ${req.method} ${req.url}`, 'streamlit-proxy');
      },
      onError: (err, req, res) => {
        log(`Proxy error: ${err}`, 'streamlit-proxy');
        res.writeHead(500, {
          'Content-Type': 'text/plain',
        });
        res.end('Proxy error connecting to Streamlit server');
      }
    })
  );
  /**
   * Get status of the Streamlit server
   */
  app.get('/api/streamlit/status', (req, res) => {
    const isRunning = streamlitProcess !== null && 
                      streamlitProcess.exitCode === null && 
                      !streamlitProcess.killed;
    
    if (isRunning) {
      res.json({ 
        running: true, 
        url: `/proxy/streamlit`,
        port: streamlitPort
      });
    } else {
      res.json({ running: false });
    }
  });

  /**
   * Start the Streamlit server
   */
  app.post('/api/streamlit/start', (req, res) => {
    // Check if process is already running
    if (streamlitProcess !== null && 
        streamlitProcess.exitCode === null && 
        !streamlitProcess.killed) {
      return res.json({ 
        running: true, 
        url: `/proxy/streamlit`,
        port: streamlitPort 
      });
    }
    
    try {
      // Start Streamlit process
      log('Starting Streamlit server...', 'streamlit');
      
      streamlitProcess = spawn('python3', [
        '-m', 'streamlit', 'run', 'fraud_map.py',
        '--server.port', streamlitPort.toString(),
        '--server.headless', 'true',
        '--browser.serverAddress', '0.0.0.0',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
      ]);
      
      // Log output from Streamlit
      streamlitProcess?.stdout?.on('data', (data) => {
        log(`Streamlit stdout: ${data}`, 'streamlit');
      });
      
      streamlitProcess?.stderr?.on('data', (data) => {
        log(`Streamlit stderr: ${data}`, 'streamlit');
      });
      
      // Handle process exit
      streamlitProcess?.on('close', (code) => {
        log(`Streamlit process exited with code ${code}`, 'streamlit');
        streamlitProcess = null;
      });
      
      // Give Streamlit some time to start
      setTimeout(() => {
        res.json({ 
          running: true, 
          url: `/proxy/streamlit`,
          port: streamlitPort 
        });
      }, 3000);
      
    } catch (error) {
      console.error('Error starting Streamlit:', error);
      res.status(500).json({ 
        running: false, 
        error: 'Failed to start Streamlit server' 
      });
    }
  });

  /**
   * Stop the Streamlit server
   */
  app.post('/api/streamlit/stop', (req, res) => {
    if (streamlitProcess) {
      log('Stopping Streamlit server...', 'streamlit');
      
      try {
        streamlitProcess.kill();
        streamlitProcess = null;
        res.json({ success: true, message: 'Streamlit server stopped' });
      } catch (error) {
        console.error('Error stopping Streamlit:', error);
        res.status(500).json({ 
          success: false, 
          error: 'Failed to stop Streamlit server' 
        });
      }
    } else {
      res.json({ success: true, message: 'Streamlit server not running' });
    }
  });
}