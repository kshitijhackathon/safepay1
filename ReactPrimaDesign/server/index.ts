import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import * as dotenv from "dotenv";
import path from "path";

dotenv.config();

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// CORS configuration
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", process.env.VERCEL_URL || "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept, Authorization"
  );
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }
  next();
});

// Logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  const server = await registerRoutes(app);

  // Error handling middleware
  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";
    res.status(status).json({ message });
    console.error(err);
  });

  if (process.env.NODE_ENV === "development") {
    await setupVite(app, server);
  } else {
    // Serve static files from the dist directory in production
    app.use(express.static(path.join(process.cwd(), "dist")));

    // Handle client-side routing
    app.get("/*", (req, res, next) => {
      if (req.path.startsWith("/api")) {
        return next();
      }
      res.sendFile(path.join(process.cwd(), "dist", "index.html"), (err) => {
        if (err) {
          console.error("Error sending index.html:", err);
          res.status(500).send("Error loading application");
        }
      });
    });
  }

  const PORT = process.env.PORT || 3000;
  const HOST = process.env.HOST || "0.0.0.0";

  if (process.env.VERCEL) {
    // Export the Express app for Vercel
    module.exports = app;
  } else {
    // Start the server locally
    server.listen(PORT, HOST, () => {
      log(`[express] Server running at http://${HOST}:${PORT}`);
    });
  }
})();
