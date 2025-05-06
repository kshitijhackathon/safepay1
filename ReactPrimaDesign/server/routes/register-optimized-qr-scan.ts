import { Express } from "express";
import optimizedQRScanRoutes from "./optimized-qr-scan";
import { createDirectQRRouter } from "./direct-qr-analysis";
import { log } from "../vite";

export function registerOptimizedQRScanRoutes(app: Express): void {
  log("Registering Optimized QR Scan Routes...", "routes");
  
  // First, try to use the Python service proxy
  // This will attempt to forward to the Python ML service
  app.use("/api/optimized-qr", optimizedQRScanRoutes);
  
  // Also register our direct TypeScript implementation
  // This ensures we always have a working QR analysis endpoint
  const directQRRouter = createDirectQRRouter();
  app.use("/api/direct-qr", directQRRouter);
  
  log("✅ QR routes registered - using direct TypeScript implementation", "qrscan");
}