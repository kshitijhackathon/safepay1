/**
 * Register Advanced QR Scan Routes
 * Registers the routes for the advanced Python-based QR scanning service
 */

import { Express } from "express";
import advancedQRScanRoutes from "./advanced-qr-scan";
import { log } from "../vite";

export function registerAdvancedQRScanRoutes(app: Express): void {
  log("Registering Advanced ML-Based QR Scan Routes...", "routes");
  
  // Register the advanced QR scan routes
  app.use("/api/advanced-qr", advancedQRScanRoutes);
  
  log("✅ Advanced QR scan routes registered", "qrscan");
}