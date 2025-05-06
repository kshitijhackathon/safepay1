import { useLocation } from "wouter";
import { ArrowLeft } from "lucide-react";
import { useTheme } from "@/hooks/useTheme";
import { ThemeToggle } from "@/components/ui/ThemeToggle";

export default function HelpPage() {
  const [, setLocation] = useLocation();
  const { theme, isDark } = useTheme();

  return (
    <div className="dark-bg-secondary min-h-screen pb-6">
      {/* Header */}
      <div className="p-4 dark-bg-primary shadow-sm flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setLocation('/')}
            className="w-8 h-8 flex items-center justify-center rounded-full dark-bg-tertiary"
          >
            <ArrowLeft size={18} className="dark-text-primary" />
          </button>
          <h1 className="text-lg font-medium dark-text-primary">Help & Info</h1>
        </div>
        <ThemeToggle />
      </div>

      {/* Content */}
      <div className="px-4 py-6 space-y-6">
        {/* Theme Settings */}
        <div className="dark-card p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-3 dark-text-primary">Theme Settings</h2>
          <p className="dark-text-secondary mb-4">
            You can toggle between light and dark mode using the switch in the header.
            Your preference is saved and will be remembered across visits.
          </p>
          <div className="flex items-center justify-between p-3 dark-bg-tertiary rounded-lg">
            <span className="dark-text-primary font-medium">Current theme</span>
            <span className="dark-text-secondary capitalize">{theme} mode</span>
          </div>
        </div>

        {/* App Features */}
        <div className="dark-card p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-3 dark-text-primary">Key Features</h2>
          
          <div className="space-y-4">
            <div className="border-b dark-border pb-3">
              <h3 className="font-medium mb-1 dark-text-primary">UPI Scam Check</h3>
              <p className="dark-text-secondary text-sm">
                Check if a UPI ID is safe before making a payment. The app will analyze the UPI ID and provide a risk score.
              </p>
            </div>
            
            <div className="border-b dark-border pb-3">
              <h3 className="font-medium mb-1 dark-text-primary">Voice Scam Detection</h3>
              <p className="dark-text-secondary text-sm">
                Record or upload a voice conversation to detect potential scam patterns using AI analysis.
              </p>
            </div>
            
            <div className="border-b dark-border pb-3">
              <h3 className="font-medium mb-1 dark-text-primary">WhatsApp Message Analysis</h3>
              <p className="dark-text-secondary text-sm">
                Upload screenshots of suspicious WhatsApp messages for AI-powered scam detection.
              </p>
            </div>
            
            <div className="border-b dark-border pb-3">
              <h3 className="font-medium mb-1 dark-text-primary">Fraud Map</h3>
              <p className="dark-text-secondary text-sm">
                View a geographical map showing scam hotspots and recent fraud activities in your area.
              </p>
            </div>
            
            <div>
              <h3 className="font-medium mb-1 dark-text-primary">SIM Swap & MFA Protection</h3>
              <p className="dark-text-secondary text-sm">
                Advanced security features to protect your account from SIM swap attacks with multi-factor authentication.
              </p>
            </div>
          </div>
        </div>
        
        {/* About */}
        <div className="dark-card p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-3 dark-text-primary">About SafePay</h2>
          <p className="dark-text-secondary mb-3">
            SafePay is a cutting-edge security app designed to protect users from UPI payment scams using advanced AI and machine learning technology.
          </p>
          <p className="dark-text-secondary">
            Version 1.0.0 • © 2025 SafePay
          </p>
        </div>
      </div>
    </div>
  );
}