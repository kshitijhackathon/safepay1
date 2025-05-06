import React from 'react';
import { EnhancedQRScanner } from '@/components/scanner/enhanced-qr-scanner';
import { useLocation } from 'wouter';
import { Keyboard } from 'lucide-react';

export default function QRScan() {
  const [, setLocation] = useLocation();
  
  const handleScan = (qrData: string) => {
    console.log('QR code scanned:', qrData);
    // Store the scan data and redirect to scan page for processing
    try {
      sessionStorage.setItem('lastScannedQR', qrData);
    } catch (error) {
      console.error('Error storing QR data:', error);
    }
    setLocation('/scan?qrData=' + encodeURIComponent(qrData));
  };
  
  const handleClose = () => {
    setLocation('/home');
  };
  
  const goToManualEntry = () => {
    setLocation('/manual-upi-entry');
  };
  
  return (
    <div className="h-screen w-full flex flex-col">
      <div className="flex-1 relative">
        <EnhancedQRScanner 
          onScan={handleScan}
          onClose={handleClose}
        />
      </div>
      
      {/* Fixed bottom button for manual entry, always visible */}
      <div className="fixed bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent z-50">
        <button
          onClick={goToManualEntry}
          className="w-full bg-white text-primary hover:bg-primary hover:text-white transition-colors flex items-center justify-center gap-2 py-4 rounded-lg font-medium border-2 border-white shadow-xl"
        >
          <Keyboard className="w-5 h-5" /> Enter UPI ID manually
        </button>
        <div className="w-full text-center mt-2 text-white text-xs opacity-80">
          Can't scan? Enter UPI ID manually
        </div>
      </div>
    </div>
  );
}