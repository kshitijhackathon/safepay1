import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { ArrowLeft } from 'lucide-react';
import jsQR from 'jsqr';

interface QRScannerProps {
  onScan: (data: string) => void;
  onClose: () => void;
  className?: string;
}

export function QRScanner({ onScan, onClose, className }: QRScannerProps) {
  const [hasFlash, setHasFlash] = useState(false);
  const [flashOn, setFlashOn] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scanIntervalRef = useRef<number | null>(null);
  
  // Setup camera when component mounts
  useEffect(() => {
    let stream: MediaStream | null = null;
    
    const setupCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          
          // Check if flash is available
          const tracks = stream.getVideoTracks();
          if (tracks.length > 0) {
            const capabilities = tracks[0].getCapabilities();
            setHasFlash('torch' in capabilities);
          }
        }
      } catch (error) {
        console.error('Error accessing camera:', error);
      }
    };
    
    setupCamera();
    
    // Clean up function
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);
  
  // Toggle flashlight
  const toggleFlash = async () => {
    if (!videoRef.current || !hasFlash) return;
    
    const stream = videoRef.current.srcObject as MediaStream;
    const tracks = stream.getVideoTracks();
    
    if (tracks.length > 0) {
      const track = tracks[0];
      const newFlashState = !flashOn;
      
      try {
        // The 'torch' property is not in standard TypeScript definitions but is supported in some browsers
        await track.applyConstraints({
          // @ts-ignore - Suppressing typescript error for torch property
          advanced: [{ torch: newFlashState }]
        });
        
        setFlashOn(newFlashState);
      } catch (error) {
        console.error('Error toggling flash:', error);
      }
    }
  };
  
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanComplete, setScanComplete] = useState(false);
  const animationFrameId = useRef<number | null>(null);
  
  // Real QR code detection function
  const detectQRCode = () => {
    if (scanComplete || !isScanning || !videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Set canvas dimensions to match video
    const width = video.videoWidth;
    const height = video.videoHeight;
    
    if (width === 0 || height === 0) {
      // Video dimensions not yet available, try again shortly
      animationFrameId.current = requestAnimationFrame(detectQRCode);
      return;
    }
    
    canvas.width = width;
    canvas.height = height;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Could not get canvas context');
      return;
    }
    
    // Draw the current video frame to the canvas
    ctx.drawImage(video, 0, 0, width, height);
    
    try {
      // Get image data for QR code analysis
      const imageData = ctx.getImageData(0, 0, width, height);
      
      // Process with jsQR
      const code = jsQR(imageData.data, imageData.width, imageData.height, {
        inversionAttempts: "dontInvert", // QR codes in well-lit conditions are dark on light background
      });
      
      if (code) {
        console.log('QR code detected:', code.data);
        
        // Process the QR code data - parse UPI info
        let upiId = '';
        
        // Check if the data is a UPI URL
        if (code.data.startsWith('upi://')) {
          // Extract UPI ID from UPI URL format (like upi://pay?pa=abc@bank&pn=Name)
          const url = new URL(code.data);
          const params = new URLSearchParams(url.search);
          upiId = params.get('pa') || '';
        } else if (code.data.includes('@')) {
          // Directly a UPI ID (like abc@bank)
          upiId = code.data;
        } else {
          // Try to check if the QR contains text with a UPI ID in it
          const match = code.data.match(/([a-zA-Z0-9\.\-\_]+@[a-zA-Z0-9]+)/);
          if (match && match[1]) {
            upiId = match[1];
          } else {
            upiId = code.data; // Use as-is if nothing else works
          }
        }
        
        console.log('Extracted UPI ID:', upiId);
        
        // Show success UI
        setScanComplete(true);
        setScanProgress(100);
        
        // Return the detected UPI ID
        setTimeout(() => {
          onScan(upiId);
        }, 800); // Show success animation briefly
        
        // Stop scanning
        return;
      }
      
      // No QR code found, update progress and continue scanning
      setScanProgress(prev => {
        // Make progress pulsate a bit to indicate active scanning
        const fluctuation = Math.sin(Date.now() / 300) * 5; // Oscillate between -5 and +5
        return Math.min(Math.max(50 + fluctuation, 40), 60); // Keep between 40-60%
      });
      
      // Continue scanning
      animationFrameId.current = requestAnimationFrame(detectQRCode);
    } catch (err) {
      const error = err as Error;
      console.error('QR scanning error:', error);
      
      setScanError('Error analyzing camera feed');
      
      // Continue scanning despite error, with a slight delay to prevent CPU overload
      setTimeout(() => {
        animationFrameId.current = requestAnimationFrame(detectQRCode);
      }, 500);
    }
  };
  
  // Start QR detection when video plays
  const handleVideoPlay = () => {
    if (!isScanning) {
      setIsScanning(true);
      detectQRCode();
    }
  };
  
  // Cleanup animation frame or timer on unmount
  useEffect(() => {
    return () => {
      if (animationFrameId.current) {
        // Clean up either requestAnimationFrame or setTimeout
        cancelAnimationFrame(animationFrameId.current);
        clearTimeout(animationFrameId.current);
      }
    };
  }, []);

  return (
    <div className={cn("relative flex flex-col h-screen bg-black", className)}>
      {/* Header */}
      <div className="w-full flex justify-between items-center p-6">
        <button 
          onClick={onClose}
          className="w-10 h-10 bg-black/50 rounded-full flex items-center justify-center"
        >
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            fill="none" 
            viewBox="0 0 24 24" 
            strokeWidth={1.5} 
            stroke="currentColor" 
            className="w-6 h-6 text-white"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              d="M6 18L18 6M6 6l12 12" 
            />
          </svg>
        </button>
        <p className="text-white font-medium">Scan QR Code</p>
        <div className="w-10"></div>
      </div>
      
      {/* Video feed */}
      <div className="flex-1 flex items-center justify-center">
        <video 
          ref={videoRef}
          autoPlay 
          playsInline 
          muted 
          className="absolute inset-0 w-full h-full object-cover"
          onPlay={handleVideoPlay}
        />
        
        {/* Scan overlay */}
        <div className="border-2 border-white rounded-3xl w-[250px] h-[250px] relative z-10 overflow-hidden flex items-center justify-center">
          <canvas ref={canvasRef} className="hidden" />
          
          {isScanning && !scanComplete && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/30">
              <div className="w-16 h-16 mb-4">
                <svg className="w-full h-full animate-pulse" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M3 9V5.25C3 4.00736 4.00736 3 5.25 3H9M9 21H5.25C4.00736 21 3 19.9926 3 18.75V15M21 15V18.75C21 19.9926 19.9926 21 18.75 21H15M15 3H18.75C19.9926 3 21 4.00736 21 5.25V9" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div className="w-3/4 bg-gray-700 rounded-full h-1.5 mb-2">
                <div 
                  className="bg-white h-1.5 rounded-full transition-all duration-100 ease-in-out" 
                  style={{ width: `${scanProgress}%` }}
                />
              </div>
              <p className="text-white text-sm">Scanning QR code...</p>

            </div>
          )}
          
          {scanComplete && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/30">
              <div className="w-16 h-16 text-green-500 animate-pulse mb-2">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-full h-full">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-white font-medium text-center">QR Code Detected!</p>
              <p className="text-white text-sm mt-1">Starting security verification...</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Controls */}
      <div className="w-full p-6 flex flex-col items-center">
        <div className="flex justify-center mb-4">
          {hasFlash && (
            <button 
              onClick={toggleFlash}
              className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center mr-4"
            >
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                strokeWidth={1.5} 
                stroke="currentColor" 
                className="w-6 h-6 text-white"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" 
                />
              </svg>
            </button>
          )}
          
          <button className="w-12 h-12 bg-white rounded-full flex items-center justify-center">
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              fill="none" 
              viewBox="0 0 24 24" 
              strokeWidth={1.5} 
              stroke="currentColor" 
              className="w-6 h-6 text-black"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" 
              />
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0zM18.75 10.5h.008v.008h-.008V10.5z" 
              />
            </svg>
          </button>
        </div>
        
        {/* QR Code scanning and Manual Entry buttons */}
        <div className="flex flex-col gap-3 w-full">
          {!isScanning && !scanComplete && (
            <button
              onClick={() => {
                // Reset everything to initial state
                setScanProgress(0);
                setScanComplete(false);
                setScanError(null);
                
                // Start scanning
                setIsScanning(true);
                
                // Give a small delay for UI to update, then start detection cycle
                setTimeout(() => {
                  detectQRCode();
                }, 100);
              }}
              className="mt-4 bg-primary text-white px-6 py-3 rounded-lg text-lg font-medium border-2 border-white shadow-lg animate-pulse"
            >
              👉 Tap to Scan QR Code
            </button>
          )}
          
          {scanError && (
            <div className="bg-red-500/70 text-white px-4 py-2 rounded-lg text-center mt-2">
              {scanError} 
              <button 
                className="underline ml-2"
                onClick={() => {
                  setScanError(null);
                  setIsScanning(false);
                }}
              >
                Retry
              </button>
            </div>
          )}
          
          {/* Manual UPI entry option */}
          <button
            onClick={onClose}
            className="text-white flex items-center justify-center gap-2 mt-2"
          >
            <ArrowLeft size={16} />
            <span>Go Back</span>
          </button>
        </div>
      </div>
    </div>
  );
}
