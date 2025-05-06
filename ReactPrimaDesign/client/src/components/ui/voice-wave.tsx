import React from 'react';

interface VoiceWaveProps {
  isRecording: boolean;
}

export const VoiceWave: React.FC<VoiceWaveProps> = ({ isRecording }) => {
  return (
    <div className="voice-wave flex items-center justify-center h-16 w-56 mb-4">
      {isRecording ? (
        <div className="flex items-center justify-center gap-1">
          {[...Array(5)].map((_, index) => (
            <div
              key={index}
              className="h-8 w-2 bg-primary rounded-full transform transition-all duration-300 ease-in-out animate-pulse"
              style={{
                animation: `pulse 0.${index + 3}s ease-in-out infinite alternate`,
                height: `${Math.max(8, Math.random() * 30)}px`
              }}
            />
          ))}
          {[...Array(5)].map((_, index) => (
            <div
              key={index + 5}
              className="h-12 w-2 bg-primary rounded-full transform transition-all duration-300 ease-in-out animate-pulse"
              style={{
                animation: `pulse 0.${index + 5}s ease-in-out infinite alternate`,
                height: `${Math.max(10, Math.random() * 40)}px`
              }}
            />
          ))}
          {[...Array(5)].map((_, index) => (
            <div
              key={index + 10}
              className="h-8 w-2 bg-primary rounded-full transform transition-all duration-300 ease-in-out animate-pulse"
              style={{
                animation: `pulse 0.${index + 2}s ease-in-out infinite alternate`,
                height: `${Math.max(8, Math.random() * 30)}px`
              }}
            />
          ))}
        </div>
      ) : (
        <div className="flex items-center justify-center gap-1">
          {[...Array(15)].map((_, index) => (
            <div
              key={index}
              className="h-2 w-2 bg-gray-300 rounded-full"
            />
          ))}
        </div>
      )}
    </div>
  );
};