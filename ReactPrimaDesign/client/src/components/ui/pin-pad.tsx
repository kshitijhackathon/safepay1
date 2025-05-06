import React, { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface PinPadProps {
  length?: number;
  onComplete?: (pin: string) => void;
  className?: string;
  resetOnComplete?: boolean;
}

export function PinPad({ 
  length = 4, 
  onComplete,
  className,
  resetOnComplete = false
}: PinPadProps) {
  const [pin, setPin] = useState<string>('');
  const [animateReset, setAnimateReset] = useState<boolean>(false);
  
  const handleNumberPress = (num: number | string) => {
    if (pin.length < length) {
      const newPin = pin + num;
      setPin(newPin);
      
      if (newPin.length === length) {
        if (onComplete) {
          onComplete(newPin);
        }
        
        if (resetOnComplete) {
          setAnimateReset(true);
          setTimeout(() => {
            setPin('');
            setAnimateReset(false);
          }, 300); // Animation duration
        }
      }
    }
  };
  
  const handleDelete = () => {
    if (pin.length > 0) {
      setPin(pin.slice(0, -1));
    }
  };
  
  // Create an array [1,2,3,4,5,6,7,8,9,'',0,'del']
  const buttons = [
    ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => ({ value: num, type: 'number' })),
    { value: '', type: 'empty' },
    { value: 0, type: 'number' },
    { value: 'del', type: 'delete' },
  ];

  return (
    <div className={cn("flex flex-col items-center", className)}>
      {/* PIN dots */}
      <div className="flex justify-center my-8">
        {Array.from({ length }).map((_, index) => (
          <div 
            key={index} 
            className={cn(
              "pin-dot transition-all duration-300",
              index < pin.length && "filled",
              animateReset && "animate-ping"
            )} 
          />
        ))}
      </div>
      
      {/* PIN pad */}
      <div className="grid grid-cols-3 gap-4 justify-items-center">
        {buttons.map((button, index) => {
          if (button.type === 'empty') {
            return <div key={index} />;
          }
          
          if (button.type === 'delete') {
            return (
              <button 
                key={index}
                onClick={handleDelete}
                className="pin-button dark:text-gray-300"
              >
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-6 h-6"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M12 9.75L14.25 12m0 0l2.25 2.25M14.25 12l2.25-2.25M14.25 12L12 14.25m-2.58 4.92l-6.375-6.375a1.125 1.125 0 010-1.59L9.42 4.83c.211-.211.498-.33.796-.33H19.5a2.25 2.25 0 012.25 2.25v10.5a2.25 2.25 0 01-2.25 2.25h-9.284c-.298 0-.585-.119-.796-.33z" 
                  />
                </svg>
              </button>
            );
          }
          
          return (
            <button 
              key={index}
              onClick={() => handleNumberPress(button.value)}
              className="pin-button dark:text-gray-300"
            >
              {button.value}
            </button>
          );
        })}
      </div>
    </div>
  );
}
