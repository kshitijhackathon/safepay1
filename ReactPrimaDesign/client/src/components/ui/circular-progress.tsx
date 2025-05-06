import React from 'react';
import { cn } from '@/lib/utils';

interface CircularProgressProps {
  value: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
  valueClassName?: string;
  color?: string;
}

export function CircularProgress({
  value,
  size = 160,
  strokeWidth = 16,
  className,
  valueClassName,
  color
}: CircularProgressProps) {
  // Ensure value is between 0 and 100
  const normalizedValue = Math.min(100, Math.max(0, value));
  
  // Calculate inner circle size
  const innerSize = size - (strokeWidth * 2);
  
  // Calculate color based on value if not provided
  const progressColor = color || (() => {
    if (value < 30) return '#43A047'; // Success/Green
    if (value < 70) return '#FFB300'; // Warning/Yellow
    return '#E53935'; // Error/Red
  })();

  return (
    <div 
      className={cn("relative flex items-center justify-center", className)}
      style={{ 
        width: size, 
        height: size,
        borderRadius: '50%',
        background: `conic-gradient(
          ${progressColor} ${normalizedValue * 3.6}deg,
          #F5F6FA 0deg
        )`
      }}
    >
      <div 
        className="absolute bg-white rounded-full flex items-center justify-center"
        style={{ 
          width: innerSize,
          height: innerSize
        }}
      >
        <span 
          className={cn("text-4xl font-bold", valueClassName)}
          style={{ color: progressColor }}
        >
          {normalizedValue}%
        </span>
      </div>
    </div>
  );
}
