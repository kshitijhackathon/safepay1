import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface NotificationProps {
  message: string;
  type?: 'info' | 'success' | 'warning' | 'error';
  duration?: number; // in milliseconds, 0 means it won't auto-dismiss
  onClose?: () => void;
  className?: string;
}

export function NotificationBar({
  message,
  type = 'info',
  duration = 0,
  onClose,
  className
}: NotificationProps) {
  const [visible, setVisible] = useState(true);

  // Background colors for different notification types
  const bgColors = {
    info: 'bg-blue-500',
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500'
  };

  const handleClose = () => {
    setVisible(false);
    if (onClose) onClose();
  };

  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(handleClose, duration);
      return () => clearTimeout(timer);
    }
  }, [duration]);

  if (!visible) return null;

  return (
    <div className={cn(
      'fixed bottom-16 left-0 right-0 z-50 p-3 text-white flex items-center justify-between',
      bgColors[type],
      className
    )}>
      <span className="font-medium text-sm">{message}</span>
      <button 
        onClick={handleClose}
        className="p-1 rounded-full hover:bg-white/20 transition-colors"
        aria-label="Close notification"
      >
        <X size={18} />
      </button>
    </div>
  );
}