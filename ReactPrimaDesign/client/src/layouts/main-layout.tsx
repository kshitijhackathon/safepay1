import React from 'react';
import { BottomNav } from '@/components/navigation/bottom-nav';
import { cn } from '@/lib/utils';
import { useLocation } from 'wouter';

interface MainLayoutProps {
  children: React.ReactNode;
  noBottomNav?: boolean;
  className?: string;
  fullHeight?: boolean; // For layouts that need to fill the entire viewport height
}

export default function MainLayout({ 
  children, 
  noBottomNav = false,
  className,
  fullHeight = false
}: MainLayoutProps) {
  const [location] = useLocation();
  const isChatPage = location === '/chat-support';
  
  return (
    <div className={cn(
      "bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-300",
      (isChatPage || fullHeight) ? "h-[100dvh] w-full overflow-hidden flex flex-col" : "min-h-screen",
      className
    )}>
      <main className={cn(
        (isChatPage || fullHeight) ? "flex-1 overflow-hidden flex flex-col" : "pb-16",
      )}>
        {children}
      </main>
      
      {/* The bottom nav renders conditionally */}
      {!noBottomNav && (
        <div className={cn(
          "w-full flex-shrink-0",
          // Ensure the bottom nav remains visible in chat by giving it higher z-index
          isChatPage && "sticky bottom-0 z-50"
        )}>
          <BottomNav />
        </div>
      )}
    </div>
  );
}
