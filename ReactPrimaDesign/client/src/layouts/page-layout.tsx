import React from 'react';
import { cn } from '@/lib/utils';

export type PageLayoutProps = {
  children: React.ReactNode;
  className?: string;
  title?: string;
  description?: string;
  icon?: React.ReactNode;
};

export function PageLayout({ children, className, title, description, icon }: PageLayoutProps) {
  return (
    <div className={cn(
      "container mx-auto px-4 py-6 max-w-6xl", 
      className
    )}>
      {(title || description) && (
        <div className="mb-6">
          {title && (
            <div className="flex items-center gap-2 mb-1">
              {icon && <div className="text-primary">{icon}</div>}
              <h1 className="text-2xl font-bold">{title}</h1>
            </div>
          )}
          {description && (
            <p className="text-muted-foreground">{description}</p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}