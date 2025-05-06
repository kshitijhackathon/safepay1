import React from 'react';
import { Switch } from '@headlessui/react';
import { useTheme } from '@/hooks/useTheme';
import { Sun, Moon } from 'lucide-react';

interface ThemeToggleProps {
  label?: string;
  className?: string;
}

/**
 * A toggle component for switching between light and dark themes
 */
export const ThemeToggle: React.FC<ThemeToggleProps> = ({
  label = 'Dark Mode',
  className = '',
}) => {
  const { isDark, toggleTheme } = useTheme();

  return (
    <div className={`flex items-center justify-between ${className}`}>
      <div className="flex items-center space-x-2">
        {isDark ? (
          <Moon className="h-5 w-5 text-indigo-300" aria-hidden="true" />
        ) : (
          <Sun className="h-5 w-5 text-amber-500" aria-hidden="true" />
        )}
        <span className="text-sm font-medium dark:text-gray-300">{label}</span>
      </div>
      
      <Switch
        checked={isDark}
        onChange={toggleTheme}
        className={`${
          isDark ? 'bg-primary' : 'bg-gray-200 dark:bg-gray-700'
        } relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary/80 focus:ring-offset-2 focus:ring-offset-background`}
      >
        <span className="sr-only">Toggle dark mode</span>
        <span
          className={`${
            isDark ? 'translate-x-6' : 'translate-x-1'
          } inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-300 ease-in-out`}
        />
      </Switch>
    </div>
  );
};