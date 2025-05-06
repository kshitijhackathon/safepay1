import React, { createContext, useState, useEffect, ReactNode } from 'react';
import Cookies from 'js-cookie';

// Define the type for the theme context value
export type ThemeContextType = {
  theme: 'light' | 'dark';
  isDark: boolean;
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
};

// Create the theme context with a default value
export const ThemeContext = createContext<ThemeContextType>({
  theme: 'light',
  isDark: false,
  toggleTheme: () => {},
  setTheme: () => {},
});

// Props type for the ThemeProvider component
interface ThemeProviderProps {
  children: ReactNode;
}

// Theme provider component that will wrap the app
export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  // State to track current theme
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  // Computed value for easier boolean checks
  const isDark = theme === 'dark';

  // Effect to initialize theme from localStorage, cookie, or system preference
  useEffect(() => {
    // First try to get theme from localStorage
    let savedTheme = localStorage.getItem('theme');
    
    // If not found in localStorage, try cookies
    if (!savedTheme) {
      const cookieTheme = Cookies.get('theme');
      if (cookieTheme) {
        savedTheme = cookieTheme as string;
      }
    }
    
    // If still not found, check system preference
    if (!savedTheme) {
      savedTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    
    // Apply theme immediately on page load, before React rendering
    if (savedTheme === 'dark') {
      document.documentElement.classList.add('dark');
      setTheme('dark');
    } else {
      document.documentElement.classList.remove('dark');
      setTheme('light');
    }
  }, []);

  // Effect to apply theme class to html element and save to storage
  useEffect(() => {
    // Get the HTML element
    const root = document.documentElement;
    
    // Apply or remove the dark class based on current theme
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    
    // Save theme preference to both localStorage and cookies for persistence
    localStorage.setItem('theme', theme);
    Cookies.set('theme', theme, { 
      expires: 200, // 200 days
      sameSite: 'Lax',
      secure: window.location.protocol === 'https:'
    });
  }, [theme]);

  // Listen for storage events for cross-tab synchronization
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'theme') {
        const newTheme = e.newValue as 'light' | 'dark';
        if (newTheme && (newTheme === 'light' || newTheme === 'dark')) {
          setTheme(newTheme);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Function to toggle between light and dark themes
  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // Provide the theme context to children components
  return (
    <ThemeContext.Provider value={{ theme, isDark, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};