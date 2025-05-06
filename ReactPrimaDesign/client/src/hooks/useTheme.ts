import { useContext } from 'react';
import { ThemeContext, ThemeContextType } from '@/contexts/ThemeContext';

/**
 * Custom hook to access theme context
 * @returns ThemeContextType - Contains the current theme, isDark flag, and toggle function
 */
export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  
  return context;
};