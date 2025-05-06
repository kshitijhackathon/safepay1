import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type AuthState = {
  isLoggedIn: boolean;
  userId: string | null;
  phoneNumber: string | null;
  isSkipped: boolean;
};

type AuthContextType = {
  authState: AuthState;
  login: (userId: string, phoneNumber: string) => void;
  logout: () => void;
  skipLogin: () => void;
  requireLogin: (callback: () => void) => void;
};

// Create a context with a default value
const AuthContext = createContext<AuthContextType | null>(null);

// Initial auth state
const initialState: AuthState = {
  isLoggedIn: false,
  userId: null,
  phoneNumber: null,
  isSkipped: false,
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const [authState, setAuthState] = useState<AuthState>(() => {
    // Try to load from localStorage on initial render
    try {
      // Make sure we're in a browser environment
      if (typeof window !== 'undefined' && window.localStorage) {
        const savedAuth = localStorage.getItem('auth');
        return savedAuth ? JSON.parse(savedAuth) : initialState;
      }
      return initialState;
    } catch (error) {
      console.error('Error loading auth state from localStorage:', error);
      return initialState;
    }
  });

  // Update localStorage whenever authState changes
  useEffect(() => {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.setItem('auth', JSON.stringify(authState));
      }
    } catch (error) {
      console.error('Error saving auth state to localStorage:', error);
    }
  }, [authState]);

  // Login function
  const login = (userId: string, phoneNumber: string) => {
    setAuthState({
      isLoggedIn: true,
      userId,
      phoneNumber,
      isSkipped: false,
    });
  };

  // Logout function
  const logout = () => {
    setAuthState(initialState);
  };

  // Skip login function
  const skipLogin = () => {
    setAuthState({
      ...initialState,
      isSkipped: true,
    });
  };

  // Function to handle features that require login
  const requireLogin = (callback: () => void) => {
    if (authState.isLoggedIn) {
      callback();
    } else {
      try {
        // Use a fixed, reliable returnUrl for scan page to avoid URL encoding issues
        if (typeof window !== 'undefined') {
          const returnUrl = '/scan';
          // Clear the callback to prevent execution
          callback = () => {};
          console.log('Login required. Redirecting to phone login page');
          window.location.href = '/phone-login?returnUrl=' + returnUrl;
        }
      } catch (error) {
        console.error('Error in requireLogin:', error);
      }
    }
  };

  return (
    <AuthContext.Provider value={{ authState, login, logout, skipLogin, requireLogin }}>
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook to use the auth context
export function useAuthState() {
  try {
    const context = useContext(AuthContext);
    if (!context) {
      console.error('useAuthState must be used within an AuthProvider');
      // Return a default implementation to prevent crashes
      return {
        authState: {
          isLoggedIn: false,
          userId: null,
          phoneNumber: null,
          isSkipped: false
        },
        login: () => console.warn('AuthProvider not found'),
        logout: () => console.warn('AuthProvider not found'),
        skipLogin: () => console.warn('AuthProvider not found'),
        requireLogin: () => console.warn('AuthProvider not found')
      };
    }
    return context;
  } catch (error) {
    console.error('Error in useAuthState:', error);
    // Return a default implementation to prevent crashes
    return {
      authState: {
        isLoggedIn: false,
        userId: null,
        phoneNumber: null,
        isSkipped: false
      },
      login: () => console.warn('AuthProvider not found'),
      logout: () => console.warn('AuthProvider not found'),
      skipLogin: () => console.warn('AuthProvider not found'),
      requireLogin: () => console.warn('AuthProvider not found')
    };
  }
}