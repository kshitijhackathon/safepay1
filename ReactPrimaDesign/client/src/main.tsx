import { createRoot } from "react-dom/client";
import React from "react";
import App from "./App";
import "./index.css";

// Initialize theme from localStorage before React loads
// This prevents the flash of wrong theme on initial load
(function initializeTheme() {
  // First check localStorage
  const savedTheme = localStorage.getItem('theme');
  
  // Apply dark mode if saved as dark
  if (savedTheme === 'dark') {
    document.documentElement.classList.add('dark');
  } else if (savedTheme === 'light') {
    document.documentElement.classList.remove('dark');
  } else {
    // If no saved preference, check system preference
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (prefersDark) {
      document.documentElement.classList.add('dark');
    }
  }
})();

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
