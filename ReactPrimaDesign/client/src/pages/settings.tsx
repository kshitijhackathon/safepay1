import React from 'react';
import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { ArrowLeft, Moon, Sun, Bell, User } from 'lucide-react';
import { Link } from 'wouter';
import { useTheme } from '@/hooks/useTheme';

/**
 * Settings page component with appearance settings and theme toggle
 */
export default function SettingsPage() {
  const { isDark } = useTheme();

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300 dark:bg-gray-900">
      <div className="container mx-auto max-w-4xl p-4">
        <header className="mb-6">
          <div className="flex items-center mb-4">
            <Link href="/home">
              <div className="flex items-center text-primary hover:text-primary/80 cursor-pointer">
                <ArrowLeft className="h-5 w-5 mr-2" />
                <span>Back</span>
              </div>
            </Link>
          </div>
          <h1 className="text-2xl font-bold dark:text-white">Settings</h1>
        </header>

        <main className="space-y-8">
          {/* Appearance Section */}
          <section className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 transition-colors duration-300">
            <h2 className="text-xl font-semibold mb-4 dark:text-white flex items-center">
              {isDark ? 
                <Moon className="h-5 w-5 mr-2 text-indigo-300" /> : 
                <Sun className="h-5 w-5 mr-2 text-amber-500" />
              }
              Appearance
            </h2>
            <div className="space-y-6">
              <div className="border-b dark:border-gray-700 pb-4">
                <ThemeToggle />
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Preview</h3>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="bg-white border border-gray-200 rounded-md p-4 shadow-sm transition-colors duration-300">
                    <h4 className="font-medium mb-2 flex items-center">
                      <Sun className="h-4 w-4 mr-2 text-amber-500" />
                      Light
                    </h4>
                    <p className="text-sm text-gray-600">
                      Preview how the app looks in light mode
                    </p>
                  </div>
                  <div className="bg-gray-900 border border-gray-800 rounded-md p-4 shadow-sm text-white transition-colors duration-300">
                    <h4 className="font-medium mb-2 flex items-center">
                      <Moon className="h-4 w-4 mr-2 text-indigo-300" />
                      Dark
                    </h4>
                    <p className="text-sm text-gray-400">
                      Preview how the app looks in dark mode
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Other Settings Sections */}
          <section className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 transition-colors duration-300">
            <h2 className="text-xl font-semibold mb-4 dark:text-white flex items-center">
              <User className="h-5 w-5 mr-2 text-primary" />
              Account
            </h2>
            <p className="text-gray-600 dark:text-gray-400">Manage your account settings and preferences</p>
          </section>

          <section className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 transition-colors duration-300">
            <h2 className="text-xl font-semibold mb-4 dark:text-white flex items-center">
              <Bell className="h-5 w-5 mr-2 text-primary" />
              Notifications
            </h2>
            <p className="text-gray-600 dark:text-gray-400">Configure how you receive notifications</p>
          </section>
        </main>
      </div>
    </div>
  );
}