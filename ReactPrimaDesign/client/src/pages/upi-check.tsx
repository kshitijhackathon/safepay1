import React from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import UpiCheckButton from '@/components/upi-check/upi-check-button';

export default function UpiCheck() {
  const [, setLocation] = useLocation();

  return (
    <div className="flex flex-col px-6 py-8 min-h-screen overflow-y-auto">
      <div className="flex justify-between items-center mb-8">
        <button
          onClick={() => setLocation('/home')}
          className="w-10 h-10 bg-[#F5F6FA] rounded-full flex items-center justify-center"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <h1 className="text-xl font-bold">UPI Risk Check</h1>
        <div className="w-10"></div>
      </div>
      
      {/* Integrated UPI Check Button that opens as dialog */}
      <div className="mb-6">
        <UpiCheckButton />
      </div>
      
      <div className="mt-6 space-y-6 pb-6">
        <div className="bg-blue-50 rounded-lg p-4 dark:bg-blue-950">
          <h3 className="font-medium text-blue-800 mb-2 dark:text-blue-300">What is UPI Scam Check?</h3>
          <p className="text-blue-700 text-sm dark:text-blue-400">
            Our advanced system scans UPI IDs for potential fraud risks. It combines machine learning, historical data, and 
            real-time threat intelligence to provide a comprehensive risk assessment.
          </p>
        </div>
        
        <div className="bg-orange-50 rounded-lg p-4 dark:bg-orange-950">
          <h3 className="font-medium text-orange-800 mb-2 dark:text-orange-300">How It Works</h3>
          <ol className="text-orange-700 text-sm space-y-2 list-decimal pl-4 dark:text-orange-400">
            <li>Enter any UPI ID you want to check</li>
            <li>Our system analyzes it against multiple data sources</li>
            <li>Review the risk assessment and recommended actions</li>
            <li>Report suspicious UPI IDs to help protect others</li>
          </ol>
        </div>
        
        <Button
          onClick={() => setLocation('/report-scam')}
          className="w-full bg-primary text-white font-semibold py-4 px-6 rounded-xl shadow-md text-center"
        >
          Report a Scam
        </Button>
      </div>
    </div>
  );
}
