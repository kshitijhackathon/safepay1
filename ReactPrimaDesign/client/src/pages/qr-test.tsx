import React from 'react';
import { QRTestGenerator } from '@/components/test/qr-test-generator';
import { ArrowLeft } from 'lucide-react';
import { Link } from 'wouter';
import { Button } from '@/components/ui/button';

export default function QRTestPage() {
  return (
    <div className="container py-6 max-w-5xl mx-auto">
      <header className="mb-6">
        <Link href="/">
          <Button variant="ghost" className="flex items-center gap-2">
            <ArrowLeft size={16} /> Back to App
          </Button>
        </Link>
        <h1 className="text-2xl font-bold mt-4">QR Code Scanner Test</h1>
        <p className="text-muted-foreground">
          This page generates test QR codes for the QR scanner. Use it to validate that the scanner 
          is working properly with different UPI IDs and QR code formats.
        </p>
      </header>
      
      <div className="grid md:grid-cols-2 gap-8">
        <div>
          <h2 className="text-xl font-semibold mb-4">Test Generator</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Generate test QR codes with different UPI IDs. To test the scanner, generate a QR code here, 
            then scan it using the QR scanner in the main app.
          </p>
          
          <QRTestGenerator />
        </div>
        
        <div className="space-y-4">
          <h2 className="text-xl font-semibold mb-4">How to Test</h2>
          <div className="bg-slate-100 dark:bg-slate-800 p-4 rounded-lg">
            <ol className="list-decimal ml-5 space-y-2">
              <li>Generate a QR code using the generator on the left</li>
              <li>Display this QR code on a separate device or screen</li>
              <li>Go to the main app and open the QR scanner</li>
              <li>Point the camera at the QR code and test if it's detected correctly</li>
              <li>Verify that the detected UPI ID matches what was generated</li>
            </ol>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900 dark:bg-opacity-20 p-4 rounded-lg">
            <h3 className="font-semibold text-lg">Testing Tips</h3>
            <ul className="list-disc ml-5 space-y-2 mt-2">
              <li>Try with different brightness levels on the device displaying the QR code</li>
              <li>Test with both low-risk and high-risk UPI IDs</li>
              <li>Test at different distances and angles</li>
              <li>If scanning fails, check for reflections or try with a different device</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900 dark:bg-opacity-20 p-4 rounded-lg">
            <h3 className="font-semibold text-lg">Available Test UPI IDs</h3>
            <div className="mt-2 space-y-2">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  Low Risk
                </span>
                <span className="text-sm">shallowfoebe@okicici, quickpay@okaxis</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                  Medium Risk
                </span>
                <span className="text-sm">mobikwik@ybl, fastcash.refund@hdfcbank</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                  High Risk
                </span>
                <span className="text-sm">instant-verify@oksbi, support.kyc@yesbank</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-10 text-center">
        <Link href="/scan">
          <Button size="lg" className="bg-primary text-white">
            Open QR Scanner
          </Button>
        </Link>
      </div>
    </div>
  );
}