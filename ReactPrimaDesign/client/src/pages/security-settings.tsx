import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { ArrowLeft, Shield, AlertTriangle, Smartphone, Lock, Fingerprint, MapPin } from 'lucide-react';
import PageHeader from '@/components/ui/page-header';

interface MfaSettings {
  userId: number;
  mfaEnabled: boolean;
  preferredChannel: 'email' | 'sms' | 'authenticator';
  biometricEnabled: boolean;
  lastVerified: string;
  securityLevel: 'low' | 'medium' | 'high';
}

export default function SecuritySettings() {
  const [, setLocation] = useLocation();
  const [loading, setLoading] = useState(true);
  const [simSwapStatus, setSimSwapStatus] = useState<{
    isChecking: boolean;
    isSwapped: boolean;
    lastChecked: string | null;
    carrier?: string;
  }>({
    isChecking: false,
    isSwapped: false,
    lastChecked: null
  });
  const [mfaSettings, setMfaSettings] = useState<MfaSettings | null>(null);
  const [showSuccessMessage, setShowSuccessMessage] = useState(false);
  const userId = 1; // For demo, typically would come from auth context

  useEffect(() => {
    // Fetch security settings on component mount
    const fetchSettings = async () => {
      try {
        const response = await fetch(`/api/security/mfa-settings/${userId}`);
        if (response.ok) {
          const data = await response.json();
          setMfaSettings(data);
        } else {
          console.error('Failed to fetch MFA settings');
        }
      } catch (error) {
        console.error('Error fetching MFA settings:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSettings();
  }, [userId]);

  const checkSimSwap = async () => {
    try {
      setSimSwapStatus(prev => ({ ...prev, isChecking: true }));
      
      // Using a demo phone number for simulation
      const phoneNumber = '9876543210';
      
      const response = await fetch(`/api/security/check-sim-swap?phoneNumber=${phoneNumber}`);
      if (response.ok) {
        const data = await response.json();
        setSimSwapStatus({
          isChecking: false,
          isSwapped: data.isSwapped,
          lastChecked: new Date().toLocaleString(),
          carrier: data.carrier
        });
        
        // If SIM swap detected, show alert
        if (data.isSwapped) {
          alert('WARNING: SIM swap detected! Your phone number may have been compromised. Please contact your carrier immediately.');
        }
      } else {
        setSimSwapStatus(prev => ({ ...prev, isChecking: false }));
        alert('Failed to check SIM swap status');
      }
    } catch (error) {
      console.error('Error checking SIM swap:', error);
      setSimSwapStatus(prev => ({ ...prev, isChecking: false }));
      alert('Error checking SIM swap status');
    }
  };

  const toggleMfa = async () => {
    if (!mfaSettings) return;
    
    try {
      const newSettings = {
        ...mfaSettings,
        mfaEnabled: !mfaSettings.mfaEnabled
      };
      
      const response = await fetch(`/api/security/mfa-settings/${userId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newSettings)
      });
      
      if (response.ok) {
        setMfaSettings(newSettings);
        setShowSuccessMessage(true);
        setTimeout(() => setShowSuccessMessage(false), 3000);
      } else {
        alert('Failed to update MFA settings');
      }
    } catch (error) {
      console.error('Error updating MFA settings:', error);
      alert('Error updating MFA settings');
    }
  };

  const toggleBiometric = async () => {
    if (!mfaSettings) return;
    
    try {
      const newSettings = {
        ...mfaSettings,
        biometricEnabled: !mfaSettings.biometricEnabled
      };
      
      const response = await fetch(`/api/security/mfa-settings/${userId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newSettings)
      });
      
      if (response.ok) {
        setMfaSettings(newSettings);
        setShowSuccessMessage(true);
        setTimeout(() => setShowSuccessMessage(false), 3000);
      } else {
        alert('Failed to update biometric settings');
      }
    } catch (error) {
      console.error('Error updating biometric settings:', error);
      alert('Error updating biometric settings');
    }
  };

  const changePreferredChannel = async (channel: 'email' | 'sms' | 'authenticator') => {
    if (!mfaSettings) return;
    
    try {
      const newSettings = {
        ...mfaSettings,
        preferredChannel: channel
      };
      
      const response = await fetch(`/api/security/mfa-settings/${userId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newSettings)
      });
      
      if (response.ok) {
        setMfaSettings(newSettings);
        setShowSuccessMessage(true);
        setTimeout(() => setShowSuccessMessage(false), 3000);
      } else {
        alert('Failed to update preferred channel');
      }
    } catch (error) {
      console.error('Error updating preferred channel:', error);
      alert('Error updating preferred channel');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-4">
        <PageHeader title="Security Settings" onBack={() => setLocation('/')} />
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <PageHeader title="Security Settings" onBack={() => setLocation('/')} />
      
      {showSuccessMessage && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
          Settings updated successfully!
        </div>
      )}
      
      {/* SIM Swap Detection */}
      <div className="bg-white rounded-lg shadow-sm p-4 mb-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center">
            <Shield className="text-blue-500 mr-2" size={20} />
            <h2 className="text-lg font-medium">SIM Swap Detection</h2>
          </div>
          <button 
            onClick={checkSimSwap}
            disabled={simSwapStatus.isChecking}
            className="px-3 py-1 bg-blue-500 text-white rounded-md text-sm hover:bg-blue-600 disabled:bg-blue-300"
          >
            {simSwapStatus.isChecking ? 'Checking...' : 'Check Now'}
          </button>
        </div>
        
        <p className="text-sm text-gray-600 mb-3">
          Verify your SIM card hasn't been compromised or moved to another device without your knowledge.
        </p>
        
        {simSwapStatus.lastChecked && (
          <div className={`p-3 rounded-md mb-2 ${simSwapStatus.isSwapped ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
            <div className="flex items-center">
              {simSwapStatus.isSwapped ? (
                <AlertTriangle className="mr-2" size={16} />
              ) : (
                <Shield className="mr-2" size={16} />
              )}
              <span className="text-sm font-medium">
                {simSwapStatus.isSwapped 
                  ? 'SIM swap detected! Please contact your carrier immediately.' 
                  : 'No SIM swap detected. Your phone number is secure.'}
              </span>
            </div>
            <div className="text-xs mt-1">
              Last checked: {simSwapStatus.lastChecked}
              {simSwapStatus.carrier && ` • Carrier: ${simSwapStatus.carrier}`}
            </div>
          </div>
        )}
      </div>
      
      {/* Multi-Factor Authentication Settings */}
      {mfaSettings && (
        <div className="bg-white rounded-lg shadow-sm p-4 mb-4">
          <div className="flex items-center mb-4">
            <Lock className="text-blue-500 mr-2" size={20} />
            <h2 className="text-lg font-medium">Multi-Factor Authentication</h2>
            <div className="ml-auto">
              <label className="relative inline-flex items-center cursor-pointer">
                <input 
                  type="checkbox" 
                  className="sr-only peer"
                  checked={mfaSettings.mfaEnabled}
                  onChange={toggleMfa}
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-500"></div>
              </label>
            </div>
          </div>
          
          <p className="text-sm text-gray-600 mb-4">
            Strengthen your account security by requiring multiple forms of verification when signing in.
          </p>
          
          {/* Verification Channels */}
          <div className="mb-4">
            <h3 className="text-sm font-medium mb-2">Preferred Verification Channel</h3>
            <div className="flex space-x-2">
              <button 
                onClick={() => changePreferredChannel('email')}
                className={`px-3 py-1.5 rounded-md text-sm flex items-center ${mfaSettings.preferredChannel === 'email' ? 'bg-blue-100 text-blue-800 border border-blue-200' : 'bg-gray-100 text-gray-700'}`}
              >
                <span>Email</span>
              </button>
              <button 
                onClick={() => changePreferredChannel('sms')}
                className={`px-3 py-1.5 rounded-md text-sm flex items-center ${mfaSettings.preferredChannel === 'sms' ? 'bg-blue-100 text-blue-800 border border-blue-200' : 'bg-gray-100 text-gray-700'}`}
              >
                <Smartphone className="mr-1" size={14} />
                <span>SMS</span>
              </button>
              <button 
                onClick={() => changePreferredChannel('authenticator')}
                className={`px-3 py-1.5 rounded-md text-sm flex items-center ${mfaSettings.preferredChannel === 'authenticator' ? 'bg-blue-100 text-blue-800 border border-blue-200' : 'bg-gray-100 text-gray-700'}`}
              >
                <span>Authenticator</span>
              </button>
            </div>
          </div>
          
          {/* Biometric Authentication */}
          <div className="pb-2 border-b border-gray-100">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Fingerprint className="text-blue-500 mr-2" size={16} />
                <span className="text-sm">Biometric Authentication</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input 
                  type="checkbox" 
                  className="sr-only peer"
                  checked={mfaSettings.biometricEnabled}
                  onChange={toggleBiometric}
                />
                <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-500"></div>
              </label>
            </div>
            <p className="text-xs text-gray-500 ml-6 mt-1">
              Use fingerprint or face recognition for an additional layer of security
            </p>
          </div>
          
          {/* Location-based Verification */}
          <div className="pt-2">
            <div className="flex items-center">
              <MapPin className="text-blue-500 mr-2" size={16} />
              <span className="text-sm">Location-based Verification</span>
              <span className="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded">Coming Soon</span>
            </div>
            <p className="text-xs text-gray-500 ml-6 mt-1">
              Get alerts when your account is accessed from unusual locations
            </p>
          </div>
        </div>
      )}
      
      {/* Security Level Summary */}
      {mfaSettings && (
        <div className="bg-white rounded-lg shadow-sm p-4">
          <h2 className="text-lg font-medium mb-3">Security Level: {mfaSettings.securityLevel.charAt(0).toUpperCase() + mfaSettings.securityLevel.slice(1)}</h2>
          
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className={`h-2.5 rounded-full ${
                mfaSettings.securityLevel === 'high' ? 'bg-green-500 w-full' : 
                mfaSettings.securityLevel === 'medium' ? 'bg-yellow-400 w-2/3' : 
                'bg-red-500 w-1/3'
              }`}
            ></div>
          </div>
          
          <div className="mt-4 text-sm text-gray-600">
            {mfaSettings.securityLevel === 'high' ? (
              <p>Your account has strong protection. Keep up the good security practices!</p>
            ) : mfaSettings.securityLevel === 'medium' ? (
              <p>Enable biometric authentication to further strengthen your account security.</p>
            ) : (
              <p>Enable MFA and biometric authentication to protect your account from unauthorized access.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}