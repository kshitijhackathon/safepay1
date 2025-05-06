import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { PinPad } from "@/components/ui/pin-pad";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useLocation } from "wouter";
import { Loader2, Fingerprint, Lock, Check, AlertCircle } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export default function SetupSecurity() {
  const [step, setStep] = useState<"choose-pin" | "confirm-pin" | "biometric">("choose-pin");
  const [pin, setPin] = useState("");
  const [confirmPin, setConfirmPin] = useState("");
  const [pinError, setPinError] = useState(false);
  const [useBiometric, setUseBiometric] = useState(false);
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  // Get userId from URL query parameter or local storage
  // For example, after OTP verification, we'll redirect with userId in the URL
  const [location] = useLocation();
  const params = new URLSearchParams(location.split('?')[1]);
  const userId = params.get('userId') || localStorage.getItem('userId') || '';
  
  // Log user ID for debugging
  console.log('Setup Security - User ID:', userId);
  
  // Auto-reset error state
  useEffect(() => {
    if (pinError) {
      const timer = setTimeout(() => {
        setPinError(false);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [pinError]);

  // Setup PIN mutation
  const setupPinMutation = useMutation({
    mutationFn: async () => {
      console.log('Setting up PIN with:', { pin });
      
      // For demo, we'll just set the PIN without requiring a user ID
      const res = await apiRequest("POST", "/api/auth/setup-pin", { 
        pin 
      });
      return await res.json();
    },
    onSuccess: () => {
      toast({
        title: "PIN Setup Complete",
        description: "Your PIN has been set successfully.",
      });
      setStep("biometric");
    },
    onError: (error: Error) => {
      toast({
        title: "PIN Setup Failed",
        description: error.message || "Failed to set up PIN. Please try again.",
        variant: "destructive",
      });
    },
  });

  // Setup biometric mutation
  const setupBiometricMutation = useMutation({
    mutationFn: async () => {
      console.log('Setting up biometric with:', { enable: useBiometric });
      
      // For demo, we'll just set biometric preferences without requiring a user ID
      const res = await apiRequest("POST", "/api/auth/biometric", { 
        enable: useBiometric,
        deviceId: "device_" + Date.now() // In a real app, get actual device ID
      });
      return await res.json();
    },
    onSuccess: () => {
      toast({
        title: "Setup Complete",
        description: "Your security preferences have been saved.",
      });
      // Redirect to home page
      setTimeout(() => {
        setLocation("/home");
      }, 1000);
    },
    onError: (error: Error) => {
      toast({
        title: "Setup Failed",
        description: error.message || "Failed to save preferences. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handlePinComplete = (pinValue: string) => {
    if (step === "choose-pin") {
      setPin(pinValue);
      setTimeout(() => {
        setStep("confirm-pin");
      }, 400); // Short delay to allow animation to complete
    } else if (step === "confirm-pin") {
      setConfirmPin(pinValue);
      if (pinValue === pin) {
        // PINs match, set up PIN
        setupPinMutation.mutate();
      } else {
        setPinError(true);
        toast({
          title: "PINs don't match",
          description: "The PINs you entered don't match. Please try again.",
          variant: "destructive",
        });
        // Reset to first step after a short delay
        setTimeout(() => {
          setPin("");
          setConfirmPin("");
          setStep("choose-pin");
        }, 1500);
      }
    }
  };

  const handleBiometricToggle = (checked: boolean) => {
    setUseBiometric(checked);
  };

  const handleFinishSetup = () => {
    setupBiometricMutation.mutate();
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 p-4">
      <Card className="w-full max-w-md mx-auto">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold text-center">
            {step === "choose-pin" ? "Create PIN" : 
             step === "confirm-pin" ? "Confirm PIN" : 
             "Enable Biometric Login"}
          </CardTitle>
          <CardDescription className="text-center">
            {step === "choose-pin" ? "Choose a 4-digit PIN for quick access" : 
             step === "confirm-pin" ? "Enter the same PIN again to confirm" : 
             "Use your fingerprint or face for faster login"}
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          {(step === "choose-pin" || step === "confirm-pin") && (
            <div className="space-y-6">
              <div className="flex justify-center">
                {pinError ? (
                  <AlertCircle className="h-12 w-12 text-destructive animate-pulse" />
                ) : (
                  <Lock className="h-12 w-12 text-primary" />
                )}
              </div>
              
              <PinPad 
                length={4} 
                onComplete={handlePinComplete}
                className="mx-auto"
                resetOnComplete={true}
              />
              
              {setupPinMutation.isPending && (
                <div className="flex justify-center my-4">
                  <Loader2 className="h-6 w-6 animate-spin text-primary" />
                </div>
              )}
            </div>
          )}
          
          {step === "biometric" && (
            <div className="space-y-6">
              <div className="flex justify-center">
                <Fingerprint className="h-16 w-16 text-primary" />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="biometric">Enable biometric login</Label>
                  <p className="text-sm text-muted-foreground">
                    Use fingerprint or face recognition for faster login
                  </p>
                </div>
                <Switch 
                  id="biometric" 
                  checked={useBiometric} 
                  onCheckedChange={handleBiometricToggle} 
                />
              </div>
              
              <Button 
                className="w-full" 
                onClick={handleFinishSetup}
                disabled={setupBiometricMutation.isPending}
              >
                {setupBiometricMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Check className="mr-2 h-4 w-4" />
                )}
                Finish Setup
              </Button>
            </div>
          )}
        </CardContent>
        
        <CardFooter className="flex justify-center">
          <p className="text-xs text-muted-foreground text-center">
            Your PIN and biometric data are securely stored on your device.
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}