import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { OtpInput } from "@/components/ui/otp-input";
import { useLocation } from "wouter";
import { Loader2, ArrowRight, Phone, LockKeyhole } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { isValidPhoneNumber } from "@/lib/upi";
import { useToast } from "@/hooks/use-toast";
import { useAuthState } from "@/hooks/use-auth-state";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function PhoneLogin() {
  const [step, setStep] = useState<"phone" | "otp">("phone");
  const [displayOtp, setDisplayOtp] = useState<string>("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [otp, setOtp] = useState<string>("");
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const { skipLogin, login } = useAuthState();

  // Check for return URL in query parameters
  const [location] = useLocation();

  // More robust URL parsing with logging
  let returnUrl = "/home";
  try {
    const queryParams = location.includes("?") ? location.split("?")[1] : "";
    console.log("Query params:", queryParams);

    if (queryParams) {
      const params = new URLSearchParams(queryParams);
      const returnParam = params.get("returnUrl");
      console.log("Found returnUrl:", returnParam);

      if (returnParam) {
        // Use decodeURIComponent to handle encoded URLs
        returnUrl = decodeURIComponent(returnParam);
      }
    }

    console.log("Final returnUrl:", returnUrl);
  } catch (error) {
    console.error("Error parsing returnUrl:", error);
  }

  // Handle Skip Login
  const handleSkipLogin = () => {
    try {
      console.log("Skipping login and navigating to", returnUrl);
      skipLogin();
      setTimeout(() => {
        setLocation(returnUrl);
      }, 500);
    } catch (error) {
      console.error("Error skipping login:", error);
      // Fallback navigation if there's an error
      setLocation("/home");
    }
  };

  // Request OTP mutation
  const requestOtpMutation = useMutation({
    mutationFn: async () => {
      try {
        const res = await apiRequest("POST", "/api/auth/request-otp", { phoneNumber });
        const data = await res.json();
        return data;
      } catch (error) {
        console.error("Error requesting OTP:", error);
        throw new Error("Failed to send OTP. Please try again.");
      }
    },
    onSuccess: (data) => {
      // Always display 123456 as the OTP
      setDisplayOtp("123456");
      toast({
        title: "OTP Sent",
        description: "Your OTP is: 123456",
        duration: 10000, // Show for 10 seconds
      });
      setStep("otp");
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to send OTP",
        variant: "destructive",
      });
    },
  });

  // Verify OTP mutation
  const verifyOtpMutation = useMutation({
    mutationFn: async (otp: string) => {
      try {
        console.log("Verifying OTP:", { phoneNumber, otp });
        const res = await apiRequest("POST", "/api/auth/verify-otp", {
          phoneNumber,
          otp,
        });

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.message || "OTP verification failed");
        }

        if (!data.success) {
          throw new Error(data.message || "OTP verification failed");
        }

        return data;
      } catch (error) {
        console.error("Error verifying OTP:", error);
        throw error;
      }
    },
    onSuccess: (data) => {
      console.log("OTP verification successful:", data);

      if (data.userId) {
        skipLogin(); // Clear any previous skipped state
        login(data.userId.toString(), phoneNumber);

        // Check if it's a new user who needs to set up security
        if (data.isNewUser) {
          // Redirect to security setup
          setTimeout(() => {
            setLocation(`/setup-security?userId=${data.userId}`);
          }, 1000);
        } else {
          // Redirect to home page for existing users or the return URL
          setTimeout(() => {
            setLocation(returnUrl);
          }, 1000);
        }
      } else {
        throw new Error("Invalid response from server");
      }
    },
    onError: (error: Error) => {
      console.error("OTP verification error:", error);
      toast({
        title: "Authentication Failed",
        description: error.message || "Invalid OTP. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handlePhoneNumberSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate phone number
    if (!isValidPhoneNumber(phoneNumber)) {
      toast({
        title: "Invalid Phone Number",
        description: "Please enter a valid 10-digit phone number",
        variant: "destructive",
      });
      return;
    }

    // Request OTP
    requestOtpMutation.mutate();
  };

  const handleOtpComplete = (otpValue: string) => {
    if (otpValue.length === 6) {
      verifyOtpMutation.mutate(otpValue);
    }
  };

  // Allow manual verification through button click
  const handleVerifyOtp = () => {
    if (otp.length === 6) {
      verifyOtpMutation.mutate(otp);
    } else {
      toast({
        title: "Invalid OTP",
        description: "Please enter a valid 6-digit OTP",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 p-4">
      <Card className="w-full max-w-md mx-auto">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold text-center">
            {step === "phone" ? "Login with Phone" : "Verify OTP"}
          </CardTitle>
          <CardDescription className="text-center">
            {step === "phone"
              ? "Enter your phone number to receive a one-time password"
              : `We've sent a 6-digit code to ${phoneNumber}`}
          </CardDescription>
        </CardHeader>

        <CardContent>
          {step === "phone" ? (
            <form onSubmit={handlePhoneNumberSubmit} className="space-y-4">
              <div className="flex rounded-md border border-input overflow-hidden">
                <div className="bg-muted px-3 py-2 flex items-center text-sm">
                  <Phone className="h-4 w-4 mr-2" />
                  +91
                </div>
                <Input
                  type="tel"
                  placeholder="10 digit mobile number"
                  value={phoneNumber}
                  onChange={(e) => {
                    // Only allow numbers
                    const value = e.target.value.replace(/\D/g, "");
                    setPhoneNumber(value);
                  }}
                  className="flex-1 border-0 focus-visible:ring-0 focus-visible:ring-offset-0"
                  maxLength={10}
                />
              </div>

              <Button
                type="submit"
                className="w-full"
                disabled={requestOtpMutation.isPending || phoneNumber.length !== 10}
              >
                {requestOtpMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <ArrowRight className="mr-2 h-4 w-4" />
                )}
                Get OTP
              </Button>

              <div className="relative mt-2">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t border-muted-foreground/20" />
                </div>
                <div className="relative flex justify-center text-xs">
                  <span className="bg-card px-2 text-muted-foreground">or</span>
                </div>
              </div>

              <Button
                type="button"
                variant="outline"
                className="w-full mt-2"
                onClick={handleSkipLogin}
              >
                Skip for now
              </Button>
            </form>
          ) : (
            <div className="space-y-4">
              {displayOtp && (
                <Alert className="mb-3 border border-primary bg-primary/5 animate-pulse py-2 px-3">
                  <div className="flex items-center justify-center">
                    <LockKeyhole className="h-4 w-4 text-primary mr-1" />
                    <AlertTitle className="text-center font-semibold text-sm">OTP Code:</AlertTitle>
                    <p className="text-lg font-bold text-primary tracking-wider ml-2">
                      {displayOtp}
                    </p>
                  </div>
                </Alert>
              )}

              <OtpInput
                length={6}
                onComplete={handleOtpComplete}
                onChange={setOtp}
                value={otp}
                className="flex justify-center gap-2"
              />

              <Button
                onClick={handleVerifyOtp}
                className="w-full mt-4"
                disabled={verifyOtpMutation.isPending || otp.length !== 6}
              >
                {verifyOtpMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <ArrowRight className="mr-2 h-4 w-4" />
                )}
                Verify OTP
              </Button>

              {verifyOtpMutation.isPending && (
                <div className="flex justify-center my-4">
                  <Loader2 className="h-6 w-6 animate-spin text-primary" />
                </div>
              )}

              <div className="text-center text-sm">
                <Button
                  variant="link"
                  className="p-0"
                  onClick={() => {
                    requestOtpMutation.mutate();
                  }}
                  disabled={requestOtpMutation.isPending || verifyOtpMutation.isPending}
                >
                  Resend OTP
                </Button>
              </div>
            </div>
          )}
        </CardContent>

        <CardFooter className="flex justify-center">
          <p className="text-xs text-muted-foreground text-center">
            By continuing, you agree to our Terms of Service and Privacy Policy.
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}
