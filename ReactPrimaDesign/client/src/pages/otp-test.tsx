import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

export default function OtpTest() {
  const [phoneNumber, setPhoneNumber] = useState("");
  const [otpResponse, setOtpResponse] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleRequestOtp = async () => {
    if (!phoneNumber) {
      toast({
        title: "Error",
        description: "Please enter a phone number",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch("/api/auth/request-otp", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ phoneNumber }),
      });

      const data = await response.json();
      setOtpResponse(data);
      
      console.log("OTP Response:", data);
      
      if (data.otp) {
        toast({
          title: "OTP Generated",
          description: `Your OTP is: ${data.otp}`,
        });
      }
    } catch (error) {
      console.error("Error requesting OTP:", error);
      toast({
        title: "Error",
        description: "Failed to request OTP",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 p-4">
      <Card className="w-full max-w-md mx-auto">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold text-center">
            OTP Test Page
          </CardTitle>
          <CardDescription className="text-center">
            Test the OTP request and display functionality
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-4">
          <div className="flex rounded-md border border-input overflow-hidden">
            <div className="bg-muted px-3 py-2 flex items-center text-sm">
              +91
            </div>
            <Input
              type="tel"
              placeholder="10 digit mobile number"
              value={phoneNumber}
              onChange={(e) => {
                // Only allow numbers
                const value = e.target.value.replace(/\D/g, '');
                setPhoneNumber(value);
              }}
              className="flex-1 border-0 focus-visible:ring-0 focus-visible:ring-offset-0"
              maxLength={10}
            />
          </div>
          
          <Button 
            onClick={handleRequestOtp} 
            className="w-full" 
            disabled={isLoading}
          >
            {isLoading ? "Requesting..." : "Request OTP"}
          </Button>
          
          {otpResponse && (
            <div className="bg-blue-50 p-3 rounded-md mt-4">
              <h3 className="font-medium mb-2">API Response:</h3>
              <pre className="bg-gray-100 p-2 rounded text-sm overflow-auto">
                {JSON.stringify(otpResponse, null, 2)}
              </pre>
            </div>
          )}
        </CardContent>
        
        <CardFooter className="flex justify-center">
          <p className="text-xs text-muted-foreground text-center">
            This is a test page to debug OTP functionality
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}