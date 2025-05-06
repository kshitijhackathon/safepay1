import React, { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";

interface OtpInputProps {
  length?: number;
  onComplete?: (otp: string) => void;
  className?: string;
  value?: string;
  onChange?: (otp: string) => void;
}

export function OtpInput({
  length = 6,
  onComplete,
  className,
  value = "",
  onChange,
}: OtpInputProps) {
  const [otp, setOtp] = useState<string[]>(Array(length).fill(""));
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  useEffect(() => {
    // Focus first input on mount
    if (inputRefs.current[0]) {
      inputRefs.current[0].focus();
    }
  }, []);

  useEffect(() => {
    // Update internal state when value prop changes
    if (value) {
      const newOtp = value.split("").slice(0, length);
      setOtp([...newOtp, ...Array(length - newOtp.length).fill("")]);
    }
  }, [value, length]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>, index: number) => {
    const value = e.target.value;

    if (isNaN(Number(value))) return;

    const newOtp = [...otp];

    // Take only the last digit if multiple are pasted
    newOtp[index] = value.slice(-1);
    setOtp(newOtp);

    // Call onChange if provided
    if (onChange) {
      onChange(newOtp.join(""));
    }

    // If value is entered and not the last input, focus next
    if (value && index < length - 1) {
      inputRefs.current[index + 1]?.focus();
    }

    // Check if OTP is complete
    if (newOtp.every((val) => val) && onComplete) {
      onComplete(newOtp.join(""));
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, index: number) => {
    // On backspace, clear current input and focus previous input
    if (e.key === "Backspace") {
      if (otp[index]) {
        // If current input has a value, clear it
        const newOtp = [...otp];
        newOtp[index] = "";
        setOtp(newOtp);
        if (onChange) {
          onChange(newOtp.join(""));
        }
      } else if (index > 0) {
        // If current input is empty, focus previous input
        inputRefs.current[index - 1]?.focus();
      }
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
    e.preventDefault();
    const pasteData = e.clipboardData.getData("text");

    if (isNaN(Number(pasteData))) return;

    // Fill OTP inputs from pasted data
    const newOtp = [...otp];
    for (let i = 0; i < Math.min(length, pasteData.length); i++) {
      newOtp[i] = pasteData[i];
    }

    setOtp(newOtp);

    // Call onChange if provided
    if (onChange) {
      onChange(newOtp.join(""));
    }

    // Focus the first empty input or the last one
    const firstEmptyIndex = newOtp.findIndex((val) => !val);
    const focusIndex = firstEmptyIndex === -1 ? length - 1 : firstEmptyIndex;
    inputRefs.current[focusIndex]?.focus();

    // Check if OTP is complete
    if (newOtp.every((val) => val) && onComplete) {
      onComplete(newOtp.join(""));
    }
  };

  return (
    <div className={cn("flex justify-between gap-2", className)}>
      {Array.from({ length }).map((_, index) => (
        <input
          key={index}
          type="text"
          inputMode="numeric"
          maxLength={1}
          ref={(el) => (inputRefs.current[index] = el)}
          value={otp[index]}
          onChange={(e) => handleChange(e, index)}
          onKeyDown={(e) => handleKeyDown(e, index)}
          onPaste={handlePaste}
          className={cn(
            "w-10 h-12 rounded-xl bg-[#F5F6FA] border border-gray-200 text-center text-xl font-bold",
            otp[index] && "border-primary",
            "focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary",
            "sm:w-11 sm:h-13" // Slightly larger on bigger screens
          )}
        />
      ))}
    </div>
  );
}
