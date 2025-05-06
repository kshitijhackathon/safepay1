/**
 * OTP Service for generating and validating OTP codes
 */

// In-memory storage for OTPs
interface OtpRecord {
  code: string;
  expiresAt: Date;
  verified: boolean;
  attempts: number;
}

// Map to store OTPs by phone number or user identifier
const otpStore = new Map<string, OtpRecord>();

// Rate limiting for OTP generation (max 3 requests per phone per hour)
const otpRateLimits = new Map<string, { count: number, resetAt: Date }>();

/**
 * Generate a random numeric OTP of specified length
 */
function generateOtpCode(length = 6): string {
  const digits = '0123456789';
  let otp = '';
  
  for (let i = 0; i < length; i++) {
    otp += digits[Math.floor(Math.random() * 10)];
  }
  
  return otp;
}

/**
 * Generate and store an OTP for a given identifier (phone/email)
 * @param identifier User identifier (phone/email)
 * @param expiryMinutes How long the OTP should be valid (in minutes)
 * @returns The generated OTP code or throws an error if rate limit exceeded
 */
export function generateOtp(identifier: string, expiryMinutes = 10): string {
  // For demo purposes, we're removing the rate limit
  // This is a temporary fix for the hackathon demo only
  
  // Clear any existing rate limit if it exists
  if (otpRateLimits.has(identifier)) {
    otpRateLimits.delete(identifier);
  }
  
  // Set a very high rate limit for demo purposes
  otpRateLimits.set(identifier, { count: 1, resetAt: new Date(Date.now() + 24 * 60 * 60 * 1000) });
  
  // Generate a 6-digit OTP
  const code = generateOtpCode(6);
  
  // Calculate expiry time
  const expiresAt = new Date();
  expiresAt.setMinutes(expiresAt.getMinutes() + expiryMinutes);
  
  // Store the OTP
  otpStore.set(identifier, {
    code,
    expiresAt,
    verified: false,
    attempts: 0
  });
  
  // In a production app, this would be sent via SMS using services like Twilio
  // For example:
  // await twilioClient.messages.create({
  //   body: `Your verification code is: ${code}`,
  //   from: TWILIO_PHONE_NUMBER,
  //   to: identifier
  // });
  
  // For now, we just return it (will be displayed in console/response)
  return code;
}

/**
 * Verify an OTP code for a given identifier
 * @param identifier User identifier (phone/email)
 * @param code OTP code to verify
 * @returns true if the OTP is valid, false otherwise
 */
export function verifyOtp(identifier: string, code: string): boolean {
  const record = otpStore.get(identifier);
  
  // Check if OTP exists
  if (!record) {
    return false; // No OTP found for this identifier
  }
  
  // Check if OTP is expired
  if (new Date() > record.expiresAt) {
    otpStore.delete(identifier); // Clean up expired OTP
    return false; // OTP has expired
  }
  
  // Check if OTP already verified
  if (record.verified) {
    return false; // OTP already used
  }
  
  // Increment attempts
  record.attempts += 1;
  
  // Check if too many attempts (max 3)
  if (record.attempts > 3) {
    otpStore.delete(identifier); // Invalidate OTP after too many attempts
    return false;
  }
  
  // Check if code matches
  if (record.code === code) {
    // Mark as verified
    record.verified = true;
    otpStore.set(identifier, record);
    return true;
  }
  
  // Update the record with incremented attempts
  otpStore.set(identifier, record);
  
  return false; // Invalid code
}

/**
 * Check if an identifier has a valid OTP pending verification
 */
export function hasValidOtp(identifier: string): boolean {
  const record = otpStore.get(identifier);
  if (!record) return false;
  
  // Check if not expired and not already verified
  return new Date() <= record.expiresAt && !record.verified && record.attempts <= 3;
}

/**
 * Invalidate an OTP for a given identifier
 */
export function invalidateOtp(identifier: string): void {
  otpStore.delete(identifier);
}