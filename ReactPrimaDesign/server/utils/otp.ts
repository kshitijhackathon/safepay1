/**
 * OTP Utilities
 * Provides functions for generating, validating, and managing OTPs for authentication
 */

/**
 * Generate a 6-digit numeric OTP
 */
export function generateOtp(): string {
  // Generate a random 6-digit number
  const otp = Math.floor(100000 + Math.random() * 900000).toString();
  return otp;
}

/**
 * Validate an OTP against a reference OTP
 */
export function validateOtp(inputOtp: string, referenceOtp: string): boolean {
  return inputOtp === referenceOtp;
}

/**
 * Check if an OTP has expired
 */
export function isOtpExpired(createdAt: Date, expirationMinutes: number = 5): boolean {
  const expirationTime = new Date(createdAt.getTime() + expirationMinutes * 60 * 1000);
  return new Date() > expirationTime;
}

/**
 * Hash an OTP for secure storage (in a real system)
 * This is a placeholder - in production, use a proper crypto library
 */
export function hashOtp(otp: string): string {
  // In production, use a proper hashing algorithm
  // For demo purposes, this is a simple hash that should NOT be used in production
  return Buffer.from(otp + 'salt').toString('base64');
}

/**
 * Send OTP via email
 * In production, integrate with an email service like SendGrid
 */
export async function sendOtpViaEmail(email: string, otp: string): Promise<boolean> {
  try {
    // In production, integrate with an email service
    console.log(`[DEMO] OTP ${otp} would be sent to ${email}`);
    return true;
  } catch (error) {
    console.error('Error sending OTP via email:', error);
    return false;
  }
}

/**
 * Send OTP via SMS
 * In production, integrate with an SMS gateway like Twilio
 */
export async function sendOtpViaSms(phoneNumber: string, otp: string): Promise<boolean> {
  try {
    // In production, integrate with an SMS gateway
    console.log(`[DEMO] OTP ${otp} would be sent to ${phoneNumber} via SMS`);
    return true;
  } catch (error) {
    console.error('Error sending OTP via SMS:', error);
    return false;
  }
}