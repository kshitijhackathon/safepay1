export type UpiRisk = {
  riskPercentage: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  reports: number;
  age: string;
  reportedFor: string;
};

// The risk levels are: Low (<30%), Medium (30-70%), High (>70%)
export function getRiskLevel(percentage: number): 'Low' | 'Medium' | 'High' {
  if (percentage < 30) return 'Low';
  if (percentage < 70) return 'Medium';
  return 'High';
}

export function getRiskColor(level: 'Low' | 'Medium' | 'High'): string {
  switch (level) {
    case 'Low':
      return '#43A047'; // Success/Green
    case 'Medium':
      return '#FFB300'; // Warning/Yellow
    case 'High':
      return '#E53935'; // Error/Red
    default:
      return '#5164BF'; // Primary/Blue
  }
}

// Function to validate a UPI ID format
export function isValidUpiId(upiId: string): boolean {
  // Basic validation - UPI IDs typically follow a pattern like username@upiProvider
  const upiPattern = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9]+$/;
  return upiPattern.test(upiId);
}

// Function to check if a phone number is valid (10 digits for India)
export function isValidPhoneNumber(phone: string): boolean {
  const phonePattern = /^[6-9]\d{9}$/;
  return phonePattern.test(phone);
}
