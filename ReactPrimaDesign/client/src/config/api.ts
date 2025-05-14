const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "https://your-app-name.onrender.com";

export const API_ENDPOINTS = {
  // Auth endpoints
  AUTH: {
    LOGIN: `${API_BASE_URL}/api/auth/login`,
    REGISTER: `${API_BASE_URL}/api/auth/register`,
    REQUEST_OTP: `${API_BASE_URL}/api/auth/request-otp`,
  },

  // UPI endpoints
  UPI: {
    CHECK: `${API_BASE_URL}/api/upi/check`,
    VALIDATE: `${API_BASE_URL}/api/upi/validate`,
  },

  // QR endpoints
  QR: {
    SCAN: `${API_BASE_URL}/api/ml/qr-scan`,
    VERIFY: `${API_BASE_URL}/api/verify-qr`,
  },

  // Voice endpoints
  VOICE: {
    PROCESS: `${API_BASE_URL}/api/process-voice`,
    ANALYZE: `${API_BASE_URL}/api/analyze-voice`,
  },

  // Scam news endpoints
  SCAM_NEWS: {
    LATEST: `${API_BASE_URL}/api/scam-news`,
  },

  // User endpoints
  USER: {
    PROFILE: `${API_BASE_URL}/api/users`,
    PAYMENT_METHODS: `${API_BASE_URL}/api/payment-methods`,
  },

  // Payment endpoints
  PAYMENT: {
    CREATE_INTENT: `${API_BASE_URL}/api/create-payment-intent`,
  },

  // Legal help endpoints
  LEGAL: {
    VOICE_TO_TEXT: `${API_BASE_URL}/api/voice-to-text`,
    GENERATE_COMPLAINT: `${API_BASE_URL}/api/generate-complaint-email`,
    SEND_COMPLAINT: `${API_BASE_URL}/api/send-complaint-email`,
  },
};
