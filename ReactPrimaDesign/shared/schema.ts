import { pgTable, text, serial, integer, timestamp, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Enum for different types of scams
export enum ScamType {
  Banking = "Banking Scam",
  Lottery = "Lottery Scam",
  KYC = "KYC Verification Scam",
  Refund = "Refund Scam",
  Phishing = "Phishing Attempt",
  Reward = "Reward Scam",
  JobScam = "Job Scam",
  Investment = "Investment Scam",
  Unknown = "Unknown Scam"
}

// User table
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  phoneNumber: text("phone_number").notNull().unique(),
  name: text("name"),
  email: text("email"),
  address: text("address"),
  dateOfBirth: text("date_of_birth"),
  profileCompleted: boolean("profile_completed").default(false),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  lastLogin: timestamp("last_login"),
  useBiometric: boolean("use_biometric").default(false),
  usePin: boolean("use_pin").default(false),
  pin: text("pin"),
  deviceId: text("device_id"),
});

// Transaction table
export const transactions = pgTable("transactions", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  description: text("description").notNull().default('Payment'),
  upiId: text("upi_id").notNull(),
  amount: integer("amount").notNull(), // Stored in paise/cents
  currency: text("currency").notNull().default('inr'),
  transactionType: text("transaction_type").notNull().default('payment'), // 'payment', 'refund', etc.
  status: text("status").notNull(), // 'completed', 'pending', 'failed', etc.
  paymentMethod: text("payment_method").default('upi'), // 'upi', 'card', etc.
  paymentIntentId: text("payment_intent_id"), // For card payments with Stripe
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

// UPI Risk reports
export const upiRiskReports = pgTable("upi_risk_reports", {
  id: serial("id").primaryKey(),
  upiId: text("upi_id").notNull(),
  reportCount: integer("report_count").notNull().default(0),
  firstReportDate: timestamp("first_report_date").defaultNow().notNull(),
  riskScore: integer("risk_score").notNull().default(0), // 0-100
  statusVerified: boolean("status_verified").default(false),
  lastChecked: timestamp("last_checked"),
});

// Scam reports
export const scamReports = pgTable("scam_reports", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  upiId: text("upi_id").notNull(),
  scamType: text("scam_type").notNull(), // Stores values from ScamType enum
  amountLost: integer("amount_lost"), // Stored in paise/cents
  description: text("description"),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

// Payment methods
export const paymentMethods = pgTable("payment_methods", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  type: text("type").notNull(), // 'upi', 'card', 'bank_account'
  name: text("name").notNull(), // Card name, Bank name, etc.
  isDefault: boolean("is_default").default(false),
  // For UPI
  upiId: text("upi_id"),
  // For cards (only store last 4 digits for security)
  cardNumber: text("card_number"), // Only last 4 digits
  expiryDate: text("expiry_date"),
  // For bank accounts
  accountNumber: text("account_number"), // Only last 4 digits
  ifscCode: text("ifsc_code"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Chat messages
export const chatMessages = pgTable("chat_messages", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  role: text("role").notNull(), // 'user', 'assistant', 'system'
  content: text("content").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
  metadata: text("metadata"),
});

// Chat feedback
export const chatFeedbacks = pgTable("chat_feedbacks", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull(),
  messageId: integer("message_id").references(() => chatMessages.id),
  rating: integer("rating"), // 1-5 star rating
  feedback: text("feedback"),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).omit({
  id: true,
  createdAt: true,
});

export const insertTransactionSchema = createInsertSchema(transactions).omit({
  id: true,
  timestamp: true,
});

export const insertUpiRiskReportSchema = createInsertSchema(upiRiskReports).omit({
  id: true,
  firstReportDate: true,
});

// Create base schema and then extend to validate scamType
const baseScamReportSchema = createInsertSchema(scamReports).omit({
  id: true,
  timestamp: true,
});

// Extend the schema to properly validate scamType as an enum
export const insertScamReportSchema = baseScamReportSchema.extend({
  scamType: z.nativeEnum(ScamType)
});

export const insertPaymentMethodSchema = createInsertSchema(paymentMethods).omit({
  id: true,
  createdAt: true,
});

export const insertChatMessageSchema = createInsertSchema(chatMessages).omit({
  id: true,
  timestamp: true,
});

export const insertChatFeedbackSchema = createInsertSchema(chatFeedbacks).omit({
  id: true,
  timestamp: true,
});

// Export types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertTransaction = z.infer<typeof insertTransactionSchema>;
export type Transaction = typeof transactions.$inferSelect;

export type InsertUpiRiskReport = z.infer<typeof insertUpiRiskReportSchema>;
export type UpiRiskReport = typeof upiRiskReports.$inferSelect;

export type InsertScamReport = z.infer<typeof insertScamReportSchema>;
export type ScamReport = typeof scamReports.$inferSelect;

export type InsertPaymentMethod = z.infer<typeof insertPaymentMethodSchema>;
export type PaymentMethod = typeof paymentMethods.$inferSelect;

export type InsertChatMessage = z.infer<typeof insertChatMessageSchema>;
export type ChatMessage = typeof chatMessages.$inferSelect;

export type InsertChatFeedback = z.infer<typeof insertChatFeedbackSchema>;
export type ChatFeedback = typeof chatFeedbacks.$inferSelect;
