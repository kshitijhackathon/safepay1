import { 
  User, InsertUser, 
  Transaction, InsertTransaction,
  UpiRiskReport, InsertUpiRiskReport,
  ScamReport, InsertScamReport,
  PaymentMethod, InsertPaymentMethod,
  ChatMessage, InsertChatMessage,
  ChatFeedback, InsertChatFeedback,
  ScamType
} from "@shared/schema";

export interface IStorage {
  // User methods
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  getUserByPhoneNumber(phoneNumber: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  updateUser(id: number, userData: Partial<InsertUser>): Promise<User | undefined>;
  
  // Transaction methods
  getTransactionsByUserId(userId: number): Promise<Transaction[]>;
  createTransaction(transaction: InsertTransaction): Promise<Transaction>;
  
  // UPI Risk methods
  getUpiRiskByUpiId(upiId: string): Promise<UpiRiskReport | undefined>;
  updateUpiRiskScore(upiId: string): Promise<void>;
  
  // Scam Report methods
  getScamReportsByUpiId(upiId: string): Promise<ScamReport[]>;
  getScamReportsByUserId(userId: number): Promise<ScamReport[]>;
  getScamReportById(id: number): Promise<ScamReport | undefined>;
  createScamReport(report: InsertScamReport): Promise<ScamReport>;
  getMostCommonScamType(upiId: string): Promise<string>;
  
  // Payment methods
  getPaymentMethodsByUserId(userId: number): Promise<PaymentMethod[]>;
  getPaymentMethod(id: number): Promise<PaymentMethod | undefined>;
  createPaymentMethod(method: InsertPaymentMethod): Promise<PaymentMethod>;
  updatePaymentMethod(id: number, data: Partial<InsertPaymentMethod>): Promise<PaymentMethod | undefined>;
  deletePaymentMethod(id: number): Promise<boolean>;
  setDefaultPaymentMethod(userId: number, methodId: number): Promise<boolean>;
  
  // Chat methods
  getChatHistoryByUserId(userId: number): Promise<ChatMessage[]>;
  saveChatMessage(userId: number, message: { role: string; content: string }): Promise<ChatMessage>;
  saveChatFeedback(userId: number, messageId: number, feedback: { rating?: number; feedback?: string }): Promise<ChatFeedback>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private transactions: Map<number, Transaction>;
  private upiRiskReports: Map<number, UpiRiskReport>;
  private scamReports: Map<number, ScamReport>;
  private paymentMethods: Map<number, PaymentMethod>;
  private chatMessages: Map<number, ChatMessage>;
  private chatFeedbacks: Map<number, ChatFeedback>;
  
  private userIdCounter: number;
  private transactionIdCounter: number;
  private upiRiskIdCounter: number;
  private scamReportIdCounter: number;
  private paymentMethodIdCounter: number;
  private chatMessageIdCounter: number;
  private chatFeedbackIdCounter: number;

  constructor() {
    this.users = new Map();
    this.transactions = new Map();
    this.upiRiskReports = new Map();
    this.scamReports = new Map();
    this.paymentMethods = new Map();
    this.chatMessages = new Map();
    this.chatFeedbacks = new Map();
    
    this.userIdCounter = 1;
    this.transactionIdCounter = 1;
    this.upiRiskIdCounter = 1;
    this.scamReportIdCounter = 1;
    this.paymentMethodIdCounter = 1;
    this.chatMessageIdCounter = 1;
    this.chatFeedbackIdCounter = 1;
    
    // Add some initial data
    this.seedData();
  }

  // User methods
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username
    );
  }
  
  async getUserByPhoneNumber(phoneNumber: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.phoneNumber === phoneNumber
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.userIdCounter++;
    const createdAt = new Date();
    
    // Create user with all required fields, providing default values for optional fields
    const user: User = { 
      id, 
      username: insertUser.username,
      password: insertUser.password,
      phoneNumber: insertUser.phoneNumber,
      createdAt,
      name: insertUser.name || null,
      email: insertUser.email || null,
      address: insertUser.address || null,
      dateOfBirth: insertUser.dateOfBirth || null,
      profileCompleted: insertUser.profileCompleted || false,
      lastLogin: null,
      useBiometric: insertUser.useBiometric || false,
      usePin: insertUser.usePin || false,
      pin: insertUser.pin || null,
      deviceId: insertUser.deviceId || null
    };
    
    this.users.set(id, user);
    return user;
  }
  
  async updateUser(id: number, userData: Partial<InsertUser>): Promise<User | undefined> {
    const existingUser = await this.getUser(id);
    
    if (!existingUser) {
      return undefined;
    }
    
    const updatedUser: User = {
      ...existingUser,
      username: userData.username || existingUser.username,
      password: userData.password || existingUser.password,
      phoneNumber: userData.phoneNumber || existingUser.phoneNumber,
      name: userData.name !== undefined ? userData.name : existingUser.name,
      email: userData.email !== undefined ? userData.email : existingUser.email,
      address: userData.address !== undefined ? userData.address : existingUser.address,
      dateOfBirth: userData.dateOfBirth !== undefined ? userData.dateOfBirth : existingUser.dateOfBirth,
      profileCompleted: userData.profileCompleted !== undefined ? userData.profileCompleted : existingUser.profileCompleted,
      useBiometric: userData.useBiometric !== undefined ? userData.useBiometric : existingUser.useBiometric,
      usePin: userData.usePin !== undefined ? userData.usePin : existingUser.usePin,
      pin: userData.pin !== undefined ? userData.pin : existingUser.pin,
      deviceId: userData.deviceId !== undefined ? userData.deviceId : existingUser.deviceId
    };
    
    this.users.set(id, updatedUser);
    return updatedUser;
  }
  
  // Transaction methods
  async getTransactionsByUserId(userId: number): Promise<Transaction[]> {
    return Array.from(this.transactions.values()).filter(
      (tx) => tx.userId === userId
    );
  }
  
  async createTransaction(insertTransaction: InsertTransaction): Promise<Transaction> {
    const id = this.transactionIdCounter++;
    const timestamp = new Date();
    
    // Ensure all required fields are present with defaults if needed
    const transaction: Transaction = {
      id,
      userId: insertTransaction.userId,
      upiId: insertTransaction.upiId,
      amount: insertTransaction.amount,
      status: insertTransaction.status,
      timestamp,
      // Provide defaults for optional fields if not provided
      description: insertTransaction.description || 'Payment',
      currency: insertTransaction.currency || 'inr',
      transactionType: insertTransaction.transactionType || 'payment',
      paymentMethod: insertTransaction.paymentMethod || 'upi',
      paymentIntentId: insertTransaction.paymentIntentId || null
    };
    
    this.transactions.set(id, transaction);
    return transaction;
  }
  
  // UPI Risk methods
  async getUpiRiskByUpiId(upiId: string): Promise<UpiRiskReport | undefined> {
    return Array.from(this.upiRiskReports.values()).find(
      (report) => report.upiId === upiId
    );
  }
  
  async updateUpiRiskScore(upiId: string): Promise<void> {
    let riskReport = await this.getUpiRiskByUpiId(upiId);
    
    if (!riskReport) {
      // Create new report if none exists
      riskReport = {
        id: this.upiRiskIdCounter++,
        upiId,
        reportCount: 1,
        firstReportDate: new Date(),
        riskScore: 30, // Initial medium-low risk score
        statusVerified: false,
        lastChecked: new Date()
      };
      this.upiRiskReports.set(riskReport.id, riskReport);
    } else {
      // Update existing report
      riskReport.reportCount += 1;
      
      // Increase risk score as reports increase
      // Max score is 100
      riskReport.riskScore = Math.min(
        100,
        riskReport.riskScore + 10 + Math.floor(Math.random() * 10)
      );
      
      this.upiRiskReports.set(riskReport.id, riskReport);
    }
  }
  
  // Scam Report methods
  async getScamReportsByUpiId(upiId: string): Promise<ScamReport[]> {
    return Array.from(this.scamReports.values()).filter(
      (report) => report.upiId === upiId
    );
  }
  
  async getScamReportsByUserId(userId: number): Promise<ScamReport[]> {
    return Array.from(this.scamReports.values()).filter(
      (report) => report.userId === userId
    ).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()); // Newest first
  }
  
  async getScamReportById(id: number): Promise<ScamReport | undefined> {
    return this.scamReports.get(id);
  }
  
  async createScamReport(insertReport: InsertScamReport): Promise<ScamReport> {
    const id = this.scamReportIdCounter++;
    const timestamp = new Date();
    const report: ScamReport = { 
      ...insertReport, 
      id, 
      timestamp,
      description: insertReport.description || null,
      amountLost: insertReport.amountLost || null
    };
    this.scamReports.set(id, report);
    return report;
  }
  
  async getMostCommonScamType(upiId: string): Promise<string> {
    const reports = await this.getScamReportsByUpiId(upiId);
    
    if (reports.length === 0) return 'N/A';
    
    // Count scam types
    const typeCounts = new Map<string, number>();
    
    for (const report of reports) {
      const count = typeCounts.get(report.scamType) || 0;
      typeCounts.set(report.scamType, count + 1);
    }
    
    // Find most common
    let mostCommon = 'N/A';
    let maxCount = 0;
    
    typeCounts.forEach((count, type) => {
      if (count > maxCount) {
        mostCommon = type;
        maxCount = count;
      }
    });
    
    return mostCommon;
  }
  
  // Payment Method methods
  async getPaymentMethodsByUserId(userId: number): Promise<PaymentMethod[]> {
    return Array.from(this.paymentMethods.values()).filter(
      (method) => method.userId === userId
    );
  }
  
  async getPaymentMethod(id: number): Promise<PaymentMethod | undefined> {
    return this.paymentMethods.get(id);
  }
  
  async createPaymentMethod(method: InsertPaymentMethod): Promise<PaymentMethod> {
    const id = this.paymentMethodIdCounter++;
    const createdAt = new Date();
    
    // Make sure only one payment method is default if this is set as default
    const isDefaultValue = method.isDefault ?? false;
    if (isDefaultValue) {
      const userMethods = await this.getPaymentMethodsByUserId(method.userId);
      // Reset all other payment methods to not default
      for (const existingMethod of userMethods) {
        if (existingMethod.isDefault) {
          existingMethod.isDefault = false;
          this.paymentMethods.set(existingMethod.id, existingMethod);
        }
      }
    }
    
    const paymentMethod: PaymentMethod = {
      id,
      userId: method.userId,
      type: method.type,
      name: method.name,
      isDefault: isDefaultValue,
      createdAt,
      // Ensure optional fields have null value if not provided
      upiId: method.upiId || null,
      cardNumber: method.cardNumber || null, 
      expiryDate: method.expiryDate || null,
      accountNumber: method.accountNumber || null,
      ifscCode: method.ifscCode || null
    };
    
    this.paymentMethods.set(id, paymentMethod);
    return paymentMethod;
  }
  
  async updatePaymentMethod(id: number, data: Partial<InsertPaymentMethod>): Promise<PaymentMethod | undefined> {
    const existingMethod = await this.getPaymentMethod(id);
    
    if (!existingMethod) {
      return undefined;
    }
    
    // If this method is being set as default, unset others
    if (data.isDefault && !existingMethod.isDefault) {
      const userMethods = await this.getPaymentMethodsByUserId(existingMethod.userId);
      // Reset all other payment methods to not default
      for (const otherMethod of userMethods) {
        if (otherMethod.id !== id && otherMethod.isDefault) {
          otherMethod.isDefault = false;
          this.paymentMethods.set(otherMethod.id, otherMethod);
        }
      }
    }
    
    const updatedMethod: PaymentMethod = {
      ...existingMethod,
      type: data.type || existingMethod.type,
      name: data.name || existingMethod.name,
      isDefault: data.isDefault !== undefined ? data.isDefault : existingMethod.isDefault,
      upiId: data.upiId !== undefined ? data.upiId : existingMethod.upiId,
      cardNumber: data.cardNumber !== undefined ? data.cardNumber : existingMethod.cardNumber,
      expiryDate: data.expiryDate !== undefined ? data.expiryDate : existingMethod.expiryDate,
      accountNumber: data.accountNumber !== undefined ? data.accountNumber : existingMethod.accountNumber,
      ifscCode: data.ifscCode !== undefined ? data.ifscCode : existingMethod.ifscCode
    };
    
    this.paymentMethods.set(id, updatedMethod);
    return updatedMethod;
  }
  
  async deletePaymentMethod(id: number): Promise<boolean> {
    const method = await this.getPaymentMethod(id);
    
    if (!method) {
      return false;
    }
    
    // If this was a default method, we need to set another one as default
    if (method.isDefault) {
      const userMethods = await this.getPaymentMethodsByUserId(method.userId);
      const otherMethods = userMethods.filter(m => m.id !== id);
      
      if (otherMethods.length > 0) {
        // Set the first available method as default
        const newDefault = otherMethods[0];
        newDefault.isDefault = true;
        this.paymentMethods.set(newDefault.id, newDefault);
      }
    }
    
    return this.paymentMethods.delete(id);
  }
  
  async setDefaultPaymentMethod(userId: number, methodId: number): Promise<boolean> {
    const method = await this.getPaymentMethod(methodId);
    
    if (!method || method.userId !== userId) {
      return false;
    }
    
    // Already default, nothing to do
    if (method.isDefault) {
      return true;
    }
    
    // Reset current default
    const userMethods = await this.getPaymentMethodsByUserId(userId);
    for (const existingMethod of userMethods) {
      if (existingMethod.id !== methodId && existingMethod.isDefault) {
        existingMethod.isDefault = false;
        this.paymentMethods.set(existingMethod.id, existingMethod);
      }
    }
    
    // Set new default
    method.isDefault = true;
    this.paymentMethods.set(methodId, method);
    
    return true;
  }
  
  // Chat methods
  async getChatHistoryByUserId(userId: number): Promise<ChatMessage[]> {
    return Array.from(this.chatMessages.values())
      .filter(message => message.userId === userId)
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }

  async saveChatMessage(userId: number, message: { role: string; content: string }): Promise<ChatMessage> {
    const id = this.chatMessageIdCounter++;
    const timestamp = new Date();
    
    const chatMessage: ChatMessage = {
      id,
      userId,
      role: message.role,
      content: message.content,
      timestamp,
      metadata: null
    };
    
    this.chatMessages.set(id, chatMessage);
    return chatMessage;
  }

  async saveChatFeedback(userId: number, messageId: number, feedback: { rating?: number; feedback?: string }): Promise<ChatFeedback> {
    const id = this.chatFeedbackIdCounter++;
    const timestamp = new Date();
    
    const chatFeedback: ChatFeedback = {
      id,
      userId,
      messageId,
      rating: feedback.rating || null,
      feedback: feedback.feedback || null,
      timestamp
    };
    
    this.chatFeedbacks.set(id, chatFeedback);
    return chatFeedback;
  }
  
  // Seed data for demo purposes
  private seedData() {
    // Add a user
    const demoUser: InsertUser = {
      username: 'rahul',
      password: 'password123',
      phoneNumber: '9876543210',
      name: 'Rahul Sharma'
    };
    this.createUser(demoUser);
    
    // Add some transactions
    const transactions: InsertTransaction[] = [
      {
        userId: 1,
        description: 'City Supermarket',
        upiId: 'citysupermarket@upi',
        amount: -85000, // In paise (850.00 INR)
        transactionType: 'payment',
        status: 'Completed'
      },
      {
        userId: 1,
        description: 'Aman Sharma',
        upiId: 'amansharma@upi',
        amount: 120000, // In paise (1,200.00 INR)
        transactionType: 'payment',
        status: 'Received'
      },
      {
        userId: 1,
        description: 'Metro Grocery',
        upiId: 'metrogrocery@upi',
        amount: -45000, // In paise (450.00 INR)
        transactionType: 'payment',
        status: 'Completed'
      }
    ];
    
    transactions.forEach(tx => this.createTransaction(tx));
    
    // Add a high-risk UPI ID
    const riskId = this.upiRiskIdCounter++;
    const firstReportDate = new Date(Date.now() - 60 * 86400000); // 60 days ago
    
    this.upiRiskReports.set(riskId, {
      id: riskId,
      upiId: 'onlineshopping123@upi',
      reportCount: 15,
      riskScore: 75,
      statusVerified: false,
      firstReportDate,
      lastChecked: new Date()
    });
    
    // Add some scam reports
    const scamReports: InsertScamReport[] = [
      {
        userId: 1,
        upiId: 'onlineshopping123@upi',
        scamType: ScamType.Unknown, // Using Unknown for previous "Fake Products"
        amountLost: 250000, // 2,500.00 INR
        description: 'I ordered products online through this UPI ID but never received the items. The seller stopped responding after payment.'
      },
      {
        userId: 1,
        upiId: 'onlineshopping123@upi',
        scamType: ScamType.Unknown, // Using Unknown for previous "Fake Products"
        amountLost: 350000, // 3,500.00 INR
        description: 'Ordered a smartphone but received a fake product.'
      },
      {
        userId: 1,
        upiId: 'onlineshopping123@upi',
        scamType: ScamType.Banking, // Using Banking for previous "Fraud"
        amountLost: 450000, // 4,500.00 INR
        description: 'They took money and blocked my number.'
      }
    ];
    
    scamReports.forEach(report => {
      const id = this.scamReportIdCounter++;
      this.scamReports.set(id, {
        userId: report.userId,
        upiId: report.upiId,
        scamType: report.scamType,
        amountLost: report.amountLost || null,
        description: report.description || null,
        id,
        timestamp: new Date(Date.now() - Math.floor(Math.random() * 30) * 86400000) // Random date in last 30 days
      });
    });
    
    // Add some payment methods
    const paymentMethods: InsertPaymentMethod[] = [
      {
        userId: 1,
        type: 'upi',
        name: 'Personal UPI',
        isDefault: true,
        upiId: 'rahul@upi'
      },
      {
        userId: 1,
        type: 'card',
        name: 'HDFC Debit Card',
        isDefault: false,
        cardNumber: '1234', // Last 4 digits only
        expiryDate: '12/25'
      },
      {
        userId: 1,
        type: 'bank_account',
        name: 'ICICI Savings Account',
        isDefault: false,
        accountNumber: '5678', // Last 4 digits only
        ifscCode: 'ICIC0001234'
      }
    ];
    
    paymentMethods.forEach(method => this.createPaymentMethod(method));
    
    // Add some initial chat messages
    const welcomeMessage = {
      userId: 1,
      role: 'system',
      content: 'Welcome to UPI SafeGuard Chat Support! How can I help you today?'
    };
    this.saveChatMessage(1, welcomeMessage);
    
    const userMessage = {
      userId: 1, 
      role: 'user',
      content: 'How can I check if a UPI ID is safe?'
    };
    this.saveChatMessage(1, userMessage);
    
    const assistantMessage = {
      userId: 1,
      role: 'assistant',
      content: 'To check if a UPI ID is safe, you can use our "Scan QR" feature to scan the QR code or enter the UPI ID manually. Our system will analyze it and show you a risk level (Low, Medium, or High) along with any reports from other users. If the UPI ID has been reported multiple times, we recommend avoiding making payments to it.'
    };
    this.saveChatMessage(1, assistantMessage);
  }
}

import { eq, and, desc, sql } from 'drizzle-orm';
import { db } from './db';
import * as schema from '@shared/schema';
import { 
  users, transactions, upiRiskReports, scamReports,
  paymentMethods, chatMessages, chatFeedbacks
} from '@shared/schema';

export class DatabaseStorage implements IStorage {
  async getUser(id: number): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user || undefined;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user || undefined;
  }

  async getUserByPhoneNumber(phoneNumber: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.phoneNumber, phoneNumber));
    return user || undefined;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db.insert(users).values(insertUser).returning();
    return user;
  }

  async updateUser(id: number, userData: Partial<InsertUser>): Promise<User | undefined> {
    const [updatedUser] = await db.update(users)
      .set(userData)
      .where(eq(users.id, id))
      .returning();
    return updatedUser;
  }

  async getTransactionsByUserId(userId: number): Promise<Transaction[]> {
    return await db.select().from(transactions)
      .where(eq(transactions.userId, userId))
      .orderBy(desc(transactions.timestamp));
  }

  async createTransaction(insertTransaction: InsertTransaction): Promise<Transaction> {
    const [transaction] = await db.insert(transactions)
      .values(insertTransaction)
      .returning();
    return transaction;
  }

  async getUpiRiskByUpiId(upiId: string): Promise<UpiRiskReport | undefined> {
    const [report] = await db.select().from(upiRiskReports)
      .where(eq(upiRiskReports.upiId, upiId));
    return report;
  }

  async updateUpiRiskScore(upiId: string): Promise<void> {
    // Implementation depends on how you want to update the risk score
    try {
      const existingReport = await this.getUpiRiskByUpiId(upiId);
      
      if (existingReport) {
        // Just update the report count
        await db.update(upiRiskReports)
          .set({ 
            reportCount: existingReport.reportCount + 1,
            // Don't try to update lastChecked column if it doesn't exist
            // lastChecked: new Date()
          })
          .where(eq(upiRiskReports.upiId, upiId));
      } else {
        // Create a new report with default values but without lastChecked
        await db.insert(upiRiskReports).values({
          upiId,
          riskScore: 0,
          reportCount: 0,
          // Don't include lastChecked
          firstReportDate: new Date(),
          statusVerified: false
        });
      }
    } catch (error) {
      console.error("Error updating UPI risk score:", error);
      // Continue without failing
    }
  }

  async getScamReportsByUpiId(upiId: string): Promise<ScamReport[]> {
    return await db.select().from(scamReports)
      .where(eq(scamReports.upiId, upiId))
      .orderBy(desc(scamReports.timestamp));
  }
  
  async getScamReportsByUserId(userId: number): Promise<ScamReport[]> {
    return await db.select().from(scamReports)
      .where(eq(scamReports.userId, userId))
      .orderBy(desc(scamReports.timestamp));
  }
  
  async getScamReportById(id: number): Promise<ScamReport | undefined> {
    const [report] = await db.select().from(scamReports)
      .where(eq(scamReports.id, id));
    return report;
  }

  async createScamReport(insertReport: InsertScamReport): Promise<ScamReport> {
    const [report] = await db.insert(scamReports)
      .values(insertReport)
      .returning();
      
    // Update the UPI risk report
    try {
      const existingReport = await this.getUpiRiskByUpiId(insertReport.upiId);
      if (existingReport) {
        await db.update(upiRiskReports)
          .set({ 
            reportCount: existingReport.reportCount + 1,
            riskScore: Math.min(100, existingReport.riskScore + 10) // Increment risk score
            // No longer trying to update lastChecked column
          })
          .where(eq(upiRiskReports.upiId, insertReport.upiId));
      } else {
        // Create a new risk report without lastChecked
        await db.insert(upiRiskReports).values({
          upiId: insertReport.upiId,
          riskScore: 30, // Initial risk score for a new report
          reportCount: 1,
          firstReportDate: new Date(),
          statusVerified: false
        });
      }
    } catch (error) {
      console.error("Error updating UPI risk report during scam report creation:", error);
      // Continue without failing
    }
    
    return report;
  }

  async getMostCommonScamType(upiId: string): Promise<string> {
    // In a real implementation, this would use SQL aggregation
    // For simplicity, we'll load all reports and count in JS
    const reports = await this.getScamReportsByUpiId(upiId);
    
    if (!reports.length) return 'Unknown';
    
    const typeCounts: Record<string, number> = {};
    let maxCount = 0;
    let commonType = 'Unknown';
    
    for (const report of reports) {
      const count = (typeCounts[report.scamType] || 0) + 1;
      typeCounts[report.scamType] = count;
      
      if (count > maxCount) {
        maxCount = count;
        commonType = report.scamType;
      }
    }
    
    return commonType;
  }

  async getPaymentMethodsByUserId(userId: number): Promise<PaymentMethod[]> {
    return await db.select().from(paymentMethods)
      .where(eq(paymentMethods.userId, userId));
  }

  async getPaymentMethod(id: number): Promise<PaymentMethod | undefined> {
    const [method] = await db.select().from(paymentMethods)
      .where(eq(paymentMethods.id, id));
    return method;
  }

  async createPaymentMethod(method: InsertPaymentMethod): Promise<PaymentMethod> {
    // If this is set as default, unset any existing defaults
    if (method.isDefault) {
      await db.update(paymentMethods)
        .set({ isDefault: false })
        .where(and(
          eq(paymentMethods.userId, method.userId),
          eq(paymentMethods.isDefault, true)
        ));
    }
    
    const [paymentMethod] = await db.insert(paymentMethods)
      .values(method)
      .returning();
    return paymentMethod;
  }

  async updatePaymentMethod(id: number, data: Partial<InsertPaymentMethod>): Promise<PaymentMethod | undefined> {
    // If this is being set as default, unset any existing defaults
    if (data.isDefault) {
      const [method] = await db.select().from(paymentMethods)
        .where(eq(paymentMethods.id, id));
      
      if (method) {
        await db.update(paymentMethods)
          .set({ isDefault: false })
          .where(and(
            eq(paymentMethods.userId, method.userId),
            eq(paymentMethods.isDefault, true),
            sql`${paymentMethods.id} != ${id}`
          ));
      }
    }
    
    const [updatedMethod] = await db.update(paymentMethods)
      .set(data)
      .where(eq(paymentMethods.id, id))
      .returning();
    return updatedMethod;
  }

  async deletePaymentMethod(id: number): Promise<boolean> {
    const result = await db.delete(paymentMethods)
      .where(eq(paymentMethods.id, id))
      .returning({ id: paymentMethods.id });
    return result.length > 0;
  }

  async setDefaultPaymentMethod(userId: number, methodId: number): Promise<boolean> {
    // First, unset all defaults for this user
    await db.update(paymentMethods)
      .set({ isDefault: false })
      .where(and(
        eq(paymentMethods.userId, userId),
        eq(paymentMethods.isDefault, true)
      ));
    
    // Then set the new default
    const result = await db.update(paymentMethods)
      .set({ isDefault: true })
      .where(and(
        eq(paymentMethods.userId, userId),
        eq(paymentMethods.id, methodId)
      ))
      .returning({ id: paymentMethods.id });
    
    return result.length > 0;
  }

  async getChatHistoryByUserId(userId: number): Promise<ChatMessage[]> {
    return await db.select().from(chatMessages)
      .where(eq(chatMessages.userId, userId))
      .orderBy(desc(chatMessages.timestamp));
  }

  async saveChatMessage(userId: number, message: { role: string; content: string }): Promise<ChatMessage> {
    const [chatMessage] = await db.insert(chatMessages)
      .values({
        userId,
        role: message.role,
        content: message.content,
        timestamp: new Date()
      })
      .returning();
    return chatMessage;
  }

  async saveChatFeedback(userId: number, messageId: number, feedback: { rating?: number; feedback?: string }): Promise<ChatFeedback> {
    const [chatFeedback] = await db.insert(chatFeedbacks)
      .values({
        userId,
        messageId, 
        rating: feedback.rating,
        feedback: feedback.feedback,
        timestamp: new Date()
      })
      .returning();
    return chatFeedback;
  }
}

// For development and testing, we'll check if a DATABASE_URL is available
// If not, we'll fall back to MemStorage
// Otherwise, use DatabaseStorage with PostgreSQL

let storageImpl: IStorage;

try {
  // Check if we have a database connection
  if (process.env.DATABASE_URL) {
    console.log('Using PostgreSQL database storage');
    storageImpl = new DatabaseStorage();
  } else {
    console.log('No DATABASE_URL found, falling back to in-memory storage');
    storageImpl = new MemStorage();
  }
} catch (error) {
  console.error('Error initializing database storage:', error);
  console.log('Falling back to in-memory storage');
  storageImpl = new MemStorage();
}

export const storage = storageImpl;
