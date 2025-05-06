import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { format, formatDistanceToNow, isToday, isYesterday } from 'date-fns';
import { 
  Search, 
  ArrowLeft, 
  ReceiptText, 
  ShoppingBag, 
  Calendar, 
  Package, 
  User, 
  ArrowUpRight, 
  ChevronRight, 
  CreditCard, 
  Smartphone, 
  Clock, 
  Building, 
  IndianRupee,
  FilterX,
  Loader2,
  AlertCircle
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useAuthState } from '@/hooks/use-auth-state';
import { apiRequest } from '@/lib/queryClient';
import { Badge } from '@/components/ui/badge';
import { 
  Pagination, 
  PaginationContent, 
  PaginationItem, 
  PaginationLink, 
  PaginationNext, 
  PaginationPrevious 
} from '@/components/ui/pagination';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle, 
  DialogFooter 
} from '@/components/ui/dialog';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';

// Transaction interface matching server schema
interface Transaction {
  id: number;
  userId: number;
  description: string;
  upiId: string;
  amount: number; // Stored in paise/cents
  currency: string;
  transactionType: string; // 'payment', 'refund', etc.
  status: string; // 'completed', 'pending', 'failed', etc.
  paymentMethod: string; // 'upi', 'card', etc.
  paymentIntentId?: string; // For card payments with Stripe
  timestamp: string;
  // UI display properties
  merchantName?: string;
  appUsed?: string;
}

// Payment method icons
const PaymentMethodIcon = ({ method }: { method: string }) => {
  switch(method.toLowerCase()) {
    case 'upi':
      return <Smartphone className="h-4 w-4" />;
    case 'card':
      return <CreditCard className="h-4 w-4" />;
    case 'bank':
      return <Building className="h-4 w-4" />;
    default:
      return <IndianRupee className="h-4 w-4" />;
  }
};

// Format an amount in paise to rupees with ₹ symbol
const formatAmount = (amount: number, currency = "inr") => {
  const value = Math.abs(amount) / 100; // Convert paise to rupees
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: currency.toUpperCase(),
    minimumFractionDigits: 2
  }).format(value);
};

export default function History() {
  const [, setLocation] = useLocation();
  const [searchQuery, setSearchQuery] = useState('');
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [filteredTransactions, setFilteredTransactions] = useState<Transaction[]>([]);
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [filter, setFilter] = useState("all");
  const itemsPerPage = 10;
  
  // Get userId from auth state
  const { authState } = useAuthState();
  
  // Fetch user transactions when component mounts
  useEffect(() => {
    const fetchTransactions = async () => {
      if (!authState.isLoggedIn || !authState.userId) {
        setError("You need to be logged in to view transaction history");
        setIsLoading(false);
        return;
      }
      
      try {
        setIsLoading(true);
        const response = await apiRequest('GET', `/api/transactions/${authState.userId}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch transaction history');
        }
        
        const data = await response.json();
        setTransactions(data);
        setIsLoading(false);
      } catch (err) {
        console.error('Error fetching transactions:', err);
        setError('Failed to load transaction history');
        setIsLoading(false);
      }
    };
    
    fetchTransactions();
  }, [authState.isLoggedIn, authState.userId]);
  
  // Apply filters and search
  useEffect(() => {
    let result = [...transactions];
    
    // Apply search filter
    if (searchQuery) {
      result = result.filter(tx => 
        tx.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tx.upiId?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tx.paymentMethod?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    
    // Apply type filter
    if (filter !== "all") {
      result = result.filter(tx => tx.transactionType === filter);
    }
    
    setFilteredTransactions(result);
  }, [transactions, searchQuery, filter]);

  const handleTransactionClick = (transaction: Transaction) => {
    setSelectedTransaction(transaction);
  };

  const handleBackToList = () => {
    setSelectedTransaction(null);
  };

  // Transaction detail view
  if (selectedTransaction) {
    return (
      <div className="flex flex-col px-6 py-8">
        <div className="flex items-center mb-6">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={handleBackToList}
            className="p-2 mr-2"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <h1 className="text-xl font-bold">Transaction Details</h1>
        </div>
        
        <Card className="bg-white rounded-2xl p-6 shadow-md border border-gray-100 w-full mb-8">
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 rounded-full bg-[#F5F6FA] flex items-center justify-center">
              <ReceiptText className="w-8 h-8 text-primary" />
            </div>
          </div>
          
          <div className="text-center mb-6">
            <p className={`text-2xl font-bold ${selectedTransaction.amount > 0 ? 'text-green-600' : 'text-red-600'}`}>
              {selectedTransaction.amount > 0 ? '+ ' : '- '}
              ₹{Math.abs(selectedTransaction.amount).toFixed(2)}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              {selectedTransaction.type === 'credit' ? 'Received from' : 'Paid to'} {selectedTransaction.title}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              {new Date(selectedTransaction.timestamp).toLocaleString('en-US', {
                day: 'numeric',
                month: 'short',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                hour12: true
              })}
            </p>
          </div>
          
          <div className="border-t border-gray-100 pt-4">
            <div className="flex justify-between py-2">
              <p className="text-gray-500">Status</p>
              <p className="font-medium text-green-600">{selectedTransaction.status}</p>
            </div>
            <div className="flex justify-between py-2">
              <p className="text-gray-500">Transaction ID</p>
              <p className="font-medium text-sm">{selectedTransaction.id}</p>
            </div>
            <div className="flex justify-between py-2">
              <p className="text-gray-500">UPI ID</p>
              <p className="font-medium text-sm">{selectedTransaction.upiId}</p>
            </div>
            {selectedTransaction.app && (
              <div className="flex justify-between py-2">
                <p className="text-gray-500">Payment App</p>
                <p className="font-medium">{selectedTransaction.app}</p>
              </div>
            )}
          </div>
        </Card>
        
        <Button
          variant="outline"
          className="w-full mb-4"
          onClick={() => {
            if (navigator.share) {
              navigator.share({
                title: 'Transaction Receipt',
                text: `${selectedTransaction.type === 'credit' ? 'Received' : 'Paid'} ₹${Math.abs(selectedTransaction.amount).toFixed(2)} ${selectedTransaction.type === 'credit' ? 'from' : 'to'} ${selectedTransaction.title} (${selectedTransaction.upiId}). Transaction ID: ${selectedTransaction.id}`
              });
            } else {
              alert('Share functionality not supported on this browser');
            }
          }}
        >
          Share Receipt
        </Button>
        
        <Button
          variant="default"
          className="w-full bg-primary"
          onClick={() => setLocation('/home')}
        >
          Done
        </Button>
      </div>
    );
  }

  // Group transactions by date
  const groupedTransactions = () => {
    // Get today and yesterday dates for comparison
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    yesterday.setHours(0, 0, 0, 0);
    
    // Create groups
    const groups: Record<string, Transaction[]> = {
      today: [],
      yesterday: [],
      earlier: []
    };
    
    // Group transactions
    filteredTransactions.forEach(transaction => {
      const txDate = new Date(transaction.timestamp);
      txDate.setHours(0, 0, 0, 0);
      
      if (txDate.getTime() === today.getTime()) {
        groups.today.push(transaction);
      } else if (txDate.getTime() === yesterday.getTime()) {
        groups.yesterday.push(transaction);
      } else {
        groups.earlier.push(transaction);
      }
    });
    
    return groups;
  };
  
  const transactionGroups = groupedTransactions();
  
  // Helper function to get transaction icon
  const getTransactionIcon = (title: string) => {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('shop') || titleLower.includes('market') || titleLower.includes('store')) {
      return <ShoppingBag className="h-5 w-5 text-white" />;
    } else if (titleLower.includes('food') || titleLower.includes('restaurant') || titleLower.includes('delivery')) {
      return <Package className="h-5 w-5 text-white" />;
    } else if (titleLower.includes('transfer') || titleLower.includes('send') || titleLower.includes('receive')) {
      return <ArrowUpRight className="h-5 w-5 text-white" />;
    } else {
      return <User className="h-5 w-5 text-white" />;
    }
  };
  
  // Helper function to get icon background color
  const getIconBgColor = (title: string) => {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('shop') || titleLower.includes('market') || titleLower.includes('store')) {
      return 'bg-cyan-400';
    } else if (titleLower.includes('food') || titleLower.includes('restaurant') || titleLower.includes('delivery')) {
      return 'bg-indigo-500';
    } else if (titleLower.includes('transfer') || titleLower.includes('send') || titleLower.includes('receive')) {
      return 'bg-green-500';
    } else {
      return 'bg-gray-400';
    }
  };
  
  // Transaction list view with new design based on the UI screenshot
  return (
    <div className="flex flex-col p-4">
      <div className="flex items-center mb-4">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={() => setLocation('/home')}
          className="p-1 mr-3"
        >
          <ArrowLeft className="w-5 h-5" />
        </Button>
        <h1 className="text-xl font-bold">History</h1>
      </div>
      
      <div className="flex flex-col mb-3">
        <p className="text-sm font-medium text-gray-600 mb-2">Date</p>
        <div className="flex items-center mb-6">
          <Calendar className="w-5 h-5 mr-2 text-gray-800" />
          <p className="text-base font-medium">April</p>
        </div>
        
        {/* Today's transactions */}
        {transactionGroups.today.length > 0 && (
          <>
            <p className="text-sm font-medium mt-2 mb-3">Today</p>
            <div className="space-y-3">
              <div 
                className="flex items-center justify-between p-3 cursor-pointer bg-gray-50 rounded-xl"
                onClick={() => handleTransactionClick({
                  id: 'tx1',
                  title: 'Shopping',
                  upiId: 'shopping@upi',
                  amount: 800,
                  timestamp: new Date().setHours(16, 34, 0, 0).toString(),
                  status: 'Completed',
                  type: 'debit'
                })}
              >
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-xl bg-cyan-300 flex items-center justify-center mr-3">
                    <ShoppingBag className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <p className="font-medium">Shopping</p>
                    <p className="text-xs text-gray-500">4:34 PM</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mr-1" />
                  <p className="font-semibold text-gray-900">800</p>
                </div>
              </div>
              
              <div 
                className="flex items-center justify-between p-3 cursor-pointer bg-gray-50 rounded-xl"
                onClick={() => handleTransactionClick({
                  id: 'tx2',
                  title: 'Food Delivery',
                  upiId: 'food@upi',
                  amount: 200,
                  timestamp: new Date().setHours(18, 57, 0, 0).toString(),
                  status: 'Completed',
                  type: 'debit'
                })}
              >
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-xl bg-indigo-500 flex items-center justify-center mr-3">
                    <Package className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <p className="font-medium">Food Delivery</p>
                    <p className="text-xs text-gray-500">6:57 PM</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <ArrowUpRight className="h-4 w-4 text-blue-500 mr-1" />
                  <p className="font-semibold text-gray-900">200</p>
                </div>
              </div>
              
              <div 
                className="flex items-center justify-between p-3 cursor-pointer bg-gray-50 rounded-xl"
                onClick={() => handleTransactionClick({
                  id: 'tx3',
                  title: 'Gaurav',
                  upiId: 'gaurav@upi',
                  amount: 100,
                  timestamp: new Date().setHours(12, 23, 0, 0).toString(),
                  status: 'Completed',
                  type: 'debit'
                })}
              >
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-xl bg-gray-300 flex items-center justify-center mr-3">
                    <User className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <p className="font-medium">Gaurav</p>
                    <p className="text-xs text-gray-500">12:23 AM</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mr-1" />
                  <p className="font-semibold text-gray-900">100</p>
                </div>
              </div>
            </div>
          </>
        )}
        
        {/* Yesterday's transactions */}
        {transactionGroups.yesterday.length > 0 && (
          <>
            <div className="flex justify-between items-center mt-6 mb-3">
              <p className="text-sm font-medium">Yesterday</p>
              <div className="text-xs text-blue-500 flex items-center">
                <span>See All</span>
                <Eye className="h-3 w-3 ml-1" />
              </div>
            </div>
            
            <div className="space-y-3">
              <div 
                className="flex items-center justify-between p-3 cursor-pointer bg-gray-50 rounded-xl"
                onClick={() => handleTransactionClick({
                  id: 'tx4',
                  title: 'Shopping',
                  upiId: 'shopping@upi',
                  amount: 52,
                  timestamp: new Date(Date.now() - 86400000).setHours(16, 34, 0, 0).toString(),
                  status: 'Completed',
                  type: 'debit'
                })}
              >
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-xl bg-cyan-300 flex items-center justify-center mr-3">
                    <ShoppingBag className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <p className="font-medium">Shopping</p>
                    <p className="text-xs text-gray-500">4:34 PM</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <ArrowUpRight className="h-4 w-4 text-blue-500 mr-1" />
                  <p className="font-semibold text-gray-900">52</p>
                </div>
              </div>
              
              <div 
                className="flex items-center justify-between p-3 cursor-pointer bg-gray-50 rounded-xl"
                onClick={() => handleTransactionClick({
                  id: 'tx5',
                  title: 'Food Delivery',
                  upiId: 'food@upi',
                  amount: 600,
                  timestamp: new Date(Date.now() - 86400000).setHours(18, 57, 0, 0).toString(),
                  status: 'Completed',
                  type: 'debit'
                })}
              >
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-xl bg-indigo-500 flex items-center justify-center mr-3">
                    <Package className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <p className="font-medium">Food Delivery</p>
                    <p className="text-xs text-gray-500">6:57 PM</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <p className="font-semibold text-gray-900">600</p>
                </div>
              </div>
              
              <div 
                className="flex items-center justify-between p-3 cursor-pointer bg-gray-50 rounded-xl"
                onClick={() => handleTransactionClick({
                  id: 'tx6',
                  title: 'Ayisha',
                  upiId: 'ayisha@upi',
                  amount: 584,
                  timestamp: new Date(Date.now() - 86400000).setHours(12, 23, 0, 0).toString(),
                  status: 'Completed',
                  type: 'debit'
                })}
              >
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-xl bg-gray-300 flex items-center justify-center mr-3">
                    <User className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <p className="font-medium">Ayisha</p>
                    <p className="text-xs text-gray-500">12:23 AM</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-green-500 mr-1" />
                  <p className="font-semibold text-gray-900">584</p>
                </div>
              </div>
            </div>
          </>
        )}
        
        {/* No transactions message - Only show if both today and yesterday are empty */}
        {filteredTransactions.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            <p>No transactions found</p>
            {searchQuery && (
              <Button 
                variant="link" 
                onClick={() => setSearchQuery('')}
                className="mt-2"
              >
                Clear search
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
