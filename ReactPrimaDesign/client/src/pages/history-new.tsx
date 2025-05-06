import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { 
  format, 
  formatDistanceToNow, 
  isToday, 
  isYesterday 
} from 'date-fns';
import { 
  Search, 
  ArrowLeft, 
  ReceiptText, 
  ShoppingBag, 
  Calendar, 
  Package, 
  User, 
  CreditCard, 
  Smartphone, 
  Building, 
  IndianRupee,
  FilterX,
  Loader2,
  AlertCircle,
  ChevronRight,
  ArrowUpRight,
  Clock
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useAuthState } from '@/hooks/use-auth-state';
import { apiRequest } from '@/lib/queryClient';
import { Badge } from '@/components/ui/badge';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle 
} from '@/components/ui/dialog';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious
} from '@/components/ui/pagination';

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
  merchantName?: string;
  appUsed?: string;
}

// Payment app colors for visual differentiation
const getPaymentAppColor = (app?: string): string => {
  if (!app) return 'bg-gray-400';
  
  const appLower = app.toLowerCase();
  if (appLower.includes('google') || appLower.includes('gpay')) return 'bg-blue-600';
  if (appLower.includes('phone') || appLower.includes('phonepe')) return 'bg-purple-600';
  if (appLower.includes('paytm')) return 'bg-sky-500';
  if (appLower.includes('amazon')) return 'bg-orange-500';
  if (appLower.includes('whatsapp')) return 'bg-green-500';
  
  return 'bg-primary';
};

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

// Format timestamp to human-readable date/time
const formatTimestamp = (timestamp: string) => {
  const date = new Date(timestamp);
  
  if (isToday(date)) {
    return {
      date: 'Today',
      time: format(date, 'h:mm a') // e.g. "2:30 PM"
    };
  } else if (isYesterday(date)) {
    return {
      date: 'Yesterday',
      time: format(date, 'h:mm a')
    };
  } else {
    return {
      date: format(date, 'dd MMM yyyy'), // e.g. "15 Apr 2025"
      time: format(date, 'h:mm a')
    };
  }
};

export default function History() {
  const [, setLocation] = useLocation();
  const [searchQuery, setSearchQuery] = useState('');
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [filteredTransactions, setFilteredTransactions] = useState<Transaction[]>([]);
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [filter, setFilter] = useState("all");
  const [isDetailOpen, setIsDetailOpen] = useState(false);
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
        tx.paymentMethod?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tx.merchantName?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tx.appUsed?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    
    // Apply type filter
    if (filter !== "all") {
      result = result.filter(tx => tx.transactionType === filter);
    }
    
    setFilteredTransactions(result);
  }, [transactions, searchQuery, filter]);
  
  // Group transactions by date
  const groupTransactionsByDate = (transactions: Transaction[]) => {
    const grouped: Record<string, Transaction[]> = {};
    
    // Calculate pagination
    const startIndex = (currentPage - 1) * itemsPerPage;
    const paginatedTransactions = transactions.slice(startIndex, startIndex + itemsPerPage);
    
    paginatedTransactions.forEach(transaction => {
      const date = new Date(transaction.timestamp);
      let groupKey: string;
      
      if (isToday(date)) {
        groupKey = 'Today';
      } else if (isYesterday(date)) {
        groupKey = 'Yesterday';
      } else {
        groupKey = format(date, 'dd MMM yyyy'); // e.g. "15 Apr 2025"
      }
      
      if (!grouped[groupKey]) {
        grouped[groupKey] = [];
      }
      
      grouped[groupKey].push(transaction);
    });
    
    return grouped;
  };
  
  const groupedTransactions = groupTransactionsByDate(filteredTransactions);
  const totalPages = Math.ceil(filteredTransactions.length / itemsPerPage);
  
  const handleTransactionClick = (transaction: Transaction) => {
    setSelectedTransaction(transaction);
    setIsDetailOpen(true);
  };
  
  // Render
  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-background p-4 border-b">
        <div className="flex items-center mb-4">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setLocation('/home')}
            className="mr-3"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <h1 className="text-xl font-bold">Transaction History</h1>
        </div>
        
        {/* Search bar */}
        <div className="flex items-center border rounded-lg p-2 mb-4 bg-muted/20">
          <Search className="w-4 h-4 text-muted-foreground mr-2" />
          <Input
            type="text"
            placeholder="Search transactions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="border-0 p-0 h-6 focus-visible:ring-0 bg-transparent"
          />
          {searchQuery && (
            <Button variant="ghost" size="sm" onClick={() => setSearchQuery('')} className="p-1 h-auto">
              <FilterX className="w-4 h-4" />
            </Button>
          )}
        </div>
        
        {/* Filters */}
        <div className="flex justify-between items-center mb-2">
          <Select
            value={filter}
            onValueChange={setFilter}
          >
            <SelectTrigger className="w-[140px] h-8 text-sm">
              <SelectValue placeholder="Filter by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Transactions</SelectItem>
              <SelectItem value="payment">Payments</SelectItem>
              <SelectItem value="refund">Refunds</SelectItem>
              <SelectItem value="transfer">Transfers</SelectItem>
            </SelectContent>
          </Select>
          
          <div className="text-sm text-muted-foreground">
            {filteredTransactions.length} transactions
          </div>
        </div>
      </div>
      
      {/* Main content */}
      <div className="flex-1 p-4">
        {isLoading ? (
          // Loading state
          <div className="space-y-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="flex items-center space-x-4">
                <Skeleton className="h-12 w-12 rounded-full" />
                <div className="space-y-2">
                  <Skeleton className="h-4 w-[200px]" />
                  <Skeleton className="h-4 w-[160px]" />
                </div>
                <Skeleton className="h-4 w-16 ml-auto" />
              </div>
            ))}
          </div>
        ) : error ? (
          // Error state
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <AlertCircle className="w-12 h-12 text-destructive mb-4" />
            <h3 className="text-lg font-medium mb-2">Failed to load transactions</h3>
            <p className="text-muted-foreground mb-6">{error}</p>
            <Button onClick={() => window.location.reload()} variant="outline">
              Retry
            </Button>
          </div>
        ) : filteredTransactions.length === 0 ? (
          // Empty state
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <ReceiptText className="w-12 h-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No transactions found</h3>
            <p className="text-muted-foreground mb-4">
              {searchQuery 
                ? "Try adjusting your search or filters" 
                : "Once you make payments, they'll appear here"}
            </p>
            {searchQuery && (
              <Button onClick={() => setSearchQuery('')} variant="outline">
                Clear search
              </Button>
            )}
          </div>
        ) : (
          // Transactions list
          <div className="space-y-6">
            {Object.entries(groupedTransactions).map(([date, transactions]) => (
              <div key={date}>
                <h2 className="text-sm font-medium text-muted-foreground mb-3">{date}</h2>
                <div className="space-y-3">
                  {transactions.map((transaction) => {
                    const { time } = formatTimestamp(transaction.timestamp);
                    return (
                      <Card 
                        key={transaction.id} 
                        className="hover:bg-accent/50 cursor-pointer transition-colors"
                        onClick={() => handleTransactionClick(transaction)}
                      >
                        <CardContent className="p-3 flex items-center">
                          {/* Transaction icon or app logo */}
                          <div className={`w-10 h-10 rounded-full ${getPaymentAppColor(transaction.appUsed)} flex items-center justify-center mr-4 flex-shrink-0`}>
                            <PaymentMethodIcon method={transaction.paymentMethod || 'upi'} />
                          </div>
                          
                          {/* Transaction details */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center">
                              <p className="font-medium truncate">{transaction.description}</p>
                              {transaction.status === 'completed' && (
                                <Badge variant="outline" className="ml-2 text-xs bg-green-50 border-green-200 text-green-700">
                                  Completed
                                </Badge>
                              )}
                              {transaction.status === 'pending' && (
                                <Badge variant="outline" className="ml-2 text-xs bg-yellow-50 border-yellow-200 text-yellow-700">
                                  Pending
                                </Badge>
                              )}
                            </div>
                            <div className="flex items-center text-xs text-muted-foreground mt-1">
                              <span className="truncate">{transaction.upiId}</span>
                              <span className="mx-1.5 w-1 h-1 bg-muted-foreground/50 rounded-full"></span>
                              <span>{time}</span>
                              {transaction.appUsed && (
                                <>
                                  <span className="mx-1.5 w-1 h-1 bg-muted-foreground/50 rounded-full"></span>
                                  <span>{transaction.appUsed}</span>
                                </>
                              )}
                            </div>
                          </div>
                          
                          {/* Amount */}
                          <div className="flex items-center ml-4">
                            <p className={`font-semibold ${transaction.amount < 0 ? 'text-red-600' : 'text-green-600'}`}>
                              {formatAmount(transaction.amount, transaction.currency)}
                            </p>
                            <ChevronRight className="h-4 w-4 text-muted-foreground ml-2" />
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </div>
            ))}
            
            {/* Pagination */}
            {totalPages > 1 && (
              <Pagination className="my-6">
                <PaginationContent>
                  <PaginationItem>
                    <Button 
                      variant="outline"
                      size="icon"
                      onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                      disabled={currentPage === 1}
                    >
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </PaginationItem>
                  
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                    <PaginationItem key={page}>
                      <PaginationLink
                        onClick={() => setCurrentPage(page)}
                        isActive={page === currentPage}
                      >
                        {page}
                      </PaginationLink>
                    </PaginationItem>
                  ))}
                  
                  <PaginationItem>
                    <Button 
                      variant="outline"
                      size="icon"
                      onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                      disabled={currentPage === totalPages}
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </PaginationItem>
                </PaginationContent>
              </Pagination>
            )}
          </div>
        )}
      </div>
      
      {/* Transaction Details Dialog */}
      <Dialog open={isDetailOpen} onOpenChange={setIsDetailOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Transaction Details</DialogTitle>
            <DialogDescription>
              Complete information about this transaction
            </DialogDescription>
          </DialogHeader>
          
          {selectedTransaction && (
            <div className="space-y-6">
              {/* Transaction amount */}
              <div className="flex flex-col items-center justify-center py-4">
                <div className={`w-16 h-16 rounded-full ${getPaymentAppColor(selectedTransaction.appUsed)} flex items-center justify-center mb-4`}>
                  <PaymentMethodIcon method={selectedTransaction.paymentMethod || 'upi'} />
                </div>
                <p className={`text-2xl font-bold ${selectedTransaction.amount < 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {formatAmount(selectedTransaction.amount, selectedTransaction.currency)}
                </p>
                <p className="text-muted-foreground mt-1">
                  {selectedTransaction.description}
                </p>
              </div>
              
              {/* Transaction details */}
              <div className="bg-muted/30 rounded-lg p-4 space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Date & Time</span>
                  <span className="font-medium">
                    {format(new Date(selectedTransaction.timestamp), 'dd MMM yyyy, h:mm a')}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-muted-foreground">UPI ID</span>
                  <span className="font-medium">{selectedTransaction.upiId}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <span className={`font-medium ${
                    selectedTransaction.status === 'completed' ? 'text-green-600' : 
                    selectedTransaction.status === 'failed' ? 'text-red-600' : 'text-yellow-600'
                  }`}>
                    {selectedTransaction.status.charAt(0).toUpperCase() + selectedTransaction.status.slice(1)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Payment Method</span>
                  <span className="font-medium">
                    {selectedTransaction.paymentMethod.toUpperCase()}
                    {selectedTransaction.appUsed && ` (${selectedTransaction.appUsed})`}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Transaction ID</span>
                  <span className="font-medium text-sm">{selectedTransaction.id}</span>
                </div>
              </div>
              
              {/* Transaction actions */}
              <div className="flex gap-2">
                <Button className="flex-1" variant="outline">Download Receipt</Button>
                <Button className="flex-1" onClick={() => setIsDetailOpen(false)}>Close</Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}