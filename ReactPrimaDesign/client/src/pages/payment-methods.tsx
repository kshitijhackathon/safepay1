import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { useQuery, useMutation } from '@tanstack/react-query';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { 
  ArrowLeft,
  CreditCard, 
  Building2, 
  Wallet,
  Plus,
  Trash2,
  CheckCircle2,
  Circle
} from 'lucide-react';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { apiRequest, queryClient } from '@/lib/queryClient';

// Payment method form schema
const paymentMethodSchema = z.object({
  type: z.enum(['upi', 'card', 'bank_account']),
  name: z.string().min(2, { message: 'Name must be at least 2 characters' }),
  // Conditionally required fields based on type
  upiId: z.string().optional()
    .refine(val => val === undefined || val === '' || /^[a-zA-Z0-9._-]+@[a-zA-Z0-9]+$/.test(val), {
      message: 'Please enter a valid UPI ID'
    }),
  cardNumber: z.string().optional()
    .refine(val => val === undefined || val === '' || /^\d{4}$/.test(val), {
      message: 'Please enter the last 4 digits of your card'
    }),
  expiryDate: z.string().optional()
    .refine(val => val === undefined || val === '' || /^\d{2}\/\d{2}$/.test(val), {
      message: 'Please enter a valid expiry date (MM/YY)'
    }),
  accountNumber: z.string().optional()
    .refine(val => val === undefined || val === '' || /^\d{4}$/.test(val), {
      message: 'Please enter the last 4 digits of your account number'
    }),
  ifscCode: z.string().optional()
    .refine(val => val === undefined || val === '' || /^[A-Z]{4}0[A-Z0-9]{6}$/.test(val), {
      message: 'Please enter a valid IFSC code'
    }),
});

// Type for the form values
type PaymentMethodFormValues = z.infer<typeof paymentMethodSchema>;

// Function to get appropriate icon for payment method type
function getPaymentMethodIcon(type: string) {
  switch (type) {
    case 'upi':
      return <Wallet className="h-5 w-5" />;
    case 'card':
      return <CreditCard className="h-5 w-5" />;
    case 'bank_account':
      return <Building2 className="h-5 w-5" />;
    default:
      return <CreditCard className="h-5 w-5" />;
  }
}

// Function to get human-readable type name
function getTypeName(type: string) {
  switch (type) {
    case 'upi':
      return 'UPI';
    case 'card':
      return 'Card';
    case 'bank_account':
      return 'Bank Account';
    default:
      return type;
  }
}

export default function PaymentMethods() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState<any>(null);

  // Get user ID from URL or localStorage (in a real app you'd get this from auth context)
  const userId = 1; // For demo purposes

  // Fetch payment methods
  const { data: paymentMethods, isLoading } = useQuery({
    queryKey: ['/api/payment-methods', userId],
    queryFn: async () => {
      const res = await fetch(`/api/payment-methods/${userId}`);
      if (!res.ok) throw new Error('Failed to load payment methods');
      return res.json();
    },
  });

  // Add payment method mutation
  const addPaymentMethodMutation = useMutation({
    mutationFn: async (data: PaymentMethodFormValues & { userId: number }) => {
      return apiRequest('POST', '/api/payment-methods', data);
    },
    onSuccess: () => {
      toast({
        title: 'Payment method added',
        description: 'Your payment method has been added successfully.',
      });
      setIsAddDialogOpen(false);
      // Invalidate query to refetch data
      queryClient.invalidateQueries({ queryKey: ['/api/payment-methods', userId] });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to add payment method',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  // Delete payment method mutation
  const deletePaymentMethodMutation = useMutation({
    mutationFn: async (methodId: number) => {
      return apiRequest('DELETE', `/api/payment-methods/${methodId}`);
    },
    onSuccess: () => {
      toast({
        title: 'Payment method deleted',
        description: 'Your payment method has been deleted successfully.',
      });
      setIsDeleteDialogOpen(false);
      // Invalidate query to refetch data
      queryClient.invalidateQueries({ queryKey: ['/api/payment-methods', userId] });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to delete payment method',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  // Set default payment method mutation
  const setDefaultPaymentMethodMutation = useMutation({
    mutationFn: async (methodId: number) => {
      return apiRequest('POST', `/api/payment-methods/${userId}/set-default/${methodId}`);
    },
    onSuccess: () => {
      toast({
        title: 'Default method updated',
        description: 'Your default payment method has been updated.',
      });
      // Invalidate query to refetch data
      queryClient.invalidateQueries({ queryKey: ['/api/payment-methods', userId] });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to update default method',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  // Form definition using react-hook-form
  const form = useForm<PaymentMethodFormValues>({
    resolver: zodResolver(paymentMethodSchema),
    defaultValues: {
      type: 'upi',
      name: '',
      upiId: '',
      cardNumber: '',
      expiryDate: '',
      accountNumber: '',
      ifscCode: '',
    },
  });

  // Get the selected payment method type
  const watchType = form.watch('type');

  // Form submission handler
  function onSubmit(data: PaymentMethodFormValues) {
    addPaymentMethodMutation.mutate({ ...data, userId });
  }

  // Function to confirm deletion of a payment method
  function confirmDelete(method: any) {
    setSelectedMethod(method);
    setIsDeleteDialogOpen(true);
  }

  // Function to set a payment method as default
  function setAsDefault(methodId: number) {
    setDefaultPaymentMethodMutation.mutate(methodId);
  }

  if (isLoading) {
    return (
      <div className="p-6 flex justify-center items-center min-h-[50vh]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="flex flex-col px-6 py-8">
      <div className="flex items-center mb-6">
        <Button 
          variant="ghost" 
          size="icon" 
          className="mr-2" 
          onClick={() => setLocation('/account')}
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-2xl font-bold">Payment Methods</h1>
      </div>

      <div className="mb-4">
        <Button 
          onClick={() => setIsAddDialogOpen(true)}
          className="w-full flex items-center justify-center mb-6"
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Payment Method
        </Button>

        {paymentMethods && paymentMethods.length > 0 ? (
          <div className="space-y-4">
            {paymentMethods.map((method: any) => (
              <Card key={method.id} className="p-4 relative">
                <div className="flex items-start justify-between">
                  <div className="flex items-start">
                    <div className="bg-primary/10 p-2 rounded mr-3">
                      {getPaymentMethodIcon(method.type)}
                    </div>
                    <div>
                      <h3 className="font-semibold">{method.name}</h3>
                      <p className="text-sm text-gray-500">{getTypeName(method.type)}</p>
                      {method.type === 'upi' && method.upiId && (
                        <p className="text-xs mt-1">{method.upiId}</p>
                      )}
                      {method.type === 'card' && method.cardNumber && (
                        <p className="text-xs mt-1">{method.cardNumber}</p>
                      )}
                      {method.type === 'bank_account' && method.accountNumber && (
                        <p className="text-xs mt-1">{method.accountNumber}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button 
                      variant="ghost" 
                      size="icon"
                      onClick={() => confirmDelete(method)}
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => method.isDefault ? null : setAsDefault(method.id)}
                      disabled={method.isDefault}
                    >
                      {method.isDefault ? (
                        <CheckCircle2 className="h-4 w-4 text-primary" />
                      ) : (
                        <Circle className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>
                {method.isDefault && (
                  <div className="absolute top-1 right-1 bg-primary text-white text-xs px-2 py-0.5 rounded">
                    Default
                  </div>
                )}
              </Card>
            ))}
          </div>
        ) : (
          <Card className="p-6 text-center">
            <p className="text-gray-500">No payment methods added yet.</p>
          </Card>
        )}
      </div>

      {/* Add Payment Method Dialog */}
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Payment Method</DialogTitle>
          </DialogHeader>
          
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              <FormField
                control={form.control}
                name="type"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Payment Type</FormLabel>
                    <Select 
                      onValueChange={field.onChange} 
                      defaultValue={field.value}
                    >
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select payment type" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="upi">UPI</SelectItem>
                        <SelectItem value="card">Debit/Credit Card</SelectItem>
                        <SelectItem value="bank_account">Bank Account</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Name</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g. My Personal UPI" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {/* UPI ID field - shown only for UPI type */}
              {watchType === 'upi' && (
                <FormField
                  control={form.control}
                  name="upiId"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>UPI ID</FormLabel>
                      <FormControl>
                        <Input placeholder="example@upi" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              )}

              {/* Card fields - shown only for Card type */}
              {watchType === 'card' && (
                <>
                  <FormField
                    control={form.control}
                    name="cardNumber"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Last 4 digits of Card</FormLabel>
                        <FormControl>
                          <Input placeholder="1234" maxLength={4} {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="expiryDate"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Expiry Date</FormLabel>
                        <FormControl>
                          <Input placeholder="MM/YY" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </>
              )}

              {/* Bank Account fields - shown only for Bank Account type */}
              {watchType === 'bank_account' && (
                <>
                  <FormField
                    control={form.control}
                    name="accountNumber"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Last 4 digits of Account</FormLabel>
                        <FormControl>
                          <Input placeholder="1234" maxLength={4} {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="ifscCode"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>IFSC Code</FormLabel>
                        <FormControl>
                          <Input placeholder="SBIN0123456" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </>
              )}

              <Button 
                type="submit" 
                className="w-full"
                disabled={addPaymentMethodMutation.isPending}
              >
                {addPaymentMethodMutation.isPending ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                ) : (
                  <Plus className="h-4 w-4 mr-2" />
                )}
                Add Payment Method
              </Button>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Deletion</DialogTitle>
          </DialogHeader>
          <p>Are you sure you want to delete this payment method?</p>
          <div className="flex justify-end space-x-2 mt-4">
            <Button variant="outline" onClick={() => setIsDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={() => deletePaymentMethodMutation.mutate(selectedMethod?.id)}
              disabled={deletePaymentMethodMutation.isPending}
            >
              {deletePaymentMethodMutation.isPending ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              ) : (
                <Trash2 className="h-4 w-4 mr-2" />
              )}
              Delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}