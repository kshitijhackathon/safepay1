import { useEffect, useState } from 'react';
import { useLocation } from 'wouter';
import { useToast } from '@/hooks/use-toast';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, CheckCircle, AlertCircle, ArrowLeft } from 'lucide-react';
import { useAuthState } from '@/hooks/use-auth-state';
import { createPaymentIntent, getStripePromise, checkPaymentStatus } from '@/lib/stripe-service';
import { loadStripe } from '@stripe/stripe-js';
import { Elements, PaymentElement, useStripe, useElements } from '@stripe/react-stripe-js';

// Stripe Checkout Form Component
function CheckoutForm({ 
  clientSecret, 
  paymentIntentId, 
  upiId, 
  amount,
  description,
  onSuccess,
  onError
}: { 
  clientSecret: string; 
  paymentIntentId: string;
  upiId: string;
  amount: number;
  description: string;
  onSuccess: () => void;
  onError: (message: string) => void;
}) {
  const stripe = useStripe();
  const elements = useElements();
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const { toast } = useToast();
  const { authState } = useAuthState();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!stripe || !elements) {
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      // Confirm payment with Stripe
      const { error, paymentIntent } = await stripe.confirmPayment({
        elements,
        confirmParams: {
          return_url: window.location.origin + '/success',
        },
        redirect: 'if_required',
      });

      if (error) {
        setErrorMessage(error.message || 'An unknown error occurred');
        onError(error.message || 'Payment failed. Please try again.');
        toast({
          title: 'Payment Failed',
          description: error.message || 'Please try again or use a different payment method',
          variant: 'destructive',
        });
      } else if (paymentIntent && paymentIntent.status === 'succeeded') {
        // Verify with our server
        const userId = authState.userId ? parseInt(authState.userId) : 1;
        const result = await checkPaymentStatus(paymentIntentId, userId);
        
        if (result.success) {
          toast({
            title: 'Payment Successful',
            description: 'Your transaction has been completed',
          });
          onSuccess();
        } else {
          toast({
            title: 'Payment Verification Failed',
            description: 'The payment was processed but could not be verified. Please contact support.',
            variant: 'destructive',
          });
        }
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'An unexpected error occurred');
      onError(err.message || 'An unexpected error occurred');
      toast({
        title: 'Error Processing Payment',
        description: err.message || 'Please try again later',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <PaymentElement />
      
      {errorMessage && (
        <div className="bg-destructive/10 p-3 rounded-md flex items-start">
          <AlertCircle className="h-5 w-5 text-destructive mr-2 mt-0.5" />
          <p className="text-sm text-destructive">{errorMessage}</p>
        </div>
      )}
      
      <Button 
        type="submit" 
        className="w-full" 
        disabled={!stripe || isLoading}
      >
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Processing...
          </>
        ) : (
          <>Pay ₹{amount.toFixed(2)}</>
        )}
      </Button>
    </form>
  );
}

// Main Checkout Page Component
export default function Checkout() {
  const [searchParams] = useLocation();
  const params = new URLSearchParams(searchParams.split('?')[1] || '');
  
  const upiId = params.get('upiId') || '';
  const amount = parseFloat(params.get('amount') || '0');
  const description = params.get('description') || 'UPI Payment';
  
  const [clientSecret, setClientSecret] = useState('');
  const [paymentIntentId, setPaymentIntentId] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [paymentComplete, setPaymentComplete] = useState(false);
  const [stripePromise, setStripePromise] = useState<Promise<any> | null>(null);
  
  const { toast } = useToast();
  const [, setLocation] = useLocation();

  // Initialize Stripe
  useEffect(() => {
    async function initStripe() {
      try {
        if (!import.meta.env.VITE_STRIPE_PUBLIC_KEY) {
          throw new Error('Stripe public key is not available');
        }
        const promise = loadStripe(import.meta.env.VITE_STRIPE_PUBLIC_KEY);
        setStripePromise(promise);
      } catch (err) {
        console.error("Failed to initialize Stripe:", err);
        setError("Failed to load payment service");
        setIsLoading(false);
      }
    }
    
    initStripe();
  }, []);

  // Create a payment intent when the component mounts
  useEffect(() => {
    if (!upiId || !amount || !stripePromise) {
      if (!upiId || !amount) {
        setError('Missing required payment information');
      }
      setIsLoading(false);
      return;
    }

    async function initPayment() {
      try {
        const data = await createPaymentIntent(amount, upiId, description);
        setClientSecret(data.clientSecret);
        setPaymentIntentId(data.paymentIntentId);
        setIsLoading(false);
      } catch (err: any) {
        console.error("Payment intent creation error:", err);
        setError(err.message || 'Could not initialize payment');
        setIsLoading(false);
        toast({
          title: 'Payment Initialization Failed',
          description: err.message || 'Please try again later',
          variant: 'destructive',
        });
      }
    }

    initPayment();
  }, [upiId, amount, description, toast, stripePromise]);

  const handlePaymentSuccess = () => {
    setPaymentComplete(true);
    // Redirect to success page after a short delay
    setTimeout(() => {
      setLocation('/success');
    }, 2000);
  };
  
  const handlePaymentError = (message: string) => {
    setError(message);
  };
  
  const handleBack = () => {
    const returnUrl = `/payment?upiId=${encodeURIComponent(upiId)}&amount=${amount}`;
    setLocation(returnUrl);
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-50">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white border-b p-4 flex items-center">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={handleBack}
          className="mr-2"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-lg font-medium">Secure Checkout</h1>
      </div>
      
      <div className="flex-1 p-4 flex items-center justify-center">
        <Card className="w-full max-w-md shadow-lg">
          <CardHeader>
            <CardTitle>Complete Your Payment</CardTitle>
            <CardDescription>
              {paymentComplete 
                ? 'Payment successfully processed!' 
                : `Pay ₹${amount.toFixed(2)} to ${upiId}`
              }
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            {isLoading ? (
              <div className="flex justify-center py-6">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            ) : error ? (
              <div className="bg-destructive/10 p-4 rounded-md text-center">
                <AlertCircle className="h-8 w-8 text-destructive mx-auto mb-2" />
                <p className="text-destructive">{error}</p>
                <Button
                  variant="outline"
                  className="mt-4"
                  onClick={handleBack}
                >
                  Return to Payment Options
                </Button>
              </div>
            ) : paymentComplete ? (
              <div className="text-center py-6">
                <CheckCircle className="h-16 w-16 text-green-500 mx-auto mb-4" />
                <p className="text-lg font-medium">Payment Successful!</p>
                <p className="text-muted-foreground mt-1">
                  Redirecting to confirmation page...
                </p>
              </div>
            ) : (clientSecret && stripePromise) ? (
              <Elements stripe={stripePromise} options={{ clientSecret }}>
                <CheckoutForm 
                  clientSecret={clientSecret}
                  paymentIntentId={paymentIntentId}
                  upiId={upiId}
                  amount={amount}
                  description={description}
                  onSuccess={handlePaymentSuccess}
                  onError={handlePaymentError}
                />
              </Elements>
            ) : null}
          </CardContent>
          
          <CardFooter className="flex flex-col space-y-2">
            <p className="text-xs text-muted-foreground text-center">
              Secure payment powered by Stripe. Your payment information is encrypted and secure.
            </p>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}