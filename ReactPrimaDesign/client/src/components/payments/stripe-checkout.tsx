import { useState } from 'react';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { Loader2, AlertCircle } from 'lucide-react';
import { loadStripe } from '@stripe/stripe-js';
import { Elements, PaymentElement, useStripe, useElements } from '@stripe/react-stripe-js';
import { apiRequest } from '@/lib/queryClient';
import { useAuthState } from '@/hooks/use-auth-state';

// Make sure to call loadStripe outside of a component's render
// to avoid recreating the Stripe object on every render
// Get the Stripe public key from the window object
const stripePromise = loadStripe((window as any).STRIPE_PUBLIC_KEY as string);

function CheckoutFormInner({ 
  clientSecret, 
  paymentIntentId, 
  upiId, 
  amount,
  description,
  onSuccess
}: { 
  clientSecret: string; 
  paymentIntentId: string;
  upiId: string;
  amount: number;
  description: string;
  onSuccess: () => void;
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
        toast({
          title: 'Payment Failed',
          description: error.message || 'Please try again or use a different payment method',
          variant: 'destructive',
        });
      } else if (paymentIntent && paymentIntent.status === 'succeeded') {
        // Verify with our server
        const userId = authState.userId || '1';
        const response = await apiRequest('GET', `/api/payment/${paymentIntentId}?userId=${userId}`);
        const result = await response.json();
        
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

export default function StripeCheckout({
  clientSecret,
  paymentIntentId,
  upiId,
  amount,
  description,
  onSuccess
}: {
  clientSecret: string;
  paymentIntentId: string;
  upiId: string;
  amount: number;
  description: string;
  onSuccess: () => void;
}) {
  return (
    <Elements stripe={stripePromise} options={{ clientSecret }}>
      <CheckoutFormInner
        clientSecret={clientSecret}
        paymentIntentId={paymentIntentId}
        upiId={upiId}
        amount={amount}
        description={description}
        onSuccess={onSuccess}
      />
    </Elements>
  );
}