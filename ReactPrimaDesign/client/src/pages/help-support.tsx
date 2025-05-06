import React from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { 
  ArrowLeft, 
  HelpCircle,
  Phone,
  Mail,
  MessageSquare,
  ExternalLink,
  FileText,
  ShieldAlert
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

export default function HelpSupport() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  const handleContactSupport = () => {
    toast({
      title: "Support ticket submitted",
      description: "We'll get back to you as soon as possible.",
    });
  };

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
        <h1 className="text-2xl font-bold">Help & Support</h1>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-md flex items-center">
            <HelpCircle className="h-5 w-5 mr-2 text-primary" />
            Common Questions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="item-1">
              <AccordionTrigger>How do I scan a QR code safely?</AccordionTrigger>
              <AccordionContent>
                <p className="text-sm text-gray-600">
                  To scan a QR code safely, use our app's built-in scanner which automatically checks the payment details for potential scams. Always verify the merchant name and amount before confirming the payment.
                </p>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-2">
              <AccordionTrigger>How can I report a suspicious UPI ID?</AccordionTrigger>
              <AccordionContent>
                <p className="text-sm text-gray-600">
                  You can report a suspicious UPI ID by going to the "UPI Check" section and using the "Report Scam" button. Provide as much detail as possible about the suspicious activity.
                </p>
                <Button 
                  variant="link" 
                  className="p-0 h-auto mt-2 text-primary"
                  onClick={() => setLocation('/report-scam')}
                >
                  Go to Report Scam
                </Button>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-3">
              <AccordionTrigger>What should I do if I sent money to a scammer?</AccordionTrigger>
              <AccordionContent>
                <p className="text-sm text-gray-600">
                  If you believe you've sent money to a scammer, immediately:
                  <ol className="list-decimal pl-5 mt-2 space-y-1">
                    <li>Contact your bank or UPI provider to report the transaction</li>
                    <li>Report the scam in our app</li>
                    <li>File a cybercrime complaint with the police</li>
                    <li>Contact our support team for guidance</li>
                  </ol>
                </p>
                <Button 
                  variant="link" 
                  className="p-0 h-auto mt-2 text-primary"
                  onClick={() => setLocation('/legal-help')}
                >
                  View Legal Help Options
                </Button>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-4">
              <AccordionTrigger>How do I reset my PIN?</AccordionTrigger>
              <AccordionContent>
                <p className="text-sm text-gray-600">
                  To reset your PIN, go to Settings &rarr; Security &rarr; PIN Login and select "Change PIN". You'll need to verify your identity before setting a new PIN.
                </p>
                <Button 
                  variant="link" 
                  className="p-0 h-auto mt-2 text-primary"
                  onClick={() => setLocation('/settings')}
                >
                  Go to Settings
                </Button>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="item-5">
              <AccordionTrigger>How do I add or remove payment methods?</AccordionTrigger>
              <AccordionContent>
                <p className="text-sm text-gray-600">
                  You can manage your payment methods by going to Account &rarr; Payment Methods. There you can add new payment methods or remove existing ones.
                </p>
                <Button 
                  variant="link" 
                  className="p-0 h-auto mt-2 text-primary"
                  onClick={() => setLocation('/payment-methods')}
                >
                  Go to Payment Methods
                </Button>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-md flex items-center">
            <ShieldAlert className="h-5 w-5 mr-2 text-primary" />
            Safety Tips
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-gray-600 space-y-3">
          <p>• Never share your OTP, PIN, or password with anyone, not even someone claiming to be from our support team.</p>
          <p>• Be cautious of unexpected payment requests, especially those with urgency.</p>
          <p>• Always verify the UPI ID before making payments to unfamiliar accounts.</p>
          <p>• Check for risk levels when scanning QR codes for payments.</p>
          <p>• Report suspicious activity immediately to prevent others from being scammed.</p>
          
          <Button 
            variant="link" 
            className="p-0 h-auto mt-2 text-primary"
            onClick={() => setLocation('/scam-news')}
          >
            <FileText className="h-4 w-4 mr-2" />
            Read Latest Scam Alerts
          </Button>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-md flex items-center">
            <MessageSquare className="h-5 w-5 mr-2 text-primary" />
            Contact Us
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center">
              <Phone className="h-5 w-5 mr-3 text-primary" />
              <div>
                <p className="font-medium">Customer Support</p>
                <p className="text-sm text-gray-600">1800-123-4567 (Toll Free)</p>
                <p className="text-xs text-gray-500">Available 24/7</p>
              </div>
            </div>
            
            <div className="flex items-center">
              <Mail className="h-5 w-5 mr-3 text-primary" />
              <div>
                <p className="font-medium">Email Support</p>
                <p className="text-sm text-gray-600">support@safepay.com</p>
                <p className="text-xs text-gray-500">Response within 24 hours</p>
              </div>
            </div>
            
            <Button 
              className="w-full mt-2"
              onClick={handleContactSupport}
            >
              <MessageSquare className="h-4 w-4 mr-2" />
              Contact Support Team
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-md flex items-center">
            <ExternalLink className="h-5 w-5 mr-2 text-primary" />
            Additional Resources
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Button 
              variant="link" 
              className="h-auto p-0 text-primary text-left w-full justify-start"
              onClick={() => setLocation('/legal-help')}
            >
              <FileText className="h-4 w-4 mr-2" />
              Legal Help & Resources
            </Button>
            
            <Button 
              variant="link" 
              className="h-auto p-0 text-primary text-left w-full justify-start"
              onClick={() => setLocation('/fraud-heatmap')}
            >
              <ShieldAlert className="h-4 w-4 mr-2" />
              View Fraud Heatmap
            </Button>
            
            <Button 
              variant="link" 
              className="h-auto p-0 text-primary text-left w-full justify-start"
              onClick={() => {
                toast({
                  title: "User Manual",
                  description: "The user manual will open in a new window.",
                });
              }}
            >
              <FileText className="h-4 w-4 mr-2" />
              Download User Manual
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}