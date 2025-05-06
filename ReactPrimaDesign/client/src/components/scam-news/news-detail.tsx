import React from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle,
  CardDescription,
  CardFooter
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  AlertOctagon, 
  AlertTriangle, 
  Shield, 
  MapPin, 
  Calendar,
  Share2,
  Flag
} from 'lucide-react';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';

interface ScamAlert {
  title: string;
  type: string;
  description: string;
  affected_areas: string[];
  risk_level: 'High' | 'Medium' | 'Low';
  date_reported: string;
  verification_status: 'Verified' | 'Investigating' | 'Unverified';
}

interface NewsDetailProps {
  alert: ScamAlert;
  onClose: () => void;
  onReportSimilar?: () => void;
}

export function NewsDetail({ alert, onClose, onReportSimilar }: NewsDetailProps) {
  const { toast } = useToast();
  
  // Render risk level badge with appropriate color
  const renderRiskBadge = (level: string) => {
    let variant: 
      | 'default'
      | 'secondary'
      | 'destructive'
      | 'outline' = 'default';
    let icon = null;
    
    if (level.toLowerCase() === 'high') {
      variant = 'destructive';
      icon = <AlertOctagon className="h-3 w-3 mr-1" />;
    } else if (level.toLowerCase() === 'medium') {
      variant = 'default';
      icon = <AlertTriangle className="h-3 w-3 mr-1" />;
    } else {
      variant = 'secondary';
      icon = <Shield className="h-3 w-3 mr-1" />;
    }
    
    return (
      <Badge variant={variant} className="flex items-center">
        {icon}
        {level}
      </Badge>
    );
  };
  
  // Handle share functionality
  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: alert.title,
        text: `${alert.title} - ${alert.description}`,
        url: window.location.href
      }).catch(error => {
        console.error('Error sharing:', error);
      });
    } else {
      // Fallback for browsers that don't support the Share API
      navigator.clipboard.writeText(`${alert.title} - ${alert.description}`);
      toast({
        title: 'Copied to clipboard',
        description: 'Alert details copied to clipboard',
      });
    }
  };
  
  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-start mb-2">
          <CardTitle className="text-xl">{alert.title}</CardTitle>
          {renderRiskBadge(alert.risk_level)}
        </div>
        <CardDescription>
          <Badge variant="outline" className="bg-blue-50 border-blue-200 text-blue-600 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-400">
            {alert.type}
          </Badge>
          <div className="flex items-center mt-2 text-sm text-gray-500 dark:text-gray-400">
            <Calendar className="h-4 w-4 mr-1" />
            {new Date(alert.date_reported).toLocaleDateString(undefined, { 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </div>
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="text-gray-700 dark:text-gray-300 text-sm leading-relaxed">
          {/* Full detailed description - expanded version of the summary shown in the card */}
          <p className="mb-4">{alert.description}</p>
          
          <p className="mb-4">
            This type of scam typically involves fraudsters impersonating legitimate entities to trick users into 
            revealing sensitive information or making payments to fraudulent accounts. Users should be vigilant and 
            verify any unexpected payment requests through official channels.
          </p>
          
          <p>
            If you encounter a similar scam, report it immediately to your bank and 
            the relevant authorities. Never share OTP, PIN, or password with anyone, even if they 
            claim to be from your bank or a government agency.
          </p>
        </div>
        
        <Separator />
        
        <div>
          <h3 className="font-medium mb-2 text-sm">Affected Areas</h3>
          <div className="flex flex-wrap gap-2">
            {alert.affected_areas.map((area, index) => (
              <Badge key={index} variant="outline" className="flex items-center">
                <MapPin className="h-3 w-3 mr-1" />
                {area}
              </Badge>
            ))}
          </div>
        </div>
        
        <Separator />
        
        <div>
          <h3 className="font-medium mb-2 text-sm">How to Stay Safe</h3>
          <ul className="list-disc pl-5 space-y-1 text-sm">
            <li>Always verify the UPI ID before making a payment</li>
            <li>Do not click on suspicious links in messages or emails</li>
            <li>Never share OTP, PIN, or passwords with anyone</li>
            <li>Check for official communication channels before responding to offers</li>
            <li>Report suspicious activities to your bank immediately</li>
          </ul>
        </div>
        
        <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg">
          <h3 className="font-medium mb-2 text-sm text-blue-700 dark:text-blue-300">Publisher Information</h3>
          <p className="text-xs text-blue-600 dark:text-blue-400">
            This alert was published and verified by the SafePay Fraud Prevention Team in collaboration with
            financial institutions and cybersecurity experts.
          </p>
        </div>
      </CardContent>
      
      <CardFooter className="flex justify-between">
        <div className="flex space-x-2">
          <Button 
            variant="outline" 
            size="sm"
            onClick={handleShare}
          >
            <Share2 className="h-4 w-4 mr-2" />
            Share
          </Button>
          
          {onReportSimilar && (
            <Button 
              variant="outline" 
              size="sm"
              onClick={onReportSimilar}
            >
              <Flag className="h-4 w-4 mr-2" />
              Report Similar
            </Button>
          )}
        </div>
        
        <Button 
          variant="ghost" 
          size="sm"
          onClick={onClose}
        >
          Close
        </Button>
      </CardFooter>
    </Card>
  );
}