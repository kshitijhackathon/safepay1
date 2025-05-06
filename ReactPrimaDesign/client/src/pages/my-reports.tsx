import React from 'react';
import { useLocation } from 'wouter';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  ArrowLeft, 
  AlertTriangle, 
  Calendar, 
  DollarSign, 
  Shield
} from 'lucide-react';
import { ScamReport } from '@shared/schema';
import { apiRequest } from '@/lib/queryClient';
import { useAuthState } from '@/hooks/use-auth-state';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription,
  DialogHeader, 
  DialogTitle
} from '@/components/ui/dialog';

interface FormattedReport extends Omit<ScamReport, 'timestamp' | 'amountLost'> {
  timestamp: string;
  amountLost: string | null;
  date: string;
  time: string;
}

export default function MyReports() {
  const [, setLocation] = useLocation();
  const [selectedReport, setSelectedReport] = React.useState<FormattedReport | null>(null);
  const [dialogOpen, setDialogOpen] = React.useState(false);
  const { authState, requireLogin } = useAuthState();
  
  // Redirect if not logged in
  React.useEffect(() => {
    if (!authState.isLoggedIn) {
      setLocation('/phone-login?returnUrl=/my-reports');
    }
  }, [authState.isLoggedIn, setLocation]);
  
  // Get user ID from auth state
  const userId = authState.isLoggedIn && authState.userId ? parseInt(authState.userId) : null;
  
  // Fetch user scam reports
  const { data: reports, isLoading, error } = useQuery({
    queryKey: ['/api/user/scam-reports', userId],
    queryFn: async () => {
      if (!userId) {
        throw new Error('User not logged in');
      }
      
      const res = await apiRequest('GET', `/api/user/${userId}/scam-reports`);
      const data = await res.json();
      
      // Format dates for display
      return data.map((report: any) => {
        const timestamp = new Date(report.timestamp);
        return {
          ...report,
          date: format(timestamp, 'MMM d, yyyy'),
          time: format(timestamp, 'h:mm a'),
        };
      });
    }
  });
  
  const handleReportClick = (report: FormattedReport) => {
    setSelectedReport(report);
    setDialogOpen(true);
  };
  
  const getScamTypeColor = (scamType: string) => {
    switch(scamType) {
      case 'Banking Scam':
        return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-300';
      case 'Lottery Scam':
        return 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-900/30 dark:text-amber-300';
      case 'KYC Verification Scam':
        return 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300';
      case 'Refund Scam':
        return 'bg-indigo-100 text-indigo-800 border-indigo-200 dark:bg-indigo-900/30 dark:text-indigo-300';
      case 'Phishing Attempt':
        return 'bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-900/30 dark:text-purple-300';
      case 'Reward Scam':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-800 dark:text-gray-300';
    }
  };
  
  return (
    <div className="flex flex-col px-6 py-8 min-h-screen overflow-y-auto">
      <div className="flex items-center mb-6">
        <Button 
          variant="ghost" 
          size="icon" 
          className="mr-2" 
          onClick={() => setLocation('/account')}
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-2xl font-bold">My Scam Reports</h1>
      </div>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-60">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      ) : error ? (
        <Card className="bg-red-50 dark:bg-red-900/20 border-red-100 dark:border-red-800">
          <CardContent className="py-6">
            <div className="flex items-center justify-center gap-2 text-red-700 dark:text-red-400">
              <AlertTriangle />
              <p>Error loading your reports. Please try again later.</p>
            </div>
          </CardContent>
        </Card>
      ) : reports && reports.length > 0 ? (
        <div className="space-y-4 pb-6">
          {reports.map((report: FormattedReport) => (
            <Card 
              key={report.id} 
              className="overflow-hidden transition-all duration-200 hover:shadow-md cursor-pointer"
              onClick={() => handleReportClick(report)}
            >
              <CardHeader className="pb-2">
                <div className="flex justify-between items-start">
                  <CardTitle className="text-base">{report.upiId}</CardTitle>
                  <Badge className={`${getScamTypeColor(report.scamType)} border`}>
                    {report.scamType}
                  </Badge>
                </div>
                <CardDescription className="flex items-center gap-1 text-xs">
                  <Calendar className="h-3 w-3" />
                  {report.date}
                </CardDescription>
              </CardHeader>
              <CardContent className="pb-4">
                <div className="flex justify-between items-center text-sm">
                  <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                    <Shield className="h-4 w-4" />
                    <span>Report #{report.id}</span>
                  </div>
                  {report.amountLost && (
                    <div className="flex items-center gap-1 font-medium">
                      <DollarSign className="h-4 w-4 text-red-500" />
                      <span>₹{report.amountLost}</span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center h-60 text-center">
          <Shield className="h-16 w-16 text-gray-300 dark:text-gray-600 mb-4" />
          <h3 className="text-xl font-medium mb-2">No Reports Yet</h3>
          <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md">
            You haven't submitted any scam reports yet. If you encounter any suspicious UPI activities, report them to help the community.
          </p>
          <Button onClick={() => setLocation('/report-scam')}>
            Report a Scam
          </Button>
        </div>
      )}
      
      {/* Report Detail Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto">
          {selectedReport && (
            <>
              <DialogHeader>
                <DialogTitle className="text-xl mb-1">Report Details</DialogTitle>
                <DialogDescription>
                  Reported on {selectedReport.date} at {selectedReport.time}
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 mt-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">UPI ID</h3>
                  <p className="text-lg font-semibold">{selectedReport.upiId}</p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Scam Type</h3>
                  <Badge className={`${getScamTypeColor(selectedReport.scamType)} border`}>
                    {selectedReport.scamType}
                  </Badge>
                </div>
                
                {selectedReport.amountLost && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Amount Lost</h3>
                    <p className="text-lg font-semibold text-red-600 dark:text-red-400">₹{selectedReport.amountLost}</p>
                  </div>
                )}
                
                {selectedReport.description && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Description</h3>
                    <div className="bg-muted p-3 rounded-md text-sm">
                      {selectedReport.description}
                    </div>
                  </div>
                )}
                
                <div className="pt-2">
                  <Button 
                    variant="outline" 
                    className="w-full" 
                    onClick={() => setDialogOpen(false)}
                  >
                    Close
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}