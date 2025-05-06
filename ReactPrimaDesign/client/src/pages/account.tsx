import React from 'react';
import { useLocation } from 'wouter';
import { useQuery } from '@tanstack/react-query';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  User,
  CreditCard,
  Settings,
  HelpCircle,
  Shield,
  Bell,
  LogOut,
  ChevronRight,
  AlertTriangle
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';
import { useAuthState } from '@/hooks/use-auth-state';

export default function Account() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const { authState, logout } = useAuthState();

  // Redirect if not logged in
  React.useEffect(() => {
    if (!authState.isLoggedIn) {
      setLocation('/login');
    }
  }, [authState.isLoggedIn, setLocation]);

  // Get user ID from auth state
  const userId = authState.isLoggedIn && authState.userId ? parseInt(authState.userId) : null;

  // Fetch user profile data
  const { data: user, isLoading } = useQuery({
    queryKey: ['/api/users', userId],
    queryFn: async () => {
      if (!userId) {
        throw new Error('User not logged in');
      }
      
      const res = await apiRequest('GET', `/api/users/${userId}`);
      if (!res.ok) throw new Error('Failed to load user profile');
      return res.json();
    },
    enabled: !!userId,
  });

  // Menu items for the account section
  const menuItems = [
    {
      icon: <User className="w-5 h-5" />,
      title: 'My Profile',
      description: 'Manage your personal information',
      path: '/profile'
    },
    {
      icon: <CreditCard className="w-5 h-5" />,
      title: 'Payment Methods',
      description: 'Manage your UPI IDs, cards and bank accounts',
      path: '/payment-methods'
    },
    {
      icon: <AlertTriangle className="w-5 h-5" />,
      title: 'My Reports',
      description: 'View all scam reports you have submitted',
      path: '/my-reports'
    },
    {
      icon: <Settings className="w-5 h-5" />,
      title: 'Settings',
      description: 'Adjust app preferences and security settings',
      path: '/settings'
    },
    {
      icon: <HelpCircle className="w-5 h-5" />,
      title: 'Help & Support',
      description: 'Get assistance and contact customer service',
      path: '/help-support'
    },
    {
      icon: <Shield className="w-5 h-5" />,
      title: 'Security Center',
      description: 'Monitor and enhance account security',
      path: '/fraud-heatmap'
    },
    {
      icon: <Bell className="w-5 h-5" />,
      title: 'Scam Alerts',
      description: 'Stay updated on recent scam trends',
      path: '/scam-news'
    }
  ];

  // Handle logout
  const handleLogout = async () => {
    try {
      await apiRequest('POST', '/api/logout');
      toast({
        title: 'Logged out',
        description: 'You have been successfully logged out.',
      });
      setLocation('/');
    } catch (error) {
      toast({
        title: 'Logout failed',
        description: 'There was an error logging out.',
        variant: 'destructive',
      });
    }
  };

  if (isLoading) {
    return (
      <div className="p-6 flex justify-center items-center min-h-[50vh]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="flex flex-col p-6">
      <h1 className="text-2xl font-bold mb-6 dark:text-white">My Account</h1>
      
      {/* User profile summary */}
      <Card className="p-4 mb-6 bg-primary/5 dark:bg-primary/10 border-none transition-colors duration-300">
        <div className="flex items-center">
          <div className="w-12 h-12 rounded-full bg-primary/25 dark:bg-primary/30 flex items-center justify-center text-primary dark:text-primary-foreground font-bold text-xl mr-4 transition-colors duration-300">
            {user?.name ? user.name.charAt(0).toUpperCase() : 'U'}
          </div>
          <div>
            <h2 className="font-bold text-lg dark:text-white">{user?.name || 'User'}</h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">{user?.phoneNumber || '+91 9876543210'}</p>
          </div>
        </div>
      </Card>
      
      {/* Menu items */}
      <div className="space-y-3">
        {menuItems.map((item, index) => (
          <Card 
            key={index} 
            className="p-4 hover:bg-gray-50 dark:hover:bg-gray-800 dark:bg-gray-850 dark:border-gray-700 transition-colors cursor-pointer"
            onClick={() => setLocation(item.path)}
          >
            <div className="flex items-center">
              <div className="bg-primary/10 dark:bg-primary/20 p-2 rounded-full mr-4 transition-colors duration-300">
                <span className="text-primary dark:text-primary-foreground">{item.icon}</span>
              </div>
              <div className="flex-1">
                <h3 className="font-medium dark:text-white">{item.title}</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">{item.description}</p>
              </div>
              <ChevronRight className="w-5 h-5 text-gray-400 dark:text-gray-500" />
            </div>
          </Card>
        ))}
        
        {/* Logout option */}
        <Button 
          variant="ghost" 
          className="w-full justify-start text-destructive hover:text-destructive hover:bg-destructive/10 dark:text-red-400 dark:hover:text-red-300 dark:hover:bg-red-900/20 mt-6 transition-colors duration-300"
          onClick={handleLogout}
        >
          <LogOut className="w-5 h-5 mr-2" />
          <span>Logout</span>
        </Button>
      </div>
    </div>
  );
}