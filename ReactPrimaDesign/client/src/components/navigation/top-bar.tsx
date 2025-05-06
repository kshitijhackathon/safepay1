import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { 
  Search, 
  Bell, 
  User,
  X,
  ArrowLeft
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { cn } from '@/lib/utils';

interface TopBarProps {
  title?: string;
  showBack?: boolean;
  showSearch?: boolean;
  showNotification?: boolean;
  showProfile?: boolean;
  className?: string;
  onSearchSubmit?: (query: string) => void;
}

export function TopBar({
  title = 'UPI Fraud Shield',
  showBack = false,
  showSearch = true,
  showNotification = true,
  showProfile = true,
  className,
  onSearchSubmit
}: TopBarProps) {
  const [, setLocation] = useLocation();
  const [isSearchActive, setIsSearchActive] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const handleBack = () => {
    window.history.back();
  };

  const handleSearchClick = () => {
    if (isSearchActive && searchQuery.trim()) {
      // Submit search
      if (onSearchSubmit) {
        onSearchSubmit(searchQuery);
      } else {
        // Default search behavior - redirect to search page with query
        setLocation(`/search?q=${encodeURIComponent(searchQuery)}`);
      }
      setIsSearchActive(false);
      setSearchQuery('');
    } else {
      // Toggle search input
      setIsSearchActive(!isSearchActive);
    }
  };

  const handleSearchCancel = () => {
    setIsSearchActive(false);
    setSearchQuery('');
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      if (onSearchSubmit) {
        onSearchSubmit(searchQuery);
      } else {
        // Default search behavior
        setLocation(`/search?q=${encodeURIComponent(searchQuery)}`);
      }
      setIsSearchActive(false);
      setSearchQuery('');
    }
  };

  const handleProfileClick = () => {
    setLocation('/account');
  };

  const handleNotificationClick = () => {
    setLocation('/notifications');
  };

  return (
    <div className={cn("sticky top-0 z-20 w-full bg-white border-b p-2", className)}>
      <div className="flex items-center gap-2 mx-auto max-w-md bg-gray-100 rounded-full px-4 py-2">
        <Avatar className="h-8 w-8 cursor-pointer" onClick={handleProfileClick}>
          <AvatarFallback className="bg-gray-200">
            <User className="h-4 w-4 text-gray-400" />
          </AvatarFallback>
        </Avatar>
        
        <div className="flex-1 font-medium">{title}</div>
        
        <Button 
          variant="ghost" 
          size="icon" 
          className="rounded-full"
          onClick={handleSearchClick}
          aria-label="Search"
        >
          <Search className="h-5 w-5" />
        </Button>
        
        <Button 
          variant="ghost" 
          size="icon" 
          className="rounded-full"
          onClick={handleNotificationClick}
          aria-label="Notifications"
        >
          <Bell className="h-5 w-5" />
        </Button>
        
        <Avatar 
          className="h-8 w-8 cursor-pointer bg-primary"
          onClick={handleProfileClick}
        >
          <AvatarFallback className="bg-primary text-white">
            <User className="h-4 w-4" />
          </AvatarFallback>
        </Avatar>
      </div>
      
      {isSearchActive && (
        <div className="absolute inset-0 bg-white px-4 py-2 flex items-center">
          <form onSubmit={handleSearchSubmit} className="flex items-center w-full gap-2">
            <Button 
              type="button" 
              variant="ghost" 
              size="icon" 
              onClick={handleSearchCancel}
              aria-label="Cancel search"
            >
              <X className="h-5 w-5" />
            </Button>
            <Input
              type="text"
              placeholder="Search for UPI IDs, scams..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1"
              autoFocus
            />
            <Button 
              type="submit" 
              variant="ghost" 
              size="icon"
              disabled={!searchQuery.trim()}
              aria-label="Submit search"
            >
              <Search className="h-5 w-5" />
            </Button>
          </form>
        </div>
      )}
    </div>
  );
}