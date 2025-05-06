import React from 'react';
import { ArrowLeft } from 'lucide-react';

interface PageHeaderProps {
  title: string;
  onBack?: () => void;
  rightContent?: React.ReactNode;
}

const PageHeader: React.FC<PageHeaderProps> = ({ title, onBack, rightContent }) => {
  return (
    <div className="flex items-center justify-between py-3 mb-4">
      <div className="flex items-center">
        {onBack && (
          <button
            onClick={onBack}
            className="p-1.5 mr-2 rounded-full hover:bg-gray-100"
            aria-label="Go back"
          >
            <ArrowLeft size={20} className="text-gray-700" />
          </button>
        )}
        <h1 className="text-xl font-semibold text-gray-800">{title}</h1>
      </div>
      {rightContent && <div>{rightContent}</div>}
    </div>
  );
};

export default PageHeader;