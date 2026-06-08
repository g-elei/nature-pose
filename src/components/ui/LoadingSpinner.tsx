import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ message = 'Loading...' }) => {
  return (
    <div role="status" aria-live="polite" className="flex flex-col items-center justify-center gap-3 p-8">
      <div aria-hidden="true" className="h-8 w-8 animate-spin rounded-full border-4 border-sage-300 border-t-forest-500" />
      <p className="text-sm text-forest-700">{message}</p>
    </div>
  );
};

export default LoadingSpinner;