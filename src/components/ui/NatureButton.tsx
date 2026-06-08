import React from 'react';

interface NatureButtonProps {
  children: React.ReactNode;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
  className?: string;
}

const NatureButton: React.FC<NatureButtonProps> = ({
  children, onClick, variant = 'primary', disabled = false, className = '',
}) => {
  const base = 'rounded-lg px-5 py-2.5 font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-forest-300 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
  const styles = {
    primary: 'bg-sand-500 text-forest-900 hover:bg-sand-400',
    secondary: 'bg-cream-200 text-forest-700 border border-sage-300 hover:bg-cream-100',
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-disabled={disabled}
      className={`${base} ${styles[variant]} ${className}`}
    >
      {children}
    </button>
  );
};

export default NatureButton;