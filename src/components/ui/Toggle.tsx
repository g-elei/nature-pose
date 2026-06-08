import React from 'react';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  id: string;
}

const Toggle: React.FC<ToggleProps> = ({ checked, onChange, label, id }) => {
  return (
    <label htmlFor={id} className="flex items-center gap-2 cursor-pointer select-none">
      <span className="text-sm text-forest-700">{label}</span>
      <button
        id={id}
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-forest-300 focus:ring-offset-2 ${
          checked ? 'bg-forest-500' : 'bg-sage-200'
        }`}
      >
        <span
          className={`inline-block h-3.5 w-3.5 rounded-full bg-cream-50 transition-transform ${
            checked ? 'translate-x-4.5' : 'translate-x-1'
          }`}
        />
      </button>
    </label>
  );
};

export default Toggle;