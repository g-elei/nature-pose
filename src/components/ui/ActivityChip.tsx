import React from 'react';
import { ActivityKind } from '../../types/session';

const ACTIVITY_META: Record<ActivityKind, { icon: string; label: string }> = {
  running: { icon: '🏃', label: 'Running' },
  walking: { icon: '🚶', label: 'Walking' },
  yoga: { icon: '🧘', label: 'Yoga' },
  sitting: { icon: '🪑', label: 'Sitting' },
};

interface ActivityChipProps {
  kind: ActivityKind;
  selected: boolean;
  onClick: () => void;
}

const ActivityChip: React.FC<ActivityChipProps> = ({ kind, selected, onClick }) => {
  const meta = ACTIVITY_META[kind];

  return (
    <button
      role="radio"
      aria-checked={selected}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onClick(); }}
      className={`flex items-center gap-1.5 rounded-full px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-forest-300 focus:ring-offset-2 ${
        selected
          ? 'bg-forest-500 text-cream-50 ring-2 ring-forest-300'
          : 'bg-cream-200 text-forest-700 hover:bg-cream-100'
      }`}
    >
      <span>{meta.icon}</span>
      <span>{meta.label}</span>
    </button>
  );
};

export default ActivityChip;