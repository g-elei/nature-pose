import React from 'react';
import { CoachAction } from '../../rl/coach';

interface FeedbackBubbleProps {
  coachActions: CoachAction[];
  severity: 'good' | 'warning' | 'critical';
}

const SEVERITY_CONFIG = {
  good: { icon: '🌿', bg: 'bg-forest-100', border: 'border-forest-400', text: 'text-forest-900' },
  warning: { icon: '⚠️', bg: 'bg-sand-100', border: 'border-sand-400', text: 'text-sand-900' },
  critical: { icon: '🚨', bg: 'bg-red-100', border: 'border-red-400', text: 'text-red-900' },
};

const FeedbackBubble: React.FC<FeedbackBubbleProps> = ({ coachActions, severity }) => {
  if (coachActions.length === 0) return null;

  const cfg = SEVERITY_CONFIG[severity];
  const primary = coachActions[0];

  return (
    <div
      role="status"
      aria-live="polite"
      className={`rounded-lg border-2 p-4 ${cfg.bg} ${cfg.border} animate-fadeIn`}
    >
      <div className="flex items-start gap-3">
        <span className="text-xl">{cfg.icon}</span>
        <div className="flex-1">
          <p className={`font-medium ${cfg.text}`}>{primary.advice}</p>
          {coachActions.length > 1 && (
            <ul className="mt-2 space-y-1">
              {coachActions.slice(1).map((a) => (
                <li key={a.actionId} className={`text-sm ${cfg.text} opacity-80`}>
                  {a.voiceCue}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default FeedbackBubble;