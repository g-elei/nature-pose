import React from 'react';
import { DBSession } from '../../types/session';
import NatureButton from '../ui/NatureButton';

const ACTIVITY_ICON: Record<string, string> = {
  running: '🏃', walking: '🚶', yoga: '🧘', sitting: '🪑',
};

interface SessionListProps {
  sessions: DBSession[];
  onExport: () => void;
}

function topCues(summary: Record<string, number>, count: number): string[] {
  return Object.entries(summary)
    .sort(([, a], [, b]) => b - a)
    .slice(0, count)
    .map(([key]) => key.replace(/_/g, ' '));
}

const SessionList: React.FC<SessionListProps> = ({ sessions, onExport }) => {
  if (sessions.length === 0) {
    return (
      <div className="text-center py-8 text-sage-500 text-sm">
        Complete a session to see your history
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-forest-700">Session History</h3>
        <NatureButton variant="secondary" onClick={onExport}>
          Download CSV
        </NatureButton>
      </div>
      <div className="overflow-x-auto rounded-lg border border-sage-200">
        <table className="w-full text-left text-sm" role="table" aria-label="Session history">
          <thead>
            <tr className="bg-cream-100 text-forest-700">
              <th scope="col" className="px-3 py-2 font-medium">Date</th>
              <th scope="col" className="px-3 py-2 font-medium">Activity</th>
              <th scope="col" className="px-3 py-2 font-medium">Score</th>
              <th scope="col" className="px-3 py-2 font-medium">Top Cues</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => {
              const cues = topCues(s.actionsSummary, 2);
              return (
                <tr key={s.id} className="border-t border-sage-100 hover:bg-cream-50 transition-colors">
                  <td className="px-3 py-2 text-sage-600">
                    {new Date(s.date).toLocaleDateString()}
                  </td>
                  <td className="px-3 py-2">
                    <span className="flex items-center gap-1">
                      <span>{ACTIVITY_ICON[s.activity] || '❓'}</span>
                      <span className="capitalize text-forest-700">{s.activity}</span>
                    </span>
                  </td>
                  <td className="px-3 py-2 font-medium text-forest-900">{s.averageScore}</td>
                  <td className="px-3 py-2 text-sage-600">
                    {cues.length > 0 ? cues.join(', ') : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default SessionList;