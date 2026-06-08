import React, { useEffect } from 'react';
import { useSession } from '../../store/SessionContext';
import ScoreChart from '../dashboard/ScoreChart';
import SessionList from '../dashboard/SessionList';
import NatureButton from '../ui/NatureButton';
import { exportCSV } from '../../lib/db';

interface DashboardScreenProps {
  onBack: () => void;
}

export const DashboardScreen: React.FC<DashboardScreenProps> = ({ onBack }) => {
  const { sessions, loadSessions } = useSession();

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleExport = () => {
    const csv = exportCSV(sessions);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `naturepose_sessions_${new Date().toISOString().slice(0,19)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-cream-100 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-forest-800">📊 Your Progress</h1>
          <div className="flex gap-3">
            <NatureButton variant="secondary" onClick={handleExport}>Export CSV</NatureButton>
            <NatureButton variant="primary" onClick={onBack}>New Session</NatureButton>
          </div>
        </div>

        {sessions.length === 0 ? (
          <div className="bg-cream-50 rounded-xl p-12 text-center text-forest-500">
            No sessions yet. Complete a coaching session to see your progress.
          </div>
        ) : (
          <>
            <div className="bg-cream-50 rounded-xl p-4 shadow mb-8">
              <ScoreChart sessions={sessions} />
            </div>
            <div className="bg-cream-50 rounded-xl p-4 shadow">
              <SessionList sessions={sessions} onExport={handleExport} />
            </div>
          </>
        )}
      </div>
    </div>
  );
};