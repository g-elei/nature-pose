import React from 'react';
import { useSettings } from '../../store/SettingsContext';
import ActivityChip from '../ui/ActivityChip';
import NatureButton from '../ui/NatureButton';
import { ActivityKind } from '../../types/session';

interface HomeScreenProps {
  onStart: () => void;
}

export const HomeScreen: React.FC<HomeScreenProps> = ({ onStart }) => {
  const { activity, setActivity } = useSettings();
  const activities: ActivityKind[] = ['sitting', 'yoga', 'walking', 'running'];

  return (
    <div className="min-h-screen bg-gradient-to-b from-cream-100 to-sage-50 flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full bg-cream-50 rounded-2xl shadow-xl p-8 text-center">
        <h1 className="text-4xl font-bold text-forest-800 mb-2">🌿 NaturePose</h1>
        <p className="text-forest-600 mb-8">Your AI coach for better movement</p>

        <div className="flex flex-wrap justify-center gap-3 mb-8" role="radiogroup" aria-label="Select activity">
          {activities.map((act) => (
              <ActivityChip
              key={act}
              kind={act}
              selected={activity === act}
              onClick={() => setActivity(act)}
            />
          ))}
        </div>

        <NatureButton variant="primary" onClick={onStart} className="w-full py-3 text-lg">
          Start Session
        </NatureButton>
      </div>
    </div>
  );
};