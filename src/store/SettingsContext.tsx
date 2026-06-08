import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { ActivityKind } from '../types/session';

interface SettingsContextValue {
  activity: ActivityKind;
  setActivity: (activity: ActivityKind) => void;
  voiceMuted: boolean;
  setVoiceMuted: (muted: boolean) => void;
}

const SettingsContext = createContext<SettingsContextValue | undefined>(undefined);

export const useSettings = () => {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider');
  return ctx;
};

interface SettingsProviderProps {
  children: ReactNode;
}

const STORAGE_KEY = 'nature-pose-settings';

export const SettingsProvider: React.FC<SettingsProviderProps> = ({ children }) => {
  const [activity, setActivity] = useState<ActivityKind>('running');
  const [voiceMuted, setVoiceMuted] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (parsed.activity) setActivity(parsed.activity);
        if (typeof parsed.voiceMuted === 'boolean') setVoiceMuted(parsed.voiceMuted);
      } catch (e) { /* ignore */ }
    }
  }, []);

  // Save to localStorage on change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ activity, voiceMuted }));
  }, [activity, voiceMuted]);

  return (
    <SettingsContext.Provider value={{ activity, setActivity, voiceMuted, setVoiceMuted }}>
      {children}
    </SettingsContext.Provider>
  );
};