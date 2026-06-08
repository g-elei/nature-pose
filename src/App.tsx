// src/App.tsx
import React from 'react';
import { SettingsProvider } from './context/SettingsContext';
import { SessionProvider } from './context/SessionContext';
import { PoseProvider } from './context/PoseContext';
import { LiveCoachDashboard } from './components/LiveCoachDashboard';

export default function App() {
  return (
    <SettingsProvider>
      <SessionProvider>
        <PoseProvider>
          <LiveCoachDashboard />
        </PoseProvider>
      </SessionProvider>
    </SettingsProvider>
  );
}