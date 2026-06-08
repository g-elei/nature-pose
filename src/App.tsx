// src/App.tsx
import React from 'react';
import { SettingsProvider } from './store/SettingsContext'; // <-- Changed from ./context/
import { SessionProvider } from './store/SessionContext';   // <-- Changed from ./context/
import { PoseProvider } from './store/PoseContext';       // <-- Changed from ./context/
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