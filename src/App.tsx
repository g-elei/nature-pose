// src/App.tsx
import { SettingsProvider } from './store/SettingsContext';
import { SessionProvider } from './store/SessionContext';
import { PoseProvider } from './store/PoseContext';
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