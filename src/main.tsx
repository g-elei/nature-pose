import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { PoseProvider } from './store/PoseContext';
import { SessionProvider } from './store/SessionContext';
import { SettingsProvider } from './store/SettingsContext';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <SettingsProvider>
      <SessionProvider>
        <PoseProvider>
          <App />
        </PoseProvider>
      </SessionProvider>
    </SettingsProvider>
  </React.StrictMode>
);