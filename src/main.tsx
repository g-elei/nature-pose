import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { PoseProvider } from './store/PoseContext';
import { SessionProvider } from './store/SessionContext';
import { SettingsProvider } from './store/SettingsContext';
import { initBackend } from './lib/tfBackend';
import './index.css';

initBackend().then((backend) => {
  console.log('TF backend initialized:', backend);
});

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