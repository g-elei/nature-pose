// src/main.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { initBackend } from './lib/tfBackend';
import './index.css';

initBackend().then((backend) => {
  console.log('TF backend initialized:', backend);
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);