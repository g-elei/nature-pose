import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite'; // <-- Add this import

export default defineConfig({
  plugins: [
    react(), 
    tailwindcss() // <-- Register the Tailwind compiler plugin
  ],
  base: '/', // <-- Fix base path to match package.json homepage
  optimizeDeps: {
    include: ['@tensorflow/tfjs-core', '@tensorflow/tfjs-backend-webgl', 'long'],
  },
});