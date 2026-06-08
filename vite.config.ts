// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite'; // 1. Add this import

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(), // 2. Add the compiler plugin right here
  ],
  base: '/',
  optimizeDeps: {
    include: ['@tensorflow/tfjs-core', '@tensorflow/tfjs-backend-webgl', 'long'],
  },
});