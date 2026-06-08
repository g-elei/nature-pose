import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/nature-pose/',
  plugins: [react()],
  optimizeDeps: {
    exclude: [
      '@tensorflow/tfjs-backend-webgl',
      '@tensorflow/tfjs-backend-cpu',
      '@tensorflow/tfjs-core',
    ],
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'tf-core': ['@tensorflow/tfjs-core'],
          'tf-webgl': ['@tensorflow/tfjs-backend-webgl'],
          'tf-pose': ['@tensorflow-models/pose-detection'],
        },
      },
    },
  },
});