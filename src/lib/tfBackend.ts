import * as tf from '@tensorflow/tfjs-core';

async function detectWebGLSupport(): Promise<boolean> {
  const canvas = document.createElement('canvas');
  const gl =
    (canvas.getContext('webgl2') as WebGLRenderingContext | null) ??
    (canvas.getContext('webgl') as WebGLRenderingContext | null);
  if (!gl) return false;
  try {
    await tf.setBackend('webgl');
    return tf.getBackend() === 'webgl';
  } catch {
    return false;
  }
}

export async function initBackend(): Promise<string> {
  const webglOk = await detectWebGLSupport();
  if (webglOk) {
    await import('@tensorflow/tfjs-backend-webgl');
    await tf.setBackend('webgl');
  } else {
    await import('@tensorflow/tfjs-backend-cpu');
    await tf.setBackend('cpu');
  }
  return tf.getBackend();
}