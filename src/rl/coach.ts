// src/rl/coach.ts
import * as tf from '@tensorflow/tfjs-core';
import { Keypoint } from '../types/pose';
import { RunningMetrics } from '../pose/analyzer';
import { headTilt, hipTilt } from '../pose/angles';

export interface CoachAction {
  actionId: string; // Used as dictionary key in SessionContext summary telemetry
  text: string;     // Read aloud by Web Speech API / displayed in UI
}

export const COACH_ACTIONS = [
  "Form is excellent. Maintain this standard!", 
  "Straighten your spine. Avoid slouching.",     
  "Drive your knees higher to increase power.",  
  "Relax your shoulders and pump your arms.",    
  "Engage your core to stabilize your hips.",    
  "Keep your gaze forward and head level."       
];

// Consistent string keys to track feedback frequency over the duration of a session
const ACTION_IDS = [
  "optimal",
  "slouching",
  "low-knee-drive",
  "poor-arm-swing",
  "hip-instability",
  "head-drop"
];

/**
 * Loads the pre-trained neural network policy parameters from your public assets folder
 */
export async function loadCoachPolicy() {
  // Pointing to your public assets folder (Vercel uses root domain routing)
  return await tf.loadLayersModel('/rl-models/coach_policy.json');
}

/**
 * Extracts raw coordinate arrays and outputs a structured 1D state tensor [1, 8]
 */
export function buildStateVector(keypoints: Keypoint[], metrics: RunningMetrics): number[] {
  const leftEar = keypoints[3] || { x: 0, y: 0, score: 0 };
  const rightEar = keypoints[4] || { x: 0, y: 0, score: 0 };
  const leftHip = keypoints[11] || { x: 0, y: 0, score: 0 };
  const rightHip = keypoints[12] || { x: 0, y: 0, score: 0 };

  const headTiltVal = headTilt(leftEar, rightEar);
  const hipTiltVal = hipTilt(leftHip, rightHip);

  return [
    Math.min(Math.max((metrics.torsoLean || 0) / 90.0, 0), 1),
    Math.min(Math.max((metrics.kneeLeft || 0) / 180.0, 0), 1),
    Math.min(Math.max((metrics.kneeRight || 0) / 180.0, 0), 1),
    Math.min(Math.max((metrics.hipExtension || 0) / 180.0, 0), 1),
    Math.min(Math.max((metrics.armSwingRatio || 0), 0), 1),
    Math.min(Math.max(hipTiltVal / 45.0, 0), 1),
    Math.min(Math.max(headTiltVal / 45.0, 0), 1),
    Math.min(Math.max((metrics.cadence || 0) / 240.0, 0), 1)
  ];
}

/**
 * Computes forward propagation and resolves the highest probability argmax index
 */
export function evaluatePoseRL(model: any, stateFeatures: number[]): CoachAction {
  return tf.tidy(() => {
    const inputTensor = tf.tensor2d([stateFeatures], [1, 8]);
    const prediction = model.predict(inputTensor) as tf.Tensor;
    const outputData = prediction.dataSync();
    
    let maxIdx = 0;
    let maxVal = outputData[0];
    for (let i = 1; i < outputData.length; i++) {
      if (outputData[i] > maxVal) {
        maxVal = outputData[i];
        maxIdx = i;
      }
    }
    
    return {
      actionId: ACTION_IDS[maxIdx] || "unknown",
      text: COACH_ACTIONS[maxIdx] || COACH_ACTIONS[0]
    };
  });
}