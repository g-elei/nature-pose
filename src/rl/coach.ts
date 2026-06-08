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
 * Bypasses the non-existent JSON model asset and returns a safe fallback token
 */
export async function loadCoachPolicy() {
  console.warn('coach_policy.json not present. Activating heuristic rule-engine.');
  // Return a mock model signature so initialization hooks resolve successfully
  return { isMockModel: true };
}

/**
 * Extracts raw coordinate arrays and outputs a structured 1D state array
 */
export function buildStateVector(keypoints: Keypoint[], metrics: RunningMetrics): number[] {
  const leftEar = keypoints[3] || { x: 0, y: 0, score: 0 };
  const rightEar = keypoints[4] || { x: 0, y: 0, score: 0 };
  const leftHip = keypoints[11] || { x: 0, y: 0, score: 0 };
  const rightHip = keypoints[12] || { x: 0, y: 0, score: 0 };

  const headTiltVal = headTilt(leftEar, rightEar);
  const hipTiltVal = hipTilt(leftHip, rightHip);

  // Normalizes body metrics consistently between 0.0 and 1.0
  return [
    Math.min(Math.max((metrics.torsoLean || 0) / 90.0, 0), 1),       // index 0
    Math.min(Math.max((metrics.kneeLeft || 0) / 180.0, 0), 1),       // index 1
    Math.min(Math.max((metrics.kneeRight || 0) / 180.0, 0), 1),      // index 2
    Math.min(Math.max((metrics.hipExtension || 0) / 180.0, 0), 1),   // index 3
    Math.min(Math.max((metrics.armSwingRatio || 0), 0), 1),          // index 4
    Math.min(Math.max(hipTiltVal / 45.0, 0), 1),                     // index 5
    Math.min(Math.max(headTiltVal / 45.0, 0), 1),                    // index 6
    Math.min(Math.max((metrics.cadence || 0) / 240.0, 0), 1)         // index 7
  ];
}

/**
 * Evaluates features using safe kinematic heuristics instead of a neural network file
 */
export function evaluatePoseRL(model: any, stateFeatures: number[]): CoachAction {
  // Reconstruct real-world values from the normalized state array
  const torsoLeanDegrees = stateFeatures[0] * 90;
  const avgKneeFlexion = ((stateFeatures[1] + stateFeatures[2]) / 2) * 180;
  const armSwingRatio = stateFeatures[4];
  const hipTiltDegrees = stateFeatures[5] * 45;
  const headTiltDegrees = stateFeatures[6] * 45;

  let chosenIdx = 0; // Default to "optimal" Form

  // Evaluate kinematic rules hierarchically
  if (torsoLeanDegrees > 25) {
    chosenIdx = 1; // slouching
  } else if (headTiltDegrees > 15) {
    chosenIdx = 5; // head-drop
  } else if (armSwingRatio < 0.25) {
    chosenIdx = 3; // poor-arm-swing
  } else if (hipTiltDegrees > 12) {
    chosenIdx = 4; // hip-instability
  } else if (avgKneeFlexion < 45 && stateFeatures[7] > 0.1) { 
    chosenIdx = 2; // low-knee-drive
  }

  return {
    actionId: ACTION_IDS[chosenIdx] || "optimal",
    text: COACH_ACTIONS[chosenIdx] || COACH_ACTIONS[0]
  };
}