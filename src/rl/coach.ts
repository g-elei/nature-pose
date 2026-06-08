// src/rl/coach.ts
import { Keypoint } from '../types/pose';
import { RunningMetrics } from '../pose/analyzer';
import { headTilt, hipTilt } from '../pose/angles';

export interface CoachAction {
  actionId: string; 
  text: string;     
  advice: string;   
  voiceCue: string; 
  severity?: 'good' | 'warning' | 'critical'; // Added to support SessionScreen check loops
}

export const COACH_ACTIONS = [
  "Form is excellent. Maintain this standard!", 
  "Straighten your spine. Avoid slouching.",     
  "Drive your knees higher to increase power.",  
  "Relax your shoulders and pump your arms.",    
  "Engage your core to stabilize your hips.",    
  "Keep your gaze forward and head level."       
];

const ACTION_IDS = [
  "optimal",
  "slouching",
  "low-knee-drive",
  "poor-arm-swing",
  "hip-instability",
  "head-drop"
];

export async function loadCoachPolicy() {
  console.warn('coach_policy.json not present. Activating heuristic rule-engine.');
  return { isMockModel: true };
}

/**
 * Fixes TS2554, TS2345, and TS2339 in SessionScreen.tsx
 * Accepts metric telemetry data and structures it into proper CoachAction objects.
 */
export function getCoachingActions(metrics: any, _activity?: any): CoachAction[] {
  const torsoLeanDegrees = metrics?.torsoLean || 0;
  let chosenIdx = 0;

  if (torsoLeanDegrees > 25) {
    chosenIdx = 1; // slouching
  }

  const textOutput = COACH_ACTIONS[chosenIdx] || COACH_ACTIONS[0];
  const severityOutput = torsoLeanDegrees > 25 ? 'warning' : 'good';

  return [
    {
      actionId: ACTION_IDS[chosenIdx] || "optimal",
      text: textOutput,
      advice: textOutput,
      voiceCue: textOutput,
      severity: severityOutput
    }
  ];
}

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

export function evaluatePoseRL(_model: any, stateFeatures: number[]): CoachAction {
  const torsoLeanDegrees = stateFeatures[0] * 90;
  const avgKneeFlexion = ((stateFeatures[1] + stateFeatures[2]) / 2) * 180;
  const armSwingRatio = stateFeatures[4];
  const hipTiltDegrees = stateFeatures[5] * 45;
  const headTiltDegrees = stateFeatures[6] * 45;

  let chosenIdx = 0; 

  if (torsoLeanDegrees > 25) {
    chosenIdx = 1; 
  } else if (headTiltDegrees > 15) {
    chosenIdx = 5; 
  } else if (armSwingRatio < 0.25) {
    chosenIdx = 3; 
  } else if (hipTiltDegrees > 12) {
    chosenIdx = 4; 
  } else if (avgKneeFlexion < 45 && stateFeatures[7] > 0.1) { 
    chosenIdx = 2; 
  }

  const textOutput = COACH_ACTIONS[chosenIdx] || COACH_ACTIONS[0];

  return {
    actionId: ACTION_IDS[chosenIdx] || "optimal",
    text: textOutput,
    advice: textOutput,   
    voiceCue: textOutput  
  };
}