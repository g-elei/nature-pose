import { Keypoint } from '../types/pose';
import {
  kneeFlexion,
  torsoLean,
  armSwingRatio,
  hipExtension,
  cadenceEstimate,
  isHeelStrike,
  hipTilt,
  headTilt
} from './angles';

// MoveNet 17‑keypoint index map
const IDX = {
  nose: 0, leftEye: 1, rightEye: 2, leftEar: 3, rightEar: 4,
  leftShoulder: 5, rightShoulder: 6, leftElbow: 7, rightElbow: 8,
  leftWrist: 9, rightWrist: 10, leftHip: 11, rightHip: 12,
  leftKnee: 13, rightKnee: 14, leftAnkle: 15, rightAnkle: 16,
};

function getKeypoint(kp: Keypoint[], idx: number): Keypoint {
  return kp[idx] || { x: 0, y: 0, score: 0 };
}

export interface RunningMetrics {
  kneeLeft: number;
  kneeRight: number;
  kneeAvg: number;
  torsoLean: number;
  armSwingRatio: number;
  cadence: number;
  isHeelStrike: boolean;
  hipExtension: number;
  score: number;
}

// Helper: compute composite score from raw metrics
function computeScore(metrics: Partial<RunningMetrics>): number {
  let penalty = 0;
  if (metrics.kneeAvg !== undefined) {
    if (metrics.kneeAvg < 60 || metrics.kneeAvg > 130) penalty += 25;
    else if (metrics.kneeAvg < 80 || metrics.kneeAvg > 110) penalty += 10;
  }
  if (metrics.torsoLean !== undefined) {
    if (metrics.torsoLean < 3 || metrics.torsoLean > 20) penalty += 25;
    else if (metrics.torsoLean < 5 || metrics.torsoLean > 15) penalty += 10;
  }
  if (metrics.isHeelStrike === true) penalty += 25;
  if (metrics.armSwingRatio !== undefined) {
    if (metrics.armSwingRatio < 0.3) penalty += 25;
    else if (metrics.armSwingRatio < 0.5) penalty += 10;
  }
  return Math.max(0, 100 - penalty);
}

export function analyzeRunning(
  kp: Keypoint[],
  ankleHistory?: number[][],
  fps?: number
): RunningMetrics {
  const leftKnee = kneeFlexion(
    getKeypoint(kp, IDX.leftHip), getKeypoint(kp, IDX.leftKnee), getKeypoint(kp, IDX.leftAnkle)
  );
  const rightKnee = kneeFlexion(
    getKeypoint(kp, IDX.rightHip), getKeypoint(kp, IDX.rightKnee), getKeypoint(kp, IDX.rightAnkle)
  );
  const lean = torsoLean(
    getKeypoint(kp, IDX.leftShoulder), getKeypoint(kp, IDX.rightShoulder),
    getKeypoint(kp, IDX.leftHip), getKeypoint(kp, IDX.rightHip)
  );
  const arm = armSwingRatio(
    getKeypoint(kp, IDX.leftWrist), getKeypoint(kp, IDX.leftShoulder)
  );
  const heel = isHeelStrike(
    getKeypoint(kp, IDX.leftAnkle), getKeypoint(kp, IDX.leftKnee)
  );
  const hipExt = hipExtension(
    getKeypoint(kp, IDX.leftShoulder), getKeypoint(kp, IDX.leftHip), getKeypoint(kp, IDX.leftKnee)
  );
  let cadence = 0;
  if (ankleHistory && fps && ankleHistory.length > 0) {
    const yVals = ankleHistory.map(frame => frame[IDX.leftAnkle]).filter(y => y > 0);
    cadence = cadenceEstimate(yVals, fps);
  }

  const metrics = {
    kneeLeft: leftKnee,
    kneeRight: rightKnee,
    kneeAvg: (leftKnee + rightKnee) / 2,
    torsoLean: lean,
    armSwingRatio: arm,
    cadence,
    isHeelStrike: heel,
    hipExtension: hipExt,
    score: 0,
  };
  metrics.score = computeScore(metrics);
  return metrics;
}

export function analyzeWalking(
  kp: Keypoint[],
  ankleHistory?: number[][],
  fps?: number
): RunningMetrics {
  // For prototype, reuse running logic
  return analyzeRunning(kp, ankleHistory, fps);
}

export function analyzeYoga(kp: Keypoint[]): RunningMetrics {
  const leftHip = getKeypoint(kp, IDX.leftHip);
  const rightHip = getKeypoint(kp, IDX.rightHip);
  const leftShoulder = getKeypoint(kp, IDX.leftShoulder);
  const rightShoulder = getKeypoint(kp, IDX.rightShoulder);
  const leftWrist = getKeypoint(kp, IDX.leftWrist);
  const hipTiltVal = hipTilt(leftHip, rightHip);
  const armSwing = armSwingRatio(leftWrist, leftShoulder);
  const lean = torsoLean(leftShoulder, rightShoulder, leftHip, rightHip);
  const metrics: RunningMetrics = {
    kneeLeft: 0, kneeRight: 0, kneeAvg: 0,
    torsoLean: lean,
    armSwingRatio: armSwing,
    cadence: 0,
    isHeelStrike: false,
    hipExtension: 0,
    score: 0,
  };
  // Score for yoga: penalize hip tilt and poor arm position
  let penalty = 0;
  if (hipTiltVal > 30) penalty += 25;
  else if (hipTiltVal > 15) penalty += 10;
  if (armSwing < 0.2) penalty += 25;
  else if (armSwing < 0.4) penalty += 10;
  metrics.score = Math.max(0, 100 - penalty);
  return metrics;
}

export function analyzeSitting(kp: Keypoint[]): RunningMetrics {
  const leftEar = getKeypoint(kp, IDX.leftEar);
  const rightEar = getKeypoint(kp, IDX.rightEar);
  const leftShoulder = getKeypoint(kp, IDX.leftShoulder);
  const rightShoulder = getKeypoint(kp, IDX.rightShoulder);
  const leftHip = getKeypoint(kp, IDX.leftHip);
  const rightHip = getKeypoint(kp, IDX.rightHip);
  const headTiltVal = headTilt(leftEar, rightEar);
  const lean = torsoLean(leftShoulder, rightShoulder, leftHip, rightHip);
  const metrics: RunningMetrics = {
    kneeLeft: 0, kneeRight: 0, kneeAvg: 0,
    torsoLean: lean,
    armSwingRatio: 0,
    cadence: 0,
    isHeelStrike: false,
    hipExtension: 0,
    score: 0,
  };
  let penalty = 0;
  if (headTiltVal > 20) penalty += 25;
  else if (headTiltVal > 8) penalty += 10;
  if (lean < 70 || lean > 110) penalty += 25;
  else if (lean < 80 || lean > 100) penalty += 10;
  metrics.score = Math.max(0, 100 - penalty);
  return metrics;
}