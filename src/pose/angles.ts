import { Keypoint } from '../types/pose';

// Angle at point B formed by A→B→C (degrees, 0–180)
export function angle3pt(
  ax: number, ay: number,
  bx: number, by: number,
  cx: number, cy: number
): number {
  const ab = Math.hypot(ax - bx, ay - by);
  const bc = Math.hypot(bx - cx, by - cy);
  const ac = Math.hypot(ax - cx, ay - cy);
  if (ab === 0 || bc === 0) return 0;
  const cos = (ab * ab + bc * bc - ac * ac) / (2 * ab * bc);
  const rad = Math.acos(Math.min(1, Math.max(-1, cos)));
  return rad * 180 / Math.PI;
}

// Forward lean of torso from vertical (degrees, 0–90)
export function torsoLean(
  leftShoulder: Keypoint, rightShoulder: Keypoint,
  leftHip: Keypoint, rightHip: Keypoint
): number {
  const midShoulder = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
  const midHip = { x: (leftHip.x + rightHip.x) / 2, y: (leftHip.y + rightHip.y) / 2 };
  const dx = midShoulder.x - midHip.x;
  const dy = midShoulder.y - midHip.y;
  const angle = Math.atan2(dx, dy) * 180 / Math.PI;
  return Math.abs(angle);
}

// Arm swing ratio: wrist horizontal displacement / shoulder vertical displacement
export function armSwingRatio(wrist: Keypoint, shoulder: Keypoint): number {
  const dx = Math.abs(wrist.x - shoulder.x);
  const dy = Math.abs(shoulder.y - wrist.y);
  return dy > 0 ? dx / dy : 0;
}

// Knee flexion angle (degrees)
export function kneeFlexion(hip: Keypoint, knee: Keypoint, ankle: Keypoint): number {
  return angle3pt(hip.x, hip.y, knee.x, knee.y, ankle.x, ankle.y);
}

// Hip extension angle (degrees)
export function hipExtension(shoulder: Keypoint, hip: Keypoint, knee: Keypoint): number {
  return angle3pt(shoulder.x, shoulder.y, hip.x, hip.y, knee.x, knee.y);
}

// Cadence estimate from ankle y‑oscillation history (steps/min)
export function cadenceEstimate(ankleHistory: number[], fps: number): number {
  if (ankleHistory.length < 2) return 0;
  let crossings = 0;
  for (let i = 1; i < ankleHistory.length; i++) {
    if ((ankleHistory[i-1] - 0.5) * (ankleHistory[i] - 0.5) < 0) crossings++;
  }
  const durationSec = (ankleHistory.length - 1) / fps;
  if (durationSec === 0) return 0;
  const stepsPerSec = crossings / 2 / durationSec;
  return Math.round(stepsPerSec * 60);
}

// Heel strike: ankle x > knee x (foot ahead of knee)
export function isHeelStrike(ankle: Keypoint, knee: Keypoint): boolean {
  return ankle.x > knee.x;
}

// Hip tilt asymmetry (difference in x between left and right hip, normalized)
export function hipTilt(leftHip: Keypoint, rightHip: Keypoint): number {
  return Math.abs(leftHip.x - rightHip.x);
}

// Head tilt (difference in x between left and right ear)
export function headTilt(leftEar: Keypoint, rightEar: Keypoint): number {
  return Math.abs(leftEar.x - rightEar.x);
}