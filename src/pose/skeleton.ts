import { Keypoint } from '../types/pose';

// COCO skeleton pairs for MoveNet 17-keypoint model (0‑based indices)
export const CONNECTIONS: [number, number][] = [
  [5, 6],   // left shoulder → right shoulder
  [5, 7],   // left shoulder → left elbow
  [7, 9],   // left elbow → left wrist
  [6, 8],   // right shoulder → right elbow
  [8, 10],  // right elbow → right wrist
  [5, 11],  // left shoulder → left hip
  [6, 12],  // right shoulder → right hip
  [11, 12], // left hip → right hip
  [11, 13], // left hip → left knee
  [13, 15], // left knee → left ankle
  [12, 14], // right hip → right knee
  [14, 16], // right knee → right ankle
];

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  color: string
): void {
  if (!keypoints.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [i, j] of CONNECTIONS) {
    const p1 = keypoints[i];
    const p2 = keypoints[j];
    if (p1 && p2 && p1.score > 0.3 && p2.score > 0.3) {
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
    }
  }
  ctx.stroke();
  ctx.restore();
}

export function drawKeypoints(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  color: string
): void {
  if (!keypoints.length) return;
  ctx.save();
  ctx.fillStyle = color;
  for (const kp of keypoints) {
    if (kp.score > 0.3) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.restore();
}