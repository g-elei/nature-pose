import { Keypoint } from '../types/pose';

export interface FramingResult {
  status: 'good' | 'too-close' | 'too-far' | 'off-center';
  boxLeft: number;
  boxTop: number;
  width: number;
  height: number;
  message: string;
}

export function checkFraming(
  keypoints: Keypoint[],
  videoW: number,
  _videoH: number
): FramingResult | null {
  // Use nose, shoulders, hips
  const nose = keypoints[0];
  const leftShoulder = keypoints[5];
  const rightShoulder = keypoints[6];
  const leftHip = keypoints[11];
  const rightHip = keypoints[12];
  const visible = [nose, leftShoulder, rightShoulder, leftHip, rightHip]
    .filter(kp => kp && kp.score > 0.3);
  if (visible.length < 3) {
    return null;
  }

  const xs = visible.map(kp => kp.x);
  const ys = visible.map(kp => kp.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const boxLeft = minX;
  const boxTop = minY;
  const width = maxX - minX;
  const height = maxY - minY;

  const widthRatio = width / videoW;
  const centerX = (minX + maxX) / 2;
  const centerXRatio = centerX / videoW;

  let status: FramingResult['status'] = 'good';
  let message = 'Good framing';

  if (widthRatio > 0.6) {
    status = 'too-close';
    message = 'Step back — you are too close';
  } else if (widthRatio < 0.1) {
    status = 'too-far';
    message = 'Move closer to the camera';
  } else if (centerXRatio < 0.2 || centerXRatio > 0.8) {
    status = 'off-center';
    message = 'Center yourself in the frame';
  }

  return { status, boxLeft, boxTop, width, height, message };
}