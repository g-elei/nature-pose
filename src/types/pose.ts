export interface Keypoint {
  x: number;
  y: number;
  score: number;
}

export interface DetectedPose {
  keypoints: Keypoint[];
}