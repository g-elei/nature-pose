import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

let detector: poseDetection.PoseDetector | null = null;

export async function createDetector(): Promise<poseDetection.PoseDetector> {
  if (detector) return detector;

  await tf.ready();

  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    }
  );

  return detector;
}