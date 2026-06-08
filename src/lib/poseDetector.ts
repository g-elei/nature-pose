import * as poseDetection from '@tensorflow-models/pose-detection';
import { isChinaNetwork } from './chinaNetwork';
import * as tf from '@tensorflow/tfjs-core';

let detector: poseDetection.PoseDetector | null = null;

export async function createDetector(): Promise<poseDetection.PoseDetector> {
  if (detector) return detector;

  await tf.ready();

  // For China deployments, place model files in public/movenet-model/
  // Download from: https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4
  // Files needed: model.json + group1-shard1of1.bin
  const modelUrl = isChinaNetwork()
    ? '/movenet-model/model.json'
    : undefined;

  try {
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        ...(modelUrl ? { modelUrl } : {}),
      }
    );
  } catch (err) {
    if (modelUrl) {
      console.warn(
        'Local model not found at /movenet-model/model.json, falling back to CDN.'
      );
      detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        }
      );
    } else {
      throw err;
    }
  }

  return detector;
}