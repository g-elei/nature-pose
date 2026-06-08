import { useEffect, useRef } from 'react';
import { createDetector } from '../lib/poseDetector';
import { analyzeRunning, analyzeWalking, analyzeYoga, analyzeSitting, RunningMetrics } from '../pose/analyzer';
import { drawSkeleton, drawKeypoints } from '../pose/skeleton';
import { ActivityKind } from '../types/session';
import { Keypoint } from '../types/pose';

interface UsePoseDetectionOptions {
  videoElement: HTMLVideoElement | null;
  canvasElement: HTMLCanvasElement | null;
  activity: ActivityKind;
  onResult: (metrics: RunningMetrics, keypoints: Keypoint[]) => void;
  enabled?: boolean;
}

export function usePoseDetection({
  videoElement,
  canvasElement,
  activity,
  onResult,
  enabled = true,
}: UsePoseDetectionOptions) {
  const frameIdRef = useRef<number | null>(null);
  const lastCallbackTimeRef = useRef(0);
  const ankleHistoryRef = useRef<number[][]>([]);
  const fpsRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(performance.now());

  useEffect(() => {
    if (!enabled || !videoElement || !canvasElement) return;

    let detector: any = null;
    let isActive = true;

    const runDetection = async () => {
      if (!isActive) return;
      detector = await createDetector();
      const ctx = canvasElement.getContext('2d');
      if (!ctx) return;

      const loop = async () => {
        if (!isActive || !videoElement || !canvasElement || !detector) {
          frameIdRef.current = null;
          return;
        }

        const now = performance.now();
        // Update FPS roughly
        frameCountRef.current++;
        if (now - lastFpsUpdateRef.current >= 1000) {
          fpsRef.current = frameCountRef.current;
          frameCountRef.current = 0;
          lastFpsUpdateRef.current = now;
        }

        // Run pose estimation
        const poses = await detector.estimatePoses(videoElement);
        if (poses.length > 0 && poses[0].keypoints) {
          const keypoints = poses[0].keypoints as Keypoint[];
          // Store ankle positions for cadence (use left ankle index 15)
          const leftAnkle = keypoints[15];
          if (leftAnkle && leftAnkle.score > 0.3) {
            ankleHistoryRef.current.push(keypoints.map(kp => kp.x));
            if (ankleHistoryRef.current.length > 100) ankleHistoryRef.current.shift();
          }

          // Compute metrics based on activity
          let metrics: RunningMetrics;
          switch (activity) {
            case 'running':
              metrics = analyzeRunning(keypoints, ankleHistoryRef.current, fpsRef.current);
              break;
            case 'walking':
              metrics = analyzeWalking(keypoints, ankleHistoryRef.current, fpsRef.current);
              break;
            case 'yoga':
              metrics = analyzeYoga(keypoints);
              break;
            case 'sitting':
              metrics = analyzeSitting(keypoints);
              break;
            default:
              metrics = analyzeRunning(keypoints);
          }

          // Throttle callback to ~5Hz (200ms)
          if (now - lastCallbackTimeRef.current >= 200) {
            onResult(metrics, keypoints);
            lastCallbackTimeRef.current = now;
          }

          // Draw skeleton every frame
          ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
          drawSkeleton(ctx, keypoints, '#2d6a4f');
          drawKeypoints(ctx, keypoints, '#b7e4c7');
        }

        frameIdRef.current = requestAnimationFrame(loop);
      };

      loop();
    };

    runDetection();

    return () => {
      isActive = false;
      if (frameIdRef.current) cancelAnimationFrame(frameIdRef.current);
    };
  }, [videoElement, canvasElement, activity, onResult, enabled]);
}