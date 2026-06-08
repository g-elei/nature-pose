// src/hooks/usePoseDetection.ts
import { useEffect, useRef } from 'react';
import { createDetector } from '../lib/poseDetector';
import { analyzeRunning, analyzeWalking, analyzeYoga, analyzeSitting, RunningMetrics } from '../pose/analyzer';
import { drawSkeleton, drawKeypoints } from '../pose/skeleton';
import { ActivityKind } from '../types/session';
import { Keypoint } from '../types/pose';
import { loadCoachPolicy, buildStateVector, evaluatePoseRL, CoachAction } from '../rl/coach';

interface UsePoseDetectionOptions {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  activity: ActivityKind;
  // Updated onResult parameter to emit the active frame inference out to your dashboard layout context
  onResult: (metrics: RunningMetrics, keypoints: Keypoint[], coachAction: CoachAction | null) => void;
  enabled?: boolean;
}

export function usePoseDetection({
  videoRef,
  canvasRef,
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
  
  const detectorRef = useRef<any>(null);
  const detectorLoadingRef = useRef(false);

  // Added references to handle neural-network lifecycle management in sync with user loop
  const coachModelRef = useRef<any>(null);
  const coachLoadingRef = useRef(false);

  useEffect(() => {
    if (!enabled) return;

    let isActive = true;

    const loop = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!isActive || !video || !canvas) {
        frameIdRef.current = requestAnimationFrame(loop);
        return;
      }

      if (video.readyState < 2) {
        frameIdRef.current = requestAnimationFrame(loop);
        return;
      }

      const now = performance.now();

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        frameIdRef.current = requestAnimationFrame(loop);
        return;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Lazily instantiate MoveNet Estimator
      if (!detectorRef.current && !detectorLoadingRef.current) {
        detectorLoadingRef.current = true;
        try {
          detectorRef.current = await createDetector();
        } catch (e) {
          console.error('Failed to create detector:', e);
        }
        detectorLoadingRef.current = false;
      }

      // Lazily instantiate pre-trained RL Coach Inference Model
      if (!coachModelRef.current && !coachLoadingRef.current) {
        coachLoadingRef.current = true;
        try {
          coachModelRef.current = await loadCoachPolicy();
        } catch (e) {
          console.error('Failed to load RL coach policy:', e);
        }
        coachLoadingRef.current = false;
      }

      const detector = detectorRef.current;
      if (!detector) {
        frameIdRef.current = requestAnimationFrame(loop);
        return;
      }

      frameCountRef.current++;
      if (now - lastFpsUpdateRef.current >= 1000) {
        fpsRef.current = frameCountRef.current;
        frameCountRef.current = 0;
        lastFpsUpdateRef.current = now;
      }

      try {
        const poses = await detector.estimatePoses(video);
        if (poses.length > 0 && poses[0].keypoints) {
          const keypoints = poses[0].keypoints as Keypoint[];

          const leftAnkle = keypoints[15];
          if (leftAnkle && leftAnkle.score > 0.3) {
            ankleHistoryRef.current.push(keypoints.map(kp => kp.x));
            if (ankleHistoryRef.current.length > 100) ankleHistoryRef.current.shift();
          }

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

          // Evaluate the frame using RL Policy model
          let currentAction: CoachAction | null = null;
          if (coachModelRef.current) {
            const stateFeatures = buildStateVector(keypoints, metrics);
            currentAction = evaluatePoseRL(coachModelRef.current, stateFeatures);
          }

          // Emit metrics and model evaluation results down to active consumer hooks every 200ms
          if (now - lastCallbackTimeRef.current >= 200) {
            onResult(metrics, keypoints, currentAction);
            lastCallbackTimeRef.current = now;
          }

          drawSkeleton(ctx, keypoints, '#00ff88');
          drawKeypoints(ctx, keypoints, '#ff4444');
        }
      } catch (e) {
        // Fallthrough protection to ensure frame drops don't completely lock thread execution
      }

      frameIdRef.current = requestAnimationFrame(loop);
    };

    frameIdRef.current = requestAnimationFrame(loop);

    return () => {
      isActive = false;
      if (frameIdRef.current) cancelAnimationFrame(frameIdRef.current);
    };
  }, [videoRef, canvasRef, activity, onResult, enabled]);
}