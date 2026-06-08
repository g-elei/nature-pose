import { useEffect, useRef } from 'react';
import { createDetector } from '../lib/poseDetector';
import { analyzeRunning, analyzeWalking, analyzeYoga, analyzeSitting, RunningMetrics } from '../pose/analyzer';
import { drawSkeleton, drawKeypoints } from '../pose/skeleton';
import { ActivityKind } from '../types/session';
import { Keypoint } from '../types/pose';

interface UsePoseDetectionOptions {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  activity: ActivityKind;
  onResult: (metrics: RunningMetrics, keypoints: Keypoint[]) => void;
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

      // Resize canvas to match video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        frameIdRef.current = requestAnimationFrame(loop);
        return;
      }

      // Draw video frame onto canvas so user sees camera feed
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Load detector lazily (once)
      if (!detectorRef.current && !detectorLoadingRef.current) {
        detectorLoadingRef.current = true;
        try {
          detectorRef.current = await createDetector();
        } catch (e) {
          console.error('Failed to create detector:', e);
        }
        detectorLoadingRef.current = false;
      }

      const detector = detectorRef.current;
      if (!detector) {
        frameIdRef.current = requestAnimationFrame(loop);
        return;
      }

      // FPS counter
      frameCountRef.current++;
      if (now - lastFpsUpdateRef.current >= 1000) {
        fpsRef.current = frameCountRef.current;
        frameCountRef.current = 0;
        lastFpsUpdateRef.current = now;
      }

      // Run pose estimation
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

          if (now - lastCallbackTimeRef.current >= 200) {
            onResult(metrics, keypoints);
            lastCallbackTimeRef.current = now;
          }

          drawSkeleton(ctx, keypoints, '#00ff88');
          drawKeypoints(ctx, keypoints, '#ff4444');
        }
      } catch (e) {
        // Silently retry next frame
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