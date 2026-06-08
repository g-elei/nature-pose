import React, { useEffect, useRef, useCallback, useState } from 'react';
import { useSettings } from '../../store/SettingsContext';
import { useSession } from '../../store/SessionContext';
import { usePose, usePoseUpdate } from '../../store/PoseContext';
import { usePoseDetection } from '../../hooks/usePoseDetection';
import { useCamera } from '../../hooks/useCamera';
import { useCanvas } from '../../hooks/useCanvas';
import { useFramingGuide } from '../../hooks/useFramingGuide';
import { useVoiceCoach } from '../../hooks/useVoiceCoach';
import { getCoachingActions } from '../../rl/coach';
import { analyzeRunning, analyzeWalking, analyzeYoga, analyzeSitting, RunningMetrics } from '../../pose/analyzer';
import { createDetector } from '../../lib/poseDetector';
import { Keypoint } from '../../types/pose';
import CameraFeed from '../camera/CameraFeed';
import FramingGuide from '../camera/FramingGuide';
import InputSelector from '../camera/InputSelector';
import FeedbackBubble from '../feedback/FeedbackBubble';
import MetricBar from '../feedback/MetricBar';
import NatureButton from '../ui/NatureButton';

const ANALYZE_FN: Record<string, typeof analyzeRunning> = {
  running: analyzeRunning,
  walking: analyzeWalking,
  yoga: analyzeYoga,
  sitting: analyzeSitting,
};

interface SessionScreenProps {
  onFinish: () => void;
}

export const SessionScreen: React.FC<SessionScreenProps> = ({ onFinish }) => {
  const { activity, voiceMuted } = useSettings();
  const { startSession, recordFrame, endSession } = useSession();
  const updatePose = usePoseUpdate();
  const { keypoints } = usePose();

  const { videoRef, start: startCamera, stop: stopCamera, error: cameraError } = useCamera();
  const { canvasRef } = useCanvas();
  const { speak } = useVoiceCoach();

  const [inputMode, setInputMode] = useState<'camera' | 'image' | 'video'>('camera');
  const [lastMetrics, setLastMetrics] = useState<RunningMetrics | null>(null);
  const [lastActions, setLastActions] = useState<any[]>([]);
  const [severity, setSeverity] = useState<'good' | 'warning' | 'critical'>('good');
  const lastActionsRef = useRef<any[]>([]);
  const uploadedVideoRef = useRef<HTMLVideoElement | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [videoDimensions, setVideoDimensions] = useState({ w: 640, h: 480 });

  const onPoseResult = useCallback((metrics: RunningMetrics, kp: Keypoint[]) => {
    setLastMetrics(metrics);
    updatePose(kp, metrics, 30);

    const actions = getCoachingActions(metrics, activity);
    setLastActions(actions);
    lastActionsRef.current = actions;

    const sev = actions.some((a) => a.severity === 'critical')
      ? 'critical'
      : actions.some((a) => a.severity === 'warning')
        ? 'warning'
        : 'good';
    setSeverity(sev);

    recordFrame(metrics.score, actions);

    if (!voiceMuted && actions.length > 0) {
      speak(actions[0].voiceCue);
    }
  }, [activity, updatePose, recordFrame, speak, voiceMuted]);

  const detectVideoRef = inputMode === 'camera' ? videoRef : uploadedVideoRef;

  usePoseDetection({
    videoRef: detectVideoRef as React.RefObject<HTMLVideoElement>,
    canvasRef,
    activity,
    onResult: onPoseResult,
    enabled: inputMode === 'camera' || inputMode === 'video',
  });

  useEffect(() => {
    startSession(activity);
    startCamera();
    return () => { stopCamera(); };
  }, []);

  useEffect(() => {
    if (videoRef.current) {
      const checkDimensions = setInterval(() => {
        const v = videoRef.current;
        if (v && v.videoWidth > 0) {
          setVideoDimensions({ w: v.videoWidth, h: v.videoHeight });
          clearInterval(checkDimensions);
        }
      }, 100);
      return () => clearInterval(checkDimensions);
    }
  }, [videoRef.current]);

  const handleImageUpload = useCallback(async (file: File) => {
    setUploadError(null);
    const img = new Image();
    img.onload = async () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx?.drawImage(img, 0, 0);
        const detector = await createDetector();
        const poses = await detector.estimatePoses(canvas);
        if (poses[0]?.keypoints) {
          const kp = (poses[0].keypoints as unknown) as Keypoint[];
          const analyze = ANALYZE_FN[activity] || analyzeRunning;
          const metrics = analyze(kp);
          updatePose(kp, metrics, 0);
          setLastMetrics(metrics);
          const actions = getCoachingActions(metrics, activity);
          setLastActions(actions);
          lastActionsRef.current = actions;
          const sev = actions.some((a) => a.severity === 'critical') ? 'critical'
            : actions.some((a) => a.severity === 'warning') ? 'warning' : 'good';
          setSeverity(sev);
          recordFrame(metrics.score, actions);
          if (!voiceMuted && actions.length > 0) speak(actions[0].voiceCue);
        } else {
          setUploadError('No person detected in image');
        }
      } catch (e) {
        setUploadError('Failed to analyze image');
      }
    };
    img.onerror = () => setUploadError('Failed to load image');
    img.src = URL.createObjectURL(file);
    setInputMode('image');
  }, [activity, recordFrame, speak, voiceMuted, updatePose]);

  const handleVideoUpload = useCallback(async (file: File) => {
    setUploadError(null);
    const video = document.createElement('video');
    video.src = URL.createObjectURL(file);
    video.muted = true;
    video.loop = false;
    video.playsInline = true;
    video.onloadeddata = () => {
      video.play().catch(() => setUploadError('Failed to play video'));
      uploadedVideoRef.current = video;
      setVideoDimensions({ w: video.videoWidth, h: video.videoHeight });
      setInputMode('video');
    };
    video.onerror = () => setUploadError('Failed to load video');
  }, []);

  const handleFinish = async () => {
    await endSession();
    onFinish();
  };

  return (
    <div className="min-h-screen bg-cream-100 flex flex-col">
      <div className="flex-1 relative">
        {(cameraError || uploadError) && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-cream-50">
            <p className="rounded-lg bg-red-100 px-4 py-2 text-sm text-red-700">
              {cameraError || uploadError}
            </p>
          </div>
        )}
        <CameraFeed
          videoRef={videoRef}
          canvasRef={canvasRef}
          width={videoDimensions.w}
          height={videoDimensions.h}
        />
        {inputMode === 'video' && uploadedVideoRef.current && (
          <video
            ref={uploadedVideoRef as React.Ref<HTMLVideoElement>}
            autoPlay
            muted
            playsInline
            className="absolute inset-0 h-full w-full object-contain"
          />
        )}
        <FramingGuide
          framing={useFramingGuide(keypoints, videoDimensions.w, videoDimensions.h)}
        />
      </div>

      <div className="space-y-3 border-t border-sage-200 bg-cream-50 p-4">
        {lastMetrics && (
          <div className="grid grid-cols-2 gap-3">
            <MetricBar
              label="Knee Angle"
              value={lastMetrics.kneeAvg}
              unit="°"
              min={0}
              max={180}
              idealMin={80}
              idealMax={110}
            />
            <MetricBar
              label="Torso Lean"
              value={lastMetrics.torsoLean}
              unit="°"
              min={0}
              max={90}
              idealMin={5}
              idealMax={15}
            />
          </div>
        )}

        {lastActions.length > 0 && (
          <FeedbackBubble coachActions={lastActions} severity={severity} />
        )}

        <NatureButton variant="primary" onClick={handleFinish} className="w-full py-3 text-center">
          Finish & Save
        </NatureButton>
      </div>

      <InputSelector
        activeMode={inputMode}
        onSelectImage={handleImageUpload}
        onSelectVideo={handleVideoUpload}
        onSelectCamera={() => setInputMode('camera')}
      />
    </div>
  );
};