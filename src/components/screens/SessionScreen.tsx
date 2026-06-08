import React, { useEffect, useRef, useCallback, useState } from 'react';
import { useSettings } from '../../store/SettingsContext';
import { useSession } from '../../store/SessionContext';
import { usePoseUpdate } from '../../store/PoseContext';
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

  const onPoseResult = useCallback((metrics: RunningMetrics, keypoints: Keypoint[]) => {
    setLastMetrics(metrics);
    updatePose(keypoints, metrics, 30);

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

  const detectVideoSource = inputMode === 'camera' ? videoRef.current : uploadedVideoRef.current;

  usePoseDetection({
    videoElement: detectVideoSource,
    canvasElement: canvasRef.current,
    activity,
    onResult: onPoseResult,
    enabled: inputMode === 'camera' || inputMode === 'video',
  });

  useEffect(() => {
    startSession(activity);
    startCamera();
    return () => { stopCamera(); };
  }, []);

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
  }, [activity, recordFrame, speak, voiceMuted]);

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
          <div className="absolute inset-0 flex items-center justify-center bg-cream-50 z-20">
            <p className="text-red-600 text-sm">{cameraError || uploadError}</p>
          </div>
        )}
        <CameraFeed videoRef={videoRef} canvasRef={canvasRef} />
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
          framing={useFramingGuide([], 640, 480)}
        />
      </div>

      <div className="bg-cream-50 border-t border-sage-200 p-4 space-y-3">
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