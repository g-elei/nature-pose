import React, { createContext, useContext, useRef, useState, useEffect, ReactNode } from 'react';
import { Keypoint } from '../types/pose';
import { RunningMetrics } from '../pose/analyzer';

interface PoseReadContextValue {
  keypoints: Keypoint[];
  metrics: RunningMetrics | null;
  fps: number;
}

type PoseUpdateFn = (kp: Keypoint[], m: RunningMetrics | null, f: number) => void;

const PoseReadContext = createContext<PoseReadContextValue | undefined>(undefined);
const PoseUpdateContext = createContext<PoseUpdateFn | undefined>(undefined);

export const usePose = () => {
  const ctx = useContext(PoseReadContext);
  if (!ctx) throw new Error('usePose must be used within PoseProvider');
  return ctx;
};

export const usePoseUpdate = () => {
  const ctx = useContext(PoseUpdateContext);
  if (!ctx) throw new Error('usePoseUpdate must be used within PoseProvider');
  return ctx;
};

interface PoseProviderProps {
  children: ReactNode;
}

export const PoseProvider: React.FC<PoseProviderProps> = ({ children }) => {
  const keypointsRef = useRef<Keypoint[]>([]);
  const metricsRef = useRef<RunningMetrics | null>(null);
  const fpsRef = useRef<number>(0);

  const [readState, setReadState] = useState<PoseReadContextValue>({
    keypoints: [],
    metrics: null,
    fps: 0,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setReadState({
        keypoints: keypointsRef.current,
        metrics: metricsRef.current,
        fps: fpsRef.current,
      });
    }, 200);
    return () => clearInterval(interval);
  }, []);

  const updatePose: PoseUpdateFn = (kp, m, f) => {
    keypointsRef.current = kp;
    metricsRef.current = m;
    fpsRef.current = f;
  };

  return (
    <PoseReadContext.Provider value={readState}>
      <PoseUpdateContext.Provider value={updatePose}>
        {children}
      </PoseUpdateContext.Provider>
    </PoseReadContext.Provider>
  );
};