import { useState, useEffect } from 'react';
import { checkFraming, FramingResult } from '../pose/framing';
import { Keypoint } from '../types/pose';

export function useFramingGuide(keypoints: Keypoint[], videoWidth: number, videoHeight: number) {
  const [framing, setFraming] = useState<FramingResult | null>(null);

  useEffect(() => {
    if (!keypoints.length || videoWidth === 0 || videoHeight === 0) {
      setFraming(null);
      return;
    }
    const result = checkFraming(keypoints, videoWidth, videoHeight);
    setFraming(result);
  }, [keypoints, videoWidth, videoHeight]);

  return framing;
}