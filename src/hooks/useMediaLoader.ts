import { useState, useCallback } from 'react';

export function useMediaLoader() {
  const [source, setSource] = useState<'image' | 'video' | null>(null);
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);

  const loadImage = useCallback((file: File): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        setImageEl(img);
        setSource('image');
        resolve(img);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }, []);

  const loadVideo = useCallback((file: File): Promise<HTMLVideoElement> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.preload = 'auto';
      video.muted = true;
      video.loop = true;
      video.onloadeddata = () => {
        setVideoEl(video);
        setSource('video');
        resolve(video);
      };
      video.onerror = reject;
      video.src = URL.createObjectURL(file);
      video.play().catch(reject);
    });
  }, []);

  const clearMedia = useCallback(() => {
    if (imageEl) URL.revokeObjectURL(imageEl.src);
    if (videoEl) URL.revokeObjectURL(videoEl.src);
    setImageEl(null);
    setVideoEl(null);
    setSource(null);
  }, [imageEl, videoEl]);

  return { loadImage, loadVideo, clearMedia, source, imageEl, videoEl };
}