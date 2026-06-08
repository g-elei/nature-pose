import { useRef, useState, useCallback, useEffect } from 'react';

export function useVoiceCoach() {
  const queueRef = useRef<string[]>([]);
  const speakingRef = useRef(false);
  const [muted, setMuted] = useState(false);

  const processQueue = useCallback(() => {
    if (!speakingRef.current && queueRef.current.length > 0 && !muted) {
      const text = queueRef.current.shift()!;
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.onend = () => {
        speakingRef.current = false;
        processQueue();
      };
      speakingRef.current = true;
      window.speechSynthesis.speak(utterance);
    }
  }, [muted]);

  const speak = useCallback((text: string) => {
    if (muted) return;
    queueRef.current.push(text);
    processQueue();
  }, [muted, processQueue]);

  const stop = useCallback(() => {
    queueRef.current = [];
    window.speechSynthesis.cancel();
    speakingRef.current = false;
  }, []);

  const toggleMute = useCallback(() => {
    setMuted(prev => !prev);
    if (!muted) stop();
  }, [muted, stop]);

  useEffect(() => {
    return () => stop();
  }, [stop]);

  return { speak, stop, muted, toggleMute };
}