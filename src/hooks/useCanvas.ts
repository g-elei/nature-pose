import { useRef, useEffect } from 'react';

export function useCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (ctx) ctxRef.current = ctx;
  }, []);

  const resizeCanvas = (width: number, height: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    if (ctxRef.current) {
      ctxRef.current.clearRect(0, 0, width, height);
    }
  };

  return { canvasRef, ctxRef, resizeCanvas };
}