import React from 'react';

interface CameraFeedProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  width?: number;
  height?: number;
}

const CameraFeed: React.FC<CameraFeedProps> = ({ videoRef, canvasRef, width = 640, height = 480 }) => {
  return (
    <div className="relative w-full bg-forest-950" style={{ aspectRatio: `${width} / ${height}` }}>
      <video
        ref={videoRef as React.Ref<HTMLVideoElement>}
        autoPlay
        playsInline
        muted
        aria-hidden="true"
        tabIndex={-1}
        className="absolute inset-0 h-full w-full object-contain"
        style={{ display: 'none' }}
      />
      <canvas
        ref={canvasRef as React.Ref<HTMLCanvasElement>}
        width={width}
        height={height}
        role="img"
        aria-label="Pose skeleton overlay"
        className="absolute inset-0 h-full w-full object-contain"
      />
    </div>
  );
};

export default CameraFeed;