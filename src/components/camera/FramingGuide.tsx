import React from 'react';
import { FramingResult } from '../../pose/framing';

interface FramingGuideProps {
  framing: FramingResult | null;
}

const FramingGuide: React.FC<FramingGuideProps> = ({ framing }) => {
  if (!framing) return null;

  return (
    <div className="pointer-events-none absolute inset-0 z-10">
      {framing.status !== 'good' && (
        <div
          role="alert"
          className="absolute left-1/2 top-4 -translate-x-1/2 rounded-lg bg-sand-500 px-4 py-2 text-sm font-medium text-forest-900 shadow-lg animate-fadeIn"
        >
          {framing.message}
        </div>
      )}
      <div
        aria-hidden="true"
        className="absolute border-2 border-dashed border-forest-400/60 rounded"
        style={{
          left: framing.boxLeft,
          top: framing.boxTop,
          width: framing.width,
          height: framing.height,
        }}
      />
    </div>
  );
};

export default FramingGuide;