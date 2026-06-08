import React, { useRef } from 'react';

interface InputSelectorProps {
  onSelectImage: (file: File) => void;
  onSelectVideo: (file: File) => void;
  onSelectCamera: () => void;
  activeMode: 'image' | 'video' | 'camera';
}

const InputSelector: React.FC<InputSelectorProps> = ({
  onSelectImage, onSelectVideo, onSelectCamera, activeMode,
}) => {
  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onSelectImage(file);
  };

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onSelectVideo(file);
  };

  const tabClass = (mode: string) =>
    `px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-forest-300 ${
      activeMode === mode
        ? 'bg-sand-500 text-forest-900'
        : 'bg-cream-200 text-forest-700 hover:bg-cream-100'
    }`;

  return (
    <div role="tablist" aria-label="Input mode" className="flex rounded-lg overflow-hidden border border-sage-300">
      <input
        ref={imageInputRef}
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        className="hidden"
        aria-label="Upload image"
      />
      <input
        ref={videoInputRef}
        type="file"
        accept="video/*"
        onChange={handleVideoChange}
        className="hidden"
        aria-label="Upload video"
      />

      <button
        role="tab"
        aria-selected={activeMode === 'image'}
        className={tabClass('image')}
        onClick={() => imageInputRef.current?.click()}
        aria-label="Upload image"
      >
        📷 Image
      </button>
      <button
        role="tab"
        aria-selected={activeMode === 'video'}
        className={tabClass('video')}
        onClick={() => videoInputRef.current?.click()}
        aria-label="Upload video"
      >
        🎥 Video
      </button>
      <button
        role="tab"
        aria-selected={activeMode === 'camera'}
        className={tabClass('camera')}
        onClick={onSelectCamera}
        aria-label="Live camera"
      >
        📹 Live
      </button>
    </div>
  );
};

export default InputSelector;