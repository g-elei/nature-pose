import React, { useState, useEffect } from 'react';
import { useCamera } from '../hooks/useCamera';
import { usePoseDetection } from '../hooks/usePoseDetection';
import { useSession } from '../context/SessionContext';
import { useVoiceCoach } from '../hooks/useVoiceCoach';
import { useSettings } from '../context/SettingsContext';
import { RunningMetrics } from '../pose/analyzer';
import { checkFraming, FramingResult } from '../pose/framing';
import { CoachAction } from '../rl/coach';

export const LiveCoachDashboard: React.FC = () => {
  // 1. Core Ecosystem Hooks
  const { videoRef, start, stop, isActive: isCameraActive, error: cameraError } = useCamera('user');
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  
  const { activity, voiceMuted, setVoiceMuted } = useSettings();
  const { currentSession, startSession, recordFrame, endSession } = useSession();
  const { speak, stop: stopVoice, toggleMute } = useVoiceCoach();

  // 2. High-Frequency UI State
  const [metrics, setMetrics] = useState<RunningMetrics | null>(null);
  const [liveAction, setLiveAction] = useState<CoachAction | null>(null);
  const [framingGuide, setFramingGuide] = useState<FramingResult | null>(null);

  // Sync vocal state with settings adjustments
  useEffect(() => {
    if (voiceMuted) {
      stopVoice();
    }
  }, [voiceMuted, stopVoice]);

  const handleMuteToggle = () => {
    toggleMute();
    setVoiceMuted(!voiceMuted);
  };

  // 3. Central Stream Telemetry Consumer Pipeline
  usePoseDetection({
    videoRef,
    canvasRef,
    activity,
    enabled: isCameraActive,
    onResult: (latestMetrics, latestKeypoints, coachAction) => {
      setMetrics(latestMetrics);
      setLiveAction(coachAction);

      // Perform camera safety window checks on the layout thread
      if (videoRef.current) {
        const frameCheck = checkFraming(latestKeypoints, videoRef.current.videoWidth, videoRef.current.videoHeight);
        setFramingGuide(frameCheck);
      }

      // Record telemetry frame directly to IndexedDB store
      if (currentSession && coachAction) {
        recordFrame(latestMetrics.score, [coachAction]);
      }

      // Voice broadcast
      if (coachAction && !voiceMuted) {
        speak(coachAction.text);
      }
    }
  });

  // Calculate semantic color themes based on live performance levels
  const formScore = metrics?.score ?? 0;
  const scoreColorClass = formScore >= 80 
    ? 'text-emerald-400 border-emerald-500/30' 
    : formScore >= 50 
      ? 'text-amber-400 border-amber-500/30' 
      : 'text-rose-400 border-rose-500/30';

  return (
    <div className="min-h-screen bg-stone-950 text-stone-100 p-4 md:p-8 font-sans antialiased selection:bg-emerald-500/30 selection:text-emerald-200">
      
      {/* HEADER CONTROLS */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8 pb-6 border-b border-stone-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-stone-100 via-emerald-100 to-emerald-400 bg-clip-text text-transparent">
              NaturePose Engine
            </h1>
          </div>
          <p className="text-stone-400 text-sm mt-1">Biomechanical alignment via neural reinforcement networks</p>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          {/* Audio Queue Mute Control */}
          <button
            onClick={handleMuteToggle}
            className={`p-3 rounded-xl border transition-all duration-300 flex items-center justify-center ${
              voiceMuted 
                ? 'bg-stone-900 border-stone-800 text-stone-500 hover:text-stone-400' 
                : 'bg-emerald-950/40 border-emerald-800/50 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.1)]'
            }`}
            title={voiceMuted ? "Unmute Voice Feedback" : "Mute Voice Feedback"}
          >
            {voiceMuted ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" /></svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" /></svg>
            )}
          </button>

          {/* Core Pipeline Switch */}
          {!currentSession ? (
            <button
              onClick={() => { start(); startSession(activity); }}
              className="flex-1 md:flex-none px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold rounded-xl transition-all shadow-[0_4px_20px_rgba(16,185,129,0.2)] hover:shadow-[0_4px_25px_rgba(16,185,129,0.3)] active:scale-[0.98]"
            >
              Initialize Coaching Session
            </button>
          ) : (
            <button
              onClick={() => { stop(); endSession(); setMetrics(null); setLiveAction(null); }}
              className="flex-1 md:flex-none px-6 py-3 bg-gradient-to-r from-rose-600 to-orange-600 hover:from-rose-500 hover:to-orange-500 text-white font-semibold rounded-xl transition-all shadow-[0_4px_20px_rgba(244,63,94,0.2)] active:scale-[0.98]"
            >
              Terminate & Save Session
            </button>
          )}
        </div>
      </header>

      {/* ERROR CORRECTION STATUS SLATE */}
      {cameraError && (
        <div className="max-w-7xl mx-auto mb-6 p-4 bg-rose-950/40 border border-rose-800/50 rounded-xl text-rose-300 flex items-center gap-3 text-sm">
          <span className="h-2 w-2 rounded-full bg-rose-500 animate-ping" />
          <strong>Hardware Notice:</strong> {cameraError}
        </div>
      )}

      {/* CORE WORKSPACE GRID */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT ASPECT SECTION: VISUAL HUD STREAM CONTAINER */}
        <section className="lg:col-span-7 xl:col-span-8 flex flex-col gap-6">
          <div className="relative w-full aspect-video rounded-2xl bg-stone-950 overflow-hidden border border-stone-800 shadow-inner group">
            
            {/* Mirror Processed Camera Sub-layers */}
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas 
              ref={canvasRef} 
              className={`w-full h-full object-cover transition-transform duration-500 scale-x-[-1] ${
                isCameraActive ? 'opacity-100' : 'opacity-20'
              }`} 
            />

            {/* PIPELINE DISCONNECTED DEFAULT VIEW SPLASH */}
            {!isCameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-stone-950/60 backdrop-blur-sm transition-opacity duration-300">
                <div className="p-4 bg-stone-900/80 rounded-2xl border border-stone-800 text-emerald-500/80 mb-3 group-hover:scale-105 transition-transform">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                </div>
                <h3 className="text-lg font-semibold text-stone-200">Optical Pipeline Offline</h3>
                <p className="text-sm text-stone-400 max-w-sm mt-1">Click the initialization trigger above to engage hardware camera resources.</p>
              </div>
            )}

            {/* LIVE DYNAMIC VISUAL HUD ALERTS */}
            {isCameraActive && framingGuide && framingGuide.status !== 'good' && (
              <div className="absolute top-4 left-4 right-4 bg-amber-950/80 backdrop-blur-md px-4 py-3 rounded-xl border border-amber-500/40 text-amber-300 text-xs md:text-sm flex items-center gap-3 animate-fade-in shadow-lg">
                <span className="flex h-2 w-2 rounded-full bg-amber-400 animate-ping" />
                <p><strong>Framing Warning:</strong> {framingGuide.message}</p>
              </div>
            )}
          </div>

          {/* NEURAL MODEL TEXT RECOVERY TEXT OUTLET */}
          <div className="bg-stone-900/60 border border-stone-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm relative overflow-hidden">
            <div className="absolute top-0 left-0 w-1.5 h-full bg-emerald-500" />
            <span className="text-xs uppercase tracking-widest font-bold text-emerald-500 block mb-1">Active Neural Target Model Directive</span>
            <h2 className="text-xl md:text-2xl font-bold tracking-tight text-white transition-all duration-300">
              {liveAction ? liveAction.text : "Awaiting motion tracking frame stream..."}
            </h2>
          </div>
        </section>

        {/* RIGHT ASPECT SECTION: METRICS TELEMETRY PANELS */}
        <section className="lg:col-span-5 xl:col-span-4 flex flex-col gap-6">
          
          {/* METRICS INTERACTIVE HEADER GAUGES */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-6">
            
            {/* COMPOSITE METRICS RADIAL CARD */}
            <div className={`bg-stone-900/40 border rounded-2xl p-6 flex flex-col items-center text-center backdrop-blur-sm transition-all duration-500 ${scoreColorClass}`}>
              <span className="text-stone-400 text-xs uppercase tracking-wider font-semibold mb-3">Kinetic Alignment Score</span>
              <div className="relative flex items-center justify-center w-28 h-28 my-2">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                  <path className="text-stone-800" strokeWidth="3" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                  <path className="transition-all duration-500 ease-out" strokeDasharray={`${formScore}, 100`} strokeWidth="3" strokeLinecap="round" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                </svg>
                <div className="absolute text-3xl font-black text-stone-100 tracking-tight">
                  {formScore}<span className="text-xs text-stone-400 font-normal">%</span>
                </div>
              </div>
              <p className="text-stone-300 text-xs mt-3 max-w-[200px]">Composite accuracy of joint angle models across tracking sets.</p>
            </div>

            {/* DYNAMIC SPATIAL CADENCE CARD */}
            <div className="bg-stone-900/40 border border-stone-800 rounded-2xl p-6 flex flex-col items-center justify-center text-center backdrop-blur-sm">
              <span className="text-stone-400 text-xs uppercase tracking-wider font-semibold mb-2">Estimated Cadence</span>
              <div className="text-5xl font-black tracking-tight text-white my-2 flex items-baseline gap-1">
                {metrics ? metrics.cadence : 0}
                <span className="text-xs font-medium text-stone-500 tracking-normal">SPM</span>
              </div>
              <div className="w-full bg-stone-800 h-1 rounded-full overflow-hidden mt-3 max-w-[150px]">
                <div 
                  className="bg-emerald-500 h-full transition-all duration-500"
                  style={{ width: `${Math.min(((metrics?.cadence ?? 0) / 200) * 100, 100)}%` }}
                />
              </div>
              <p className="text-stone-400 text-[11px] mt-2">Steps per minute calculated from ankle vertical velocity mapping.</p>
            </div>
          </div>

          {/* LINEAR EXTENSION SPATIAL TRACKERS SLATE */}
          <div className="bg-stone-900/40 border border-stone-800 rounded-2xl p-6 flex flex-col gap-5 backdrop-blur-sm">
            <h3 className="text-sm font-bold text-stone-200 uppercase tracking-wide border-b border-stone-800 pb-3">Real-time Joint Metrics</h3>
            
            {/* TRACKER ELEMENT: TORSO LEAN */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-stone-400 font-medium">Torso Forward Lean</span>
                <span className="text-stone-200 font-bold">{metrics ? Math.round(metrics.torsoLean) : 0}°</span>
              </div>
              <div className="bg-stone-800 h-2 rounded-lg overflow-hidden">
                <div 
                  className="bg-teal-500 h-full transition-all duration-300"
                  style={{ width: `${Math.min(((metrics?.torsoLean ?? 0) / 45) * 100, 100)}%` }}
                />
              </div>
            </div>

            {/* TRACKER ELEMENT: LEFT KNEE COMPRESSION */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-stone-400 font-medium">Left Knee Extension</span>
                <span className="text-stone-200 font-bold">{metrics ? Math.round(metrics.kneeLeft) : 0}°</span>
              </div>
              <div className="bg-stone-800 h-2 rounded-lg overflow-hidden">
                <div 
                  className="bg-emerald-400 h-full transition-all duration-300"
                  style={{ width: `${Math.min(((metrics?.kneeLeft ?? 0) / 180) * 100, 100)}%` }}
                />
              </div>
            </div>

            {/* TRACKER ELEMENT: