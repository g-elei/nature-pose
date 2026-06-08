import { useEffect, useState, useRef } from 'react';
import { useSettings } from '../store/SettingsContext';
import { useSession } from '../store/SessionContext';
import { usePose, usePoseUpdate } from '../store/PoseContext';
import { useCamera } from '../hooks/useCamera';
import { useCanvas } from '../hooks/useCanvas';
import { usePoseDetection } from '../hooks/usePoseDetection';
import { useVoiceCoach } from '../hooks/useVoiceCoach';
import { loadCoachPolicy, evaluatePoseRL } from '../rl/coach';
import { ActivityKind } from '../types/session';
import { RunningMetrics } from '../pose/analyzer';

export function LiveCoachDashboard() {
  // Contexts
  const { activity, setActivity, voiceMuted, setVoiceMuted } = useSettings();
  const { currentSession, sessions, startSession, recordFrame, endSession } = useSession();
  // From: const { metrics, keypoints } = usePose();
const { metrics } = usePose();
  const updatePose = usePoseUpdate();

  // Custom Hooks
  const { videoRef, start: startCamera, stop: stopCamera, isActive: isCameraActive, error: cameraError } = useCamera('user');
  const { canvasRef, resizeCanvas } = useCanvas();
  const { speak, stop: stopVoice } = useVoiceCoach();

  // States
const [model, setModel] = useState<any>(null);
  const [coachFeedback, setCoachFeedback] = useState<string>('Awaiting motion input to coach your form...');
  const lastFeedbackTime = useRef<number>(0);

  // Load the AI Coach Policy on mount
  useEffect(() => {
    async function initModel() {
      try {
        const loadedModel = await loadCoachPolicy();
        setModel(loadedModel);
        console.log('AI Coach reinforcement learning model loaded successfully.');
      } catch (err) {
        console.error('Failed to load RL coach policy, falling back to rule-based cues:', err);
      }
    }
    initModel();
  }, []);

  // Sync voice muted status from application settings
  useEffect(() => {
    if (voiceMuted) {
      stopVoice();
    }
  }, [voiceMuted, stopVoice]);

  // Handle frame analysis results
  const handlePoseResult = (latestMetrics: RunningMetrics, latestKeypoints: any[]) => {
    // 1. Update the shared pose context tracking state
    updatePose(latestKeypoints, latestMetrics, 0);

    // 2. Evaluate Form via the AI RL Model or fall back to standard metrics
    let actionText = 'Form is stable. Maintain your rhythm.';
    let actionId = 'stable';

    if (model && latestMetrics) {
      // Normalize metrics into standard input vector for the policy neural network
      const inputVector = [
        latestMetrics.kneeLeft || 0,
        latestMetrics.kneeRight || 0,
        latestMetrics.torsoLean || 0,
        latestMetrics.armSwingRatio || 0,
        latestMetrics.hipExtension || 0,
      ];
      
      const actionIndex = evaluatePoseRL(model, inputVector);
      

      //  Directly read the properties from the coach action object
if (actionIndex) {
  actionId = actionIndex.actionId; 
  actionText = actionIndex.text;
}
    } else if (latestMetrics) {
      // Rule-based feedback fallback if model is still loading
      if (latestMetrics.torsoLean > 15) {
        actionId = 'posture';
        actionText = 'Reduce forward lean. Engage your core.';
      } else if (latestMetrics.score < 70 && latestMetrics.armSwingRatio < 0.3) {
        actionId = 'arms';
        actionText = 'Pump your arms in line with your movement direction.';
      }
    }

    setCoachFeedback(actionText);

    // 3. Trigger Voice Feedback every 4 seconds max to avoid speaking overlaps
    const now = performance.now();
    if (!voiceMuted && now - lastFeedbackTime.current > 4000) {
      speak(actionText);
      lastFeedbackTime.current = now;
    }

    // 4. Append frame metrics to database if a recording session is running
    if (currentSession) {
      recordFrame(latestMetrics.score || 0, [{ 
  actionId, 
  text: actionText, 
  advice: actionText, 
  voiceCue: actionText 
}]);
    }
  };

  // Connect pose detection loop to refs
  usePoseDetection({
    videoRef,
    canvasRef,
    activity,
    onResult: handlePoseResult,
    enabled: isCameraActive,
  });

  // Adjust canvas bounds on camera activation
  useEffect(() => {
    if (isCameraActive && videoRef.current) {
      const timer = setTimeout(() => {
        const w = videoRef.current?.videoWidth || 640;
        const h = videoRef.current?.videoHeight || 480;
        resizeCanvas(w, h);
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [isCameraActive, videoRef, resizeCanvas]);

  // Manage Live Track controls
  const toggleCamera = async () => {
    if (isCameraActive) {
      if (currentSession) await handleEndSession();
      stopCamera();
    } else {
      await startCamera();
    }
  };

  const handleStartSession = () => {
    if (!isCameraActive) return;
    startSession(activity);
  };

  const handleEndSession = async () => {
    await endSession();
  };

  // Export Session History log to CSV formatted spreadsheet
  const exportHistoryToCSV = () => {
    if (sessions.length === 0) return;
    const headers = ['ID', 'Date', 'Activity', 'Duration (s)', 'Average Score'];
    const rows = sessions.map(s => [
      s.id || '',
      s.date,
      s.activity,
      Math.round(s.durationMs / 1000),
      s.averageScore,
    ]);

    const csvContent = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `NaturePose_History_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-[#f6f7f1] text-[#1e211a] font-sans antialiased selection:bg-[#3d8f3d] selection:text-white">
      {/* Accessibility Skip Link */}
      <a href="#main-dashboard" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-[#2e722e] text-[#fdfbf7] px-4 py-2 rounded shadow">
        Skip to main content
      </a>

      {/* Navigation Header */}
      <header className="bg-[#193919] text-[#fdfbf7] shadow-md border-b-4 border-[#3d8f3d]" role="banner">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <span className="text-3xl" aria-hidden="true">🌿</span>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">NaturePose</h1>
              <p className="text-xs text-[#b4bda0]">Eco-Themed Real-Time AI Running Form Coach</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setVoiceMuted(!voiceMuted)}
              className={`flex items-center gap-2 px-4 py-2 rounded-full font-medium transition focus:outline-none focus:ring-2 focus:ring-[#85c985] text-sm ${
                voiceMuted ? 'bg-amber-800 text-amber-100 hover:bg-amber-700' : 'bg-[#2e722e] text-[#fdfbf7] hover:bg-[#245a24]'
              }`}
              aria-label={voiceMuted ? "Unmute voice coaching" : "Mute voice coaching"}
            >
              <span>{voiceMuted ? '🔇 Voice Off' : '🔊 Voice On'}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main id="main-dashboard" className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 space-y-6" role="main">
        
        {/* Step 1: Configuration Tab Row */}
        <section className="bg-white rounded-xl shadow-sm border border-[#e8ebe0] p-4" aria-labelledby="activity-heading">
          <h2 id="activity-heading" className="text-sm font-semibold uppercase tracking-wider text-[#626b50] mb-3">
            Select Coaching Activity
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2" role="tablist" aria-label="Activity tracking modes">
            {(['running', 'walking', 'yoga', 'sitting'] as ActivityKind[]).map((mode) => (
              <button
                key={mode}
                role="tab"
                aria-selected={activity === mode}
                onClick={() => setActivity(mode)}
                disabled={!!currentSession}
                className={`py-3 px-4 rounded-lg font-medium text-sm border transition capitalize ${
                  activity === mode
                    ? 'bg-[#2e722e] text-[#fdfbf7] border-[#1d471d] shadow-sm'
                    : 'bg-[#fdfbf7] border-[#d2d7c2] text-[#4e5540] hover:bg-[#e8ebe0] disabled:opacity-40'
                }`}
              >
                {mode === 'running' && '🏃 '}
                {mode === 'walking' && '🚶 '}
                {mode === 'yoga' && '🧘 '}
                {mode === 'sitting' && '🪑 '}
                {mode}
              </button>
            ))}
          </div>
          {currentSession && (
            <p className="text-xs text-amber-700 mt-2 font-medium">
              ⚠️ Complete your active recording session to change tracking activities.
            </p>
          )}
        </section>

        {/* Multi-Column Feed Dashboard View */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Vision Panel Feed Box */}
          <div className="lg:col-span-7 space-y-4">
            <div className="bg-white rounded-xl shadow-sm border border-[#e8ebe0] p-4 flex flex-col">
              <div className="flex justify-between items-center mb-3">
                <h3 className="font-bold text-[#245a24] text-lg flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 inline-block animate-pulse"></span>
                  Vision Tracking Feed
                </h3>
                <span className="text-xs bg-[#e8ebe0] text-[#4e5540] px-2.5 py-1 rounded-md font-medium">
                  {isCameraActive ? '📷 Active' : '💤 Offline'}
                </span>
              </div>

              {/* Viewfinder Canvas Stack */}
              <div className="relative w-full aspect-video bg-neutral-900 rounded-lg overflow-hidden flex items-center justify-center border border-neutral-800">
                {!isCameraActive && (
                  <div className="absolute text-center px-4">
                    <p className="text-neutral-400 text-sm mb-3">Camera capture is currently stopped</p>
                    <button
                      onClick={toggleCamera}
                      className="bg-[#2e722e] text-white hover:bg-[#245a24] px-5 py-2 rounded-lg text-sm font-medium transition focus:ring-2 focus:ring-offset-2 focus:ring-[#3d8f3d]"
                    >
                      Initialize Camera Feed
                    </button>
                  </div>
                )}

                {cameraError && (
                  <div className="absolute inset-0 bg-red-950/90 text-red-200 p-4 flex items-center justify-center text-center text-sm">
                    ❌ {cameraError}. Please verify hardware options and site permissions.
                  </div>
                )}

                <video
                  ref={videoRef}
                  className={`w-full h-full object-cover ${isCameraActive ? 'block' : 'hidden'}`}
                  muted
                  playsInline
                />
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full object-cover pointer-events-none"
                />
              </div>

              {/* Live Session Controls */}
              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  onClick={toggleCamera}
                  className={`px-4 py-2.5 rounded-lg text-sm font-medium transition ${
                    isCameraActive 
                      ? 'bg-neutral-200 text-neutral-800 hover:bg-neutral-300' 
                      : 'bg-[#2e722e] text-white hover:bg-[#245a24]'
                  }`}
                >
                  {isCameraActive ? 'Turn Off Camera' : 'Start Camera'}
                </button>

                {isCameraActive && (
                  <button
                    onClick={currentSession ? handleEndSession : handleStartSession}
                    className={`px-5 py-2.5 rounded-lg text-sm font-medium transition shadow-sm ${
                      currentSession
                        ? 'bg-red-600 text-white hover:bg-red-700 font-bold animate-pulse'
                        : 'bg-emerald-700 text-white hover:bg-emerald-800'
                    }`}
                  >
                    {currentSession ? '⏹️ Stop & Save Session' : '⏺️ Record Form Metrics'}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Metrics & AI Coaching Analytics Column */}
          <div className="lg:col-span-5 space-y-4">
            
            {/* AI Coaching Box */}
            <div className="bg-[#1d471d] text-[#fdfbf7] rounded-xl shadow-sm p-5 border-l-4 border-[#85c985]">
              <h3 className="text-xs font-bold uppercase tracking-widest text-[#b4bda0] mb-2">
                🤖 AI Form Reinforcement Coach
              </h3>
              <div className="bg-[#193919] rounded-lg p-4 border border-[#245a24]">
                <p className="text-lg font-medium leading-relaxed italic text-[#f2e4ce]">
                  "{coachFeedback}"
                </p>
              </div>
              <div className="mt-3 flex items-center justify-between text-xs text-[#b4bda0]">
                <span>Strategy: Deep Q-Policy Network</span>
                <span>Status: {model ? 'Model Connected' : 'Loading Model Weights...'}</span>
              </div>
            </div>

            {/* Metrics Checklist Display */}
            <div className="bg-white rounded-xl shadow-sm border border-[#e8ebe0] p-5 space-y-4">
              <div className="flex justify-between items-baseline border-b border-[#e8ebe0] pb-2">
                <h3 className="font-bold text-[#1e211a] text-base">Real-Time Kinematics</h3>
                <span className="text-sm font-bold text-[#2e722e]">
                  Form Score: {metrics ? metrics.score : '--'}/100
                </span>
              </div>

              {metrics ? (
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="bg-[#f6f7f1] p-3 rounded-lg border border-[#e8ebe0]">
                    <span className="block text-xs text-[#626b50] font-medium">Avg Knee Flexion</span>
                    <span className="text-lg font-bold text-[#245a24]">{Math.round(metrics.kneeAvg || 0)}°</span>
                  </div>
                  <div className="bg-[#f6f7f1] p-3 rounded-lg border border-[#e8ebe0]">
                    <span className="block text-xs text-[#626b50] font-medium">Forward Torso Lean</span>
                    <span className="text-lg font-bold text-[#245a24]">{Math.round(metrics.torsoLean || 0)}°</span>
                  </div>
                  <div className="bg-[#f6f7f1] p-3 rounded-lg border border-[#e8ebe0]">
                    <span className="block text-xs text-[#626b50] font-medium">Arm Swing Extent</span>
                    <span className="text-lg font-bold text-[#245a24]">{Math.round((metrics.armSwingRatio || 0) * 100)}%</span>
                  </div>
                  <div className="bg-[#f6f7f1] p-3 rounded-lg border border-[#e8ebe0]">
                    <span className="block text-xs text-[#626b50] font-medium">Cadence Count</span>
                    <span className="text-lg font-bold text-[#245a24]">{metrics.cadence || 0} SPM</span>
                  </div>
                  <div className="bg-[#f6f7f1] p-3 rounded-lg border border-[#e8ebe0] col-span-2 flex justify-between items-center">
                    <span className="text-xs text-[#626b50] font-medium">Heel Strike Landing Over-stride</span>
                    <span className={`px-2.5 py-0.5 rounded text-xs font-bold ${metrics.isHeelStrike ? 'bg-amber-100 text-amber-800' : 'bg-green-100 text-green-800'}`}>
                      {metrics.isHeelStrike ? '⚠️ Heel Strike Detected' : '✅ Optimal Forefoot/Midfoot'}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="py-8 text-center text-sm text-[#97a27e]">
                  No tracking vectors found. Step into frame to monitor joint mechanics.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* History Log Section */}
        <section className="bg-white rounded-xl shadow-sm border border-[#e8ebe0] p-5" aria-labelledby="history-heading">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center border-b border-[#e8ebe0] pb-3 mb-4 gap-3">
            <div>
              <h2 id="history-heading" className="text-lg font-bold text-[#1e211a]">
                Eco-Training Session History
              </h2>
              <p className="text-xs text-[#626b50]">Your localized database logs saved to this client</p>
            </div>
            {sessions.length > 0 && (
              <button
                onClick={exportHistoryToCSV}
                className="bg-white border border-[#d2d7c2] hover:bg-[#f6f7f1] text-[#4e5540] px-3 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-1.5"
              >
                📥 Export Logs to CSV
              </button>
            )}
          </div>

          {sessions.length === 0 ? (
            <div className="text-center py-8 text-[#97a27e] text-sm">
              No historical data in local storage. Record your first movement set to begin!
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse" role="table">
                <thead>
                  <tr className="border-b border-[#d2d7c2] text-[#626b50] bg-[#f6f7f1]">
                    <th className="p-3 font-semibold">Date Timestamp</th>
                    <th className="p-3 font-semibold">Activity</th>
                    <th className="p-3 font-semibold">Duration</th>
                    <th className="p-3 font-semibold">Avg Competency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#e8ebe0]">
                  {sessions.slice().reverse().map((session) => (
                    <tr key={session.id} className="hover:bg-[#fdfbf7] transition">
                      <td className="p-3 text-xs tabular-nums text-[#4e5540]">{session.date}</td>
                      <td className="p-3 capitalize font-medium">{session.activity}</td>
                      <td className="p-3 text-xs tabular-nums">
                        {Math.floor(session.durationMs / 60000)}m {Math.round((session.durationMs % 60000) / 1000)}s
                      </td>
                      <td className="p-3 font-bold text-[#2e722e]">
                        {session.averageScore}/100
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}