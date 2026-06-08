import React, { createContext, useContext, useState, useCallback, useRef, ReactNode } from 'react';
import { DBSession, ActivityKind } from '../types/session';
import { saveSession, getAllSessions } from '../lib/db';
import { CoachAction } from '../rl/coach';

interface SessionContextValue {
  currentSession: Omit<DBSession, 'id'> | null;
  sessions: DBSession[];
  startSession: (activity: ActivityKind) => void;
  recordFrame: (score: number, actions: CoachAction[]) => void;
  endSession: () => Promise<void>;
  loadSessions: () => Promise<void>;
}

const SessionContext = createContext<SessionContextValue | undefined>(undefined);

export const useSession = () => {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error('useSession must be used within SessionProvider');
  return ctx;
};

interface SessionProviderProps {
  children: ReactNode;
}

export const SessionProvider: React.FC<SessionProviderProps> = ({ children }) => {
  const [currentSession, setCurrentSession] = useState<Omit<DBSession, 'id'> | null>(null);
  const [sessions, setSessions] = useState<DBSession[]>([]);

  const startTimeRef = useRef<number>(0);
  const scoreSumRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const actionsSummaryRef = useRef<Record<string, number>>({});

  const startSession = useCallback((activity: ActivityKind) => {
    startTimeRef.current = Date.now();
    scoreSumRef.current = 0;
    frameCountRef.current = 0;
    actionsSummaryRef.current = {};

    setCurrentSession({
      date: new Date().toISOString(),
      activity,
      durationMs: 0,
      averageScore: 0,
      actionsSummary: {},
    });
  }, []);

  const recordFrame = useCallback((score: number, actions: CoachAction[]) => {
    if (!currentSession) return;

    scoreSumRef.current += score;
    frameCountRef.current++;

    for (const action of actions) {
      actionsSummaryRef.current[action.actionId] =
        (actionsSummaryRef.current[action.actionId] || 0) + 1;
    }

    setCurrentSession(prev => {
      if (!prev) return null;
      return {
        ...prev,
        actionsSummary: { ...actionsSummaryRef.current },
      };
    });
  }, [currentSession]);

  const loadSessions = useCallback(async () => {
    const all = await getAllSessions();
    setSessions(all);
  }, []);

  const endSession = useCallback(async () => {
    if (!currentSession) return;

    const durationMs = Date.now() - startTimeRef.current;
    const averageScore = frameCountRef.current > 0
      ? Math.round(scoreSumRef.current / frameCountRef.current)
      : 0;

    const finalSession: DBSession = {
      ...currentSession,
      durationMs,
      averageScore,
      actionsSummary: actionsSummaryRef.current,
    };

    await saveSession(finalSession);
    setCurrentSession(null);
    await loadSessions();
  }, [currentSession, loadSessions]);

  return (
    <SessionContext.Provider value={{
      currentSession,
      sessions,
      startSession,
      recordFrame,
      endSession,
      loadSessions,
    }}>
      {children}
    </SessionContext.Provider>
  );
};