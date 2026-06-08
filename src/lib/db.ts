import { openDB } from 'idb';
import type { DBSession } from '../types/session';

const DB_NAME = 'nature-pose';
const DB_VERSION = 1;
const STORE_NAME = 'sessions';

let dbPromise: ReturnType<typeof openDB> | null = null;

export function openDatabase() {
  if (dbPromise) return dbPromise;

  dbPromise = openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, {
          keyPath: 'id',
          autoIncrement: true,
        });
      }
    },
  });

  return dbPromise;
}

export async function saveSession(session: DBSession): Promise<void> {
  const db = await openDatabase();
  await db.add(STORE_NAME, session);
}

export async function getAllSessions(): Promise<DBSession[]> {
  const db = await openDatabase();
  const tx = db.transaction(STORE_NAME, 'readonly');
  const store = tx.store;
  const all = await store.getAll();
  return all.sort(
    (a: DBSession, b: DBSession) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );
}

function topCues(summary: Record<string, number>, count: number): string[] {
  return Object.entries(summary)
    .sort(([, a], [, b]) => b - a)
    .slice(0, count)
    .map(([key]) => key);
}

export function exportCSV(sessions: DBSession[]): string {
  const header = 'date,activity,durationSec,avgScore,topCue1,topCue2';
  const rows = sessions.map((s) => {
    const cues = topCues(s.actionsSummary, 2);
    return [
      s.date,
      s.activity,
      Math.round(s.durationMs / 1000),
      s.averageScore.toFixed(1),
      cues[0] ?? '',
      cues[1] ?? '',
    ].join(',');
  });
  return [header, ...rows].join('\n');
}