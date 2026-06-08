export type ActivityKind = 'sitting' | 'yoga' | 'walking' | 'running';

export interface DBSession {
  id?: number;
  date: string;
  activity: ActivityKind;
  durationMs: number;
  averageScore: number;
  actionsSummary: Record<string, number>;
}