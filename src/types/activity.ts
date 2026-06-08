import type { ActivityKind } from './session';

export interface ActivityConfig {
  kind: ActivityKind;
  label: string;
  icon: string;
  metrics: string[];
  skeletonColor: string;
}