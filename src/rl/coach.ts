import { RunningMetrics } from '../pose/analyzer';
import { ActivityKind } from '../types/session';

/*
 * ── Real DQN (future upgrade) ──
 * State: [kneeAvg, torsoLean, armSwing, heelStrike, activity, prevAction]
 * Actions: same actionId strings
 * Reward: +1 if next-state score improves, -1 if degrades, +5 if composite >80
 * Model: 2 dense layers (64, 32), epsilon‑greedy with epsilon=0.1
 * Training: requires 10k+ labeled transitions from recorded sessions
 *
 * ── Simulated RL (current) ──
 * Same input/output interface. Deterministic rule‑based thresholds.
 * Zero training data. Swap path: replace function body with model.predict().
 */

export interface CoachAction {
  actionId: string;
  advice: string;
  voiceCue: string;
  severity: 'good' | 'warning' | 'critical';
  scoreDelta: number;   // penalty applied to composite score
}

export function getCoachingActions(
  metrics: RunningMetrics,
  activity: ActivityKind
): CoachAction[] {
  const actions: CoachAction[] = [];

  if (activity === 'running') {
    const { kneeAvg, torsoLean, isHeelStrike, armSwingRatio } = metrics;
    if (kneeAvg < 60 || kneeAvg > 130) {
      actions.push({
        actionId: 'knee_critical',
        advice: 'Your knee angle is extreme – focus on a mid‑range bend',
        voiceCue: 'Knee angle extreme',
        severity: 'critical',
        scoreDelta: -25,
      });
    } else if (kneeAvg < 80 || kneeAvg > 110) {
      actions.push({
        actionId: 'knee_warning',
        advice: 'Slightly adjust your knee bend',
        voiceCue: 'Adjust knee bend',
        severity: 'warning',
        scoreDelta: -10,
      });
    }

    if (torsoLean < 3 || torsoLean > 20) {
      actions.push({
        actionId: 'torso_critical',
        advice: 'Your torso lean is too extreme – keep a neutral spine',
        voiceCue: 'Neutral spine',
        severity: 'critical',
        scoreDelta: -25,
      });
    } else if (torsoLean < 5 || torsoLean > 15) {
      actions.push({
        actionId: 'torso_warning',
        advice: 'Slightly adjust your forward lean',
        voiceCue: 'Adjust lean',
        severity: 'warning',
        scoreDelta: -10,
      });
    }

    if (isHeelStrike) {
      actions.push({
        actionId: 'heel_strike',
        advice: 'Land on your midfoot, not your heel',
        voiceCue: 'Midfoot landing',
        severity: 'critical',
        scoreDelta: -25,
      });
    }

    if (armSwingRatio < 0.3) {
      actions.push({
        actionId: 'arms_critical',
        advice: 'Pump your arms more',
        voiceCue: 'Pump arms',
        severity: 'critical',
        scoreDelta: -25,
      });
    } else if (armSwingRatio < 0.5) {
      actions.push({
        actionId: 'arms_warning',
        advice: 'Increase your arm swing',
        voiceCue: 'More arm swing',
        severity: 'warning',
        scoreDelta: -10,
      });
    }
  } else if (activity === 'yoga') {
    if (metrics.torsoLean < 80 || metrics.torsoLean > 100) {
      actions.push({
        actionId: 'yoga_spine',
        advice: 'Lengthen your spine, avoid rounding',
        voiceCue: 'Lengthen spine',
        severity: 'warning',
        scoreDelta: -10,
      });
    }
    if (metrics.armSwingRatio < 0.2) {
      actions.push({
        actionId: 'yoga_arms',
        advice: 'Reach through your arms more',
        voiceCue: 'Reach arms',
        severity: 'warning',
        scoreDelta: -10,
      });
    }
  } else if (activity === 'sitting') {
    if (metrics.torsoLean < 70 || metrics.torsoLean > 110) {
      actions.push({
        actionId: 'sit_spine',
        advice: 'Sit upright with your back against the chair',
        voiceCue: 'Sit up straight',
        severity: 'critical',
        scoreDelta: -25,
      });
    } else if (metrics.torsoLean < 80 || metrics.torsoLean > 100) {
      actions.push({
        actionId: 'sit_slight',
        advice: 'Straighten your back a little',
        voiceCue: 'Straighten back',
        severity: 'warning',
        scoreDelta: -10,
      });
    }
  }

  if (actions.length === 0) {
    actions.push({
      actionId: 'good_job',
      advice: 'Great form! Keep it up',
      voiceCue: 'Looking good',
      severity: 'good',
      scoreDelta: 0,
    });
  }

  return actions;
}