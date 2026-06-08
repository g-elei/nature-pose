import React from 'react';

interface MetricBarProps {
  label: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  idealMin?: number;
  idealMax?: number;
}

const MetricBar: React.FC<MetricBarProps> = ({ label, value, unit, min, max, idealMin, idealMax }) => {
  const pct = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-forest-700">{label}</span>
        <span className="text-forest-900">
          {value.toFixed(1)}<span className="text-sage-500 ml-0.5">{unit}</span>
        </span>
      </div>
      <div
        role="progressbar"
        aria-valuenow={Math.round(value)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
        className="relative h-3 w-full rounded-full bg-sage-200 overflow-hidden"
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-forest-400 transition-all duration-200"
          style={{ width: `${pct}%` }}
        />
        {idealMin !== undefined && idealMax !== undefined && (
          <div
            className="absolute inset-y-0 rounded-full bg-forest-500/20"
            style={{
              left: `${((idealMin - min) / (max - min)) * 100}%`,
              width: `${((idealMax - idealMin) / (max - min)) * 100}%`,
            }}
          />
        )}
      </div>
    </div>
  );
};

export default MetricBar;