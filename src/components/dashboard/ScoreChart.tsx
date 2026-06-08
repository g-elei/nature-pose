import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
} from 'chart.js';
import { DBSession } from '../../types/session';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Filler);

interface ScoreChartProps {
  sessions: DBSession[];
}

const ScoreChart: React.FC<ScoreChartProps> = ({ sessions }) => {
  if (sessions.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-sage-500 text-sm">
        No sessions yet
      </div>
    );
  }

  const reversed = [...sessions].reverse();
  const labels = reversed.map((s) => new Date(s.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }));
  const scores = reversed.map((s) => s.averageScore);

  return (
    <div className="h-48" role="img" aria-label="Line chart of session scores over time">
      <Line
        data={{
          labels,
          datasets: [
            {
              label: 'Score',
              data: scores,
              borderColor: '#d0a06a',
              backgroundColor: 'rgba(208, 160, 106, 0.2)',
              pointBackgroundColor: '#3d8f3d',
              pointBorderColor: '#d0a06a',
              pointRadius: 4,
              borderWidth: 2,
              fill: true,
              tension: 0.3,
            },
          ],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: true } },
          scales: {
            x: { ticks: { color: '#7b8665', maxRotation: 0 }, grid: { color: 'rgba(123, 134, 101, 0.15)' } },
            y: { min: 0, max: 100, ticks: { color: '#7b8665' }, grid: { color: 'rgba(123, 134, 101, 0.15)' } },
          },
        }}
      />
    </div>
  );
};

export default ScoreChart;