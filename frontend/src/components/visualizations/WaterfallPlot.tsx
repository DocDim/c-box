import React from 'react';
import Plot from 'react-plotly.js';
import { FeatureContribution } from '../../types/api';

interface WaterfallPlotProps {
  baseValue: number;
  prediction: number;
  contributions: FeatureContribution[];
}

export const WaterfallPlot: React.FC<WaterfallPlotProps> = ({
  baseValue,
  prediction,
  contributions
}) => {
  // Take top 10 most impactful features for clean chart rendering
  const topFeatures = contributions.slice(0, 10).reverse();

  const labels = topFeatures.map(f => `${f.feature} (${f.actual_value})`);
  const values = topFeatures.map(f => f.shap_value);
  const colors = topFeatures.map(f => f.shap_value >= 0 ? '#ef4444' : '#3b82f6');

  return (
    <div className="bg-slate-800/80 p-4 rounded-xl border border-slate-700 shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">Local Feature Attributions (SHAP)</h3>
          <p className="text-sm text-slate-400">
            Baseline $E[f(x)]$: <span className="text-slate-200 font-mono">{baseValue.toFixed(4)}</span> | 
            Model Output $f(x)$: <span className="text-emerald-400 font-mono font-bold ml-1">{prediction.toFixed(4)}</span>
          </p>
        </div>
      </div>
      <Plot
        data={[
          {
            type: 'bar',
            orientation: 'h',
            y: labels,
            x: values,
            marker: { color: colors },
          }
        ]}
        layout={{
          autosize: true,
          height: 380,
          margin: { l: 140, r: 20, t: 20, b: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: {
            title: { text: 'SHAP Value (Impact on Prediction)' },
            zeroline: true,
            zerolinecolor: '#64748b',
            gridcolor: '#334155',
            tickfont: { color: '#94a3b8' },
            titlefont: { color: '#cbd5e1' }
          },
          yaxis: {
            automargin: true,
            tickfont: { color: '#cbd5e1' }
          }
        }}
        useResizeHandler={true}
        className="w-full"
      />
    </div>
  );
};