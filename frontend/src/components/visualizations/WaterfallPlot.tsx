import React from 'react';
import Plot from 'react-plotly.js';
import { FeatureContribution } from '../../types/api';
import { Target, Activity } from 'lucide-react';

interface WaterfallPlotProps {
  baseValue: number;
  prediction: number;
  predictedClass?: string;
  probabilities?: Record<string, number>;
  contributions: FeatureContribution[];
}

export const WaterfallPlot: React.FC<WaterfallPlotProps> = ({
  baseValue,
  prediction,
  predictedClass,
  probabilities,
  contributions
}) => {
  const topFeatures = contributions.slice(0, 10).reverse();

  const labels = topFeatures.map(f => `${f.feature} (${f.actual_value})`);
  const values = topFeatures.map(f => f.shap_value);
  const colors = topFeatures.map(f => f.shap_value >= 0 ? '#38bdf8' : '#f87171');

  return (
    <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-lg space-y-4">
      {/* Top Banner: Prediction and Probabilities */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/90 border border-slate-700">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <Target size={22} />
          </div>
          <div>
            <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Model Prediction</span>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold text-white">
                {predictedClass ? `Class: ${predictedClass}` : prediction.toFixed(4)}
              </span>
              <span className="text-xs text-emerald-400 font-mono font-bold">
                ({(prediction * 100).toFixed(1)}% confidence)
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <span className="text-slate-400 block text-[10px] uppercase font-sans">Baseline E[f(x)]</span>
            <span className="text-slate-300 font-semibold">{baseValue.toFixed(4)}</span>
          </div>
          <div className="text-right border-l border-slate-700 pl-4">
            <span className="text-slate-400 block text-[10px] uppercase font-sans">Attribution Sum</span>
            <span className="text-indigo-400 font-semibold">
              {(values.reduce((a, b) => a + b, 0)).toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      {/* Class Probabilities Distribution */}
      {probabilities && Object.keys(probabilities).length > 1 && (
        <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-700/60 text-xs">
          <span className="text-slate-400 text-[11px] font-semibold block mb-2 flex items-center gap-1.5">
            <Activity size={13} className="text-slate-400" />
            Class Probability Distribution
          </span>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {Object.entries(probabilities).map(([cls, prob]) => (
              <div key={cls} className="bg-slate-800/80 px-2.5 py-1.5 rounded border border-slate-700 flex justify-between items-center">
                <span className="font-mono text-slate-300 truncate mr-2">{cls}</span>
                <span className="font-mono font-bold text-emerald-400">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Feature Attribution Chart */}
      <div>
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
          Local Feature Attributions (SHAP)
        </h4>
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
            height: 320,
            margin: { l: 140, r: 20, t: 10, b: 35 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
              title: { text: 'Impact on Prediction (SHAP Value)', font: { size: 11, color: '#94a3b8' } },
              zeroline: true,
              zerolinecolor: '#64748b',
              gridcolor: '#334155',
              tickfont: { color: '#94a3b8', size: 10 },
            },
            yaxis: {
              automargin: true,
              tickfont: { color: '#cbd5e1', size: 11 }
            }
          }}
          useResizeHandler={true}
          className="w-full"
        />
      </div>
    </div>
  );
};