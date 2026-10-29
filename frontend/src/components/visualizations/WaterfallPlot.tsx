import React from 'react';
import Plot from 'react-plotly.js';
import { InstanceExplainResponse } from '../../types/api';
import { Target, Activity } from 'lucide-react';
import { fmt, fmtMs } from '../../utils/format';
import { useChartColors } from '../../hooks/useTheme';

// SHAP's convention: red pushes the output up, blue pushes it down
export const SHAP_POSITIVE = '#046c1f';
export const SHAP_NEGATIVE = '#dd2100';
const MAX_BARS = 10;

interface WaterfallPlotProps {
  result: InstanceExplainResponse;
}

const valueLabel = (v: number | string | null) =>
  v === null ? '—' : typeof v === 'number' ? String(Number(v.toFixed(4))) : v;

export const WaterfallPlot: React.FC<WaterfallPlotProps> = ({ result }) => {
  const c = useChartColors();
  const contributions = result.feature_contributions;
  const shown = contributions.slice(0, MAX_BARS);
  const rest = contributions.slice(MAX_BARS);
  const restSum = rest.reduce((a, c) => a + c.shap_value, 0);
  const attributionSum = contributions.reduce((a, c) => a + c.shap_value, 0);

  // Bottom-to-top: remaining features, then smallest to largest, then f(x)
  const ordered = [...shown].reverse();
  const labels = [
    ...(rest.length ? [`${rest.length} other features`] : []),
    ...ordered.map((c) => `${c.feature} = ${valueLabel(c.actual_value)}`),
    'f(x)',
  ];
  const deltas = [...(rest.length ? [restSum] : []), ...ordered.map((c) => c.shap_value), 0];
  const measures = [...deltas.slice(0, -1).map(() => 'relative'), 'total'];

  const isClassification = result.task_type === 'classification';
  const unitsNote =
    result.output_units === 'log-odds'
      ? 'Attributions are in log-odds. Upload a background dataset to explain in probability space.'
      : `Attributions are in ${result.output_units}.`;

  const trace: any = {
    type: 'waterfall',
    orientation: 'h',
    base: result.base_value,
    measure: measures,
    y: labels,
    x: deltas,
    text: deltas.map((d, i) => (i === deltas.length - 1 ? fmt(result.explained_output, 3) : `${d >= 0 ? '+' : ''}${fmt(d, 3)}`)),
    textposition: 'outside',
    cliponaxis: false,
    increasing: { marker: { color: SHAP_POSITIVE } },
    decreasing: { marker: { color: SHAP_NEGATIVE } },
    totals: { marker: { color: '#009ddd' } },
    connector: { line: { color: c.connector, width: 1 } },
    hovertemplate: '%{y}<br>%{x:.4f}<extra></extra>',
  };

  return (
    <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-lg space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/90 border border-slate-700">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-primary-500/20 text-primary-400 border border-primary-500/30">
            <Target size={22} />
          </div>
          <div>
            <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Model Prediction</span>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold text-slate-50">
                {isClassification ? `Class ${result.predicted_class}` : fmt(result.prediction)}
              </span>
              {isClassification && (
                <span className="text-xs text-secondary-400 font-mono font-bold">
                  P = {(result.prediction * 100).toFixed(1)}%
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-right">
            <span className="text-slate-400 block text-[10px] uppercase font-sans">Baseline E[f(x)]</span>
            <span className="text-slate-300 font-semibold">{fmt(result.base_value)}</span>
          </div>
          <div className="text-right border-l border-slate-700 pl-4">
            <span className="text-slate-400 block text-[10px] uppercase font-sans">Σ SHAP (all features)</span>
            <span className="text-primary-400 font-semibold">{attributionSum >= 0 ? '+' : ''}{fmt(attributionSum)}</span>
          </div>
          <div className="text-right border-l border-slate-700 pl-4">
            <span className="text-slate-400 block text-[10px] uppercase font-sans">f(x)</span>
            <span className="text-slate-50 font-semibold">{fmt(result.explained_output)}</span>
          </div>
        </div>
      </div>

      {result.probabilities && Object.keys(result.probabilities).length > 1 && (
        <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-700/60 text-xs">
          <span className="text-slate-400 text-[11px] font-semibold block mb-2 flex items-center gap-1.5">
            <Activity size={13} className="text-slate-400" />
            Class Probability Distribution
          </span>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {Object.entries(result.probabilities).map(([cls, prob]) => (
              <div key={cls} className="bg-slate-800/80 px-2.5 py-1.5 rounded border border-slate-700 flex justify-between items-center">
                <span className="font-mono text-slate-300 truncate mr-2">{cls}</span>
                <span className="font-mono font-bold text-secondary-400">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
          Local SHAP Waterfall{isClassification ? ` · explains class ${result.explained_class}` : ''}
        </h4>
        <p className="text-[11px] text-slate-500 mb-2">
          {unitsNote} {result.explainer_type} · additivity gap {result.additivity_gap.toExponential(1)} · computed in {fmtMs(result.shap_ms)}
        </p>
        <Plot
          data={[trace]}
          layout={{
            autosize: true,
            height: 60 + labels.length * 28,
            margin: { l: 10, r: 60, t: 10, b: 35 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            showlegend: false,
            xaxis: {
              title: { text: `Model output (${result.output_units})`, font: { size: 11, color: c.tick } },
              gridcolor: c.grid,
              tickfont: { color: c.tick, size: 10 },
              zeroline: false,
            },
            yaxis: { automargin: true, tickfont: { color: c.label, size: 11 } },
          }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler={true}
          className="w-full"
        />
      </div>
    </div>
  );
};
