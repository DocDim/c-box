import React from 'react';
import Plot from 'react-plotly.js';
import { InstanceExplainResponse } from '../../types/api';
import { fmt } from '../../utils/format';
import { SHAP_NEGATIVE, SHAP_POSITIVE } from './WaterfallPlot';
import { useChartColors } from '../../hooks/useTheme';

interface ForcePlotProps {
  result: InstanceExplainResponse;
}

/**
 * SHAP force plot: red segments push the output up and end at f(x); blue segments
 * push it down and start at f(x). Segment width is |phi|.
 */
export const ForcePlot: React.FC<ForcePlotProps> = ({ result }) => {
  const c = useChartColors();
  const fx = result.explained_output;
  const positives = result.feature_contributions
    .filter((c) => c.shap_value > 0)
    .sort((a, b) => a.shap_value - b.shap_value);
  const negatives = result.feature_contributions
    .filter((c) => c.shap_value < 0)
    .sort((a, b) => a.shap_value - b.shap_value);

  const segments = (
    items: typeof positives,
    start: number,
    color: string
  ) => {
    let cursor = start;
    const bases: number[] = [];
    const widths: number[] = [];
    const labels: string[] = [];
    items.forEach((c) => {
      const w = Math.abs(c.shap_value);
      bases.push(cursor);
      widths.push(w);
      labels.push(`${c.feature} = ${c.actual_value ?? '—'}<br>SHAP ${c.shap_value >= 0 ? '+' : ''}${c.shap_value.toFixed(4)}`);
      cursor += w;
    });
    const total = Math.abs(items.reduce((a, c) => a + c.shap_value, 0)) || 1;
    return {
      type: 'bar',
      orientation: 'h',
      y: items.map(() => 'force'),
      x: widths,
      base: bases,
      marker: { color, line: { color: c.outline, width: 1 } },
      text: items.map((c) => (Math.abs(c.shap_value) / total > 0.12 ? c.feature : '')),
      textposition: 'inside',
      insidetextanchor: 'middle',
      textfont: { color: '#fff', size: 10 },
      hovertext: labels,
      hoverinfo: 'text',
    } as any;
  };

  const posTotal = positives.reduce((a, c) => a + c.shap_value, 0);
  const data = [
    segments(positives, fx - posTotal, SHAP_POSITIVE),
    // Largest |phi| sits next to f(x) on both sides, as in SHAP's force plot
    segments(negatives, fx, SHAP_NEGATIVE),
  ];

  return (
    <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-lg">
      <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">Force Plot</h4>
      <p className="text-[11px] text-slate-500 mb-1">
        <span style={{ color: SHAP_POSITIVE }}>■</span> higher &nbsp;
        <span style={{ color: SHAP_NEGATIVE }}>■</span> lower · base value {fmt(result.base_value, 3)} → f(x) {fmt(fx, 3)}
      </p>
      <Plot
        data={data}
        layout={{
          autosize: true,
          height: 140,
          barmode: 'overlay',
          showlegend: false,
          margin: { l: 10, r: 10, t: 30, b: 30 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: { gridcolor: c.grid, tickfont: { color: c.tick, size: 10 }, zeroline: false },
          yaxis: { visible: false },
          shapes: [
            { type: 'line', x0: fx, x1: fx, yref: 'paper', y0: 0, y1: 1, line: { color: c.strong, width: 2 } },
            { type: 'line', x0: result.base_value, x1: result.base_value, yref: 'paper', y0: 0, y1: 1, line: { color: c.tick, width: 1, dash: 'dot' } },
          ],
          annotations: [
            { x: fx, y: 1.15, yref: 'paper', text: `f(x) = ${fmt(fx, 3)}`, showarrow: false, font: { color: c.strong, size: 11 } },
            { x: result.base_value, y: -0.35, yref: 'paper', text: 'base value', showarrow: false, font: { color: c.tick, size: 10 } },
          ],
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler={true}
        className="w-full"
      />
    </div>
  );
};
