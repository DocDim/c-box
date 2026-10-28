import React, { useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { BarChart3 } from 'lucide-react';
import { Dataset, GlobalExplainResponse, ModelArtifact } from '../../types/api';
import { errorMessage, explainService } from '../../services/api';
import { fmtMs } from '../../utils/format';
import { useChartColors } from '../../hooks/useTheme';

const MAX_BEESWARM_FEATURES = 12;
// Low feature value -> blue, high -> green (brand take on the SHAP summary plot colormap)
const VALUE_COLORSCALE: [number, string][] = [[0, '#009ddd'], [0.5, '#e48d12'], [1, '#00927e']];

type View = 'bar' | 'beeswarm' | 'dependence';

interface SummaryPlotProps {
  model: ModelArtifact;
  datasets: Dataset[];
}

export const SummaryPlot: React.FC<SummaryPlotProps> = ({ model, datasets }) => {
  const c = useChartColors();
  const axisStyle = { gridcolor: c.grid, tickfont: { color: c.tick, size: 10 }, zeroline: false };
  const [samples, setSamples] = useState(200);
  const [classIndex, setClassIndex] = useState<string>('');
  const [result, setResult] = useState<GlobalExplainResponse | null>(null);
  const [view, setView] = useState<View>('beeswarm');
  const [dependenceFeature, setDependenceFeature] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Global behaviour is always computed on the model's background (reference) data
  const background = datasets.find((d) => d.id === model.background_dataset_id) ?? null;

  useEffect(() => {
    setResult(null);
    setClassIndex('');
    setError('');
  }, [model.id, model.background_dataset_id]);

  const compute = async () => {
    if (!background) return;
    setLoading(true);
    setError('');
    try {
      const res = await explainService.explainGlobal(
        model.id, samples, classIndex === '' ? undefined : Number(classIndex)
      );
      setResult(res);
      setDependenceFeature(res.global_importance[0]?.feature ?? '');
    } catch (err) {
      setError(errorMessage(err, 'Global SHAP computation failed.'));
    } finally {
      setLoading(false);
    }
  };

  const plot = useMemo(() => {
    if (!result) return null;
    const { beeswarm, global_importance } = result;

    if (view === 'bar') {
      const top = [...global_importance].reverse();
      return {
        data: [{
          type: 'bar', orientation: 'h',
          y: top.map((g) => g.feature), x: top.map((g) => g.mean_abs_shap),
          marker: { color: '#009ddd' },
          hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
        }],
        layout: { xaxis: { ...axisStyle, title: { text: 'mean(|SHAP value|)', font: { size: 11, color: c.tick } } } },
        height: 60 + top.length * 26,
      };
    }

    if (view === 'beeswarm') {
      const count = Math.min(MAX_BEESWARM_FEATURES, beeswarm.features.length);
      const xs: number[] = [];
      const ys: number[] = [];
      const colors: number[] = [];
      const hover: string[] = [];
      for (let f = 0; f < count; f++) {
        const vals = beeswarm.feature_values[f];
        const numeric = vals.filter((v): v is number => v !== null);
        const lo = Math.min(...numeric);
        const hi = Math.max(...numeric);
        const yBase = count - 1 - f; // most important feature on top
        beeswarm.shap_values[f].forEach((phi, i) => {
          const v = vals[i];
          xs.push(phi);
          // Deterministic jitter keeps the swarm stable across renders
          ys.push(yBase + (((i * 2654435761) % 1000) / 1000 - 0.5) * 0.6);
          colors.push(v === null || hi === lo ? 0.5 : (v - lo) / (hi - lo));
          hover.push(`${beeswarm.features[f]} = ${v ?? '—'}<br>SHAP ${phi.toFixed(4)}`);
        });
      }
      return {
        data: [{
          type: 'scattergl', mode: 'markers', x: xs, y: ys,
          marker: {
            size: 5, color: colors, colorscale: VALUE_COLORSCALE, cmin: 0, cmax: 1, opacity: 0.8,
            colorbar: { title: { text: 'Feature value', side: 'right', font: { color: c.tick, size: 10 } },
              tickvals: [0, 1], ticktext: ['Low', 'High'], tickfont: { color: c.tick, size: 10 }, thickness: 10 },
          },
          hovertext: hover, hoverinfo: 'text',
        }],
        layout: {
          xaxis: { ...axisStyle, zeroline: true, zerolinecolor: c.zero,
            title: { text: 'SHAP value (impact on model output)', font: { size: 11, color: c.tick } } },
          yaxis: { tickvals: Array.from({ length: count }, (_, f) => count - 1 - f),
            ticktext: beeswarm.features.slice(0, count), tickfont: { color: c.label, size: 11 },
            automargin: true, gridcolor: c.gridFaint },
        },
        height: 80 + count * 34,
      };
    }

    const f = Math.max(0, beeswarm.features.indexOf(dependenceFeature));
    return {
      data: [{
        type: 'scattergl', mode: 'markers',
        x: beeswarm.feature_values[f], y: beeswarm.shap_values[f],
        marker: { size: 6, opacity: 0.8, color: beeswarm.shap_values[f].map((p) => (p >= 0 ? '#00927e' : '#009ddd')) },
        hovertemplate: `${beeswarm.features[f]} = %{x}<br>SHAP %{y:.4f}<extra></extra>`,
      }],
      layout: {
        xaxis: { ...axisStyle, title: { text: beeswarm.features[f], font: { size: 11, color: c.tick } } },
        yaxis: { ...axisStyle, zeroline: true, zerolinecolor: c.zero,
          title: { text: `SHAP value for ${beeswarm.features[f]}`, font: { size: 11, color: c.tick } } },
      },
      height: 360,
    };
  }, [result, view, dependenceFeature, c]);

  const classes = model.class_labels ?? [];

  return (
    <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-lg space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-700 pb-3">
        <h3 className="text-sm font-semibold text-slate-50 flex items-center gap-2">
          <BarChart3 size={18} className="text-primary-400" /> Global Model Behaviour (dataset-wide SHAP)
        </h3>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-400">
            Background: <span className="text-slate-200">{background ? `${background.name} (${background.n_rows} rows)` : 'none'}</span>
          </span>
          <select
            value={samples}
            onChange={(e) => setSamples(Number(e.target.value))}
            className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-slate-50"
          >
            {[50, 100, 200, 300].map((n) => <option key={n} value={n}>{n} samples</option>)}
          </select>
          {classes.length > 2 && (
            <select
              value={classIndex}
              onChange={(e) => setClassIndex(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-slate-50"
            >
              <option value="">All classes (averaged)</option>
              {classes.map((c, i) => <option key={c} value={i}>Class {c}</option>)}
            </select>
          )}
          <button
            onClick={compute}
            disabled={loading || !background}
            className="bg-primary-600 hover:bg-primary-500 disabled:bg-gray-500 text-white font-medium px-3 py-1.5 rounded-lg"
          >
            {loading ? 'Computing…' : 'Compute'}
          </button>
        </div>
      </div>

      {error && <div className="text-xs text-rose-400 bg-rose-950/40 p-2.5 rounded-lg border border-rose-800">{error}</div>}

      {!background && (
        <p className="text-xs text-accent-400">
          This model has no background data. {model.is_owner
            ? 'Upload it on the Models page (Manage → Background data).'
            : 'Ask the model owner to upload it.'}
        </p>
      )}
      {background && !result && !error && (
        <p className="text-xs text-slate-500 italic">
          Computes SHAP values over a sample of up to 300 background rows to show which features drive the model overall.
        </p>
      )}

      {result && plot && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex gap-1 text-xs">
              {(['bar', 'beeswarm', 'dependence'] as View[]).map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className={`px-3 py-1 rounded-md capitalize ${view === v ? 'bg-primary-600 text-white' : 'bg-gray-600 hover:bg-gray-500 text-white'}`}
                >
                  {v === 'bar' ? 'Importance' : v}
                </button>
              ))}
              {view === 'dependence' && (
                <select
                  value={dependenceFeature}
                  onChange={(e) => setDependenceFeature(e.target.value)}
                  className="ml-2 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-slate-50"
                >
                  {result.beeswarm.features.map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              )}
            </div>
            <span className="text-[11px] text-slate-500">
              {result.samples_evaluated} samples · {result.explained_output} · {result.explainer_type} · {fmtMs(result.shap_ms)}
            </span>
          </div>
          <Plot
            data={plot.data as any}
            layout={{
              autosize: true,
              height: plot.height,
              margin: { l: 10, r: 10, t: 10, b: 45 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              showlegend: false,
              yaxis: { automargin: true, tickfont: { color: c.label, size: 11 } },
              ...(plot.layout as any),
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler={true}
            className="w-full"
          />
        </>
      )}
    </div>
  );
};
