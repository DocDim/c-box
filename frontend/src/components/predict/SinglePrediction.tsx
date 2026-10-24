import React, { useEffect, useState } from 'react';
import { Activity, Play, Sparkles } from 'lucide-react';
import { errorMessage, explainService } from '../../services/api';
import { InstanceSource, ModelArtifact, PredictResponse } from '../../types/api';
import { fmt } from '../../utils/format';

interface SinglePredictionProps {
  model: ModelArtifact;
  columns: string[];
  allowExplain: boolean;
  explaining: boolean;
  onExplain: (source: InstanceSource) => void;
}

/** Predicts (and optionally explains) one data point entered through a form (FR-3.1). */
export const SinglePrediction: React.FC<SinglePredictionProps> = ({
  model,
  columns,
  allowExplain,
  explaining,
  onExplain,
}) => {
  const [features, setFeatures] = useState<Record<string, any>>({});
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setFeatures({});
    setResult(null);
    setError('');
  }, [model.id]);

  const setValue = (col: string, val: string) => {
    const num = Number(val);
    setFeatures((prev) => ({ ...prev, [col]: val.trim() !== '' && !Number.isNaN(num) ? num : val }));
    setResult(null);
  };

  const validSource = (): InstanceSource | null => {
    const missing = columns.filter((c) => features[c] === undefined || features[c] === null || features[c] === '');
    if (missing.length) {
      setError(`Fill in all features (missing: ${missing.slice(0, 5).join(', ')}${missing.length > 5 ? '…' : ''}).`);
      return null;
    }
    setError('');
    const subset: Record<string, any> = {};
    columns.forEach((c) => { subset[c] = features[c]; });
    return { instance_features: subset };
  };

  const predict = async (e: React.FormEvent) => {
    e.preventDefault();
    const source = validSource();
    if (!source) return;
    setLoading(true);
    try {
      setResult(await explainService.predict(model.id, source));
    } catch (err) {
      setError(errorMessage(err, 'Inference failed. Check feature values.'));
    } finally {
      setLoading(false);
    }
  };

  const explain = () => {
    const source = validSource();
    if (source) onExplain(source);
  };

  return (
    <div className="space-y-4">
      <p className="text-xs text-slate-400">Enter one value per feature for the data point to predict.</p>

      {error && <div className="text-xs text-rose-400 bg-rose-950/40 p-2.5 rounded-lg border border-rose-800">{error}</div>}

      <form onSubmit={predict} className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-3 max-h-80 overflow-y-auto p-1">
          {columns.map((col) => (
            <div key={col} className="bg-slate-900/60 p-2 rounded-lg border border-slate-700/80">
              <label className="block text-[11px] font-mono text-slate-400 truncate mb-1" title={col}>{col}</label>
              <input
                type="text"
                value={features[col] ?? ''}
                onChange={(e) => setValue(col, e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono focus:outline-none focus:border-indigo-500"
                placeholder="value"
              />
            </div>
          ))}
        </div>

        <div className="flex flex-wrap justify-end gap-2">
          <button
            type="submit"
            disabled={loading || columns.length === 0}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 text-white text-xs font-semibold px-4 py-2 rounded-lg flex items-center gap-1.5"
          >
            <Play size={14} /> {loading ? 'Running…' : 'Predict'}
          </button>
          {allowExplain && (
            <button
              type="button"
              onClick={explain}
              disabled={explaining || columns.length === 0}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 text-white text-xs font-semibold px-4 py-2 rounded-lg flex items-center gap-1.5"
            >
              <Sparkles size={14} /> {explaining ? 'Explaining…' : 'Predict & Explain'}
            </button>
          )}
        </div>
      </form>

      {result && (
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-700 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Prediction</span>
            <span className="text-sm font-mono font-bold text-emerald-400">
              {result.predicted_class !== null && result.predicted_class !== undefined
                ? `Class ${result.predicted_class} · P = ${fmt(result.prediction)}`
                : `Predicted value: ${fmt(result.prediction)}`}
            </span>
          </div>
          {result.probabilities && (
            <div>
              <span className="text-[11px] text-slate-400 font-semibold mb-1.5 flex items-center gap-1">
                <Activity size={12} /> Class probabilities
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {Object.entries(result.probabilities).map(([cls, p]) => (
                  <div key={cls} className="bg-slate-800 p-2 rounded border border-slate-700/80 flex justify-between text-xs">
                    <span className="font-mono text-slate-300">{cls}</span>
                    <span className="font-mono text-emerald-400 font-semibold">{(p * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
