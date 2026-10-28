import React, { useState } from 'react';
import { Database, RefreshCw } from 'lucide-react';
import { Dataset, ModelArtifact } from '../../types/api';
import { errorMessage, modelService } from '../../services/api';
import { fmtDate } from '../../utils/format';

interface BackgroundDataCardProps {
  model: ModelArtifact;
  background: Dataset | null;
  onModelUpdated: (model: ModelArtifact) => void;
}

/**
 * The model's background data is SHAP's reference distribution: it feeds explanations and
 * global insights and is never predicted on. Only the model owner can upload or replace it.
 */
export const BackgroundDataCard: React.FC<BackgroundDataCardProps> = ({ model, background, onModelUpdated }) => {
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [inputKey, setInputKey] = useState(0);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    if (background && !window.confirm(`Replace "${background.name}"? The current background data will be deleted.`)) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      onModelUpdated(await modelService.replaceBackground(model.id, file, target.trim() || undefined));
      setNotice(background ? 'Background data replaced.' : 'Background data uploaded.');
      setFile(null);
      setTarget('');
      setInputKey((k) => k + 1);
    } catch (err) {
      setError(errorMessage(err, 'Background upload failed.'));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-3 space-y-3">
      <h4 className="text-xs font-semibold text-slate-50 flex items-center gap-1.5">
        <Database size={14} className="text-secondary-400" /> Background data
      </h4>
      <p className="text-[11px] text-slate-400">
        Reference distribution for SHAP explanations and Global Insights. Predictions never run on it.
      </p>

      {background ? (
        <dl className="text-xs grid grid-cols-[auto,1fr] gap-x-3 gap-y-0.5">
          <dt className="text-slate-500">File</dt><dd className="text-slate-200 truncate">{background.name}</dd>
          <dt className="text-slate-500">Rows</dt><dd className="text-slate-200">{background.n_rows}</dd>
          <dt className="text-slate-500">Target</dt><dd className="text-slate-200">{background.target_column ?? 'none'}</dd>
          <dt className="text-slate-500">Loaded</dt><dd className="text-slate-200">{fmtDate(background.created_at)}</dd>
        </dl>
      ) : (
        <p className="text-xs text-accent-400">
          No background data. Non-tree models cannot be explained and Global Insights are unavailable.
        </p>
      )}

      {model.is_owner ? (
        <form onSubmit={submit} className="space-y-2">
          {error && <div className="text-xs text-rose-400 bg-rose-950/40 p-2 rounded border border-rose-800">{error}</div>}
          {notice && <div className="text-xs text-secondary-300">{notice}</div>}
          <input
            key={inputKey}
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="w-full text-xs text-slate-400 file:mr-3 file:py-1 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-secondary-600 file:text-white cursor-pointer"
          />
          <div className="flex gap-2">
            <input
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="Target column (optional)"
              className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-50"
            />
            <button
              type="submit"
              disabled={!file || busy}
              className="bg-secondary-600 hover:bg-secondary-500 disabled:bg-gray-500 text-white text-xs font-semibold px-3 py-1.5 rounded-lg flex items-center gap-1"
            >
              <RefreshCw size={12} /> {busy ? 'Validating…' : background ? 'Replace' : 'Upload'}
            </button>
          </div>
        </form>
      ) : (
        <p className="text-[11px] text-slate-500 italic">Only the model owner can load or update the background data.</p>
      )}
    </div>
  );
};
