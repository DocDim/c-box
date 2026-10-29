import React, { useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight, Download, FileSpreadsheet, Play, Sparkles } from 'lucide-react';
import { datasetService, errorMessage, explainService } from '../../services/api';
import { BatchPredictResponse, Dataset, DatasetPreview, ModelArtifact } from '../../types/api';
import { fmt, fmtMs, fmtPct } from '../../utils/format';

const PAGE_SIZE = 25;

interface BatchPredictionProps {
  model: ModelArtifact;
  datasets: Dataset[];
  onDatasetAdded: (dataset: Dataset) => void;
  allowExplain: boolean;
  explaining: boolean;
  explainedRow: { datasetId: number; rowIndex: number } | null;
  onExplainRow: (datasetId: number, rowIndex: number) => void;
}

const sameLabel = (actual: any, predicted?: string | null) => {
  if (actual === null || actual === undefined || predicted === null || predicted === undefined) return null;
  const a = typeof actual === 'number' && Number.isInteger(actual) ? String(actual) : String(actual);
  return a === predicted;
};

/** Predicts every row of a dataset the user picks or uploads on this page. */
export const BatchPrediction: React.FC<BatchPredictionProps> = ({
  model,
  datasets,
  onDatasetAdded,
  allowExplain,
  explaining,
  explainedRow,
  onExplainRow,
}) => {
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState('');
  const [uploading, setUploading] = useState(false);
  const [inputKey, setInputKey] = useState(0);
  const [result, setResult] = useState<BatchPredictResponse | null>(null);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setResult(null);
    setError('');
    setDatasetId((current) => (current && datasets.some((d) => d.id === current) ? current : datasets[0]?.id ?? null));
  }, [model.id, datasets]);

  const run = async (dsId: number, pageIndex: number) => {
    setLoading(true);
    setError('');
    try {
      setResult(await explainService.predictBatch(model.id, dsId, pageIndex * PAGE_SIZE, PAGE_SIZE));
      setPage(pageIndex);
    } catch (err) {
      setError(errorMessage(err, 'Batch prediction failed.'));
    } finally {
      setLoading(false);
    }
  };

  const upload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('model_id', String(model.id));
      if (target.trim()) form.append('target_column', target.trim());
      const preview: DatasetPreview = await datasetService.uploadDataset(form);
      const { extra_columns: _e, summary: _s, preview_rows: _p, ...dataset } = preview;
      onDatasetAdded(dataset);
      setDatasetId(dataset.id);
      setFile(null);
      setTarget('');
      setInputKey((k) => k + 1);
      await run(dataset.id, 0);
    } catch (err) {
      setError(errorMessage(err, 'Dataset upload failed.'));
    } finally {
      setUploading(false);
    }
  };

  const dataset = datasets.find((d) => d.id === datasetId) ?? null;
  const totalPages = result ? Math.max(1, Math.ceil(result.total_rows / PAGE_SIZE)) : 1;
  const featureCols = result?.rows[0] ? Object.keys(result.rows[0].features) : [];
  const isClassification = result?.task_type === 'classification';

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-3 space-y-2">
          <label className="block text-xs font-medium text-slate-300">Use a dataset already loaded for this model</label>
          <div className="flex gap-2">
            <select
              value={datasetId ?? ''}
              onChange={(e) => { setDatasetId(Number(e.target.value)); setResult(null); }}
              className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-50"
            >
              {datasets.length === 0 && <option value="">No datasets yet: upload one</option>}
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} · {d.n_rows} rows
                </option>
              ))}
            </select>
            <button
              onClick={() => datasetId && run(datasetId, 0)}
              disabled={!datasetId || loading}
              className="bg-secondary-600 hover:bg-secondary-500 disabled:bg-gray-500 text-white text-xs font-semibold px-3 py-1.5 rounded-lg flex items-center gap-1"
            >
              <Play size={13} /> {loading ? 'Running…' : 'Run batch'}
            </button>
          </div>
        </div>

        <form onSubmit={upload} className="bg-slate-900/50 border border-slate-700 rounded-lg p-3 space-y-2">
          <label className="block text-xs font-medium text-slate-300 flex items-center gap-1.5">
            <FileSpreadsheet size={14} className="text-secondary-400" /> Or load a new CSV to predict
          </label>
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
              placeholder="Target column (optional, enables accuracy)"
              className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-50"
            />
            <button
              type="submit"
              disabled={!file || uploading}
              className="bg-primary-600 hover:bg-primary-500 disabled:opacity-50 disabled:hover:bg-primary-600 disabled:cursor-not-allowed text-white text-xs font-semibold px-3 py-1.5 rounded-lg"
            >
              {uploading ? 'Uploading…' : 'Upload & run'}
            </button>
          </div>
        </form>
      </div>

      {error && <div className="text-xs text-rose-400 bg-rose-950/40 p-2.5 rounded-lg border border-rose-800">{error}</div>}

      {result && dataset && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-3">
              <div className="text-[10px] uppercase text-slate-400">Rows predicted</div>
              <div className="text-lg font-mono text-slate-50">{result.total_rows}</div>
              <div className="text-[10px] text-slate-500">in {fmtMs(result.predict_ms)}</div>
            </div>
            {result.metrics?.accuracy !== undefined && (
              <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-3">
                <div className="text-[10px] uppercase text-slate-400">Accuracy vs “{result.target_column}”</div>
                <div className="text-lg font-mono text-secondary-400">{fmtPct(result.metrics.accuracy)}</div>
              </div>
            )}
            {result.metrics?.mae !== undefined && (
              <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-3">
                <div className="text-[10px] uppercase text-slate-400">MAE / RMSE</div>
                <div className="text-lg font-mono text-slate-50">{fmt(result.metrics.mae, 3)} / {fmt(result.metrics.rmse, 3)}</div>
                {result.metrics.r2 !== undefined && <div className="text-[10px] text-slate-500">R² {fmt(result.metrics.r2, 3)}</div>}
              </div>
            )}
            {result.prediction_stats && (
              <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-3">
                <div className="text-[10px] uppercase text-slate-400">Prediction range</div>
                <div className="text-sm font-mono text-slate-50">{fmt(result.prediction_stats.min, 2)} … {fmt(result.prediction_stats.max, 2)}</div>
                <div className="text-[10px] text-slate-500">mean {fmt(result.prediction_stats.mean, 3)}</div>
              </div>
            )}
            {result.class_counts && (
              <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-3 col-span-2">
                <div className="text-[10px] uppercase text-slate-400 mb-1">Predicted class distribution</div>
                {Object.entries(result.class_counts).map(([cls, n]) => (
                  <div key={cls} className="flex items-center gap-2 text-[11px]">
                    <span className="w-16 font-mono text-slate-300 truncate">{cls}</span>
                    <div className="flex-1 bg-slate-800 rounded h-2">
                      <div className="bg-primary-500 h-2 rounded" style={{ width: `${(n / result.total_rows) * 100}%` }} />
                    </div>
                    <span className="w-10 text-right font-mono text-slate-400">{n}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
            <span>
              Rows {result.offset}–{Math.min(result.offset + PAGE_SIZE, result.total_rows) - 1} of {result.total_rows}
              {allowExplain && ' · click Explain to compute SHAP for a row'}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => explainService.exportBatchCsv(model.id, dataset.id, `${dataset.name.replace(/\.csv$/i, '')}_predictions.csv`).catch((err) => setError(errorMessage(err)))}
                className="bg-gray-600 hover:bg-gray-500 text-white px-2.5 py-1 rounded-md flex items-center gap-1"
              >
                <Download size={12} /> Export CSV
              </button>
              <button disabled={page === 0 || loading} onClick={() => run(dataset.id, page - 1)} className="p-1 rounded bg-gray-600 hover:bg-gray-500 text-white disabled:opacity-40" aria-label="Previous page">
                <ChevronLeft size={14} />
              </button>
              <span className="font-mono">{page + 1} / {totalPages}</span>
              <button disabled={page + 1 >= totalPages || loading} onClick={() => run(dataset.id, page + 1)} className="p-1 rounded bg-gray-600 hover:bg-gray-500 text-white disabled:opacity-40" aria-label="Next page">
                <ChevronRight size={14} />
              </button>
            </div>
          </div>

          <div className={`overflow-x-auto ${loading ? 'opacity-50' : ''}`}>
            <table className="w-full text-left text-xs text-slate-300 border-collapse">
              <thead>
                <tr className="border-b border-slate-700 bg-slate-900/50 text-slate-400">
                  <th className="p-2">#</th>
                  <th className="p-2">{isClassification ? 'Predicted' : 'Prediction'}</th>
                  {isClassification && <th className="p-2">P</th>}
                  {result.target_column && <th className="p-2 text-accent-400">{result.target_column}</th>}
                  {featureCols.map((c) => <th key={c} className="p-2 font-mono whitespace-nowrap">{c}</th>)}
                  {allowExplain && <th className="p-2 text-right">SHAP</th>}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row) => {
                  const match = sameLabel(row.actual, row.predicted_class);
                  const active = explainedRow?.datasetId === dataset.id && explainedRow.rowIndex === row.row_index;
                  return (
                    <tr key={row.row_index} className={`border-b border-slate-800/60 ${active ? 'bg-primary-950/60' : 'hover:bg-slate-700/30'}`}>
                      <td className="p-2 font-mono text-slate-500">{row.row_index}</td>
                      <td className="p-2 font-mono text-slate-50">{row.predicted_class ?? fmt(row.prediction, 3)}</td>
                      {isClassification && <td className="p-2 font-mono text-secondary-400">{fmtPct(row.prediction)}</td>}
                      {result.target_column && (
                        <td className={`p-2 font-mono ${match === false ? 'text-rose-400' : match ? 'text-secondary-400' : ''}`}>
                          {row.actual === null || row.actual === undefined ? '—' : String(row.actual)}
                          {match === false && ' ✗'}
                        </td>
                      )}
                      {featureCols.map((c) => (
                        <td key={c} className="p-2 font-mono whitespace-nowrap">{row.features[c] === null ? '—' : String(row.features[c])}</td>
                      ))}
                      {allowExplain && (
                        <td className="p-2 text-right">
                          <button
                            disabled={explaining}
                            onClick={() => onExplainRow(dataset.id, row.row_index)}
                            className={`px-2.5 py-1 rounded text-[11px] font-medium flex items-center gap-1 ml-auto ${
                              active ? 'bg-primary-600 text-white' : 'bg-gray-600 hover:bg-gray-500 text-white'
                            }`}
                          >
                            <Sparkles size={11} /> {active ? 'Explained' : 'Explain'}
                          </button>
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};
