import React, { useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { errorMessage, modelService } from '../../services/api';
import { ModelArtifact } from '../../types/api';

interface RegisterModelFormProps {
  onRegistered: (model: ModelArtifact) => void;
}

const inputClass =
  'w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500';
const fileClass =
  'w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:text-white cursor-pointer';

/** Registers a model together with its background (reference) dataset in one atomic request. */
export const RegisterModelForm: React.FC<RegisterModelFormProps> = ({ onRegistered }) => {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [backgroundFile, setBackgroundFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [domainContext, setDomainContext] = useState('Clinical Healthcare');
  const [targetColumn, setTargetColumn] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [inputKey, setInputKey] = useState(0);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelFile || !backgroundFile) {
      setError('Select both the model file and its background dataset (.csv).');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', modelFile);
      formData.append('background_file', backgroundFile);
      formData.append('name', name || modelFile.name);
      formData.append('domain_context', domainContext);
      if (targetColumn.trim()) formData.append('target_column', targetColumn.trim());
      onRegistered(await modelService.uploadModel(formData));
      setModelFile(null);
      setBackgroundFile(null);
      setName('');
      setTargetColumn('');
      setInputKey((k) => k + 1);
    } catch (err) {
      setError(errorMessage(err, 'Failed to register the model.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-md">
      <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
        <UploadCloud size={18} className="text-indigo-400" /> Register a Model
      </h3>
      <p className="text-[11px] text-slate-400 mb-4">
        The background dataset is the reference distribution SHAP compares predictions against. It must contain
        the model's features; a single extra column is treated as the target.
      </p>
      {error && <div className="text-xs text-rose-400 bg-rose-950/40 p-2.5 rounded-lg mb-3 border border-rose-800">{error}</div>}

      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1">Model Name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Heart Disease RF" className={inputClass} />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1">Domain of Expertise</label>
          <input
            value={domainContext}
            onChange={(e) => setDomainContext(e.target.value)}
            placeholder="e.g. Clinical Cardiology, Credit Risk Scoring"
            className={inputClass}
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1">
            1. Model file <span className="text-slate-500">(.skops recommended, .joblib, .pkl · ≤ 250 MB)</span>
          </label>
          <input
            key={`m${inputKey}`}
            type="file"
            accept=".skops,.joblib,.pkl"
            onChange={(e) => setModelFile(e.target.files?.[0] || null)}
            className={`${fileClass} file:bg-indigo-600 hover:file:bg-indigo-500`}
          />
          <p className="text-[10px] text-slate-500 mt-1">Pickle formats can run code when loaded. Upload only trusted models.</p>
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1">
            2. Background dataset <span className="text-slate-500">(.csv · ≤ 100 MB)</span>
          </label>
          <input
            key={`b${inputKey}`}
            type="file"
            accept=".csv"
            onChange={(e) => setBackgroundFile(e.target.files?.[0] || null)}
            className={`${fileClass} file:bg-emerald-600 hover:file:bg-emerald-500`}
          />
          <input
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            placeholder="Target column (optional, auto-detected)"
            className={`${inputClass} mt-2`}
          />
        </div>

        <div className="md:col-span-2 flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 text-white font-medium px-5 py-2 rounded-lg text-xs transition shadow"
          >
            {loading ? 'Validating model and background…' : 'Register Model'}
          </button>
        </div>
      </form>
    </div>
  );
};
