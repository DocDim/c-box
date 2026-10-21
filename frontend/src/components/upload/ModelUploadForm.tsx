import React, { useState } from 'react';
import { modelService } from '../../services/api';
import { ModelArtifact } from '../../types/api';
import { UploadCloud } from 'lucide-react';

interface ModelUploadFormProps {
  onModelUploaded: (model: ModelArtifact) => void;
}

export const ModelUploadForm: React.FC<ModelUploadFormProps> = ({ onModelUploaded }) => {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [domainContext, setDomainContext] = useState('Clinical Healthcare');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a .joblib or .pkl model file.');
      return;
    }
    setError('');
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', name || file.name);
      formData.append('domain_context', domainContext);

      const model = await modelService.uploadModel(formData);
      onModelUploaded(model);
      setFile(null);
      setName('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload and validate model.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-md">
      <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
        <UploadCloud size={18} className="text-indigo-400" />
        Upload Pre-Trained Model
      </h3>
      {error && (
        <div className="text-xs text-rose-400 bg-rose-950/40 p-2.5 rounded-lg mb-3 border border-rose-800">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1">Model Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Random Forest Classifier"
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1">Domain of Expertise</label>
          <input
            type="text"
            value={domainContext}
            onChange={(e) => setDomainContext(e.target.value)}
            placeholder="e.g. Clinical Healthcare, Credit Risk Scoring"
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1">Binary File (.joblib, .pkl)</label>
          <input
            type="file"
            accept=".joblib,.pkl"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 text-white font-medium py-2 rounded-lg text-xs transition shadow flex items-center justify-center gap-1.5"
        >
          {loading ? 'Validating Artifact...' : 'Ingest Model'}
        </button>
      </form>
    </div>
  );
};