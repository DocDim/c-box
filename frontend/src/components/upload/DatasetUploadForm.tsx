import React, { useState } from 'react';
import { modelService } from '../../services/api';
import { DatasetPreview } from '../../types/api';
import { FileSpreadsheet } from 'lucide-react';

interface DatasetUploadFormProps {
  onDatasetUploaded: (dataset: DatasetPreview) => void;
}

export const DatasetUploadForm: React.FC<DatasetUploadFormProps> = ({ onDatasetUploaded }) => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a .csv file.');
      return;
    }
    setError('');
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      const preview = await modelService.uploadDataset(formData);
      onDatasetUploaded(preview);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload CSV dataset.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-md">
      <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
        <FileSpreadsheet size={18} className="text-emerald-400" />
        Upload Evaluation Dataset
      </h3>
      {error && <div className="text-xs text-rose-400 bg-rose-950/40 p-2.5 rounded-lg mb-3 border border-rose-800">{error}</div>}

      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label className="block text-xs font-medium text-slate-300 mb-1">CSV File</label>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-emerald-600 file:text-white hover:file:bg-emerald-500 cursor-pointer"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 text-white font-medium py-2 rounded-lg text-xs transition shadow flex items-center justify-center gap-1.5"
        >
          {loading ? 'Parsing Records...' : 'Ingest CSV Dataset'}
        </button>
      </form>
    </div>
  );
};