import React, { useState } from 'react';
import { FlaskConical, Table2, TextCursorInput } from 'lucide-react';
import { Dataset, InstanceExplainResponse, InstanceSource, ModelArtifact } from '../types/api';
import { canExplain, modelFeatureColumns } from '../utils/format';
import { SinglePrediction } from '../components/predict/SinglePrediction';
import { BatchPrediction } from '../components/predict/BatchPrediction';
import { WaterfallPlot } from '../components/visualizations/WaterfallPlot';
import { ForcePlot } from '../components/visualizations/ForcePlot';

type Mode = 'single' | 'batch';

interface PredictPageProps {
  models: ModelArtifact[];
  model: ModelArtifact | null;
  datasets: Dataset[];
  onSelectModel: (model: ModelArtifact) => void;
  onDatasetAdded: (dataset: Dataset) => void;
  explanation: InstanceExplainResponse | null;
  explaining: boolean;
  explainError: string;
  onExplain: (source: InstanceSource) => void;
}

export const PredictPage: React.FC<PredictPageProps> = ({
  models,
  model,
  datasets,
  onSelectModel,
  onDatasetAdded,
  explanation,
  explaining,
  explainError,
  onExplain,
}) => {
  const [mode, setMode] = useState<Mode>('single');
  // Background data is SHAP reference only: just its column names are used, never its rows
  const background = datasets.find((d) => d.id === model?.background_dataset_id) ?? null;
  const evaluationDatasets = datasets.filter((d) => d.purpose === 'evaluation');
  const allowExplain = canExplain(model?.assigned_role);
  const schemaSource = background ?? evaluationDatasets[0];
  const columns = modelFeatureColumns(model, schemaSource?.columns, schemaSource?.target_column);
  const explainedRow =
    explanation && explanation.dataset_id !== null && explanation.dataset_id !== undefined &&
    explanation.row_index !== null && explanation.row_index !== undefined
      ? { datasetId: explanation.dataset_id, rowIndex: explanation.row_index }
      : null;

  return (
    <div className="space-y-6">
      <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-md space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="flex-1 min-w-[16rem]">
            <label className="block text-xs font-medium text-slate-300 mb-1 flex items-center gap-1.5">
              <FlaskConical size={14} className="text-primary-400" /> Model used for this prediction
            </label>
            <select
              value={model?.id ?? ''}
              onChange={(e) => {
                const m = models.find((x) => x.id === Number(e.target.value));
                if (m) onSelectModel(m);
              }}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-50"
            >
              {models.length === 0 && <option value="">No models available. Register one first.</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} · {m.estimator_type} · {m.assigned_role}
                </option>
              ))}
            </select>
          </div>
          <div className="flex bg-slate-900 border border-slate-700 rounded-lg p-1 text-xs">
            <button
              onClick={() => setMode('single')}
              className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 ${mode === 'single' ? 'bg-primary-600 text-white' : 'text-slate-400 hover:text-slate-50'}`}
            >
              <TextCursorInput size={14} /> Single data point
            </button>
            <button
              onClick={() => setMode('batch')}
              className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 ${mode === 'batch' ? 'bg-primary-600 text-white' : 'text-slate-400 hover:text-slate-50'}`}
            >
              <Table2 size={14} /> Batch (dataset)
            </button>
          </div>
        </div>

        {model && !allowExplain && (
          <div className="text-xs text-accent-300 bg-accent-950/40 border border-accent-800 rounded-lg p-2.5">
            VIEWER access: predictions only. SHAP and LLM explanations require EXPLAINER access.
          </div>
        )}

        {model && (mode === 'single' ? (
          <SinglePrediction
            model={model}
            columns={columns}
            allowExplain={allowExplain}
            explaining={explaining}
            onExplain={onExplain}
          />
        ) : (
          <BatchPrediction
            model={model}
            datasets={evaluationDatasets}
            onDatasetAdded={onDatasetAdded}
            allowExplain={allowExplain}
            explaining={explaining}
            explainedRow={explainedRow}
            onExplainRow={(datasetId, rowIndex) => onExplain({ dataset_id: datasetId, row_index: rowIndex })}
          />
        ))}
      </div>

      {explainError && (
        <div className="text-xs text-rose-400 bg-rose-950/40 p-3 rounded-lg border border-rose-800">{explainError}</div>
      )}

      {explanation && model && explanation.model_id === model.id && (
        <div className={`grid grid-cols-1 2xl:grid-cols-2 gap-6 items-start ${explaining ? 'opacity-60' : ''}`}>
          <WaterfallPlot result={explanation} />
          <ForcePlot result={explanation} />
        </div>
      )}
    </div>
  );
};
