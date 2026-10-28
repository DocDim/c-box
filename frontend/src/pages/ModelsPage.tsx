import React, { useEffect, useState } from 'react';
import { ChevronDown, ChevronUp, FlaskConical, Trash2 } from 'lucide-react';
import { Dataset, ModelArtifact } from '../types/api';
import { datasetService, errorMessage, modelService } from '../services/api';
import { BackgroundDataCard } from '../components/models/BackgroundDataCard';
import { canManage, fmtDate } from '../utils/format';
import { RegisterModelForm } from '../components/models/RegisterModelForm';
import { SharingPanel } from '../components/models/SharingPanel';
import { ModelMetadataPanel } from '../components/models/ModelMetadataPanel';

interface ModelsPageProps {
  models: ModelArtifact[];
  activeModel: ModelArtifact | null;
  onModelRegistered: (model: ModelArtifact) => void;
  onModelUpdated: (model: ModelArtifact) => void;
  onModelDeleted: (modelId: number) => void;
  onActivate: (model: ModelArtifact) => void;
  onPredict: (model: ModelArtifact) => void;
  datasetsVersion: number;
}

export const ModelsPage: React.FC<ModelsPageProps> = ({
  models,
  activeModel,
  onModelRegistered,
  onModelUpdated,
  onModelDeleted,
  onActivate,
  onPredict,
  datasetsVersion,
}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    datasetService.listDatasets(undefined, 'background').then(setDatasets).catch(() => setDatasets([]));
  }, [models, datasetsVersion]);

  const remove = async (m: ModelArtifact) => {
    if (!window.confirm(`Delete "${m.name}"? Shares and explanation history are removed too.`)) return;
    try {
      await modelService.deleteModel(m.id);
      onModelDeleted(m.id);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  return (
    <div className="space-y-6">
      <RegisterModelForm onRegistered={onModelRegistered} />

      <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-md">
        <h3 className="text-sm font-semibold text-slate-50 mb-3">Your Models</h3>
        {error && <div className="text-xs text-rose-400 mb-2">{error}</div>}
        {models.length === 0 ? (
          <p className="text-xs text-slate-500 italic">No models yet. Register one above, or ask a colleague to share one.</p>
        ) : (
          <ul className="space-y-2">
            {models.map((m) => {
              const isActive = activeModel?.id === m.id;
              const background = datasets.find((d) => d.id === m.background_dataset_id);
              const open = expanded === m.id;
              return (
                <li key={m.id} className={`rounded-lg border ${isActive ? 'border-primary-500 bg-primary-950/30' : 'border-slate-700 bg-slate-900/40'}`}>
                  <div className="flex flex-wrap items-center gap-3 p-3">
                    <div className="flex-1 min-w-[12rem]">
                      <div className="text-sm text-slate-50 font-medium">
                        {m.name} {isActive && <span className="ml-1 text-[10px] uppercase text-primary-300">active</span>}
                      </div>
                      <div className="text-[11px] text-slate-400">
                        {m.estimator_type} · {m.task_type} · {m.n_features ?? '?'} features · {m.domain_context}
                      </div>
                      <div className="text-[11px] text-slate-500">
                        {m.is_owner ? 'Owned by you' : `Shared by ${m.owner_email ?? 'owner'}`} · role{' '}
                        <span className="text-secondary-400">{m.assigned_role}</span> · background{' '}
                        <span className={background ? 'text-slate-300' : 'text-accent-400'}>{background ? background.name : 'none'}</span>
                        {' · '}added {fmtDate(m.created_at)}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {!isActive && (
                        <button onClick={() => onActivate(m)} className="text-xs px-3 py-1.5 rounded-md bg-accent-600 hover:bg-accent-500 text-white">
                          Set active
                        </button>
                      )}
                      <button onClick={() => onPredict(m)} className="text-xs px-3 py-1.5 rounded-md bg-primary-600 hover:bg-primary-500 text-white flex items-center gap-1">
                        <FlaskConical size={12} /> Predict
                      </button>
                      {canManage(m.assigned_role) && (
                        <button
                          onClick={() => setExpanded(open ? null : m.id)}
                          className="text-xs px-2 py-1.5 rounded-md bg-secondary-600 hover:bg-secondary-500 text-white flex items-center gap-1"
                        >
                          Manage {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </button>
                      )}
                    </div>
                  </div>

                  {open && (
                    <div className="border-t border-slate-700 p-3 grid grid-cols-1 lg:grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <BackgroundDataCard model={m} background={background ?? null} onModelUpdated={onModelUpdated} />
                        {m.is_owner && (
                          <button
                            onClick={() => remove(m)}
                            className="text-[11px] bg-rose-600 hover:bg-rose-500 text-white px-2.5 py-1 rounded-md flex items-center gap-1"
                          >
                            <Trash2 size={12} /> Delete model
                          </button>
                        )}
                      </div>
                      <SharingPanel model={m} />
                      {m.is_owner && (
                        <div className="lg:col-span-2">
                          <ModelMetadataPanel model={m} onModelUpdated={onModelUpdated} />
                        </div>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
};
