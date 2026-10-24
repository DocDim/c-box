import React from 'react';
import { BarChart3, ClipboardCheck, Database, FlaskConical, Layers } from 'lucide-react';
import { Dataset, ModelArtifact } from '../../types/api';
import { Page } from '../../hooks/useHashRoute';
import { canExplain, canManage } from '../../utils/format';

interface SidebarProps {
  page: Page;
  onNavigate: (page: Page) => void;
  activeModel: ModelArtifact | null;
  activeDatasets: Dataset[];
}

const NAV: { page: Page; label: string; icon: React.ReactNode; show: (m: ModelArtifact | null) => boolean }[] = [
  { page: 'models', label: 'Models', icon: <Layers size={16} />, show: () => true },
  { page: 'predict', label: 'Predict & Explain', icon: <FlaskConical size={16} />, show: () => true },
  { page: 'insights', label: 'Global Insights', icon: <BarChart3 size={16} />, show: (m) => canExplain(m?.assigned_role) },
  { page: 'audit', label: 'Audit & Evaluation', icon: <ClipboardCheck size={16} />, show: (m) => canManage(m?.assigned_role) },
];

const Row: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div className="flex justify-between gap-3 py-1 border-b border-slate-800 last:border-0">
    <dt className="text-slate-500 shrink-0">{label}</dt>
    <dd className="text-slate-200 text-right truncate">{children}</dd>
  </div>
);

export const Sidebar: React.FC<SidebarProps> = ({ page, onNavigate, activeModel, activeDatasets }) => {
  const background = activeDatasets.find((d) => d.id === activeModel?.background_dataset_id);

  return (
    <aside className="space-y-5">
      <nav className="space-y-1">
        {NAV.filter((n) => n.show(activeModel)).map((n) => (
          <button
            key={n.page}
            onClick={() => onNavigate(n.page)}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition ${
              page === n.page ? 'bg-indigo-600 text-white shadow' : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            {n.icon} {n.label}
          </button>
        ))}
      </nav>

      <section className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 space-y-3">
        <h3 className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Active Model</h3>
        {!activeModel ? (
          <p className="text-xs text-slate-500 italic">
            No active model. Register one or pick one on the Models or Predict page.
          </p>
        ) : (
          <>
            <div>
              <div className="text-sm font-semibold text-white break-words">{activeModel.name}</div>
              <div className="text-xs text-indigo-300">{activeModel.domain_context}</div>
            </div>
            <dl className="text-xs">
              <Row label="Your role"><span className="text-emerald-400 font-medium">{activeModel.assigned_role}</span></Row>
              {!activeModel.is_owner && <Row label="Owner">{activeModel.owner_email ?? '—'}</Row>}
              <Row label="Estimator">{activeModel.estimator_type}</Row>
              <Row label="Task">{activeModel.task_type}</Row>
              {activeModel.class_labels && <Row label="Classes">{activeModel.class_labels.join(', ')}</Row>}
              <Row label="Format">.{activeModel.file_format}</Row>
              <Row label="Features">{activeModel.n_features ?? '?'}</Row>
            </dl>

            <div className="text-xs">
              <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                <Database size={12} /> Background data
              </div>
              {background ? (
                <div className="text-slate-200">
                  {background.name}
                  <span className="text-slate-500"> · {background.n_rows} rows</span>
                  {background.target_column && <div className="text-slate-500">target: {background.target_column}</div>}
                </div>
              ) : (
                <div className="text-amber-400">None. Non-tree models cannot be explained.</div>
              )}
            </div>

            {activeModel.feature_names && (
              <details className="text-xs">
                <summary className="cursor-pointer text-slate-400 hover:text-white">Feature names</summary>
                <ul className="mt-1 max-h-40 overflow-y-auto font-mono text-[11px] text-slate-300 space-y-0.5">
                  {activeModel.feature_names.map((f) => <li key={f} className="truncate" title={f}>{f}</li>)}
                </ul>
              </details>
            )}
          </>
        )}
      </section>
    </aside>
  );
};
