import React, { useEffect, useState } from 'react';
import { ClipboardCheck, Download, RefreshCw } from 'lucide-react';
import { AuditList, AuditMetrics, ModelArtifact } from '../../types/api';
import { auditService, errorMessage } from '../../services/api';
import { fmt, fmtDate, fmtMs, fmtPct } from '../../utils/format';
import { renderMarkdown } from '../../utils/markdown';

interface AuditPanelProps {
  model: ModelArtifact;
  refreshKey: number;
}

const Stat: React.FC<{ label: string; value: string; hint?: string }> = ({ label, value, hint }) => (
  <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-3">
    <div className="text-[10px] uppercase tracking-wider text-slate-400">{label}</div>
    <div className="text-lg font-semibold text-white font-mono">{value}</div>
    {hint && <div className="text-[10px] text-slate-500">{hint}</div>}
  </div>
);

export const AuditPanel: React.FC<AuditPanelProps> = ({ model, refreshKey }) => {
  const [metrics, setMetrics] = useState<AuditMetrics | null>(null);
  const [trail, setTrail] = useState<AuditList | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState('');

  const load = async () => {
    setError('');
    try {
      const [m, t] = await Promise.all([auditService.metrics(model.id), auditService.explanations(model.id, 20)]);
      setMetrics(m);
      setTrail(t);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model.id, refreshKey]);

  return (
    <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-md space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <ClipboardCheck size={18} className="text-teal-400" /> Audit Trail &amp; Evaluation Metrics
        </h3>
        <div className="flex gap-2">
          <button onClick={load} className="text-xs bg-slate-900 border border-slate-700 text-slate-300 hover:text-white px-2.5 py-1 rounded-md flex items-center gap-1">
            <RefreshCw size={12} /> Refresh
          </button>
          <button
            onClick={() => auditService.exportCsv(model.id).catch((err) => setError(errorMessage(err)))}
            className="text-xs bg-teal-700 hover:bg-teal-600 text-white px-2.5 py-1 rounded-md flex items-center gap-1"
          >
            <Download size={12} /> Export CSV
          </button>
        </div>
      </div>
      {error && <div className="text-xs text-rose-400">{error}</div>}

      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Explanations" value={String(metrics.explanations)} hint={`${metrics.narratives} narratives · ${metrics.followups} follow-ups`} />
          <Stat label="SHAP latency p95" value={fmtMs(metrics.shap_ms.p95)} hint={`mean ${fmtMs(metrics.shap_ms.mean)} · target < 2.5 s`} />
          <Stat label="LLM first token p95" value={fmtMs(metrics.llm_ttft_ms.p95)} hint={`total mean ${fmtMs(metrics.llm_latency_ms.mean)} · target < 4 s`} />
          <Stat label="Zero-fabrication rate" value={fmtPct(metrics.faithfulness_zero_fabrication_rate)} hint={`numbers verified ${fmtPct(metrics.faithfulness_numbers_verified_rate)}`} />
          <Stat label="4-part structure" value={fmtPct(metrics.narrative_structure_complete_rate)} hint="narratives with all sections" />
          <Stat label="Direction errors" value={fmtPct(metrics.narrative_direction_error_rate)} hint="narratives with a factor in the wrong section" />
          <Stat label="Features cited / answer" value={fmt(metrics.mean_features_mentioned ?? null, 1)} />
          <Stat label="Max additivity gap" value={metrics.max_additivity_gap === null || metrics.max_additivity_gap === undefined ? '—' : metrics.max_additivity_gap.toExponential(1)} hint="|E[f(x)] + Σφ − f(x)|" />
        </div>
      )}

      {trail && trail.items.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left text-slate-300">
            <thead className="text-slate-400 border-b border-slate-700">
              <tr>
                <th className="p-2">When</th><th className="p-2">User</th><th className="p-2">Source</th>
                <th className="p-2">Prediction</th><th className="p-2">SHAP</th><th className="p-2">LLM answers</th>
              </tr>
            </thead>
            <tbody>
              {trail.items.map((item) => {
                const answers = item.messages.filter((m) => m.role === 'assistant');
                const flagged = answers.some((m) => m.faithfulness?.fabricated || m.faithfulness?.direction_errors);
                return (
                  <React.Fragment key={item.explanation_id}>
                    <tr
                      onClick={() => setExpanded(expanded === item.explanation_id ? null : item.explanation_id)}
                      className="border-b border-slate-800 hover:bg-slate-700/40 cursor-pointer"
                    >
                      <td className="p-2 whitespace-nowrap">{fmtDate(item.created_at)}</td>
                      <td className="p-2">{item.user_email}</td>
                      <td className="p-2 font-mono">{item.row_index !== null && item.row_index !== undefined ? `row #${item.row_index}` : 'manual'}</td>
                      <td className="p-2 font-mono">{item.predicted_class !== null && item.predicted_class !== undefined ? `class ${item.predicted_class} · ` : ''}{fmt(item.prediction, 3)}</td>
                      <td className="p-2 font-mono">{fmtMs(item.shap_ms)}</td>
                      <td className="p-2">
                        {answers.length}
                        {flagged && <span className="ml-2 text-amber-400">⚠ grounding issues</span>}
                      </td>
                    </tr>
                    {expanded === item.explanation_id && (
                      <tr className="bg-slate-900/60">
                        <td colSpan={6} className="p-3 space-y-2">
                          <div className="font-mono text-[11px] text-slate-400 break-all">
                            input hash {item.input_hash} · {item.explainer_type} · units {item.output_units}
                          </div>
                          <div className="font-mono text-[11px] text-slate-300 break-all">{JSON.stringify(item.input_features)}</div>
                          {item.messages.map((m) =>
                            m.role === 'user' ? (
                              <div key={m.id} className="text-[11px] text-indigo-300">
                                <span className="font-semibold">Q: </span>{m.content}
                              </div>
                            ) : (
                              <div key={m.id} className="text-[11px] text-slate-300 border-l-2 border-slate-700 pl-3">
                                <div className="text-[10px] uppercase tracking-wider text-slate-500">
                                  {m.kind} · {m.llm_model} · {fmtMs(m.latency_ms)}
                                </div>
                                {m.faithfulness?.fabricated && (
                                  <div className="text-amber-400 my-1">
                                    ⚠ Unverified numbers: {m.faithfulness.unverified_numbers.join(', ')}
                                  </div>
                                )}
                                {m.faithfulness?.direction_errors && (
                                  <div className="text-amber-400 my-1">
                                    ⚠ Wrong section: {(m.faithfulness.misplaced_features ?? [])
                                      .map((x) => `${x.feature} (actually ${x.actual_effect})`).join('; ')}
                                  </div>
                                )}
                                {renderMarkdown(m.content)}
                              </div>
                            )
                          )}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
          {trail.total > trail.items.length && (
            <p className="text-[11px] text-slate-500 mt-2">Showing latest {trail.items.length} of {trail.total}; export CSV for the full trail.</p>
          )}
        </div>
      )}
      {trail && trail.items.length === 0 && <p className="text-xs text-slate-500 italic">No explanations recorded yet.</p>}
    </div>
  );
};
