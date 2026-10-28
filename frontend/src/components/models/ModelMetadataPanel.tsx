import React, { useEffect, useState } from 'react';
import { BookOpenText, Save } from 'lucide-react';
import { ModelArtifact } from '../../types/api';
import { errorMessage, modelService } from '../../services/api';

interface ModelMetadataPanelProps {
  model: ModelArtifact;
  onModelUpdated: (model: ModelArtifact) => void;
}

const inputClass =
  'w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-50 focus:outline-none focus:border-primary-500';

const Field: React.FC<{ label: string; hint: string; children: React.ReactNode }> = ({ label, hint, children }) => (
  <label className="block space-y-1">
    <span className="text-xs font-medium text-slate-300">{label}</span>
    <span className="block text-[11px] text-slate-500">{hint}</span>
    {children}
  </label>
);

/** Owner-only editor for the business context the LLM uses to write for non-technical readers. */
export const ModelMetadataPanel: React.FC<ModelMetadataPanelProps> = ({ model, onModelUpdated }) => {
  const [outcome, setOutcome] = useState('');
  const [decision, setDecision] = useState('');
  const [classes, setClasses] = useState<Record<string, string>>({});
  const [glossary, setGlossary] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setOutcome(model.outcome_description ?? '');
    setDecision(model.decision_context ?? '');
    setClasses(model.class_descriptions ?? {});
    setGlossary(model.feature_glossary ?? {});
  }, [model]);

  useEffect(() => {
    setError('');
    setSaved(false);
  }, [model.id]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    setSaved(false);
    try {
      onModelUpdated(await modelService.updateMetadata(model.id, {
        feature_glossary: glossary,
        outcome_description: outcome.trim() || null,
        class_descriptions: classes,
        decision_context: decision.trim() || null,
      }));
      setSaved(true);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const features = model.feature_names ?? [];
  const classLabels = model.class_labels ?? [];

  return (
    <form onSubmit={handleSave} className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-md space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-slate-50 flex items-center gap-2">
          <BookOpenText size={18} className="text-primary-400" /> Business Context for AI Narratives
        </h3>
        <p className="text-[11px] text-slate-400 mt-1">
          Used to explain predictions in your readers' language. Decision-makers see these names instead of raw
          column names. All fields are optional.
        </p>
      </div>
      {error && <div className="text-xs text-rose-400 bg-rose-950/40 p-2 rounded-lg border border-rose-800">{error}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="space-y-3">
          <Field label="What the model predicts" hint="e.g. Risk that the patient has heart disease">
            <textarea rows={2} maxLength={1000} value={outcome} onChange={(e) => setOutcome(e.target.value)} className={inputClass} />
          </Field>
          <Field
            label="Decision it supports"
            hint="The decision and the actions available, e.g. Refer to a cardiologist, or schedule routine follow-up"
          >
            <textarea rows={3} maxLength={2000} value={decision} onChange={(e) => setDecision(e.target.value)} className={inputClass} />
          </Field>
          {classLabels.length > 0 && (
            <Field label="Class meanings" hint="Readable name for each predicted class">
              <div className="space-y-1.5">
                {classLabels.map((c) => (
                  <div key={c} className="flex items-center gap-2">
                    <span className="w-20 shrink-0 text-[11px] font-mono text-slate-400 truncate" title={c}>{c}</span>
                    <input
                      maxLength={255}
                      value={classes[c] ?? ''}
                      onChange={(e) => setClasses({ ...classes, [c]: e.target.value })}
                      placeholder="e.g. high cardiac risk"
                      className={inputClass}
                    />
                  </div>
                ))}
              </div>
            </Field>
          )}
        </div>

        <Field label="Feature glossary" hint="Plain-language name for each feature">
          {features.length === 0 ? (
            <p className="text-[11px] text-slate-500">Feature names were not detected for this model.</p>
          ) : (
            <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
              {features.map((f) => (
                <div key={f} className="flex items-center gap-2">
                  <span className="w-36 shrink-0 text-[11px] font-mono text-slate-400 truncate" title={f}>{f}</span>
                  <input
                    maxLength={255}
                    value={glossary[f] ?? ''}
                    onChange={(e) => setGlossary({ ...glossary, [f]: e.target.value })}
                    placeholder={f.replace(/_/g, ' ')}
                    className={inputClass}
                  />
                </div>
              ))}
            </div>
          )}
        </Field>
      </div>

      <div className="flex items-center justify-end gap-3">
        {saved && <span className="text-[11px] text-accent-400">Saved. New narratives use this context.</span>}
        <button
          type="submit"
          disabled={busy}
          className="bg-primary-600 hover:bg-primary-500 disabled:bg-gray-500 text-white text-xs font-semibold px-4 py-1.5 rounded-lg flex items-center gap-1.5"
        >
          <Save size={13} /> {busy ? 'Saving…' : 'Save context'}
        </button>
      </div>
    </form>
  );
};
