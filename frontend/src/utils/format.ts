import { ModelArtifact, Role } from '../types/api';

export const canExplain = (role?: Role | null) =>
  role === 'OWNER' || role === 'ADMIN' || role === 'EXPLAINER';

export const canManage = (role?: Role | null) => role === 'OWNER' || role === 'ADMIN';

export const fmt = (value: number | null | undefined, digits = 4) =>
  value === null || value === undefined || Number.isNaN(value) ? '—' : value.toFixed(digits);

export const fmtMs = (value: number | null | undefined) =>
  value === null || value === undefined ? '—' : value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(0)} ms`;

export const fmtPct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`;

/** Backend timestamps are naive UTC; mark them as UTC before converting to local time. */
export const fmtDate = (iso: string) =>
  new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`).toLocaleString();

/** Feature inputs for a model: fitted names, else dataset columns, else positional names. */
export const modelFeatureColumns = (
  model: ModelArtifact | null,
  datasetColumns?: string[] | null,
  targetColumn?: string | null
): string[] => {
  if (!model) return [];
  if (model.feature_names && model.feature_names.length > 0) return model.feature_names;
  if (datasetColumns && datasetColumns.length > 0) {
    return datasetColumns.filter((c) => c !== targetColumn);
  }
  return Array.from({ length: model.n_features ?? 0 }, (_, i) => `x${i}`);
};
