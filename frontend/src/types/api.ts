export interface User {
  id: number;
  email: string;
  created_at: string;
}

export type Role = 'OWNER' | 'ADMIN' | 'EXPLAINER' | 'VIEWER';
export type TaskType = 'classification' | 'regression' | 'unknown';

export interface ModelArtifact {
  id: number;
  name: string;
  domain_context: string;
  estimator_type: string;
  owner_id: number;
  owner_email?: string | null;
  is_owner: boolean;
  assigned_role: Role;
  created_at: string;
  task_type: TaskType;
  file_format: string;
  n_features?: number | null;
  feature_names?: string[] | null;
  class_labels?: string[] | null;
  background_dataset_id?: number | null;
  /** Business context the LLM prompt layer uses (owner-edited). */
  feature_glossary?: Record<string, string> | null;
  outcome_description?: string | null;
  class_descriptions?: Record<string, string> | null;
  decision_context?: string | null;
}

export interface ModelMetadataUpdate {
  feature_glossary: Record<string, string>;
  outcome_description: string | null;
  class_descriptions: Record<string, string>;
  decision_context: string | null;
}

/** Reader the LLM narrative is written for. */
export type Audience = 'decision_maker' | 'domain_analyst' | 'data_scientist';

export interface SharedUser {
  user_id: number;
  email: string;
  role: Exclude<Role, 'OWNER'>;
  granted_at: string;
}

export interface ColumnSummary {
  dtype: string;
  missing: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  unique?: number;
}

export interface Dataset {
  id: number;
  name: string;
  owner_id: number;
  model_id?: number | null;
  n_rows: number;
  columns: string[];
  target_column?: string | null;
  created_at: string;
  /** 'background' = SHAP reference data (never predicted on); 'evaluation' = data to predict. */
  purpose: 'background' | 'evaluation';
  is_background: boolean;
}

export interface DatasetPreview extends Dataset {
  extra_columns: string[];
  summary: Record<string, ColumnSummary>;
  preview_rows: Record<string, any>[];
}

export interface DatasetRows {
  dataset_id: number;
  offset: number;
  limit: number;
  total_rows: number;
  columns: string[];
  rows: Record<string, any>[];
}

export interface FeatureContribution {
  feature: string;
  actual_value: number | string | null;
  shap_value: number;
}

/** Either manual features, or a dataset row reference (optionally with what-if overrides). */
export interface InstanceSource {
  instance_features?: Record<string, any>;
  dataset_id?: number;
  row_index?: number;
}

export interface InstanceExplainResponse {
  explanation_id: string;
  model_id: number;
  task_type: TaskType;
  explainer_type: string;
  prediction: number;
  predicted_class?: string | null;
  explained_class?: string | null;
  probabilities?: Record<string, number> | null;
  base_value: number;
  explained_output: number;
  output_units: string;
  additivity_gap: number;
  feature_contributions: FeatureContribution[];
  input_features: Record<string, any>;
  dataset_id?: number | null;
  row_index?: number | null;
  shap_ms: number;
  created_at: string;
}

export interface PredictResponse {
  model_id: number;
  task_type: TaskType;
  prediction: number;
  predicted_class?: string | null;
  probabilities?: Record<string, number> | null;
  input_features: Record<string, any>;
}

export interface GlobalExplainResponse {
  model_id: number;
  dataset_id: number;
  samples_evaluated: number;
  explained_output: string;
  explainer_type: string;
  base_value: number;
  global_importance: { feature: string; mean_abs_shap: number }[];
  beeswarm: {
    features: string[];
    shap_values: number[][];
    feature_values: (number | null)[][];
  };
  shap_ms: number;
}

export interface Faithfulness {
  numbers_checked: number;
  numbers_verified: number;
  unverified_numbers: string[];
  fabricated: boolean;
  features_in_context: number;
  features_mentioned: string[];
  sections_found?: string[];
  structure_complete?: boolean;
  /** Factors presented in the wrong section (e.g. a lowering factor listed as an upward pressure). */
  misplaced_features?: { feature: string; section: string; actual_effect: string }[];
  direction_errors?: boolean;
}

export interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  kind: 'narrative' | 'followup';
  audience?: Audience | null;
  content: string;
  llm_model?: string | null;
  latency_ms?: number | null;
  ttft_ms?: number | null;
  faithfulness?: Faithfulness | null;
  created_at: string;
}

export interface ConversationResponse {
  explanation_id: string;
  messages: ChatMessage[];
}

export interface LatencyStats {
  count: number;
  mean?: number | null;
  p50?: number | null;
  p95?: number | null;
  max?: number | null;
}

export interface AuditMetrics {
  model_id: number;
  explanations: number;
  llm_responses: number;
  narratives: number;
  followups: number;
  shap_ms: LatencyStats;
  llm_latency_ms: LatencyStats;
  llm_ttft_ms: LatencyStats;
  max_additivity_gap?: number | null;
  faithfulness_zero_fabrication_rate?: number | null;
  faithfulness_numbers_verified_rate?: number | null;
  narrative_structure_complete_rate?: number | null;
  narrative_direction_error_rate?: number | null;
  mean_features_mentioned?: number | null;
}

export interface AuditExplanation {
  explanation_id: string;
  user_email: string;
  created_at: string;
  explainer_type: string;
  task_type: TaskType;
  predicted_class?: string | null;
  prediction: number;
  base_value: number;
  output_units: string;
  additivity_gap: number;
  shap_ms: number;
  input_hash: string;
  input_features: Record<string, any>;
  dataset_id?: number | null;
  row_index?: number | null;
  feature_contributions: FeatureContribution[];
  messages: ChatMessage[];
}

export interface AuditList {
  model_id: number;
  total: number;
  items: AuditExplanation[];
}

export interface BatchPredictionRow {
  row_index: number;
  features: Record<string, any>;
  prediction?: number | null;
  predicted_class?: string | null;
  probabilities?: Record<string, number> | null;
  actual?: any;
}

export interface BatchPredictResponse {
  model_id: number;
  dataset_id: number;
  task_type: TaskType;
  total_rows: number;
  offset: number;
  limit: number;
  target_column?: string | null;
  class_counts?: Record<string, number> | null;
  prediction_stats?: Record<string, number> | null;
  metrics?: Record<string, number> | null;
  predict_ms: number;
  rows: BatchPredictionRow[];
}
