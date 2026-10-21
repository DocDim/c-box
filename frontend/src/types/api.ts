export interface User {
  id: number;
  email: string;
  created_at: string;
}

export interface ModelArtifact {
  id: number;
  name: string;
  domain_context: string;
  estimator_type: string;
  owner_id: number;
  is_owner: boolean;
  assigned_role: string;
  created_at: string;
}

export interface DatasetPreview {
  filename: string;
  total_rows: number;
  total_columns: number;
  columns: string[];
  preview_rows: Record<string, any>[];
}

export interface FeatureContribution {
  feature: string;
  actual_value: number | string;
  shap_value: number;
}

export interface InstanceExplainResponse {
  model_id: number;
  prediction: number;
  base_value: number;
  feature_contributions: FeatureContribution[];
}

export interface ChatResponse {
  explanation: string;
}