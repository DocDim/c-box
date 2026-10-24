import axios from 'axios';
import {
  AuditList,
  BatchPredictResponse,
  AuditMetrics,
  ChatMessage,
  ConversationResponse,
  Dataset,
  DatasetPreview,
  DatasetRows,
  GlobalExplainResponse,
  InstanceExplainResponse,
  InstanceSource,
  ModelArtifact,
  PredictResponse,
  Role,
  SharedUser,
  User,
} from '../types/api';

const API_BASE = '/api/v1';

const api = axios.create({
  baseURL: API_BASE,
});

const authHeader = (): Record<string, string> => {
  const token = localStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
};

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/** Saves a blob response as a file download. */
const downloadBlob = (data: Blob, filename: string) => {
  const url = URL.createObjectURL(data);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

/** Extracts a readable message from axios / fetch errors (FastAPI `detail` can be a list). */
export const errorMessage = (err: any, fallback = 'Request failed.'): string => {
  const detail = err?.response?.data?.detail ?? err?.detail;
  if (Array.isArray(detail)) return detail.map((d: any) => d.msg ?? String(d)).join('; ');
  if (typeof detail === 'string') return detail;
  return err?.message || fallback;
};

export const authService = {
  login: async (formData: FormData) => {
    const res = await api.post('/auth/login', formData);
    localStorage.setItem('token', res.data.access_token);
    return res.data;
  },
  register: async (data: { email: string; password: string }) => {
    return (await api.post('/auth/register', data)).data;
  },
  getMe: async (): Promise<User> => {
    return (await api.get('/auth/me')).data;
  },
  logout: () => {
    localStorage.removeItem('token');
  }
};

export const modelService = {
  listModels: async (): Promise<ModelArtifact[]> => {
    return (await api.get('/models/')).data;
  },
  getModel: async (modelId: number): Promise<ModelArtifact> => {
    return (await api.get(`/models/${modelId}`)).data;
  },
  uploadModel: async (formData: FormData): Promise<ModelArtifact> => {
    return (await api.post('/models/upload', formData)).data;
  },
  deleteModel: async (modelId: number): Promise<void> => {
    await api.delete(`/models/${modelId}`);
  },
  /** Owner only: uploads or replaces the model's background (SHAP reference) data. */
  replaceBackground: async (modelId: number, file: File, targetColumn?: string): Promise<ModelArtifact> => {
    const form = new FormData();
    form.append('file', file);
    if (targetColumn) form.append('target_column', targetColumn);
    return (await api.post(`/models/${modelId}/background`, form)).data;
  },
  listShares: async (modelId: number): Promise<SharedUser[]> => {
    return (await api.get(`/models/${modelId}/shares`)).data;
  },
  grantShare: async (modelId: number, userEmail: string, role: Exclude<Role, 'OWNER'>): Promise<SharedUser> => {
    return (await api.post(`/models/${modelId}/share`, { user_email: userEmail, role })).data;
  },
  revokeShare: async (modelId: number, userId: number): Promise<void> => {
    await api.delete(`/models/${modelId}/share/${userId}`);
  },
};

export const datasetService = {
  uploadDataset: async (formData: FormData): Promise<DatasetPreview> => {
    return (await api.post('/datasets/upload', formData)).data;
  },
  listDatasets: async (modelId?: number, purpose?: Dataset['purpose']): Promise<Dataset[]> => {
    return (await api.get('/datasets/', { params: { model_id: modelId, purpose } })).data;
  },
  getRows: async (datasetId: number, offset: number, limit: number): Promise<DatasetRows> => {
    return (await api.get(`/datasets/${datasetId}/rows`, { params: { offset, limit } })).data;
  },
  deleteDataset: async (datasetId: number): Promise<void> => {
    await api.delete(`/datasets/${datasetId}`);
  },
};

export const explainService = {
  predict: async (modelId: number, source: InstanceSource): Promise<PredictResponse> => {
    return (await api.post('/predict/', { model_id: modelId, ...source })).data;
  },
  predictBatch: async (modelId: number, datasetId: number, offset: number, limit: number): Promise<BatchPredictResponse> => {
    return (await api.post('/predict/batch', { model_id: modelId, dataset_id: datasetId, offset, limit })).data;
  },
  exportBatchCsv: async (modelId: number, datasetId: number, filename: string): Promise<void> => {
    const res = await api.get('/predict/batch/export.csv', {
      params: { model_id: modelId, dataset_id: datasetId },
      responseType: 'blob',
    });
    downloadBlob(res.data, filename);
  },
  explainInstance: async (modelId: number, source: InstanceSource): Promise<InstanceExplainResponse> => {
    return (await api.post('/explain/instance', { model_id: modelId, ...source })).data;
  },
  /** Global SHAP over the model's background data. */
  explainGlobal: async (modelId: number, maxSamples: number, classIndex?: number): Promise<GlobalExplainResponse> => {
    return (await api.get(`/explain/global/${modelId}`, {
      params: { max_samples: maxSamples, class_index: classIndex },
    })).data;
  },
  getConversation: async (explanationId: string): Promise<ConversationResponse> => {
    return (await api.get(`/explain/${explanationId}/conversation`)).data;
  },

  /**
   * Streams a grounded narrative (question omitted) or follow-up answer over SSE.
   * Calls onToken for every token and resolves with the persisted message.
   */
  chatStream: async (
    explanationId: string,
    question: string | undefined,
    onToken: (text: string) => void,
    signal?: AbortSignal
  ): Promise<ChatMessage> => {
    const res = await fetch(`${API_BASE}/explain/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeader() },
      body: JSON.stringify({ explanation_id: explanationId, question }),
      signal,
    });
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => ({}));
      throw new Error(errorMessage(body, `Chat request failed (HTTP ${res.status}).`));
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalMessage: ChatMessage | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf('\n\n');

        let event = 'message';
        let data = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event: ')) event = line.slice(7).trim();
          else if (line.startsWith('data: ')) data += line.slice(6);
        }
        if (!data) continue;
        const parsed = JSON.parse(data);
        if (event === 'token') onToken(parsed.text);
        else if (event === 'done') finalMessage = parsed as ChatMessage;
        else if (event === 'error') throw new Error(parsed.detail || 'LLM generation failed.');
      }
    }
    if (!finalMessage) throw new Error('The LLM stream ended unexpectedly.');
    return finalMessage;
  },
};

export const auditService = {
  metrics: async (modelId: number): Promise<AuditMetrics> => {
    return (await api.get(`/audit/models/${modelId}/metrics`)).data;
  },
  explanations: async (modelId: number, limit = 20, offset = 0): Promise<AuditList> => {
    return (await api.get(`/audit/models/${modelId}/explanations`, { params: { limit, offset } })).data;
  },
  exportCsv: async (modelId: number): Promise<void> => {
    const res = await api.get(`/audit/models/${modelId}/export.csv`, { responseType: 'blob' });
    downloadBlob(res.data, `model_${modelId}_audit.csv`);
  },
};

export default api;
