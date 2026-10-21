import axios from 'axios';
import { 
  ModelArtifact, 
  DatasetPreview, 
  InstanceExplainResponse, 
  ChatResponse 
} from '../types/api';

const api = axios.create({
  baseURL: '/api/v1',
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const authService = {
  login: async (formData: FormData) => {
    const res = await api.post('/auth/login', formData);
    localStorage.setItem('token', res.data.access_token);
    return res.data;
  },
  register: async (data: { email: string; password: string }) => {
    return (await api.post('/auth/register', data)).data;
  },
  getMe: async () => {
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
  uploadModel: async (formData: FormData): Promise<ModelArtifact> => {
    return (await api.post('/upload/model', formData)).data;
  },
  uploadDataset: async (formData: FormData): Promise<DatasetPreview> => {
    return (await api.post('/upload/dataset', formData)).data;
  },
  explainInstance: async (modelId: number, features: Record<string, any>): Promise<InstanceExplainResponse> => {
    return (await api.post('/explain/instance', {
      model_id: modelId,
      instance_features: features
    })).data;
  },
  getChatExplanation: async (
    modelId: number,
    prediction: number,
    baseValue: number,
    contributions: any[],
    userQuery?: string
  ): Promise<ChatResponse> => {
    return (await api.post('/chat/chat', {
      model_id: modelId,
      prediction,
      base_value: baseValue,
      feature_contributions: contributions,
      user_query: userQuery
    })).data;
  }
};

export default api;