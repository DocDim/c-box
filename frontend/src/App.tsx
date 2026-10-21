import React, { useState, useEffect } from 'react';
import { Navbar } from './components/common/Navbar';
import { ModelUploadForm } from './components/upload/ModelUploadForm';
import { DatasetUploadForm } from './components/upload/DatasetUploadForm';
import { DatasetTable } from './components/data/DatasetTable';
import { WaterfallPlot } from './components/visualizations/WaterfallPlot';
import { ExplanationPanel } from './components/chat/ExplanationPanel';
import { ChatFollowUp } from './components/chat/ChatFollowUp';

import { authService, modelService } from './services/api';
import { User, ModelArtifact, DatasetPreview, InstanceExplainResponse } from './types/api';

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [models, setModels] = useState<ModelArtifact[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelArtifact | null>(null);
  const [dataset, setDataset] = useState<DatasetPreview | null>(null);
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null);

  const [explainResult, setExplainResult] = useState<InstanceExplainResponse | null>(null);
  const [explanationText, setExplanationText] = useState<string>('');
  const [loadingExplain, setLoadingExplain] = useState(false);

  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('token');
      if (token) {
        try {
          const me = await authService.getMe();
          setUser(me);
          const modelList = await modelService.listModels();
          setModels(modelList);
          if (modelList.length > 0) setSelectedModel(modelList[0]);
        } catch {
          authService.logout();
          setUser(null);
        }
      }
    };
    initAuth();
  }, []);

  const handleAuthSuccess = async (authenticatedUser: User) => {
    setUser(authenticatedUser);
    const modelList = await modelService.listModels();
    setModels(modelList);
    if (modelList.length > 0) setSelectedModel(modelList[0]);
  };

  const handleLogout = () => {
    authService.logout();
    setUser(null);
    setModels([]);
    setSelectedModel(null);
    setDataset(null);
    setExplainResult(null);
    setExplanationText('');
  };

  const handleSelectRow = async (rowIndex: number, rowData: Record<string, any>) => {
    if (!selectedModel) return;
    setSelectedRowIndex(rowIndex);
    setLoadingExplain(true);
    setExplanationText('');

    try {
      // 1. Compute Local SHAP Attributions & Prediction
      const res = await modelService.explainInstance(selectedModel.id, rowData);
      setExplainResult(res);

      // 2. Compute Ollama Gemma Grounded Narrative
      const chatRes = await modelService.getChatExplanation(
        selectedModel.id,
        res.prediction,
        res.base_value,
        res.feature_contributions
      );
      setExplanationText(chatRes.explanation);
    } catch (err: any) {
      setExplanationText(`Error computing explanation: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoadingExplain(false);
    }
  };

  const handleFollowUpQuestion = async (query: string) => {
    if (!selectedModel || !explainResult) return;
    setLoadingExplain(true);
    try {
      const chatRes = await modelService.getChatExplanation(
        selectedModel.id,
        explainResult.prediction,
        explainResult.base_value,
        explainResult.feature_contributions,
        query
      );
      setExplanationText(prev => `${prev}\n\n--- Follow-up Q: ${query} ---\n${chatRes.explanation}`);
    } catch (err: any) {
      setExplanationText(prev => `${prev}\n\nError answering follow-up: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoadingExplain(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      <Navbar user={user} onAuthSuccess={handleAuthSuccess} onLogout={handleLogout} />

      <main className="max-w-7xl mx-auto px-6 py-6 flex-1 w-full space-y-6">
        {!user ? (
          <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-10 text-center max-w-xl mx-auto my-12 shadow-xl">
            <h2 className="text-2xl font-bold text-white mb-2">Welcome to CrystalBox</h2>
            <p className="text-sm text-slate-400 mb-6">
              Connect your pre-trained models with on-premise SHAP and Ollama Gemma interpretations.
            </p>
            <p className="text-xs text-indigo-400 font-mono">
              Please sign in or register at the top right to access your sandboxed workspace.
            </p>
          </div>
        ) : (
          <>
            {/* Top Row: Ingestion Controls */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <ModelUploadForm
                onModelUploaded={(m) => {
                  setModels(prev => [m, ...prev]);
                  setSelectedModel(m);
                }}
              />
              <DatasetUploadForm onDatasetUploaded={(d) => setDataset(d)} />

              <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-md flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white mb-2">Active Model Context</h3>
                  {models.length > 0 ? (
                    <div className="space-y-2">
                      <select
                        value={selectedModel?.id || ''}
                        onChange={(e) => {
                          const m = models.find(x => x.id === Number(e.target.value));
                          if (m) setSelectedModel(m);
                        }}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-xs text-white"
                      >
                        {models.map(m => (
                          <option key={m.id} value={m.id}>
                            {m.name} ({m.estimator_type})
                          </option>
                        ))}
                      </select>
                      <div className="text-xs text-slate-400 space-y-1">
                        <p>Domain: <span className="text-indigo-400 font-medium">{selectedModel?.domain_context}</span></p>
                        <p>Role: <span className="text-emerald-400 font-medium">{selectedModel?.assigned_role}</span></p>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 italic">No models uploaded yet.</p>
                  )}
                </div>
              </div>
            </div>

            {/* Middle Row: Dataset Inspection */}
            {dataset && (
              <DatasetTable
                dataset={dataset}
                selectedRowIndex={selectedRowIndex}
                onSelectRow={handleSelectRow}
              />
            )}

            {/* Bottom Row: Visualizations & Gemma Narrative */}
            {explainResult && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                <WaterfallPlot
                  baseValue={explainResult.base_value}
                  prediction={explainResult.prediction}
                  predictedClass={explainResult.predicted_class}
                  probabilities={explainResult.probabilities}
                  contributions={explainResult.feature_contributions}
                />
                <div className="flex flex-col h-full">
                  <ExplanationPanel explanation={explanationText} loading={loadingExplain} />
                  <ChatFollowUp
                    onSendMessage={handleFollowUpQuestion}
                    loading={loadingExplain}
                    disabled={!explanationText}
                  />
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}