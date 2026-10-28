import { useEffect, useState } from 'react';
import { Navbar } from './components/common/Navbar';
import { Sidebar } from './components/layout/Sidebar';
import { ChatSidebar } from './components/layout/ChatSidebar';
import { ModelsPage } from './pages/ModelsPage';
import { PredictPage } from './pages/PredictPage';
import { LoginPage } from './pages/LoginPage';
import { SummaryPlot } from './components/visualizations/SummaryPlot';
import { AuditPanel } from './components/audit/AuditPanel';
import { useHashRoute } from './hooks/useHashRoute';

import { authService, datasetService, errorMessage, explainService, modelService } from './services/api';
import { Dataset, InstanceExplainResponse, InstanceSource, ModelArtifact, User } from './types/api';
import { canExplain, canManage } from './utils/format';

const ACTIVE_MODEL_KEY = 'activeModelId';

const readStoredModelId = (): number | undefined => {
  try {
    const raw = localStorage.getItem(ACTIVE_MODEL_KEY);
    return raw ? Number(raw) : undefined;
  } catch {
    return undefined;
  }
};

export default function App() {
  const [page, navigate] = useHashRoute();
  const [user, setUser] = useState<User | null>(null);
  // Only wait on a session check when a token is stored; otherwise go straight to login
  const [checkingSession, setCheckingSession] = useState(() => !!localStorage.getItem('token'));
  const [models, setModels] = useState<ModelArtifact[]>([]);
  const [activeModel, setActiveModel] = useState<ModelArtifact | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsVersion, setDatasetsVersion] = useState(0);

  const [explanation, setExplanation] = useState<InstanceExplainResponse | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState('');
  const [auditRefresh, setAuditRefresh] = useState(0);

  const loadModels = async (preferId?: number) => {
    const list = await modelService.listModels();
    setModels(list);
    const wanted = preferId ?? readStoredModelId();
    setActiveModel(list.find((m) => m.id === wanted) ?? list[0] ?? null);
  };

  useEffect(() => {
    if (!localStorage.getItem('token')) return;
    authService.getMe()
      .then(async (me) => {
        setUser(me);
        await loadModels();
      })
      .catch(() => {
        authService.logout();
        setUser(null);
      })
      .finally(() => setCheckingSession(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Datasets of the active model drive the sidebar metadata and the batch page
  useEffect(() => {
    setExplanation(null);
    setExplainError('');
    if (!activeModel) {
      setDatasets([]);
      return;
    }
    try {
      localStorage.setItem(ACTIVE_MODEL_KEY, String(activeModel.id));
    } catch {
      // storage unavailable: the choice simply is not remembered
    }
    let cancelled = false;
    datasetService.listDatasets(activeModel.id)
      .then((list) => !cancelled && setDatasets(list))
      .catch(() => !cancelled && setDatasets([]));
    return () => {
      cancelled = true;
    };
  }, [activeModel?.id, activeModel?.background_dataset_id]);

  // Pages the active model's role cannot use fall back to the Models page
  useEffect(() => {
    if (!activeModel) return;
    if ((page === 'insights' && !canExplain(activeModel.assigned_role)) ||
        (page === 'audit' && !canManage(activeModel.assigned_role))) {
      navigate('models');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, activeModel]);

  const handleLogout = () => {
    authService.logout();
    setUser(null);
    setModels([]);
    setActiveModel(null);
    setDatasets([]);
    setExplanation(null);
  };

  const replaceModel = (updated: ModelArtifact) => {
    setModels((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
    setActiveModel((prev) => (prev && prev.id === updated.id ? updated : prev));
    setDatasetsVersion((v) => v + 1);
  };

  const handleExplain = async (source: InstanceSource) => {
    if (!activeModel || !canExplain(activeModel.assigned_role)) return;
    setExplaining(true);
    setExplainError('');
    try {
      setExplanation(await explainService.explainInstance(activeModel.id, source));
      setAuditRefresh((k) => k + 1);
    } catch (err) {
      setExplainError(errorMessage(err, 'SHAP explanation failed.'));
    } finally {
      setExplaining(false);
    }
  };

  const renderPage = () => {
    switch (page) {
      case 'predict':
        return (
          <PredictPage
            models={models}
            model={activeModel}
            datasets={datasets}
            onSelectModel={setActiveModel}
            onDatasetAdded={(d) => {
              setDatasets((prev) => [d, ...prev]);
              setDatasetsVersion((v) => v + 1);
            }}
            explanation={explanation}
            explaining={explaining}
            explainError={explainError}
            onExplain={handleExplain}
          />
        );
      case 'insights':
        return activeModel ? <SummaryPlot model={activeModel} datasets={datasets} /> : null;
      case 'audit':
        return activeModel ? <AuditPanel model={activeModel} refreshKey={auditRefresh} /> : null;
      default:
        return (
          <ModelsPage
            models={models}
            activeModel={activeModel}
            datasetsVersion={datasetsVersion}
            onModelRegistered={(m) => {
              setModels((prev) => [m, ...prev]);
              setActiveModel(m);
              setDatasetsVersion((v) => v + 1);
            }}
            onModelUpdated={replaceModel}
            onModelDeleted={() => loadModels()}
            onActivate={setActiveModel}
            onPredict={(m) => {
              setActiveModel(m);
              navigate('predict');
            }}
          />
        );
    }
  };

  if (checkingSession) {
    return <div className="min-h-screen bg-slate-900" />;
  }

  if (!user) {
    return <LoginPage onAuthSuccess={async (u) => { setUser(u); await loadModels(); }} />;
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      <Navbar user={user} onLogout={handleLogout} />

      <div className="flex-1 w-full px-4 py-5 grid grid-cols-1 lg:grid-cols-[16rem_minmax(0,1fr)] xl:grid-cols-[16rem_minmax(0,1fr)_26rem] gap-5 items-start">
        <div className="lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto">
          <Sidebar page={page} onNavigate={navigate} activeModel={activeModel} activeDatasets={datasets} />
        </div>

        <main className="min-w-0">{renderPage()}</main>

        <div className="lg:col-span-2 xl:col-span-1 xl:sticky xl:top-20 xl:max-h-[calc(100vh-6rem)] xl:overflow-y-auto">
          <ChatSidebar explanation={explanation} activeModel={activeModel} />
        </div>
      </div>
    </div>
  );
}
