import React from 'react';
import { MessageSquare } from 'lucide-react';
import { InstanceExplainResponse, ModelArtifact } from '../../types/api';
import { ChatPanel } from '../chat/ChatPanel';
import { canExplain } from '../../utils/format';

interface ChatSidebarProps {
  explanation: InstanceExplainResponse | null;
  activeModel: ModelArtifact | null;
}

export const ChatSidebar: React.FC<ChatSidebarProps> = ({ explanation, activeModel }) => {
  if (!explanation) {
    return (
      <div className="bg-slate-800/80 rounded-xl border border-slate-700 p-6 text-center space-y-3">
        <MessageSquare className="mx-auto text-indigo-400" size={28} />
        <h3 className="text-sm font-semibold text-white">Grounded AI Narrative</h3>
        <p className="text-xs text-slate-400">
          {canExplain(activeModel?.assigned_role)
            ? 'Explain a prediction on the Predict page. The narrative and follow-up chat appear here.'
            : 'Explanations require EXPLAINER access to the active model.'}
        </p>
      </div>
    );
  }

  const source =
    explanation.row_index !== null && explanation.row_index !== undefined
      ? `dataset row #${explanation.row_index}`
      : 'manual input';

  return (
    <div className="space-y-2">
      <p className="text-[11px] text-slate-400 px-1">
        Discussing <span className="text-slate-200">{source}</span>
        {explanation.predicted_class !== null && explanation.predicted_class !== undefined
          ? <> · predicted class <span className="text-slate-200">{explanation.predicted_class}</span></>
          : null}
      </p>
      <ChatPanel key={explanation.explanation_id} explanationId={explanation.explanation_id} />
    </div>
  );
};
