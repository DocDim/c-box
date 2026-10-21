import React from 'react';
import { Bot, Sparkles } from 'lucide-react';

interface ExplanationPanelProps {
  explanation: string;
  loading: boolean;
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({ explanation, loading }) => {
  return (
    <div className="bg-slate-800/80 rounded-xl border border-slate-700 p-5 shadow-lg flex flex-col h-full">
      <div className="flex items-center justify-between pb-3 border-b border-slate-700 mb-3">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Bot size={18} className="text-indigo-400" />
          Gemma AI Narrative Explanation
        </h3>
        <span className="text-[11px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800 px-2 py-0.5 rounded">
          Localhost Ollama
        </span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-10 space-y-3">
            <Sparkles className="animate-spin text-indigo-400" size={24} />
            <p className="text-xs text-slate-400">Synthesizing grounded SHAP attribution narrative...</p>
          </div>
        ) : explanation ? (
          <div className="text-xs leading-relaxed text-slate-200 whitespace-pre-wrap font-sans">
            {explanation}
          </div>
        ) : (
          <p className="text-xs text-slate-500 italic py-6 text-center">
            Select a row above to compute SHAP values and generate an on-premise Gemma explanation.
          </p>
        )}
      </div>
    </div>
  );
};