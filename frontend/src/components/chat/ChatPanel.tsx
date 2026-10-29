import React, { useEffect, useRef, useState } from 'react';
import { Bot, Send, ShieldAlert, ShieldCheck, Sparkles, User as UserIcon } from 'lucide-react';
import { Audience, ChatMessage, Faithfulness } from '../../types/api';
import { errorMessage, explainService } from '../../services/api';
import { fmtMs } from '../../utils/format';
import { renderMarkdown } from '../../utils/markdown';

interface ChatPanelProps {
  explanationId: string;
  llmLabel?: string;
}

const AUDIENCES: { value: Audience; label: string; hint: string }[] = [
  { value: 'decision_maker', label: 'Decision-maker', hint: 'Plain language, business terms, one recommended action' },
  { value: 'domain_analyst', label: 'Analyst', hint: 'Values and effect sizes, with points to verify' },
  { value: 'data_scientist', label: 'Data scientist', hint: 'SHAP values, units, diagnostics and caveats' },
];
const AUDIENCE_KEY = 'chatAudience';

const readAudience = (): Audience => {
  try {
    const saved = localStorage.getItem(AUDIENCE_KEY);
    if (AUDIENCES.some((a) => a.value === saved)) return saved as Audience;
  } catch {
    // storage unavailable: use the default reader
  }
  return 'decision_maker';
};

const FaithfulnessBadge: React.FC<{ f: Faithfulness; kind: string }> = ({ f, kind }) => {
  const ok = !f.fabricated && !f.direction_errors;
  return (
    <div className={`mt-2 text-[10px] rounded-md px-2 py-1 border flex flex-wrap items-center gap-x-3 gap-y-1 ${
      ok ? 'border-secondary-800 bg-secondary-950/40 text-secondary-300' : 'border-accent-700 bg-accent-950/40 text-accent-300'
    }`}>
      <span className="flex items-center gap-1 font-semibold">
        {ok ? <ShieldCheck size={12} /> : <ShieldAlert size={12} />}
        {f.fabricated
          ? `Unverified numbers: ${f.unverified_numbers.join(', ')}`
          : f.direction_errors ? 'Numbers grounded, but factor directions are wrong' : 'All numbers and directions grounded in SHAP context'}
      </span>
      {f.misplaced_features && f.misplaced_features.length > 0 && (
        <span className="basis-full">
          Wrong section: {f.misplaced_features.map((m) => `${m.feature} (actually ${m.actual_effect})`).join('; ')}
        </span>
      )}
      <span>{f.numbers_verified}/{f.numbers_checked} numbers verified</span>
      <span>{f.features_mentioned.length}/{f.features_in_context} key features cited</span>
      {kind === 'narrative' && f.sections_found && (
        <span>{f.sections_found.length}/4 sections{f.structure_complete ? '' : ' (incomplete)'}</span>
      )}
    </div>
  );
};

export const ChatPanel: React.FC<ChatPanelProps> = ({ explanationId, llmLabel = 'Gemma via Ollama' }) => {
  const [audience, setAudience] = useState<Audience>(readAudience);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState<string | null>(null);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const run = async (question?: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setError('');
    setStreamingText('');
    setPendingQuestion(question ?? null);
    try {
      const final = await explainService.chatStream(
        explanationId, audience, question, (t) => setStreamingText((prev) => (prev ?? '') + t), controller.signal
      );
      setMessages((prev) => {
        const next = [...prev];
        if (question) {
          next.push({
            id: -Date.now(), role: 'user', kind: 'followup', audience, content: question, created_at: new Date().toISOString(),
          });
        }
        if (!next.some((m) => m.id === final.id)) next.push(final);
        return next;
      });
    } catch (err: any) {
      if (err?.name !== 'AbortError') setError(errorMessage(err, 'LLM request failed.'));
    } finally {
      if (abortRef.current === controller) {
        setStreamingText(null);
        setPendingQuestion(null);
      }
    }
  };

  useEffect(() => {
    let cancelled = false;
    setMessages([]);
    setError('');
    explainService
      .getConversation(explanationId, audience)
      .then((conv) => {
        if (cancelled) return;
        setMessages(conv.messages);
        if (!conv.messages.some((m) => m.kind === 'narrative')) run();
      })
      .catch((err) => !cancelled && setError(errorMessage(err)));
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [explanationId, audience]);

  const chooseAudience = (next: Audience) => {
    setAudience(next);
    try {
      localStorage.setItem(AUDIENCE_KEY, next);
    } catch {
      // the choice just won't be remembered
    }
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'nearest' });
  }, [messages, streamingText]);

  const busy = streamingText !== null;
  const hasNarrative = messages.some((m) => m.kind === 'narrative');

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q || busy) return;
    setQuery('');
    run(q);
  };

  return (
    <div className="bg-slate-800/80 rounded-xl border border-slate-700 p-5 shadow-lg flex flex-col h-full min-h-[28rem]">
      <div className="pb-3 border-b border-slate-700 mb-3 space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-50 flex items-center gap-2">
            <Bot size={18} className="text-primary-400" />
            Grounded AI Narrative
          </h3>
          <span className="text-[11px] font-mono bg-primary-950 text-primary-300 border border-primary-800 px-2 py-0.5 rounded">
            {llmLabel}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px]">
          <span className="text-slate-400">Written for</span>
          <div className="flex bg-slate-900 border border-slate-700 rounded-lg p-0.5">
            {AUDIENCES.map((a) => (
              <button
                key={a.value}
                type="button"
                title={a.hint}
                onClick={() => chooseAudience(a.value)}
                className={`px-2 py-1 rounded-md transition ${
                  audience === a.value ? 'bg-primary-600 text-white' : 'text-slate-400 hover:text-slate-50'
                }`}
              >
                {a.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-4 pr-1 max-h-[36rem]">
        {messages.map((m) => (
          <div key={m.id} className={m.role === 'user' ? 'flex justify-end' : ''}>
            {m.role === 'user' ? (
              <div className="max-w-[85%] bg-primary-600/30 border border-primary-700 rounded-lg px-3 py-2 text-xs text-primary-100 flex gap-2">
                <UserIcon size={14} className="shrink-0 mt-0.5" /> {m.content}
              </div>
            ) : (
              <div className="text-xs leading-relaxed text-slate-200">
                {renderMarkdown(m.content)}
                {m.faithfulness && <FaithfulnessBadge f={m.faithfulness} kind={m.kind} />}
                <div className="text-[10px] text-slate-500 mt-1">
                  {m.llm_model} · first token {fmtMs(m.ttft_ms)} · total {fmtMs(m.latency_ms)}
                </div>
              </div>
            )}
          </div>
        ))}

        {pendingQuestion && (
          <div className="flex justify-end">
            <div className="max-w-[85%] bg-primary-600/30 border border-primary-700 rounded-lg px-3 py-2 text-xs text-primary-100">
              {pendingQuestion}
            </div>
          </div>
        )}

        {busy && (
          <div className="text-xs leading-relaxed text-slate-200">
            {streamingText ? renderMarkdown(streamingText) : (
              <div className="flex items-center gap-2 text-slate-400 py-4">
                <Sparkles className="animate-spin text-primary-400" size={16} />
                {hasNarrative ? 'Answering…' : 'Synthesizing grounded SHAP narrative…'}
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="text-xs text-rose-400 bg-rose-950/40 p-2.5 rounded-lg border border-rose-800 space-y-2">
            <p>{error}</p>
            {!hasNarrative && (
              <button onClick={() => run()} className="underline text-rose-300 hover:text-slate-50">Retry</button>
            )}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSend} className="flex items-center gap-2 mt-3">
        <input
          type="text"
          maxLength={1000}
          disabled={!hasNarrative || busy}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={hasNarrative ? 'e.g., Which feature change would flip this decision?' : 'Waiting for the narrative…'}
          className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-50 focus:outline-none focus:border-primary-500 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!hasNarrative || busy || !query.trim()}
          className="bg-primary-600 hover:bg-primary-500 disabled:bg-gray-500 text-white px-3 py-2 rounded-lg text-xs transition flex items-center gap-1 shadow"
        >
          <Send size={14} /> Send
        </button>
      </form>
    </div>
  );
};
