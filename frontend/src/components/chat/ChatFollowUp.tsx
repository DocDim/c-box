import React, { useState } from 'react';
import { Send } from 'lucide-react';

interface ChatFollowUpProps {
  onSendMessage: (query: string) => void;
  loading: boolean;
  disabled: boolean;
}

export const ChatFollowUp: React.FC<ChatFollowUpProps> = ({ onSendMessage, loading, disabled }) => {
  const [query, setQuery] = useState('');

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading || disabled) return;
    onSendMessage(query.trim());
    setQuery('');
  };

  return (
    <form onSubmit={handleSend} className="flex items-center gap-2 mt-3">
      <input
        type="text"
        disabled={disabled || loading}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={
          disabled
            ? "Generate an explanation first to ask follow-up questions..."
            : "e.g., Which feature change would flip this decision?"
        }
        className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500 disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={disabled || loading || !query.trim()}
        className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 text-white px-3 py-2 rounded-lg text-xs transition flex items-center gap-1 shadow"
      >
        <Send size={14} /> Send
      </button>
    </form>
  );
};