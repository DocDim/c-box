import React, { useEffect, useState } from 'react';
import { Share2, UserMinus } from 'lucide-react';
import { ModelArtifact, Role, SharedUser } from '../../types/api';
import { errorMessage, modelService } from '../../services/api';

interface SharingPanelProps {
  model: ModelArtifact;
}

type GrantRole = Exclude<Role, 'OWNER'>;

export const SharingPanel: React.FC<SharingPanelProps> = ({ model }) => {
  const [shares, setShares] = useState<SharedUser[]>([]);
  const [email, setEmail] = useState('');
  const [role, setRole] = useState<GrantRole>('EXPLAINER');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const isOwner = model.is_owner;

  const refresh = async () => {
    try {
      setShares(await modelService.listShares(model.id));
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  useEffect(() => {
    setError('');
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model.id]);

  const handleGrant = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      await modelService.grantShare(model.id, email.trim(), role);
      setEmail('');
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const handleRevoke = async (share: SharedUser) => {
    setError('');
    try {
      await modelService.revokeShare(model.id, share.user_id);
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const roles: GrantRole[] = isOwner ? ['VIEWER', 'EXPLAINER', 'ADMIN'] : ['VIEWER', 'EXPLAINER'];

  return (
    <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-md space-y-3">
      <h3 className="text-sm font-semibold text-slate-50 flex items-center gap-2">
        <Share2 size={18} className="text-accent-400" /> Access Sharing
      </h3>
      <p className="text-[11px] text-slate-400">
        VIEWER: predictions only · EXPLAINER: SHAP + LLM · ADMIN: also manage sharing &amp; audit
      </p>
      {error && <div className="text-xs text-rose-400 bg-rose-950/40 p-2 rounded-lg border border-rose-800">{error}</div>}

      <form onSubmit={handleGrant} className="flex flex-wrap gap-2">
        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="colleague@domain.com"
          className="flex-1 min-w-[10rem] bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-50 focus:outline-none focus:border-accent-500"
        />
        <select
          value={role}
          onChange={(e) => setRole(e.target.value as GrantRole)}
          className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-50"
        >
          {roles.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
        <button
          type="submit"
          disabled={busy}
          className="bg-accent-600 hover:bg-accent-500 disabled:bg-gray-500 text-white text-xs font-medium px-3 py-1.5 rounded-lg"
        >
          Grant / Update
        </button>
      </form>

      {shares.length === 0 ? (
        <p className="text-xs text-slate-500 italic">Not shared with anyone yet.</p>
      ) : (
        <ul className="divide-y divide-slate-700/60 text-xs">
          {shares.map((s) => (
            <li key={s.user_id} className="flex items-center justify-between py-1.5">
              <span className="text-slate-200 truncate">{s.email}</span>
              <span className="flex items-center gap-2">
                <span className="font-mono text-[11px] text-accent-300">{s.role}</span>
                {(isOwner || s.role !== 'ADMIN') && (
                  <button
                    onClick={() => handleRevoke(s)}
                    title="Revoke access"
                    className="text-slate-400 hover:text-rose-400"
                  >
                    <UserMinus size={14} />
                  </button>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
