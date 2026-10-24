import React, { useState } from 'react';
import { User } from '../../types/api';
import { authService } from '../../services/api';
import { LogIn, LogOut, User as UserIcon } from 'lucide-react';

interface NavbarProps {
  user: User | null;
  onAuthSuccess: (user: User) => void;
  onLogout: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ user, onAuthSuccess, onLogout }) => {
  const [showModal, setShowModal] = useState(false);
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      if (isRegister) {
        await authService.register({ email, password });
      }
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);
      await authService.login(formData);
      const currentUser = await authService.getMe();
      onAuthSuccess(currentUser);
      setShowModal(false);
      setEmail('');
      setPassword('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Authentication failed. Please check credentials.');
    }
  };

  return (
    <>
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-30">
        <div className="px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-md">
              CB
            </div>
            <div>
              <span className="font-bold text-lg text-white tracking-wide">CrystalBox</span>
              <span className="text-xs text-indigo-400 block -mt-1 font-mono">SHAP + Gemma XAI</span>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {user ? (
              <div className="flex items-center space-x-4">
                <span className="text-xs text-slate-300 bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-full flex items-center gap-2">
                  <UserIcon size={14} className="text-indigo-400" />
                  {user.email}
                </span>
                <button
                  onClick={onLogout}
                  className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5 border border-slate-700 transition"
                >
                  <LogOut size={14} /> Logout
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowModal(true)}
                className="text-xs bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-4 py-2 rounded-lg flex items-center gap-1.5 shadow-md transition"
              >
                <LogIn size={14} /> Sign In / Register
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Auth Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4">
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 w-full max-w-sm shadow-xl relative">
            <h2 className="text-lg font-bold text-white mb-2">
              {isRegister ? 'Create an Account' : 'Sign In to CrystalBox'}
            </h2>
            <p className="text-xs text-slate-400 mb-4">
              {isRegister
                ? 'Register to isolate model uploads and manage permissions.'
                : 'Authenticate using your registered credentials.'}
            </p>

            {error && (
              <div className="mb-4 text-xs text-rose-400 bg-rose-950/40 border border-rose-800 p-2.5 rounded-lg">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Email</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500"
                  placeholder="analyst@domain.com"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Password</label>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500"
                  placeholder="••••••••"
                />
              </div>
              <button
                type="submit"
                className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-2 rounded-lg text-sm transition mt-2 shadow"
              >
                {isRegister ? 'Register' : 'Sign In'}
              </button>
            </form>

            <div className="mt-4 text-center">
              <button
                type="button"
                onClick={() => { setIsRegister(!isRegister); setError(''); }}
                className="text-xs text-indigo-400 hover:underline"
              >
                {isRegister ? 'Already have an account? Sign In' : 'Need an account? Register here'}
              </button>
            </div>

            <button
              onClick={() => setShowModal(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white text-sm"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </>
  );
};