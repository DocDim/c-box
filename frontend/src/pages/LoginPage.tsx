import React, { useState } from 'react';
import { User } from '../types/api';
import { authService } from '../services/api';
import { LogIn, UserPlus } from 'lucide-react';
import { ThemeToggle } from '../components/common/ThemeToggle';

interface LoginPageProps {
  onAuthSuccess: (user: User) => void;
}

export const LoginPage: React.FC<LoginPageProps> = ({ onAuthSuccess }) => {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      if (isRegister) {
        await authService.register({ email, password });
      }
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);
      await authService.login(formData);
      onAuthSuccess(await authService.getMe());
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Authentication failed. Please check credentials.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center px-4 py-12 font-sans">
      <ThemeToggle className="absolute top-4 right-4" />
      <div className="w-full max-w-sm">
        <div className="flex flex-col items-center mb-8">
          <img src="/logo.png" alt="cbox" className="h-40 w-auto" />
        </div>

        <div className="bg-slate-800/80 border border-slate-700 rounded-2xl p-6 shadow-xl">
          <h1 className="text-lg font-bold text-slate-50 mb-1">
            {isRegister ? 'Create an Account' : 'Sign In to CrystalBox'}
          </h1>
          <p className="text-xs text-slate-400 mb-5">
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
                autoFocus
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-50 focus:outline-none focus:border-primary-500"
                placeholder="analyst@domain.com"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Password</label>
              <input
                type="password"
                required
                autoComplete={isRegister ? 'new-password' : 'current-password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-50 focus:outline-none focus:border-primary-500"
                placeholder="••••••••"
              />
            </div>
            <button
              type="submit"
              disabled={submitting}
              className="w-full bg-primary-600 hover:bg-primary-500 disabled:opacity-60 text-white font-semibold py-2 rounded-lg text-sm transition mt-2 shadow flex items-center justify-center gap-1.5"
            >
              {isRegister ? <UserPlus size={14} /> : <LogIn size={14} />}
              {submitting ? 'Please wait…' : isRegister ? 'Register' : 'Sign In'}
            </button>
          </form>

          <div className="mt-4 text-center">
            <button
              type="button"
              onClick={() => { setIsRegister(!isRegister); setError(''); }}
              className="text-xs text-primary-400 hover:underline"
            >
              {isRegister ? 'Already have an account? Sign In' : 'Need an account? Register here'}
            </button>
          </div>
        </div>

        <p className="text-xs text-slate-500 text-center mt-6">
          Connect your pre-trained models with on-premise SHAP and Ollama Gemma interpretations.
        </p>
      </div>
    </div>
  );
};
