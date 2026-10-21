import React from 'react';

export default function App() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-slate-900 text-slate-100 p-6">
      <div className="max-w-2xl text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight text-white">
          CrystalBox Platform
        </h1>
        <p className="text-slate-400">
        Predictive Model Explainability
        </p>
        <div className="p-4 rounded-xl bg-slate-800 border border-slate-700 text-emerald-400 font-mono text-sm">
          ✓ Frontend connected & TypeScript environment ready
        </div>
      </div>
    </div>
  );
}