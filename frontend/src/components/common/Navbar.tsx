import React from 'react';
import { User } from '../../types/api';
import { LogOut, User as UserIcon } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';

interface NavbarProps {
  user: User;
  onLogout: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ user, onLogout }) => (
  <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-30">
    <div className="px-6 h-16 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <img src="/logo.png" alt="cbox" className="h-12 w-auto" />
      </div>

      <div className="flex items-center space-x-4">
        <span className="text-xs text-slate-300 bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-full flex items-center gap-2">
          <UserIcon size={14} className="text-primary-400" />
          {user.email}
        </span>
        <ThemeToggle />
        <button
          onClick={onLogout}
          className="text-xs bg-primary-600 hover:bg-primary-500 text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition"
        >
          <LogOut size={14} /> Logout
        </button>
      </div>
    </div>
  </header>
);
