import { useSyncExternalStore } from 'react';

export type Theme = 'light' | 'dark';
const STORAGE_KEY = 'theme';

const initial = (): Theme => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    // storage unavailable: fall back to the OS preference
  }
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
};

const apply = (theme: Theme) => {
  document.documentElement.classList.toggle('light', theme === 'light');
  document.documentElement.style.colorScheme = theme;
};

let current = initial();
apply(current);
const listeners = new Set<() => void>();

export const setTheme = (theme: Theme) => {
  current = theme;
  apply(theme);
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // preference just won't persist
  }
  listeners.forEach((l) => l());
};

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

/** Global light/dark theme: toggles the `light` class on <html>, which swaps the palette CSS variables. */
export const useTheme = (): [Theme, () => void] => {
  const theme = useSyncExternalStore(subscribe, () => current);
  return [theme, () => setTheme(theme === 'dark' ? 'light' : 'dark')];
};

/** Plotly can't read CSS variables, so charts take their neutral colors from here. */
export const CHART_COLORS = {
  dark: {
    grid: '#334155', gridFaint: '#1e293b', zero: '#64748b', connector: '#475569',
    tick: '#94a3b8', label: '#cbd5e1', strong: '#e2e8f0', outline: '#0f172a',
  },
  light: {
    grid: '#e2e8f0', gridFaint: '#f1f5f9', zero: '#94a3b8', connector: '#94a3b8',
    tick: '#64748b', label: '#334155', strong: '#1e293b', outline: '#ffffff',
  },
} as const;

export const useChartColors = () => CHART_COLORS[useTheme()[0]];
