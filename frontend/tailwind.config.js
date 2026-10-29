/** @type {import('tailwindcss').Config} */

// Each shade reads an RGB CSS variable (defined in src/index.css) so the
// whole palette flips when <html> gets the `light` class.
const SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];
const themed = (name) =>
  Object.fromEntries(SHADES.map((s) => [s, `rgb(var(--${name}-${s}) / <alpha-value>)`]));

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Brand palette — 500 is the exact brand color.
        primary: { DEFAULT: '#009ddd', ...themed('primary') },
        secondary: { DEFAULT: '#e48d12', ...themed('secondary') },
        accent: { DEFAULT: '#00927e', ...themed('accent') },
        slate: themed('slate'),
        rose: themed('rose'),
      },
    },
  },
  plugins: [],
}
