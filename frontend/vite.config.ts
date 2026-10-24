import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The backend port is configurable so the UI can target a non-default API (e.g. during tests)
const apiTarget = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8001'

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.VITE_PORT ?? 3001),
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
