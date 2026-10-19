import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `python -m aether studio` serves the API on :8765; `pnpm dev` proxies to it.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8765', ws: true },
      '/files': { target: 'http://127.0.0.1:8765' },
    },
  },
})
