import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import type { ServerResponse } from 'http'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:9999',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        timeout: 120000,
        proxyTimeout: 120000,
        configure: (proxy) => {
          // When the backend worker dies mid-request, http-proxy emits an 'error' event
          // (e.g. "socket hang up", ECONNRESET, ECONNREFUSED). Vite's default handler
          // returns a plain-text 500. We piggyback a custom header BEFORE Vite writes
          // the status line so that the frontend can distinguish proxy-level failures
          // from real application 500s and trigger a safe retry.
          proxy.on('error', (err, _req, res) => {
            const isTransient =
              err.message?.includes('socket hang up') ||
              err.code === 'ECONNRESET' ||
              err.code === 'ECONNREFUSED' ||
              err.code === 'ETIMEDOUT'
            if (isTransient) {
              const serverRes = res as ServerResponse
              if (!serverRes.headersSent) {
                // setHeader() buffers the header; Vite's own handler calls writeHead()
                // immediately after, which flushes this header together with the 500 status.
                serverRes.setHeader('X-Proxy-Error', 'backend-unavailable')
              }
            }
          })
        },
      },
    },
  },
})
