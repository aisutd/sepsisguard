import react from '@vitejs/plugin-react' //brings react plugin
import { defineConfig } from 'vite' //helper that wraps settings

// https://vite.dev/config/
export default defineConfig({ //hands these settings to vite
  plugins: [react()], //activates plugin

  server: {

    proxy: {
      "/health": "http://localhost:8000" //forwards requests to this server
    }
  }
})
