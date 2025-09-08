import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // You can add more configuration options here as needed, for example:
  // server: { port: 3000 },  // to change the default dev server port
  // build: { outDir: 'dist' },  // specify build output directory
})
