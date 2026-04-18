import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ws": { target: "http://localhost:8001", ws: true },
      "/api": { target: "http://localhost:8001" },
      "/health": { target: "http://localhost:8001" },
    },
  },
});
