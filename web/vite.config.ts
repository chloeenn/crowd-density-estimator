/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev, /api is proxied to the FastAPI server so no CORS setup is needed locally.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/api": "http://localhost:8000" } },
  test: { environment: "jsdom", setupFiles: ["./src/test-setup.ts"], globals: true },
});
