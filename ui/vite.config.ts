import path from "node:path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Vite config for the orgonzola UI. The UI is the view layer only; the Tauri shell serves it from
// `devUrl` (dev) or `frontendDist` (build). strictPort so the shell's devUrl stays correct.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    rollupOptions: {
      output: {
        // Split heavy vendor libraries (chat panel, charts, markdown) into their own chunks to stay
        // under the chunk-size warning.
        manualChunks(id) {
          if (!id.includes("node_modules")) return
          if (id.includes("@assistant-ui")) return "assistant"
          if (id.includes("recharts") || id.includes("/d3-") || id.includes("victory-vendor"))
            return "charts"
          if (
            id.includes("react-markdown") ||
            id.includes("remark") ||
            id.includes("rehype") ||
            id.includes("micromark") ||
            id.includes("mdast") ||
            id.includes("hast") ||
            id.includes("unified") ||
            id.includes("/vfile") ||
            id.includes("property-information") ||
            id.includes("decode-named-character-reference")
          )
            return "markdown"
          if (id.includes("@tanstack")) return "table"
          // React stays in the main chunk: every split chunk depends on it, so its own chunk would
          // create a cycle.
        },
      },
    },
  },
})
