import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Static, client-only build. base: "./" so the bundle works from any subfolder.
// Output lands at ../demo (landing/demo/).
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../demo",
    emptyOutDir: true,
  },
});
