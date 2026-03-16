import { defineConfig } from "vite";
import { resolve } from "path";

function resolveBase(): string {
  const explicitBase = process.env.VITE_BASE_PATH;
  if (explicitBase) {
    return explicitBase.endsWith("/") ? explicitBase : `${explicitBase}/`;
  }

  if (process.env.GITHUB_ACTIONS === "true") {
    const repo = process.env.GITHUB_REPOSITORY?.split("/")[1];
    if (repo) {
      return repo.endsWith(".github.io") ? "/" : `/${repo}/`;
    }
  }

  return "/";
}

export default defineConfig({
  base: resolveBase(),
  root: ".",
  publicDir: "public",
  build: {
    outDir: "dist",
    target: "es2022",
  },
  resolve: {
    alias: {
      "attnres-wasm": resolve(__dirname, "crate/pkg"),
    },
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
