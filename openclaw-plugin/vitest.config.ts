import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: [
      {
        find: "openclaw/plugin-sdk/agent-harness-runtime",
        replacement: new URL("./test/stubs/openclaw/plugin-sdk/agent-harness-runtime.ts", import.meta.url).pathname,
      },
      {
        find: "openclaw/plugin-sdk",
        replacement: new URL("./test/stubs/openclaw/plugin-sdk.ts", import.meta.url).pathname,
      },
      {
        find: "@mariozechner/pi-ai",
        replacement: new URL("./test/stubs/pi-ai.ts", import.meta.url).pathname,
      },
    ],
  },
});
