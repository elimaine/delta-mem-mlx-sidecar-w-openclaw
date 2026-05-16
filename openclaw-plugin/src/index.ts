import { definePluginEntry } from "openclaw/plugin-sdk";
import { resolvePluginConfig } from "./config.js";
import { createOpenCodeLocalAgentHarness } from "./examples/opencode-local/harness.js";

export default definePluginEntry({
  id: "embedded-harness-examples",
  register(api) {
    const config = resolvePluginConfig((api as { pluginConfig?: unknown }).pluginConfig);
    // Config is the source of truth. Register only harnesses explicitly enabled by config/defaults.
    const opencode = config.harnesses.opencodeLocal;
    if (opencode?.enabled) {
      api.registerAgentHarness(createOpenCodeLocalAgentHarness({
        id: opencode.runtimeId,
        bin: opencode.binPath,
        baseUrl: opencode.baseUrl,
      }));
    }
  },
});

export { createOpenCodeLocalAgentHarness } from "./examples/opencode-local/harness.js";

export { preflightHarness } from "./preflight.js";
export { resolvePluginConfig } from "./config.js";
export type { EmbeddedHarnessPluginConfig, ExampleHarnessConfig, InstallStrategy } from "./config.js";
