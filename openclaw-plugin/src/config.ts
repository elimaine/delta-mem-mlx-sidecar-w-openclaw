export type InstallStrategy = "path" | "npx" | "bundled";

export type ExampleHarnessConfig = {
  enabled?: boolean;
  runtimeId?: string;
  installStrategy?: InstallStrategy;
  binPath?: string;
  packageSpec?: string;
  baseUrl?: string;
  timeoutSeconds?: number;
  maxTurns?: number;
  mode?: "act" | "plan";
  json?: boolean;
  doubleCheckCompletion?: boolean;
};

export type EmbeddedHarnessPluginConfig = {
  harnesses?: {
    opencodeLocal?: ExampleHarnessConfig;
  };
};

export const DEFAULT_PLUGIN_CONFIG: Required<EmbeddedHarnessPluginConfig> = {
  harnesses: {
    opencodeLocal: {
      enabled: true,
      runtimeId: "opencode-local",
      installStrategy: "path",
      binPath: "opencode",
      baseUrl: "http://127.0.0.1:8765/v1",
    },
  },
};

export function resolvePluginConfig(input: unknown): Required<EmbeddedHarnessPluginConfig> {
  const raw = isRecord(input) ? input : {};
  const harnesses = isRecord(raw.harnesses) ? raw.harnesses : {};
  return {
    harnesses: {
      opencodeLocal: { ...DEFAULT_PLUGIN_CONFIG.harnesses.opencodeLocal, ...(isRecord(harnesses.opencodeLocal) ? harnesses.opencodeLocal : {}) },
    },
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
