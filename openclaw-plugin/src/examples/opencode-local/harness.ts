import fs from "node:fs/promises";
import path from "node:path";
import type { AgentHarness } from "openclaw/plugin-sdk/agent-harness-runtime";
import { mapOpenClawModelToOpenCodeLocal } from "../../core/model-mapping.js";
import { runProcessHarnessAttempt } from "../../core/process-harness.js";

const DEFAULT_PROVIDER_IDS = new Set(["delta-mem-mlx"]);
const DEFAULT_BIN = "opencode";

export type OpenCodeLocalAgentHarnessOptions = {
  id?: string;
  label?: string;
  providerIds?: Iterable<string>;
  bin?: string;
  pure?: boolean;
  format?: "default" | "json";
  ensureDefaultConfig?: boolean;
  baseUrl?: string;
};

export function createOpenCodeLocalAgentHarness(options?: OpenCodeLocalAgentHarnessOptions): AgentHarness {
  const id = options?.id ?? "opencode-local";
  const providerIds = new Set([...(options?.providerIds ?? DEFAULT_PROVIDER_IDS)].map((x) => x.trim().toLowerCase()));
  const bin = options?.bin ?? DEFAULT_BIN;
  const pure = options?.pure ?? true;
  const format = options?.format ?? "default";
  const ensureDefaultConfig = options?.ensureDefaultConfig ?? true;
  const baseUrl = options?.baseUrl ?? "http://127.0.0.1:8765/v1";
  return {
    id,
    label: options?.label ?? "OpenCode local embedded harness",
    supports(ctx) {
      if (ctx.requestedRuntime && ctx.requestedRuntime !== id) return { supported: false, reason: "different forced runtime" };
      const provider = ctx.provider.trim().toLowerCase();
      if (providerIds.has(provider)) return { supported: true, priority: 90 };
      return { supported: false, reason: `provider is not one of: ${[...providerIds].toSorted().join(", ")}` };
    },
    async runAttempt(params) {
      const model = mapOpenClawModelToOpenCodeLocal(params);
      if (ensureDefaultConfig) await ensureOpencodeConfig(params.workspaceDir, baseUrl, model);
      const args = ["run"];
      if (pure) args.push("--pure");
      args.push("--model", model, "--dir", params.workspaceDir, "--format", format, params.prompt);
      return runProcessHarnessAttempt(params, { bin, args, hostCwd: params.workspaceDir, sandboxCwd: "/workspace" }, id);
    },
  };
}

async function ensureOpencodeConfig(workspaceDir: string, baseUrl: string, modelRef: string): Promise<void> {
  const configPath = path.join(workspaceDir, "opencode.json");
  try { await fs.access(configPath); return; } catch {}
  const modelId = modelRef.startsWith("delta-mem-mlx/") ? modelRef.slice("delta-mem-mlx/".length) : modelRef;
  const config = {
    $schema: "https://opencode.ai/config.json",
    provider: {
      "delta-mem-mlx": {
        npm: "@ai-sdk/openai-compatible",
        name: "Delta-Mem MLX Sidecar",
        options: { baseURL: baseUrl },
        models: { [modelId]: { name: modelId, tool_call: true, tools: true, reasoning: false, limit: { context: 128000, output: 4096 } } },
      },
    },
  };
  await fs.writeFile(configPath, `${JSON.stringify(config, null, 2)}\n`, "utf8");
}
