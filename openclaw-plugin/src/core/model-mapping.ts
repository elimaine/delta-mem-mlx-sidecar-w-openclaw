import type { AgentHarnessAttemptParams } from "openclaw/plugin-sdk/agent-harness-runtime";

export type HarnessModelSelection = {
  provider: string;
  modelId: string;
};

export function requireRunModel(params: AgentHarnessAttemptParams): HarnessModelSelection {
  const provider = params.provider?.trim();
  const modelId = params.modelId?.trim();
  if (!provider || !modelId) {
    throw new Error("embedded harness requires OpenClaw to pass params.provider and params.modelId");
  }
  return { provider, modelId };
}

export function mapOpenClawModelToOpenCodeLocal(params: AgentHarnessAttemptParams): string {
  const { provider, modelId } = requireRunModel(params);
  if (provider === "delta-mem-mlx") return `delta-mem-mlx/${modelId}`;
  throw new Error(`OpenCode local example only maps delta-mem-mlx models; got ${provider}/${modelId}`);
}

export function mapOpenClawModelToCliProviderArgs(params: AgentHarnessAttemptParams): string[] {
  const { provider, modelId } = requireRunModel(params);
  // Direct CLI tools often have their own provider names. Keep this mapping explicit.
  if (provider === "delta-mem-mlx") return ["--provider", "openai-compatible", "--model", modelId];
  return ["--provider", provider, "--model", modelId];
}
