import type { AssistantMessage, Usage } from "@mariozechner/pi-ai";

export type AgentHarness = {
  id: string;
  label: string;
  supports(ctx: { provider: string; requestedRuntime?: string }): { supported: boolean; priority?: number; reason?: string };
  runAttempt(params: AgentHarnessAttemptParams): Promise<AgentHarnessAttemptResult>;
};

export type AgentHarnessAttemptParams = {
  provider?: string;
  modelId?: string;
  workspaceDir: string;
  prompt: string;
  runId: string;
  sessionId: string;
  sessionKey?: string;
  sandboxSessionKey?: string;
  timeoutMs: number;
  abortSignal?: AbortSignal;
  config?: unknown;
};

export type AgentHarnessAttemptResult = {
  timedOut: boolean;
  timeoutMs?: number;
  aborted: boolean;
  promptError?: string;
  promptErrorType?: "error";
  sessionIdUsed: string;
  messagesSnapshot: AssistantMessage[];
  assistantTexts: string[];
  toolMetas: unknown[];
  lastAssistant?: AssistantMessage;
  usage: Usage;
  startedAtMs: number;
  completedAtMs: number;
  replayMetadata: { hadPotentialSideEffects: boolean; replaySafe: boolean };
  itemLifecycle: unknown[];
};

export type SandboxContext = {
  enabled: boolean;
  containerWorkdir: string;
  backend?: {
    runShellCommand(args: {
      script: string;
      signal?: AbortSignal;
      allowFailure: boolean;
    }): Promise<{ stdout: Buffer; stderr: Buffer; code: number }>;
  };
};

export function formatErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function resolveSandboxContext(_args?: unknown): Promise<SandboxContext | undefined> {
  return undefined;
}
