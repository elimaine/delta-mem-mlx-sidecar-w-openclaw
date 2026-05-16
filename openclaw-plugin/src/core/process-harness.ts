import { spawn } from "node:child_process";
import type { AssistantMessage, Usage } from "@mariozechner/pi-ai";
import type {
  AgentHarnessAttemptParams,
  AgentHarnessAttemptResult,
} from "openclaw/plugin-sdk/agent-harness-runtime";
import {
  formatErrorMessage,
  resolveSandboxContext,
} from "openclaw/plugin-sdk/agent-harness-runtime";

const ZERO_USAGE: Usage = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

export type ProcessHarnessCommand = {
  bin: string;
  args: string[];
  hostCwd: string;
  sandboxCwd?: string;
};

export async function runProcessHarnessAttempt(
  params: AgentHarnessAttemptParams,
  command: ProcessHarnessCommand,
  assistantIdPrefix: string,
): Promise<AgentHarnessAttemptResult> {
  const start = Date.now();
  const sandbox = await resolveAttemptSandbox(params);
  const actual = sandbox?.enabled
    ? { ...command, args: command.args.map((arg) => arg === command.hostCwd ? (command.sandboxCwd ?? sandbox.containerWorkdir) : arg) }
    : command;
  const { stdout, stderr, exitCode, signal, timedOut, externalAbort } = await runProcess(params, actual, sandbox);
  const assistantText = stdout.trim();
  const promptError = exitCode === 0 && !signal ? undefined : buildPromptError({ exitCode, signal, stderr, stdout });
  const lastAssistant = assistantText ? buildAssistantMessage(assistantText, params, assistantIdPrefix) : undefined;
  return {
    timedOut,
    timeoutMs: timedOut ? params.timeoutMs : undefined,
    aborted: externalAbort,
    promptError,
    promptErrorType: promptError ? "error" : undefined,
    sessionIdUsed: params.sessionId,
    messagesSnapshot: lastAssistant ? [lastAssistant] : [],
    assistantTexts: assistantText ? [assistantText] : [],
    toolMetas: [],
    lastAssistant,
    usage: ZERO_USAGE,
    startedAtMs: start,
    completedAtMs: Date.now(),
    replayMetadata: { hadPotentialSideEffects: true, replaySafe: false },
    itemLifecycle: [],
  };
}

type ResolvedAttemptSandbox = Awaited<ReturnType<typeof resolveSandboxContext>>;

async function resolveAttemptSandbox(params: AgentHarnessAttemptParams): Promise<ResolvedAttemptSandbox> {
  const sandboxSessionKey = params.sandboxSessionKey?.trim() || params.sessionKey?.trim() || params.sessionId;
  return resolveSandboxContext({ config: params.config, sessionKey: sandboxSessionKey, workspaceDir: params.workspaceDir });
}

async function runProcess(
  params: AgentHarnessAttemptParams,
  command: ProcessHarnessCommand,
  sandbox: ResolvedAttemptSandbox,
): Promise<{ stdout: string; stderr: string; exitCode: number | null; signal: NodeJS.Signals | null; timedOut: boolean; externalAbort: boolean }> {
  if (sandbox?.enabled) {
    const script = buildSandboxShellScript([command.bin, ...command.args]);
    const result = await sandbox.backend?.runShellCommand({ script, signal: params.abortSignal, allowFailure: true });
    if (!result) throw new Error("selected sandbox does not expose command execution");
    return { stdout: result.stdout.toString("utf8"), stderr: result.stderr.toString("utf8"), exitCode: result.code, signal: null, timedOut: false, externalAbort: Boolean(params.abortSignal?.aborted) };
  }
  return runHostProcess(params, command);
}

async function runHostProcess(
  params: AgentHarnessAttemptParams,
  command: ProcessHarnessCommand,
): Promise<{ stdout: string; stderr: string; exitCode: number | null; signal: NodeJS.Signals | null; timedOut: boolean; externalAbort: boolean }> {
  let stdout = "", stderr = "";
  let exitCode: number | null = null, signal: NodeJS.Signals | null = null;
  let timedOut = false, externalAbort = false;
  await new Promise<void>((resolve) => {
    const child = spawn(command.bin, command.args, { cwd: command.hostCwd, env: process.env, stdio: ["ignore", "pipe", "pipe"] });
    const timeout = setTimeout(() => { timedOut = true; child.kill("SIGTERM"); }, Math.max(1, params.timeoutMs));
    const abort = () => { externalAbort = true; child.kill("SIGTERM"); };
    params.abortSignal?.addEventListener("abort", abort, { once: true });
    child.stdout.setEncoding("utf8"); child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => { stderr += `${stderr ? "\n" : ""}${formatErrorMessage(error)}`; });
    child.on("close", (code, closeSignal) => { clearTimeout(timeout); params.abortSignal?.removeEventListener("abort", abort); exitCode = code; signal = closeSignal; resolve(); });
  });
  return { stdout, stderr, exitCode, signal, timedOut, externalAbort };
}

function buildSandboxShellScript(argv: string[]): string {
  const home = "/tmp/openclaw-embedded-harness-home";
  return [
    `export HOME=${shellQuoteArg(home)}`,
    `export XDG_CACHE_HOME=${shellQuoteArg(`${home}/.cache`)}`,
    `export XDG_CONFIG_HOME=${shellQuoteArg(`${home}/.config`)}`,
    `export XDG_DATA_HOME=${shellQuoteArg(`${home}/.local/share`)}`,
    `mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"`,
    argv.map(shellQuoteArg).join(" "),
  ].join(" && ");
}

function shellQuoteArg(arg: string): string { return /^[A-Za-z0-9_/:=.,@%+-]+$/u.test(arg) ? arg : `'${arg.replaceAll("'", `'\\''`)}'`; }

function buildPromptError(params: { exitCode: number | null; signal: NodeJS.Signals | null; stderr: string; stdout: string }): string {
  const pieces = [`harness process exited with code ${params.exitCode ?? "null"}${params.signal ? ` signal ${params.signal}` : ""}`];
  if (params.stderr.trim()) pieces.push(`stderr:\n${params.stderr.trim()}`);
  if (params.stdout.trim()) pieces.push(`stdout:\n${params.stdout.trim()}`);
  return pieces.join("\n\n");
}

function buildAssistantMessage(text: string, params: AgentHarnessAttemptParams, prefix: string): AssistantMessage {
  return { id: `${params.runId}:${prefix}:assistant`, role: "assistant", content: [{ type: "text", text }], usage: ZERO_USAGE };
}
