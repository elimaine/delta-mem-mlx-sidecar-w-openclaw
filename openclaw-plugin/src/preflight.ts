import { spawnSync } from "node:child_process";
import type { ExampleHarnessConfig } from "./config.js";

export type PreflightResult = {
  ok: boolean;
  checks: Array<{ name: string; ok: boolean; detail: string }>;
};

export function preflightHarness(config: ExampleHarnessConfig): PreflightResult {
  const checks: PreflightResult["checks"] = [];
  const bin = config.binPath?.trim();
  checks.push({ name: "runtime id", ok: Boolean(config.runtimeId?.trim()), detail: config.runtimeId || "missing" });
  checks.push({ name: "install strategy", ok: Boolean(config.installStrategy), detail: config.installStrategy || "missing" });
  checks.push({ name: "binary path", ok: Boolean(bin), detail: bin || "missing" });
  if (bin) checks.push(checkBinary(bin));
  if (config.installStrategy === "npx") {
    checks.push({ name: "package spec", ok: Boolean(config.packageSpec?.trim()), detail: config.packageSpec || "missing" });
  }
  return { ok: checks.every((check) => check.ok), checks };
}

function checkBinary(bin: string): { name: string; ok: boolean; detail: string } {
  const result = spawnSync(bin, ["--version"], { encoding: "utf8", timeout: 10_000 });
  if (result.error) return { name: `${bin} --version`, ok: false, detail: result.error.message };
  return { name: `${bin} --version`, ok: result.status === 0, detail: (result.stdout || result.stderr || `exit ${result.status}`).trim() };
}
