import { describe, expect, it } from "vitest";
import { preflightHarness } from "../src/preflight.js";
import { resolvePluginConfig } from "../src/config.js";

describe("plugin config", () => {
  it("defaults to bundled delta-mem sidecar example enabled", () => {
    const cfg = resolvePluginConfig({});
    expect(cfg.harnesses.opencodeLocal.enabled).toBe(true);
    expect(cfg.harnesses.opencodeLocal.baseUrl).toBe("http://127.0.0.1:8765/v1");
  });

  it("merges user config", () => {
    const cfg = resolvePluginConfig({ harnesses: { opencodeLocal: { enabled: false, runtimeId: "custom-opencode" } } });
    expect(cfg.harnesses.opencodeLocal.enabled).toBe(false);
    expect(cfg.harnesses.opencodeLocal.runtimeId).toBe("custom-opencode");
  });

  it("preflight reports missing binary", () => {
    const result = preflightHarness({ runtimeId: "x", installStrategy: "path", binPath: "definitely-not-a-real-bin-openclaw" });
    expect(result.ok).toBe(false);
  });
});
