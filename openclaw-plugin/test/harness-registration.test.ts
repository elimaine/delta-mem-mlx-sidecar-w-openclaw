import { describe, expect, it } from "vitest";
import plugin from "../src/index.js";
import { createOpenCodeLocalAgentHarness } from "../src/examples/opencode-local/harness.js";

describe("embedded harness examples", () => {
  it("registers only config-enabled harnesses by default", () => {
    const registered: string[] = [];
    plugin.register({ registerAgentHarness: (h) => registered.push(h.id), pluginConfig: {} } as never);
    expect(registered).toEqual(["opencode-local"]);
    expect(registered).not.toContain("pi");
    expect(registered).not.toContain("codex");
  });

  it("can disable the bundled OpenCode example via config", () => {
    const registered: string[] = [];
    plugin.register({
      registerAgentHarness: (h) => registered.push(h.id),
      pluginConfig: { harnesses: { opencodeLocal: { enabled: false } } },
    } as never);
    expect(registered).toEqual([]);
  });

  it("opencode supports forced opencode-local runtime for delta-mem provider", () => {
    const harness = createOpenCodeLocalAgentHarness();
    expect(harness.supports({ provider: "delta-mem-mlx", requestedRuntime: "opencode-local" })).toMatchObject({ supported: true });
  });
});
