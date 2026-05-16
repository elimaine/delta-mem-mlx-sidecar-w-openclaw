import { describe, expect, it } from "vitest";
import { mapOpenClawModelToCliProviderArgs, mapOpenClawModelToOpenCodeLocal } from "../src/core/model-mapping.js";

describe("model mapping", () => {
  it("maps delta-mem OpenClaw model to OpenCode local model ref", () => {
    expect(mapOpenClawModelToOpenCodeLocal({ provider: "delta-mem-mlx", modelId: "qwen2.5-0.5b-mlx-test" } as never)).toBe(
      "delta-mem-mlx/qwen2.5-0.5b-mlx-test",
    );
  });

  it("maps delta-mem OpenClaw model to generic CLI provider/model flags", () => {
    expect(mapOpenClawModelToCliProviderArgs({ provider: "delta-mem-mlx", modelId: "qwen2.5-0.5b-mlx-test" } as never)).toEqual([
      "--provider",
      "openai-compatible",
      "--model",
      "qwen2.5-0.5b-mlx-test",
    ]);
  });

  it("fails loudly when required model info is missing", () => {
    expect(() => mapOpenClawModelToOpenCodeLocal({ provider: "delta-mem-mlx" } as never)).toThrow(/provider and params.modelId/);
  });
});
