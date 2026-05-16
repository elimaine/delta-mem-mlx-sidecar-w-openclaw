---
name: openclaw-embedded-harness
description: Use when creating, installing, configuring, qualifying, debugging, or enabling an OpenClaw embedded harness plugin. Covers harness CLI install, pregiddyup/preflight checks, config-source install, model tool-use qualification, plugin creation, embeddedHarness config, and smoke testing.
---

# OpenClaw Embedded Harness

Use this skill to help a user create or enable an embedded harness plugin.

A harness is backend/runtime code. This skill is the operating guide for agents and humans.

## Ground rules

- Config is source of truth. Guided install should produce config, not hidden state.
- Use the effective OpenClaw run model (`provider` / `modelId`), not hidden model env vars.
- Do not trust model cards. Qualify tool use through the exact harness/backend/model route.
- Do not bypass sandbox. If the target agent is sandboxed, the harness CLI must run inside that sandbox.
- Do not reuse existing runtime ids such as `pi` or `codex` unless intentionally replacing them.
- Host-verify file writes before claiming success.

## Expected config shape

Plugin config enables/registers harnesses:

```json
{
  "plugins": {
    "embedded-harness-examples": {
      "harnesses": {
        "opencodeLocal": {
          "enabled": true,
          "runtimeId": "opencode-local",
          "installStrategy": "path",
          "binPath": "opencode",
          "baseUrl": "http://127.0.0.1:11434/v1"
        }
      }
    }
  }
}
```

Agent config selects the runtime:

```json
{
  "embeddedHarness": {
    "runtime": "opencode-local",
    "fallback": "none"
  }
}
```

Use OpenClaw's config docs/schema before editing real config. Prefer config UI/API for normal installs.

## Pregiddyup

“Pregiddyup” is the preflight/checklist phase before enabling a harness.

Run/verify:

1. Harness CLI is installed or installable.
   - path mode: `<bin> --version`
   - npx mode: `npx -y <package> --help`
2. CLI help exposes the flags the harness will use.
3. Backend/model server is reachable if relevant.
4. Effective OpenClaw model maps to the CLI's provider/model flags.
5. Model supports tool/file operations through this exact route.
6. Sandbox command path works inside the target agent sandbox.
7. Writable CLI home/cache dirs are inside sandbox or safe temp dirs.
8. Runtime id does not collide with `pi`, `codex`, or another installed harness.

If pregiddyup fails:

- Binary missing: switch install strategy, set `binPath`, or install/cache package.
- Help flags missing: pin a compatible CLI version or adjust harness command builder.
- Model missing: check exact provider/model id and backend model list.
- Tools unsupported: try a different backend route; do not assume another route works.
- Sandbox path failure: bind the minimum needed tool path read-only; do not loosen broad fs scope.
- Auth failure: configure the wrapped CLI/provider, then rerun exact smoke.
- No file on disk despite success text: mark model/harness unqualified; inspect logs/stdout/stderr.

## End-to-end checklist

1. Pick harness route: direct CLI, ACP bridge, local server, or hosted API.
2. Install or choose install strategy for the harness CLI.
3. Call the CLI directly with `--help`; write down verified command shape.
4. Pick a unique runtime id, e.g. `mytool-local`.
5. Add or generate a harness plugin using the embedded harness creator/template.
6. Add config schema/defaults for install strategy, runtime id, binary/package, backend URL, and mode.
7. Run pregiddyup.
8. Qualify model:
   - exact text smoke
   - tiny tool/file smoke
   - one-file bug fix with test
   - small multi-file/import-path fix
   - honesty check against disk state
9. Enable plugin config.
10. Set target agent `embeddedHarness.runtime` with `fallback: "none"`.
11. Restart/reload OpenClaw.
12. Run doctor and confirm plugin errors are zero.
13. Run sandboxed exact-response smoke.
14. Run sandboxed file-write smoke and host-verify.
15. Only then promote the harness/model pair.

## Docs to consult

Start with local OpenClaw docs and source before guessing:

- `/srv/clawfactory/bot_repos/sandy/code/docs`
- `/srv/clawfactory/bot_repos/sandy/code/src/agents/harness/`
- `/srv/clawfactory/bot_repos/sandy/code/src/plugin-sdk/agent-harness-runtime.ts`
- bundled plugin examples in this repo: `docs/config-install.md`, `docs/guided-install.md`, `docs/model-and-harness-qualification.md`, `docs/harness-id-safety.md`

If docs are stale or missing, inspect the current source/types and verify with a live smoke.
