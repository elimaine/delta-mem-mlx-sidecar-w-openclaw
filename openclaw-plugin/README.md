# OpenClaw Custom Embedded Harness Plugin

Build and ship custom OpenClaw embedded harnesses without patching OpenClaw core.

This repo is a plugin template plus one installable example harness. The important idea is not the example itself; it is the pattern:

- runtime behavior lives in a plugin
- config is the durable source of truth
- guided install is a friendly setup layer
- agents opt in with `embeddedHarness.runtime`
- every model/harness route gets qualified before promotion

## Why this exists

Embedded harnesses let an OpenClaw agent run through a custom runtime instead of the default PI runner. That runtime might wrap a local coding CLI, an ACP bridge, a local model server, or a company-specific execution service.

Use this when you want an agent like Pike/Kavi/etc. to use a specific coding harness reliably, with sandbox-aware execution and explicit fallback behavior.

## Getting started

Install the plugin, run the guided setup, then point an agent at the registered harness runtime.

### 1. Pull/install the plugin

Use your normal OpenClaw plugin install flow. Example shape:

`openclaw plugin add github:clawfactory-code/openclaw-custom-embedded-harness-plugin`

If your OpenClaw build uses a different plugin install command, use that command with this repo URL:

`https://github.com/clawfactory-code/openclaw-custom-embedded-harness-plugin`

### 2. Run guided setup

The guided installer should ask which harness you want, run pregiddyup checks, and write/preview config.

Example shape:

`openclaw plugin setup openclaw-custom-embedded-harness-plugin`

The wizard should cover:

- harness runtime id, e.g. `opencode-local` or your custom id
- install strategy: `path`, `npx`, or `bundled`
- binary/package path
- backend URL from plugin config if needed
- target agent
- model/tool qualification smokes
- final config patch

### 3. Or install by config

Config is the source of truth. Guided setup is only a nicer way to produce this config.

Plugin config example:

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

Agent config example:

```json
{
  "embeddedHarness": {
    "runtime": "opencode-local",
    "fallback": "none"
  }
}
```

### 4. Run pregiddyup

Pregiddyup is the preflight before trusting the harness.

Minimum checks:

- CLI exists: `opencode --version` or equivalent
- CLI help exposes the flags used by the harness
- backend/model server is reachable
- effective OpenClaw model maps to the wrapped CLI
- model supports tools through this exact route
- sandbox can run the CLI
- exact-response smoke passes
- file-write smoke passes and is host-verified

### 5. Smoke test the agent

After enabling the agent runtime:

1. restart/reload OpenClaw
2. run doctor and confirm plugin errors are zero
3. ask the agent to write a fresh marker file
4. verify the file from the host
5. run one small coding task with tests before promotion

## Example harness

This repo includes one concrete example: `opencode-local`.

It demonstrates:

- registering an embedded harness runtime
- running a local CLI
- preserving sandbox boundaries
- creating minimal local CLI config when missing
- mapping OpenClaw's effective model into the wrapped CLI route
- marking runs as side-effecting / non-replay-safe

Treat it as an installation example and reference implementation. Users are expected to add their own harness under `src/examples/<your-harness>/` or in a separate plugin package.

## Model selection

Do not hide model choice in env vars by default. Prefer the OpenClaw-configured/effective model supplied to the harness attempt. If a harness cannot map `provider/modelId` to the wrapped CLI, fail loudly and ask for an explicit config mapping.

If a user switches model through OpenClaw UI/slash-command routing and OpenClaw passes that effective model into the run, the harness should use that switched model.

## Qualification rules

Do not assume a model supports tools because its model card says so. Qualify every chosen model through the actual harness/backend path.

Required checks:

1. Plain smoke: exact response, e.g. `READY`.
2. Structured tool support through the chosen backend, not only raw chat.
3. File read/write through the harness route.
4. One-file bug fix with host-side test verification.
5. Small multi-file/import-path fix.
6. Honesty check: report only files that actually exist on disk.
7. Safety check against untrusted instructions.
8. Sequential runs on local GPU hosts; no concurrent battery tests.

## Docs

- `docs/config-install.md` — config-source install
- `docs/guided-install.md` — guided install flow
- `docs/model-and-harness-qualification.md` — model/harness qualification
- `docs/harness-id-safety.md` — avoiding runtime id collisions
- `skills/openclaw-embedded-harness/SKILL.md` — bundled companion skill for agents

## Current status

This is a starter template. Before publishing broadly, wire the plugin to your actual OpenClaw plugin packaging/install flow and run the full pregiddyup + smoke checklist in your target environment.
