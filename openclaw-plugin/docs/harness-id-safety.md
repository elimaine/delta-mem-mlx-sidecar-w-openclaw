# Harness ID Safety

Harness ids are global runtime ids. Do not reuse an existing id unless intentionally replacing that runtime.

Known built-ins / local examples:

- `pi` — built-in fallback runner
- `codex` — bundled Codex app-server harness
- `opencode-local` — example direct OpenCode CLI harness

Guidelines:

1. Use a unique, descriptive id.
2. Prefer `<tool>-<transport>` such as `opencode-local` or `<tool>-<transport>`.
3. Test plugin registration in a clean process and assert all registered ids are unique.
4. Do not silently override `pi`, `codex`, or another installed plugin's id.
5. In config, use explicit `embeddedHarness: { runtime: "<id>", fallback: "none" }` while testing.
