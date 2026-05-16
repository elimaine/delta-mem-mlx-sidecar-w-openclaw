# Guided Install

Guided install is a UX layer over config install. It should ask questions, run preflights, and emit config.

Recommended flow:

1. Pick harness example: bundled `opencode-local` or a user-provided harness.
2. Pick install strategy: `path`, `npx`, or `bundled`.
3. Check binary/package availability.
4. Check CLI help contains expected flags.
5. Ask which agent should use the runtime.
6. Write/preview config patch.
7. Restart/reload OpenClaw.
8. Run exact-response smoke.
9. Run file-write smoke and host-verify.
10. Run one small coding task before promotion.

The guided installer must not be the only source of truth. The final durable state is config.
