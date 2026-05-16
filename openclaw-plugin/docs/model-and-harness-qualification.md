# Model and Harness Qualification

A model is qualified only for a specific route:

`agent -> embedded harness -> CLI/backend -> model server -> exact model tag`

Changing any part of that route requires retesting.

## Model qualifiers

Record these before promotion:

- exact model id/tag
- quantization
- backend and version
- context/output limits
- thinking/reasoning settings
- tool-call support result
- file-edit smoke result
- coding battery score
- known failures

## Harness qualifiers

Record these before promotion:

- harness id
- plugin package/version
- wrapped binary/API
- command shape
- sandbox execution path
- required binds/env vars
- model config source and override precedence
- session/persistence strategy
- fallback strategy
- output parsing strategy
- side-effect metadata

## Promotion rule

Promote a model/harness pair only after a host-verified file-write smoke and at least one coding task with tests pass. Model claims and final assistant text are not enough.


## Slash-command / UI model switches

A harness should use the effective model in the run params, not a hidden env var. If OpenClaw slash commands or UI routing switch the model before the embedded run starts, the harness should see the switched `provider` / `modelId` and map that value to the wrapped CLI.

If the wrapped CLI cannot represent that model/provider pair, fail loudly with a mapping error rather than silently falling back to a different model.
