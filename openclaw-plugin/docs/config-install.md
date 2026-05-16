# Config Install

Config is the source of truth. Guided install should produce a config patch; the plugin should read config and register only enabled harnesses.

Example plugin config:

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

Agent runtime config then selects the registered id:

```json
{
  "embeddedHarness": {
    "runtime": "opencode-local",
    "fallback": "none"
  }
}
```

Model selection is not configured in plugin env vars. The harness maps OpenClaw's effective run model (`provider` / `modelId`) to the wrapped CLI.
