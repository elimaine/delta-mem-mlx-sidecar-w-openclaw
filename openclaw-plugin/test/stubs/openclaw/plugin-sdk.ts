type PluginApi = {
  registerAgentHarness(harness: unknown): void;
  pluginConfig?: unknown;
};

type PluginEntry = {
  id: string;
  register(api: PluginApi): void;
};

export function definePluginEntry<T extends PluginEntry>(entry: T): T {
  return entry;
}
