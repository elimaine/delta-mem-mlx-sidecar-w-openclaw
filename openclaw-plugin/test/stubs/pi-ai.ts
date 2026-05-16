export type Usage = {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  totalTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    total: number;
  };
};

export type AssistantMessage = {
  id: string;
  role: "assistant";
  content: Array<{ type: "text"; text: string }>;
  usage: Usage;
};
