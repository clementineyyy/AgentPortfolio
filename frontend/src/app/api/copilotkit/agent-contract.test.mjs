import assert from "node:assert/strict";
import test from "node:test";
import { HttpAgent } from "@ag-ui/client";

test("CopilotKit runtime agent supports setting conversation messages", () => {
  const agent = new HttpAgent({ url: "http://localhost:8000/crewai-agent" });

  assert.equal(typeof agent.setMessages, "function");
  assert.doesNotThrow(() => agent.setMessages([]));
});
