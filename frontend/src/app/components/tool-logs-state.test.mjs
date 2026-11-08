import assert from "node:assert/strict"
import test from "node:test"

import { getToolLogs } from "./tool-logs-state.mjs"

test("co-agent state without tool_logs produces an empty list", () => {
  assert.deepEqual(getToolLogs(undefined), [])
  assert.deepEqual(getToolLogs(null), [])
  assert.deepEqual(getToolLogs({}), [])
})

test("invalid tool_logs values produce an empty list", () => {
  assert.deepEqual(getToolLogs({ tool_logs: null }), [])
  assert.deepEqual(getToolLogs({ tool_logs: {} }), [])
})

test("valid tool logs are preserved", () => {
  const logs = [{ id: 1, message: "Loading market data", status: "processing" }]

  assert.equal(getToolLogs({ tool_logs: logs }), logs)
})
