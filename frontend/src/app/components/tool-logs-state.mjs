/**
 * Return the tool log list from an optional co-agent state.
 * @param {{ tool_logs?: unknown } | null | undefined} state
 * @returns {Array<{ id: string | number, message: string, status: "processing" | "completed" }>}
 */
export function getToolLogs(state) {
  const logs = state?.tool_logs
  return Array.isArray(logs) ? logs : []
}
