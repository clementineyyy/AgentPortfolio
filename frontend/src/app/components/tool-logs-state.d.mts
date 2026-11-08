export interface ToolLogStateItem {
  id: string | number
  message: string
  status: "processing" | "completed"
}

export function getToolLogs(
  state: { tool_logs?: unknown } | null | undefined,
): ToolLogStateItem[]
