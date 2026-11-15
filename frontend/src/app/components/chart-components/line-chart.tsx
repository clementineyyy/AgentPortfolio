"use client"

import { useMemo, useState } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { parsePerformanceDateValue, planPerformanceTicks } from "./performance-axis.mjs"

interface LineChartData {
  date: string
  portfolio: number
  spy: number | null | undefined
}

interface LineChartComponentProps {
  data: LineChartData[] | [] | undefined
  size?: "normal" | "small"
}

export function LineChartComponent({ data, size = "normal" }: LineChartComponentProps) {
  const [chartWidth, setChartWidth] = useState(640)
  const height = size === "small" ? 120 : 192 // h-30 or h-48
  const padding = size === "small" ? "p-2" : "p-4"
  const fontSize = size === "small" ? 8 : 10
  const tooltipFontSize = size === "small" ? "9px" : "11px"
  const legendFontSize = size === "small" ? "9px" : "11px"
  const chartData = useMemo(() => {
    const rows = Array.isArray(data) ? data : []
    const timestamps = rows.map((point) => parsePerformanceDateValue(point.date))
    const usesDateScale = rows.length > 0 && timestamps.every((timestamp) => timestamp !== null)
    return rows.map((point, index) => ({
      ...point,
      axisValue: usesDateScale ? timestamps[index] ?? index : index,
    })).sort((first, second) => first.axisValue - second.axisValue)
  }, [data])
  const axisPlan = useMemo(
    () => planPerformanceTicks(chartData, Math.max(1, chartWidth - 80)),
    [chartData, chartWidth],
  )
  const tickLabels = useMemo(
    () => new Map(axisPlan.ticks.map((tick) => [tick.value, tick.label])),
    [axisPlan.ticks],
  )
  const tooltipDates = useMemo(
    () => new Map(chartData.map((point) => [point.axisValue, point.date])),
    [chartData],
  )
  return (
    <div className={`bg-white border border-[#D8D8E5] rounded-xl ${padding}`}>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%" onResize={(width) => setChartWidth(width)}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E8E8EF" />
            <XAxis
              dataKey="axisValue"
              type="number"
              scale={axisPlan.usesDateScale ? "time" : "linear"}
              domain={["dataMin", "dataMax"]}
              ticks={axisPlan.ticks.map((tick) => tick.value)}
              tickFormatter={(value) => tickLabels.get(Number(value)) ?? ""}
              minTickGap={8}
              stroke="#575758"
              fontSize={fontSize}
              fontFamily="Plus Jakarta Sans"
            />
            <YAxis
              stroke="#575758"
              fontSize={fontSize}
              fontFamily="Plus Jakarta Sans"
              tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
            />
            <Tooltip
              labelFormatter={(value) => tooltipDates.get(Number(value)) ?? String(value)}
              contentStyle={{
                backgroundColor: "white",
                border: "1px solid #D8D8E5",
                color: "#575758",
                borderRadius: "8px",
                fontSize: tooltipFontSize,
                fontFamily: "Plus Jakarta Sans",
              }}
              formatter={(value, name) => [
                typeof value === "number" ? `$${value.toLocaleString()}` : "N/A",
                String(name ?? "").toLowerCase() === "portfolio" ? "Portfolio" : "SPY",
              ]}
            />
            <Legend
              wrapperStyle={{
                fontSize: legendFontSize,
                fontFamily: "Plus Jakarta Sans",
                fontWeight: 500,
              }}
            />
            <Line type="monotone" dataKey="portfolio" stroke="#FF003C" strokeWidth={2} name="Portfolio" dot={false} />
            <Line type="monotone" dataKey="spy" stroke="#BEC9FF" strokeWidth={2} name="SPY" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
