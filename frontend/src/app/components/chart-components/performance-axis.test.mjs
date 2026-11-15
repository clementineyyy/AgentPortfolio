import assert from "node:assert/strict"
import test from "node:test"

import { planPerformanceTicks } from "./performance-axis.mjs"

function dailyDates(start, count) {
  const first = new Date(`${start}T00:00:00Z`)
  return Array.from({ length: count }, (_, index) => {
    const date = new Date(first)
    date.setUTCDate(date.getUTCDate() + index)
    return { date: date.toISOString().slice(0, 10) }
  })
}

test("short ranges use month/day labels and retain actual endpoint dates", () => {
  const data = dailyDates("2026-08-13", 93)
  const result = planPerformanceTicks(data, 640)

  assert.equal(result.ticks[0].value, Date.parse("2026-08-13T00:00:00Z"))
  assert.equal(result.ticks[0].label, "8/13")
  assert.equal(result.ticks.at(-1).value, Date.parse("2026-11-13T00:00:00Z"))
  assert.equal(result.ticks.at(-1).label, "11/13")
  assert.ok(result.ticks.every((tick) => data.some((point) => Date.parse(`${point.date}T00:00:00Z`) === tick.value)))
})

test("wider charts can show more real dates than narrow charts", () => {
  const data = dailyDates("2022-11-14", 1461)
  const narrow = planPerformanceTicks(data, 320)
  const wide = planPerformanceTicks(data, 960)

  assert.ok(wide.ticks.length > narrow.ticks.length)
  assert.ok(wide.ticks.every((tick) => data.some((point) => Date.parse(`${point.date}T00:00:00Z`) === tick.value)))
})

test("multi-year ranges mark years without inventing dates", () => {
  const data = [
    { date: "2022-11-14" },
    { date: "2023-01-03" },
    { date: "2024-01-02" },
    { date: "2025-01-02" },
    { date: "2026-01-02" },
    { date: "2026-11-13" },
  ]
  const result = planPerformanceTicks(data, 640)

  assert.ok(result.ticks.some((tick) => tick.label === "Jan ’24"))
  assert.ok(result.ticks.every((tick) => data.some((point) => Date.parse(`${point.date}T00:00:00Z`) === tick.value)))
})

test("very narrow charts omit a colliding start label and keep the latest date", () => {
  const data = [{ date: "2026-01-01" }, { date: "2026-01-02" }]
  const result = planPerformanceTicks(data, 20)

  assert.deepEqual(result.ticks, [{ value: Date.parse("2026-01-02T00:00:00Z"), label: "1/2" }])
})

test("legacy month-year dates remain supported", () => {
  const data = [{ date: "Jan 2023" }, { date: "Apr 2023" }, { date: "Jul 2023" }, { date: "Oct 2023" }, { date: "Jan 2024" }]
  const result = planPerformanceTicks(data, 640)

  const knownMonthStarts = [
    Date.UTC(2023, 0, 1), Date.UTC(2023, 3, 1), Date.UTC(2023, 6, 1),
    Date.UTC(2023, 9, 1), Date.UTC(2024, 0, 1),
  ]
  assert.ok(result.ticks.every((tick) => knownMonthStarts.includes(tick.value)))
  assert.ok(result.ticks.some((tick) => tick.label.includes("’24")))
})
