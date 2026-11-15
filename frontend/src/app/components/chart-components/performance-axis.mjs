const DAY_MS = 24 * 60 * 60 * 1000
const MIN_LABEL_SPACING_PX = 80
const NICE_INTERVALS_DAYS = [1, 7, 14, 21, 30, 60, 90, 180, 365, 730, 1095, 1825]

const MONTHS = new Map([
  ["jan", 0], ["january", 0], ["feb", 1], ["february", 1], ["mar", 2], ["march", 2],
  ["apr", 3], ["april", 3], ["may", 4], ["jun", 5], ["june", 5], ["jul", 6], ["july", 6],
  ["aug", 7], ["august", 7], ["sep", 8], ["sept", 8], ["september", 8], ["oct", 9],
  ["october", 9], ["nov", 10], ["november", 10], ["dec", 11], ["december", 11],
])

/** @param {string} value */
function parsePerformanceDate(value) {
  if (typeof value !== "string") return null

  const isoDate = /^(\d{4})-(\d{2})-(\d{2})(?:$|T)/.exec(value)
  if (isoDate) {
    const [, yearText, monthText, dayText] = isoDate
    const timestamp = Date.UTC(Number(yearText), Number(monthText) - 1, Number(dayText))
    const date = new Date(timestamp)
    if (date.getUTCFullYear() === Number(yearText) && date.getUTCMonth() === Number(monthText) - 1 && date.getUTCDate() === Number(dayText)) {
      return timestamp
    }
    return null
  }

  const monthYear = /^([A-Za-z]+)\s+(\d{4})$/.exec(value.trim())
  if (monthYear) {
    const month = MONTHS.get(monthYear[1].toLowerCase())
    if (month !== undefined) return Date.UTC(Number(monthYear[2]), month, 1)
  }

  const timestamp = Date.parse(value)
  return Number.isFinite(timestamp) ? timestamp : null
}

/** @param {number} timestamp */
function monthName(timestamp) {
  return new Date(timestamp).toLocaleString("en-US", { month: "short", timeZone: "UTC" })
}

/**
 * @param {number} timestamp
 * @param {number} intervalDays
 * @param {number} minTimestamp
 * @param {number} maxTimestamp
 * @param {number | undefined} previousYear
 */
function formatTick(timestamp, intervalDays, minTimestamp, maxTimestamp, previousYear) {
  const date = new Date(timestamp)
  const year = date.getUTCFullYear()
  const yearChanged = previousYear !== undefined && previousYear !== year
  const crossesYear = new Date(minTimestamp).getUTCFullYear() !== new Date(maxTimestamp).getUTCFullYear()
  const showYear = yearChanged || (previousYear === undefined && crossesYear)
  const yearSuffix = ` ’${String(year).slice(-2)}`
  const month = monthName(timestamp)

  if (intervalDays >= 300) return `${month}${yearSuffix}`
  if (maxTimestamp - minTimestamp <= 120 * DAY_MS) {
    return `${date.getUTCMonth() + 1}/${date.getUTCDate()}${showYear ? yearSuffix : ""}`
  }
  if (intervalDays >= 25) return `${month}${showYear ? yearSuffix : ""}`
  return `${month} ${date.getUTCDate()}${showYear ? yearSuffix : ""}`
}

/** @param {{ date: string }[]} data @param {number} target */
function nearestDatePoint(data, target) {
  let low = 0
  let high = data.length - 1
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (data[middle].timestamp < target) low = middle + 1
    else high = middle
  }
  if (low === 0) return data[0]
  const before = data[low - 1]
  const after = data[low]
  return target - before.timestamp <= after.timestamp - target ? before : after
}

/**
 * Selects actual data dates for x-axis labels. Width is the usable plot width in CSS pixels.
 * Invalid or legacy labels that cannot be parsed use their source order and original text.
 *
 * @param {{ date: string }[]} data
 * @param {number} width
 * @returns {{ ticks: { value: number, label: string }[], intervalDays: number, usesDateScale: boolean }}
 */
export function planPerformanceTicks(data, width) {
  if (!Array.isArray(data) || data.length === 0) return { ticks: [], intervalDays: 1, usesDateScale: false }

  const parsed = data.map((point) => parsePerformanceDate(point.date))
  const usesDateScale = parsed.every((timestamp) => timestamp !== null)
  const points = usesDateScale
    ? data.map((point, index) => ({ timestamp: /** @type {number} */ (parsed[index]), date: point.date }))
      .sort((first, second) => first.timestamp - second.timestamp)
    : data.map((point, index) => ({ timestamp: index, date: point.date }))
  const distinctPoints = points.filter((point, index) => index === 0 || point.timestamp !== points[index - 1].timestamp)
  if (distinctPoints.length === 1) {
    return { ticks: [{ value: distinctPoints[0].timestamp, label: distinctPoints[0].date }], intervalDays: 1, usesDateScale }
  }

  const first = distinctPoints[0]
  const last = distinctPoints[distinctPoints.length - 1]
  const span = last.timestamp - first.timestamp
  const plotWidth = Math.max(1, Number.isFinite(width) ? width : 1)
  const maxLabels = Math.max(2, Math.floor(plotWidth / MIN_LABEL_SPACING_PX) + 1)
  const targetIntervalDays = usesDateScale ? span / DAY_MS / (maxLabels - 1) : distinctPoints.length / (maxLabels - 1)
  const intervalDays = NICE_INTERVALS_DAYS.find((days) => days >= targetIntervalDays) ?? NICE_INTERVALS_DAYS[NICE_INTERVALS_DAYS.length - 1]

  /** @type {{ timestamp: number, date: string }[]} */
  const candidates = [first]
  if (usesDateScale && intervalDays >= 300) {
    const firstYear = new Date(first.timestamp).getUTCFullYear()
    const yearStride = Math.max(1, Math.round(intervalDays / 365))
    let previousYear = firstYear
    for (const point of distinctPoints) {
      const year = new Date(point.timestamp).getUTCFullYear()
      if (year !== previousYear && (year - firstYear) % yearStride === 0 && point.timestamp !== last.timestamp) candidates.push(point)
      previousYear = year
    }
  } else {
    const step = usesDateScale ? intervalDays * DAY_MS : Math.max(1, Math.ceil(distinctPoints.length / maxLabels))
    for (let target = first.timestamp + step; target < last.timestamp; target += step) {
      const point = usesDateScale ? nearestDatePoint(distinctPoints, target) : distinctPoints[Math.round(target)]
      if (point && point.timestamp > first.timestamp && point.timestamp < last.timestamp && candidates.at(-1)?.timestamp !== point.timestamp) candidates.push(point)
    }
  }
  candidates.push(last)

  const minGap = span * MIN_LABEL_SPACING_PX / plotWidth
  if (span <= minGap) {
    // When both endpoints cannot fit without colliding, keep the most recent one.
    const tick = { value: last.timestamp, label: usesDateScale ? formatTick(last.timestamp, intervalDays, first.timestamp, last.timestamp, undefined) : last.date }
    return { ticks: [tick], intervalDays, usesDateScale }
  }

  /** @type {{ timestamp: number, date: string }[]} */
  const selected = [first]
  for (const point of candidates.slice(1, -1)) {
    if (point.timestamp - selected[selected.length - 1].timestamp >= minGap && last.timestamp - point.timestamp >= minGap) selected.push(point)
  }
  while (selected.length > 1 && last.timestamp - selected[selected.length - 1].timestamp < minGap) selected.pop()
  selected.push(last)

  let previousYear
  const ticks = selected.map((point) => {
    const year = new Date(point.timestamp).getUTCFullYear()
    const label = usesDateScale ? formatTick(point.timestamp, intervalDays, first.timestamp, last.timestamp, previousYear) : point.date
    previousYear = year
    return { value: point.timestamp, label }
  })
  return { ticks, intervalDays, usesDateScale }
}

/** @param {string} value */
export function parsePerformanceDateValue(value) {
  return parsePerformanceDate(value)
}
