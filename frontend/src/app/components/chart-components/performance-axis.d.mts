export interface PerformanceTick {
  value: number
  label: string
}

export interface PerformanceAxisPlan {
  ticks: PerformanceTick[]
  intervalDays: number
  usesDateScale: boolean
}

export function planPerformanceTicks(data: { date: string }[], width: number): PerformanceAxisPlan
export function parsePerformanceDateValue(value: string): number | null
