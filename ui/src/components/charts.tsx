import { usePrefersReducedMotion } from "@/lib/useReducedMotion"
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

// Themed chart kit on Recharts: wrappers fix palette, axes, tooltip, gradients, and animation.

// Chart palette (Tailwind hex values; SVG needs real colors, not class names).
export const CHART_COLORS = {
  blue: "#60a5fa",
  amber: "#f59e0b",
  emerald: "#10b981",
  red: "#ef4444",
  violet: "#8b5cf6",
  slate: "#94a3b8",
} as const

export type TrendSeries = { key: string; label: string; color: string }

// A multi-series area trend chart. `data` is rows keyed by each series' `key` plus a `label` field
// for the x-axis. `yDomain` fixes the y range (e.g. `[0, 100]`); omit it to fit the data.
export function TrendChart({
  data,
  series,
  height = 180,
  yDomain,
}: {
  data: Array<Record<string, number | string>>
  series: TrendSeries[]
  height?: number
  yDomain?: [number, number]
}) {
  // Recharts animates in JS, so the stylesheet's reduced-motion block cannot reach it. Drop the
  // entry sweep and gliding tooltip instead.
  const reducedMotion = usePrefersReducedMotion()
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <defs>
          {series.map((s) => (
            <linearGradient key={s.key} id={`grad-${s.key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={s.color} stopOpacity={0.4} />
              <stop offset="95%" stopColor={s.color} stopOpacity={0} />
            </linearGradient>
          ))}
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis dataKey="label" tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} />
        <YAxis
          tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
          width={32}
          allowDecimals={false}
          domain={yDomain}
        />
        <Tooltip
          isAnimationActive={!reducedMotion}
          contentStyle={{
            fontSize: 12,
            borderRadius: 8,
            border: "1px solid hsl(var(--border))",
            background: "hsl(var(--background))",
          }}
        />
        {series.map((s) => (
          <Area
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={s.color}
            fill={`url(#grad-${s.key})`}
            strokeWidth={2}
            isAnimationActive={!reducedMotion}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  )
}

export type PhaseSegment = { label: string; value: number; color: string }

// A horizontal stacked bar for a few labeled segments (e.g. pickup vs review), with tooltips. Cheap
// enough for one per table row.
export function PhaseBar({ segments, height = 8 }: { segments: PhaseSegment[]; height?: number }) {
  const total = segments.reduce((sum, s) => sum + Math.max(0, s.value), 0)
  return (
    <div
      className="flex w-full overflow-hidden rounded bg-muted/40"
      style={{ height }}
      role="img"
      aria-label={segments.map((s) => `${s.label}: ${s.value}`).join(", ")}
    >
      {total > 0 &&
        segments.map((s) => (
          <div
            key={s.label}
            className="h-full transition-all"
            style={{
              width: `${(Math.max(0, s.value) / total) * 100}%`,
              backgroundColor: s.color,
            }}
            title={`${s.label}: ${s.value}`}
          />
        ))}
    </div>
  )
}
