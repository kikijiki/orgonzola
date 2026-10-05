import {
  type BoardPeopleStatsView,
  type PersonActivityView,
  type PersonPickupView,
  type PersonStatsView,
  type WorkPatternView,
  commands,
} from "@/bindings"
import { DataTable } from "@/components/DataTable"
import { useToast } from "@/components/Toast"
import { Card, EmptyState, NameTag } from "@/components/primitives"
import { avatarUrl } from "@/lib/forge"
import { relativeAge } from "@/lib/format"
import { usePrefersReducedMotion } from "@/lib/useReducedMotion"
import { cn } from "@/lib/utils"
import type { ColumnDef } from "@tanstack/react-table"
import { useCallback, useEffect, useRef, useState } from "react"

// Comparison window: how far back the per-person stats look.
const WINDOWS = [
  { label: "Last 2 weeks", since: 14, until: 0 },
  { label: "Last month", since: 30, until: 0 },
  { label: "Last 3 months", since: 90, until: 0 },
  { label: "Last 6 months", since: 180, until: 0 },
]

// The board's people as a side-by-side comparison: throughput, flow, review participation,
// current load, risk/hygiene and an off-hours share. Bases: UTC-only off-hours/active-days;
// head-commit CI attribution; PR size only over PRs with synced files; resolved = closing links.
// Selecting a row opens that person's in-flight PRs and a when-they-work heatmap.
// The roster comes from the returned stats: a team board's stats cover its chosen people, a
// repo/org board's contributors are discovered from activity server-side.
export function BoardPeople({
  boardId,
  webBase,
  focusLogin,
}: {
  boardId: string
  webBase: string | null
  focusLogin?: string | null
}) {
  const [win, setWin] = useState(1) // default "Last month"
  const [stats, setStats] = useState<BoardPeopleStatsView | null>(null)
  const [statsError, setStatsError] = useState<string | null>(null)
  const [pickup, setPickup] = useState<PersonPickupView[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [activity, setActivity] = useState<PersonActivityView | null>(null)
  const [pattern, setPattern] = useState<WorkPatternView | null>(null)
  const toast = useToast()
  // The detail card can render far below the table. `revealPending` is set only by an explicit
  // selection (row click, pickup click, focusLogin), so a board/window reload never scrolls.
  const detailRef = useRef<HTMLDivElement | null>(null)
  const revealPending = useRef(false)
  const prefersReducedMotion = usePrefersReducedMotion()

  const selectAndReveal = useCallback((login: string) => {
    revealPending.current = true
    setSelected(login)
  }, [])

  // `block: "nearest"` scrolls the minimum needed, and not at all if the card is visible.
  useEffect(() => {
    if (revealPending.current && selected) {
      detailRef.current?.scrollIntoView({
        behavior: prefersReducedMotion ? "auto" : "smooth",
        block: "nearest",
      })
      revealPending.current = false
    }
  }, [selected, prefersReducedMotion])

  useEffect(() => {
    let ignore = false
    const w = WINDOWS[win]
    setStats(null)
    setStatsError(null)
    setPickup(null)
    void commands.boardPeopleStats(boardId, w.since, w.until).then((res) => {
      if (!ignore) {
        if (res.status === "ok") setStats(res.data)
        else setStatsError(res.error)
      }
    })
    void commands.boardPickup(boardId, w.since, w.until).then((res) => {
      if (!ignore) {
        if (res.status === "ok") setPickup(res.data)
        else toast("Could not load the pickup panel", "error")
      }
    })
    return () => {
      ignore = true
    }
  }, [boardId, win, toast])

  // Keep a valid selection: first person by default, cleared if they drop out.
  useEffect(() => {
    if (stats == null) return
    const roster = stats.people.map((s) => s.login)
    setSelected((prev) => (prev && roster.includes(prev) ? prev : (roster[0] ?? null)))
  }, [stats])

  // Focus the person from an in-app "go to this user" click, if in the roster, and reveal them.
  useEffect(() => {
    if (focusLogin && stats?.people.some((s) => s.login === focusLogin)) selectAndReveal(focusLogin)
  }, [focusLogin, stats, selectAndReveal])

  useEffect(() => {
    if (!selected) {
      setActivity(null)
      setPattern(null)
      return
    }
    let ignore = false
    const w = WINDOWS[win]
    void commands.personActivity(boardId, selected).then((res) => {
      if (!ignore) {
        if (res.status === "ok") setActivity(res.data)
        else toast("Could not load activity for this person", "error")
      }
    })
    void commands.personWorkPattern(boardId, selected, w.since, w.until).then((res) => {
      if (!ignore) {
        if (res.status === "ok") setPattern(res.data)
        else toast("Could not load work pattern for this person", "error")
      }
    })
    return () => {
      ignore = true
    }
  }, [boardId, selected, win, toast])

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <select
          className="h-9 rounded-md border border-border bg-background px-2 text-sm"
          value={win}
          onChange={(e) => setWin(Number(e.target.value))}
        >
          {WINDOWS.map((w, i) => (
            <option key={w.label} value={i}>
              {w.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-muted-foreground">
          Click a row for one person's in-flight work and when they work.
        </p>
      </div>

      {pickup != null && pickup.length > 0 && (
        <PickupPanel pickup={pickup} webBase={webBase} onSelect={selectAndReveal} />
      )}

      {statsError != null ? (
        <EmptyState title="Could not load people stats" hint={statsError} />
      ) : stats == null ? (
        <Card>
          <p className="text-sm text-muted-foreground">loading...</p>
        </Card>
      ) : stats.people.length === 0 ? (
        <EmptyState
          title="No one active in this window"
          hint="Widen the window above, or sync more activity. People are discovered from the board's repos; a team board can also pin a roster under Settings."
        />
      ) : (
        <div className="space-y-3">
          <StatsGuide />
          <DataTable
            columns={statColumns(webBase)}
            data={stats.people}
            initialSorting={[{ id: "commits", desc: true }]}
            onRowClick={(r) => selectAndReveal(r.login)}
            emptyMessage="no activity in this window"
          />
        </div>
      )}

      {stats != null && stats.people.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Team off-hours:{" "}
          <span className="font-mono">
            {stats.total_events > 0
              ? `${Math.round((stats.off_hours_events / stats.total_events) * 100)}%`
              : "n/a"}
          </span>{" "}
          of the team's work events fell on a weekend or outside 07:00-20:00 UTC. A team-wide
          sustainability read, not a per-person figure; timezones are unknown, so read it as a
          pattern, not a clock.
        </p>
      )}

      {selected && (
        // The plain div carries the scroll-target ref: Card does not forward a ref.
        <div ref={detailRef}>
          <Card className="space-y-3">
            <h2 className="text-sm font-medium">
              {selected} <span className="font-normal text-muted-foreground">- when they work</span>
            </h2>
            {pattern ? <Heatmap buckets={pattern.buckets} /> : <Skeleton />}

            <h3 className="pt-1 text-sm font-medium text-muted-foreground">
              In flight ({activity?.open_prs.length ?? 0})
            </h3>
            {activity == null ? (
              <Skeleton />
            ) : activity.open_prs.length === 0 ? (
              <p className="text-sm text-muted-foreground">nothing open</p>
            ) : (
              <ul className="space-y-1 text-sm">
                {activity.open_prs.map((pr) => (
                  <li key={`${pr.repo}#${pr.number}`}>
                    <span className="font-mono text-muted-foreground">
                      {pr.repo}#{pr.number}
                    </span>{" "}
                    {pr.title}
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}

// The "who can pick this up" panel: each person's current load, last activity and areas of
// context, sortable by name / load / recency. An operational view, not a ranking. Clicking a
// card opens that person's detail.
function PickupPanel({
  pickup,
  webBase,
  onSelect,
}: {
  pickup: PersonPickupView[]
  webBase: string | null
  onSelect: (login: string) => void
}) {
  const [sort, setSort] = useState<"name" | "load" | "recent">("name")
  const sorted = [...pickup].sort((a, b) => {
    if (sort === "load") return a.open_prs - b.open_prs || a.login.localeCompare(b.login)
    if (sort === "recent")
      return (
        (b.last_active ?? "").localeCompare(a.last_active ?? "") || a.login.localeCompare(b.login)
      )
    return a.login.localeCompare(b.login)
  })
  return (
    <Card className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-sm font-medium text-muted-foreground">Who can pick this up</h2>
          <p className="text-xs text-muted-foreground">
            Each person's current in-flight load, when they were last active, and the areas they
            have context in - to match new work to who has room and who knows the code. A
            descriptive read, not a score or a ranking.
          </p>
        </div>
        <label className="flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
          sort
          <select
            className="h-8 rounded-md border border-border bg-background px-2 text-sm"
            value={sort}
            onChange={(e) => setSort(e.target.value as "name" | "load" | "recent")}
          >
            <option value="name">name</option>
            <option value="load">least loaded</option>
            <option value="recent">most recently active</option>
          </select>
        </label>
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {sorted.map((p) => (
          <button
            key={p.login}
            type="button"
            onClick={() => onSelect(p.login)}
            // overflow-hidden zeroes this grid item's automatic minimum size so unbreakable content
            // cannot widen the grid track, and clips residual overflow at the card boundary.
            className="space-y-1 overflow-hidden rounded-md border border-border p-2 text-left hover:bg-muted/50"
          >
            <div className="flex items-center justify-between gap-2">
              <NameTag
                login={p.login}
                src={avatarUrl(webBase, p.login)}
                className="text-sm font-medium"
              />
              <span
                className={cn(
                  "shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px]",
                  p.open_prs <= 1
                    ? "bg-emerald-100 text-emerald-800"
                    : "bg-slate-100 text-slate-700",
                )}
                title="currently-open authored PRs"
              >
                {p.open_prs} in flight
              </span>
            </div>
            <p className="text-[11px] text-muted-foreground">
              {p.last_active ? `last active ${relativeAge(p.last_active)}` : "no recent activity"}
            </p>
            {p.areas.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {p.areas.map((a) => (
                  <span
                    key={a.area}
                    // min-w-0 max-w-full truncate: one area value can be an unbreakable path
                    // segment, which flex-wrap cannot break.
                    className="min-w-0 max-w-full truncate rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
                    title={`${a.area} - ${a.changes} change(s) here`}
                  >
                    {a.area}
                  </span>
                ))}
              </div>
            )}
          </button>
        ))}
      </div>
    </Card>
  )
}

// Right-aligned numeric stat cell, dimmed at zero.
function num(value: number) {
  return (
    <span
      className={cn("block text-right font-mono text-sm", value === 0 && "text-muted-foreground")}
    >
      {value}
    </span>
  )
}

// Plain-language definition per stat column: source for both the header tooltips and the
// column legend. Each entry is { header, tip }; `person` is excluded.
const STAT_GLOSSARY: { header: string; tip: string }[] = [
  { header: "commits", tip: "Commits they authored in the selected window." },
  { header: "opened", tip: "Pull requests they opened in the window." },
  { header: "merged", tip: "Pull requests of theirs that were merged in the window." },
  {
    header: "PR size",
    tip: "Average lines changed (additions + deletions) across their merged pull requests. Blank when none have file data.",
  },
  {
    header: "issues resolved",
    tip: "Issues closed by their merged pull requests (work they finished, not issues they opened).",
  },
  {
    header: "cycle time",
    tip: "Median time from opening a pull request to merging it - how fast their work lands.",
  },
  {
    header: "active days",
    tip: "Distinct days they had activity in the window - a consistency read, not hours worked.",
  },
  { header: "reviews", tip: "Reviews they gave on other people's pull requests in the window." },
  { header: "approvals", tip: "Reviews of theirs that approved a pull request." },
  {
    header: "review lag",
    tip: "Median time from a pull request opening to this person's first review on it - how responsive they are. Lower is faster.",
  },
  { header: "in flight", tip: "Open pull requests they have authored right now (current load)." },
  {
    header: "self-merges",
    tip: "Pull requests they merged with no approving review - a review-hygiene signal.",
  },
  {
    header: "CI fails",
    tip: "Failed CI runs (automated test/build checks) on commits they authored. Attributed by commit, so read it as a signal, not a verdict.",
  },
  {
    header: "off-hours",
    tip: "Share of their activity on a weekend or outside 07:00-20:00 UTC. Timezones are unknown, so read it as a pattern.",
  },
]

// Look up a column's tooltip from the glossary by header text.
function tipFor(header: string): string {
  return STAT_GLOSSARY.find((g) => g.header === header)?.tip ?? ""
}

function statColumns(webBase: string | null): ColumnDef<PersonStatsView, unknown>[] {
  return [
    {
      accessorKey: "login",
      header: "person",
      cell: ({ row }) => (
        // max-w-*: table auto-layout sizes columns from cell content, so NameTag needs an explicit
        // max width to give its inner `truncate` a box to clip against.
        <NameTag
          login={row.original.login}
          src={avatarUrl(webBase, row.original.login)}
          className="max-w-[12rem] font-medium"
        />
      ),
    },
    {
      accessorKey: "commits",
      header: "commits",
      meta: { tip: tipFor("commits") },
      cell: ({ row }) => num(row.original.commits),
    },
    {
      accessorKey: "prs_opened",
      header: "opened",
      meta: { tip: tipFor("opened") },
      cell: ({ row }) => num(row.original.prs_opened),
    },
    {
      accessorKey: "prs_merged",
      header: "merged",
      meta: { tip: tipFor("merged") },
      cell: ({ row }) => num(row.original.prs_merged),
    },
    {
      accessorKey: "avg_pr_churn",
      header: "PR size",
      meta: { tip: tipFor("PR size") },
      cell: ({ row }) => fmtLines(row.original.avg_pr_churn),
    },
    {
      accessorKey: "issues_closed",
      header: "issues resolved",
      meta: { tip: tipFor("issues resolved") },
      cell: ({ row }) => num(row.original.issues_closed),
    },
    {
      accessorKey: "median_cycle_time_secs",
      header: "cycle time",
      meta: { tip: tipFor("cycle time") },
      cell: ({ row }) => fmtCycle(row.original.median_cycle_time_secs),
    },
    {
      accessorKey: "active_days",
      header: "active days",
      meta: { tip: tipFor("active days") },
      cell: ({ row }) => num(row.original.active_days),
    },
    {
      accessorKey: "reviews_given",
      header: "reviews",
      meta: { tip: tipFor("reviews") },
      cell: ({ row }) => num(row.original.reviews_given),
    },
    {
      accessorKey: "approvals_given",
      header: "approvals",
      meta: { tip: tipFor("approvals") },
      cell: ({ row }) => num(row.original.approvals_given),
    },
    {
      accessorKey: "median_review_latency_secs",
      header: "review lag",
      meta: { tip: tipFor("review lag") },
      cell: ({ row }) => fmtCycle(row.original.median_review_latency_secs),
    },
    {
      accessorKey: "open_prs",
      header: "in flight",
      meta: { tip: tipFor("in flight") },
      cell: ({ row }) => num(row.original.open_prs),
    },
    {
      accessorKey: "self_merges",
      header: "self-merges",
      meta: { tip: tipFor("self-merges") },
      cell: ({ row }) => num(row.original.self_merges),
    },
    {
      accessorKey: "ci_failures",
      header: "CI fails",
      meta: { tip: tipFor("CI fails") },
      cell: ({ row }) => num(row.original.ci_failures),
    },
    {
      // Off-hours share (weekend or outside 07:00-20:00 UTC). UTC-only, so read it as a pattern.
      id: "off_hours",
      accessorFn: (p) => (p.total_events > 0 ? p.off_hours_events / p.total_events : -1),
      header: "off-hours",
      meta: { tip: tipFor("off-hours") },
      cell: ({ row }) =>
        row.original.total_events > 0
          ? `${Math.round((row.original.off_hours_events / row.original.total_events) * 100)}%`
          : "-",
    },
  ]
}

// Framing and column legend above the People table: the figures are an operational read, not
// a ranking, and the reader can expand every column's meaning.
function StatsGuide() {
  const [open, setOpen] = useState(false)
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">
        These are operational signals to spot load, risk, and who has room to take on work - not a
        performance score or a ranking. Every figure is for the selected window only.
      </p>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-xs font-medium text-muted-foreground underline-offset-2 hover:underline"
      >
        {open ? "Hide column meanings" : "What do these columns mean?"}
      </button>
      {open && (
        <dl className="grid grid-cols-1 gap-x-6 gap-y-1 rounded-md border border-border bg-muted/30 p-3 text-xs sm:grid-cols-2">
          {STAT_GLOSSARY.map((g) => (
            <div key={g.header} className="flex gap-2">
              <dt className="shrink-0 font-medium text-foreground">{g.header}</dt>
              <dd className="text-muted-foreground">{g.tip}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

// Format a median PR cycle time (seconds): hours under a day, else days to one decimal.
function fmtCycle(secs: number | null): string {
  if (secs == null) return "-"
  if (secs < 3600) return `${Math.round(secs / 60)}m`
  if (secs < 86400) return `${Math.round(secs / 3600)}h`
  return `${(secs / 86400).toFixed(1)}d`
}

// Format an average line count: plain under 1000, else "1.2k". "-" when unmeasured.
function fmtLines(lines: number | null): string {
  if (lines == null) return "-"
  if (lines < 1000) return `${lines}`
  return `${(lines / 1000).toFixed(1)}k`
}

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
const HOURS = Array.from({ length: 24 }, (_, h) => h)

// 7x24 weekday-by-hour heatmap of when a person is active. Opacity scales with the busiest cell;
// weekend rows are tinted.
function Heatmap({ buckets }: { buckets: number[] }) {
  const max = Math.max(1, ...buckets)
  const total = buckets.reduce((n, c) => n + c, 0)
  if (total === 0) {
    return <p className="text-sm text-muted-foreground">no activity in this window</p>
  }
  return (
    <div className="space-y-1">
      <div className="flex gap-1 pl-8 text-[10px] text-muted-foreground">
        {[0, 6, 12, 18].map((h) => (
          <span key={h} className="w-[calc(6*0.875rem)]">
            {h}:00
          </span>
        ))}
      </div>
      {DAYS.map((day, d) => (
        <div key={day} className="flex items-center gap-1">
          <span className="w-7 text-[10px] text-muted-foreground">{day}</span>
          <div className={cn("flex gap-px rounded", (d === 0 || d === 6) && "bg-amber-50")}>
            {HOURS.map((h) => {
              const n = buckets[d * 24 + h]
              return (
                <span
                  key={`${day}-${h}`}
                  title={`${day} ${h}:00 UTC - ${n} event(s)`}
                  className="size-3.5 rounded-[2px]"
                  style={{
                    backgroundColor:
                      n === 0 ? "var(--muted)" : `rgba(37, 99, 235, ${0.15 + 0.85 * (n / max)})`,
                  }}
                />
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

function Skeleton() {
  return <div className="h-24 animate-pulse rounded-md bg-muted" />
}
