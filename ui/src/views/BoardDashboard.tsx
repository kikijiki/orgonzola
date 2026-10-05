import {
  type BoardCyclePhasesView,
  type BoardScorecardView,
  type BugFlowView,
  type DoraView,
  type EpicProgressView,
  type FlowMetricsView,
  type FlowTrendView,
  type InvestmentDistributionView,
  type RepoOverview,
  type SayDoView,
  type ScorecardView,
  type TrendDeltasView,
  type TrendPointView,
  type VelocityPointView,
  commands,
} from "@/bindings"
import { useToast } from "@/components/Toast"
import { CHART_COLORS, PhaseBar, TrendChart } from "@/components/charts"
import { RepoLink } from "@/components/links"
import {
  Card,
  EmptyState,
  IndexBadge,
  type IndexInfo,
  Skeleton,
  SkeletonCard,
} from "@/components/primitives"
import { COMMAND_TIMEOUT_MS, settledResult, withTimeout } from "@/lib/async"
import { attentionCount, formatDuration, repoLabel } from "@/lib/format"
import { ArrowDown, ArrowUp, ChevronRight, Minus } from "lucide-react"
import { useEffect, useState } from "react"

// Read-only dashboard over the board's per-repo digests: totals, per-repo WIP and cycle time,
// and trend sparklines over daily snapshots. Bars and sparklines are CSS/SVG, no chart library.
export function BoardDashboard({
  boardId,
  repos: allRepos,
  reposError,
  people,
  indexStates,
  dataVersion,
  webBase,
  onRepoChanged,
}: {
  boardId: string
  repos: RepoOverview[] | null
  // Set when the parent's overview fetch failed, to tell "loading" from "failed to load".
  reposError?: string | null
  people: number
  indexStates?: Record<string, IndexInfo>
  // Bumped when a sync/index pass completes, so the analytics refetch.
  dataVersion?: number
  webBase: string | null
  // Called after a per-repo setting changes, so the parent can refetch the overview.
  onRepoChanged?: () => void
}) {
  // Fold observed contributor forks out too, so they do not pad the bars. Pinned forks stay.
  const repos =
    allRepos == null ? null : allRepos.filter((r) => !(r.is_fork && r.ownership === "observed"))

  const [trend, setTrend] = useState<TrendPointView[]>([])
  const [dora, setDora] = useState<DoraView | null>(null)
  const [deltas, setDeltas] = useState<TrendDeltasView | null>(null)
  const [scorecard, setScorecard] = useState<BoardScorecardView | null>(null)
  const [flow, setFlow] = useState<FlowMetricsView | null>(null)
  const [bugs, setBugs] = useState<BugFlowView | null>(null)
  const [flowTrend, setFlowTrend] = useState<FlowTrendView[]>([])
  const [cyclePhases, setCyclePhases] = useState<BoardCyclePhasesView | null>(null)
  const [sayDo, setSayDo] = useState<SayDoView[]>([])
  const [investment, setInvestment] = useState<InvestmentDistributionView | null>(null)
  const [epics, setEpics] = useState<EpicProgressView[]>([])
  const [velocity, setVelocity] = useState<VelocityPointView[]>([])
  // True until the first analytics batch for this board resolves. Reset only on a board switch.
  const [loading, setLoading] = useState(true)
  const toast = useToast()
  // biome-ignore lint/correctness/useExhaustiveDependencies: show the skeleton only on a board switch
  useEffect(() => {
    setLoading(true)
  }, [boardId])
  // biome-ignore lint/correctness/useExhaustiveDependencies: dataVersion is a refetch trigger, not read
  useEffect(() => {
    // Drop a stale response if the board switches before this fetch resolves.
    let ignore = false
    // withTimeout bounds each call; allSettled + settledResult turn one rejection into that call's
    // error Result; finally clears `loading` on every path.
    void Promise.allSettled([
      withTimeout(commands.boardScorecard(boardId), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardTrend(boardId), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardDora(boardId), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardTrendSummary(boardId), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardFlowMetrics(boardId, 30, 0), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardBugFlow(boardId, 30, 0), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardFlowTrend(boardId), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardCyclePhases(boardId), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardSayDo(boardId), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardVelocityTrend(boardId), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardInvestment(boardId, 30, 0), COMMAND_TIMEOUT_MS),
      withTimeout(commands.boardEpics(boardId), COMMAND_TIMEOUT_MS),
    ])
      .then(([scR, trR, drR, suR, flR, bfR, ftR, cpR, sdR, vtR, ivR, epR]) => {
        if (ignore) return
        const sc = settledResult(scR)
        const tr = settledResult(trR)
        const dr = settledResult(drR)
        const su = settledResult(suR)
        const fl = settledResult(flR)
        const bf = settledResult(bfR)
        const ft = settledResult(ftR)
        const cp = settledResult(cpR)
        const sd = settledResult(sdR)
        const vt = settledResult(vtR)
        const iv = settledResult(ivR)
        const ep = settledResult(epR)
        if (sc.status === "ok") setScorecard(sc.data)
        if (tr.status === "ok") setTrend(tr.data)
        if (dr.status === "ok") setDora(dr.data)
        if (su.status === "ok") setDeltas(su.data)
        if (fl.status === "ok") setFlow(fl.data)
        if (bf.status === "ok") setBugs(bf.data)
        if (ft.status === "ok") setFlowTrend(ft.data)
        if (cp.status === "ok") setCyclePhases(cp.data)
        if (sd.status === "ok") setSayDo(sd.data)
        if (vt.status === "ok") setVelocity(vt.data)
        if (iv.status === "ok") setInvestment(iv.data)
        if (ep.status === "ok") setEpics(ep.data)
        const anyFailed = [sc, tr, dr, su, fl, bf, ft, cp, sd, vt, iv, ep].some(
          (r) => r.status !== "ok",
        )
        if (anyFailed) toast("Some dashboard analytics failed to load", "error")
      })
      .finally(() => {
        if (!ignore) setLoading(false)
      })
    return () => {
      ignore = true
    }
  }, [boardId, dataVersion, toast])
  if (repos == null) {
    // reposError distinguishes "failed to load" from "still loading".
    if (reposError != null) {
      return <EmptyState title="Could not load the board overview" hint={reposError} />
    }
    return <DashboardSkeleton />
  }
  if (repos.length === 0) {
    return (
      <EmptyState
        title="Nothing to chart yet"
        hint="Add people (or pin repos) under Settings, then Sync now."
      />
    )
  }

  const totals = repos.reduce(
    (acc, r) => ({
      wip: acc.wip + r.digest.wip,
      stale: acc.stale + r.digest.stale_open_prs,
      mwr: acc.mwr + r.digest.merged_without_review,
      attention: acc.attention + attentionCount(r),
    }),
    { wip: 0, stale: 0, mwr: 0, attention: 0 },
  )
  const maxWip = Math.max(1, ...repos.map((r) => r.digest.wip))
  const byWip = [...repos].sort((a, b) => b.digest.wip - a.digest.wip)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3 md:grid-cols-6">
        <Stat label="in flight" value={totals.wip} />
        <Stat label="needs attention" value={totals.attention} />
        <Stat label="stale PRs" value={totals.stale} />
        <Stat label="merged w/o review" value={totals.mwr} />
        <Stat label="repos" value={repos.length} />
        <Stat label="people" value={people} />
      </div>

      {loading ? (
        <>
          <SkeletonCard rows={3} />
          <SkeletonCard rows={2} />
        </>
      ) : (
        <>
          {scorecard && scorecard.repos.length > 0 && <ScorecardCard sc={scorecard} />}

          {deltas?.deltas.some((d) => d.delta !== 0 || d.anomaly) && <DeltaStrip deltas={deltas} />}

          {dora && <DoraTiles dora={dora} />}

          {flow && (flow.completed > 0 || flow.in_progress > 0) && <FlowCard flow={flow} />}

          {cyclePhases &&
            cyclePhases.median_pickup_secs != null &&
            cyclePhases.median_review_secs != null && <DeliveryCycleCard phases={cyclePhases} />}

          {bugs && (bugs.opened > 0 || bugs.closed > 0) && <BugFlowCard bugs={bugs} />}

          {sayDo.map((sd) => (
            <SayDoCard key={sd.sprint_id} sd={sd} />
          ))}

          {sayDo.length > 0 && <VelocityTrendCard points={velocity} />}

          {investment && investment.total_count > 0 && <InvestmentCard dist={investment} />}

          {epics.length > 0 && <EpicProgressCard epics={epics} />}
        </>
      )}

      {trend.length >= 2 && (
        <Card className="space-y-3">
          <h2 className="text-sm font-medium text-muted-foreground">
            Trend ({trend[0].captured_on} - {trend[trend.length - 1].captured_on})
          </h2>
          <TrendChart
            data={trend.map((t) => ({
              label: t.captured_on.slice(5),
              wip: t.wip,
              attention: t.attention_count,
            }))}
            series={[
              { key: "wip", label: "in flight", color: CHART_COLORS.blue },
              { key: "attention", label: "needs attention", color: CHART_COLORS.amber },
            ]}
          />
        </Card>
      )}

      {flowTrend.length >= 2 && (
        <Card className="space-y-3">
          <h2 className="text-sm font-medium text-muted-foreground">
            Work-item flow trend ({flowTrend[0].captured_on} -{" "}
            {flowTrend[flowTrend.length - 1].captured_on})
          </h2>
          <TrendChart
            data={flowTrend.map((t) => ({
              label: t.captured_on.slice(5),
              lead_days: t.lead_time_secs != null ? Math.round(t.lead_time_secs / 86400) : 0,
              wip: t.wip,
              bug_net: t.bug_net,
            }))}
            series={[
              { key: "lead_days", label: "lead time (days)", color: CHART_COLORS.emerald },
              { key: "wip", label: "in flight", color: CHART_COLORS.blue },
              { key: "bug_net", label: "bug net", color: CHART_COLORS.red },
            ]}
          />
          <p className="text-[10px] text-muted-foreground">
            Recorded daily from the board's linked issues (lead time = new {"->"} done; bug net =
            opened - closed). Accrues over days the app is open.
          </p>
        </Card>
      )}

      <Card className="space-y-3">
        <h2 className="text-sm font-medium text-muted-foreground">
          Work in flight (PRs) and cycle time, by repo
        </h2>
        <ul className="space-y-3">
          {byWip.map((r) => (
            <li key={r.repo_id} className="space-y-1">
              {/* flex-wrap + min-w-0 on the name, shrink-0 on the controls: a narrow row wraps. */}
              <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 text-sm">
                <span className="flex min-w-0 flex-1 items-center gap-2 font-medium">
                  <RepoLink webBase={webBase} fullName={r.full_name} className="min-w-0" />
                  <span className="shrink-0">
                    <IndexBadge {...indexStates?.[r.repo_id]} />
                  </span>
                  <IndexCodeToggle repo={r} onChanged={onRepoChanged} />
                </span>
                {/* Separate spans so each piece wraps on its own. */}
                <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5 font-mono text-xs text-muted-foreground">
                  <span>{r.digest.wip} in flight</span>
                  <span>cycle {formatDuration(r.digest.median_cycle_time_secs)}</span>
                  {(r.digest.median_pickup_secs != null || r.digest.median_review_secs != null) && (
                    <span>
                      pickup {formatDuration(r.digest.median_pickup_secs)} / review{" "}
                      {formatDuration(r.digest.median_review_secs)}
                    </span>
                  )}
                  {r.digest.review_wait > 0 && <span>{r.digest.review_wait} awaiting review</span>}
                  {attentionCount(r) > 0 && <span>{attentionCount(r)} flagged</span>}
                </span>
              </div>
              <Bar value={r.digest.wip} max={maxWip} className="bg-blue-400" />
              {/* Cycle time split into pickup (amber) and review (slate). */}
              <PhaseBar
                segments={[
                  {
                    label: `pickup ${formatDuration(r.digest.median_pickup_secs)}`,
                    value: r.digest.median_pickup_secs ?? 0,
                    color: CHART_COLORS.amber,
                  },
                  {
                    label: `review ${formatDuration(r.digest.median_review_secs)}`,
                    value: r.digest.median_review_secs ?? 0,
                    color: CHART_COLORS.slate,
                  },
                ]}
              />
            </li>
          ))}
        </ul>
      </Card>
    </div>
  )
}

// Per-repo code-indexing toggle. Off skips the code fetch and embed and drops the existing index.
function IndexCodeToggle({ repo, onChanged }: { repo: RepoOverview; onChanged?: () => void }) {
  const [busy, setBusy] = useState(false)
  return (
    <button
      type="button"
      disabled={busy}
      onClick={async () => {
        setBusy(true)
        const res = await commands.setRepoIndexCode(repo.repo_id, !repo.index_code)
        setBusy(false)
        if (res.status === "ok") onChanged?.()
      }}
      className="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground hover:bg-muted disabled:opacity-50"
      title={
        repo.index_code
          ? "Code search on - click to stop indexing this repo's code"
          : "Code search off - click to index this repo's code"
      }
    >
      {repo.index_code ? "code: on" : "code: off"}
    </button>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-border px-3 py-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-lg">{value}</div>
    </div>
  )
}

// Work-item flow: lead time and WIP from the status history of linked issues.
// Shown only when the board has issue activity.
function FlowCard({ flow }: { flow: FlowMetricsView }) {
  const tiles = [
    {
      label: "median lead time",
      value: flow.completed > 0 ? formatDuration(flow.median_lead_time_secs) : "-",
      // Cycle decomposition: wait = before work started, active = execution.
      sub:
        flow.median_wait_secs != null && flow.median_active_secs != null
          ? `${formatDuration(flow.median_wait_secs)} wait + ${formatDuration(flow.median_active_secs)} active`
          : `${flow.completed} completed`,
    },
    { label: "in progress", value: String(flow.in_progress), sub: "work items" },
    {
      label: "median age in progress",
      value: flow.in_progress > 0 ? formatDuration(flow.median_in_progress_age_secs) : "-",
      sub: "current",
    },
  ]
  return (
    <Card className="space-y-2">
      <h2 className="text-sm font-medium text-muted-foreground">Work-item flow (30-day window)</h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {tiles.map((t) => (
          <div key={t.label} className="rounded-md border border-border px-3 py-2">
            <div className="text-xs text-muted-foreground">{t.label}</div>
            <div className="font-mono text-lg">{t.value}</div>
            <div className="text-[10px] text-muted-foreground">{t.sub}</div>
          </div>
        ))}
      </div>
      {flow.median_wait_secs != null && flow.median_active_secs != null && (
        <div className="space-y-1">
          <PhaseBar
            segments={[
              {
                label: "waiting to start",
                value: flow.median_wait_secs,
                color: CHART_COLORS.amber,
              },
              { label: "active", value: flow.median_active_secs, color: CHART_COLORS.emerald },
            ]}
            height={10}
          />
          <p className="text-[10px] text-muted-foreground">
            cycle stages -{" "}
            {Math.round(
              (flow.median_wait_secs /
                Math.max(1, flow.median_wait_secs + flow.median_active_secs)) *
                100,
            )}
            % waiting to start, the rest active
          </p>
        </div>
      )}
      <p className="text-[10px] text-muted-foreground">
        From linked GitHub issues: new {"->"} in progress (first linked PR) {"->"} done.
      </p>
    </Card>
  )
}

// Delivery cycle stages: median PR pickup (open -> first review) vs review (-> merge) as a
// proportional bar. Shown only when both medians exist.
function DeliveryCycleCard({ phases }: { phases: BoardCyclePhasesView }) {
  const pickup = phases.median_pickup_secs ?? 0
  const review = phases.median_review_secs ?? 0
  return (
    <Card className="space-y-2">
      <h2 className="text-sm font-medium text-muted-foreground">Delivery cycle (PRs)</h2>
      <PhaseBar
        segments={[
          { label: "pickup (to first review)", value: pickup, color: CHART_COLORS.amber },
          { label: "review (to merge)", value: review, color: CHART_COLORS.blue },
        ]}
        height={10}
      />
      <p className="text-[10px] text-muted-foreground">
        pickup {formatDuration(phases.median_pickup_secs)} + review{" "}
        {formatDuration(phases.median_review_secs)} - a true board median over merged PRs.
      </p>
    </Card>
  )
}

// Bug inflow vs outflow: bug-labeled issues opened vs closed in the window. Positive net means
// a growing backlog. Shown only when the board has bugs.
function BugFlowCard({ bugs }: { bugs: BugFlowView }) {
  const net = bugs.opened - bugs.closed
  return (
    <Card className="space-y-2">
      <h2 className="text-sm font-medium text-muted-foreground">Bugs (30-day window)</h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-border px-3 py-2">
          <div className="text-xs text-muted-foreground">opened</div>
          <div className="font-mono text-lg">{bugs.opened}</div>
        </div>
        <div className="rounded-md border border-border px-3 py-2">
          <div className="text-xs text-muted-foreground">closed</div>
          <div className="font-mono text-lg">{bugs.closed}</div>
        </div>
        <div className="rounded-md border border-border px-3 py-2">
          <div className="text-xs text-muted-foreground">net</div>
          <div className={`font-mono text-lg ${net > 0 ? "text-red-700" : "text-emerald-700"}`}>
            {net > 0 ? `+${net}` : net}
          </div>
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground">
        Bug-labeled issues (any label containing "bug"). A positive net means the backlog is
        growing.
      </p>
    </Card>
  )
}

// Sprint say-do: committed at sprint start vs delivered, plus scope churn (items added or pulled
// after commit). Count-based. The ratio is shown only when something was committed.
function SayDoCard({ sd }: { sd: SayDoView }) {
  const pct = sd.ratio == null ? null : Math.round(sd.ratio * 100)
  const open = sd.committed - sd.delivered
  return (
    <Card className="space-y-2">
      <div className="flex items-baseline justify-between">
        <div>
          <h2 className="text-sm font-medium text-foreground">
            Commitment vs delivery (say-do) - {sd.sprint_name}
            {sd.state ? <span className="ml-1 text-[10px]">({sd.state})</span> : null}
          </h2>
          <p className="text-xs text-muted-foreground">
            How much of what the sprint committed to at its start actually got delivered.
          </p>
        </div>
        {pct != null && <span className="font-mono text-lg">{pct}%</span>}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-md border border-border px-3 py-2">
          <div className="text-xs text-muted-foreground">committed</div>
          <div className="font-mono text-lg">{sd.committed}</div>
        </div>
        <div className="rounded-md border border-border px-3 py-2">
          <div className="text-xs text-muted-foreground">delivered</div>
          <div className="font-mono text-lg text-emerald-700">{sd.delivered}</div>
        </div>
        <div className="rounded-md border border-border px-3 py-2">
          <div className="text-xs text-muted-foreground">added</div>
          <div className="font-mono text-lg">{sd.added > 0 ? `+${sd.added}` : sd.added}</div>
        </div>
        <div className="rounded-md border border-border px-3 py-2">
          <div className="text-xs text-muted-foreground">removed</div>
          <div className="font-mono text-lg">{sd.removed > 0 ? `-${sd.removed}` : sd.removed}</div>
        </div>
      </div>
      {sd.committed > 0 && (
        <PhaseBar
          segments={[
            { label: "delivered", value: sd.delivered, color: CHART_COLORS.emerald },
            { label: "carryover", value: open, color: CHART_COLORS.amber },
          ]}
          height={10}
        />
      )}
      <p className="text-[10px] text-muted-foreground">
        Committed = the sprint's issues when first seen active; delivered = those now done.
        Added/removed is scope changed after commit.
        {sd.committed_on ? ` Snapshot ${sd.committed_on}.` : ""}
      </p>
    </Card>
  )
}

// Sprints needed before the say-do ratio is drawn as a trend; two points are not a trend.
const MIN_VELOCITY_SPRINTS = 3

// Say-do velocity trend: delivered/committed ratio across sprints. Points arrive oldest-first from
// `board_velocity_trend` and are not re-sorted. The y-axis is fixed to 0-100, and each point
// carries its counts so a 1-of-1 sprint is not read as 20-of-20.
function VelocityTrendCard({ points }: { points: VelocityPointView[] }) {
  return (
    <Card className="space-y-2">
      <div>
        <h2 className="text-sm font-medium text-foreground">Say-do velocity trend</h2>
        <p className="text-xs text-muted-foreground">
          Delivery ratio across sprints. Heading up means improving predictability.
        </p>
      </div>
      {points.length < MIN_VELOCITY_SPRINTS ? (
        <p className="text-xs text-muted-foreground">
          Not enough history yet: a trend needs {MIN_VELOCITY_SPRINTS} sprints with a frozen
          commitment, and this board has {points.length}. The chart appears once more sprints have
          started.
        </p>
      ) : (
        <>
          <TrendChart
            data={points.map((p) => ({
              label: p.sprint_name,
              ratio: Math.round(p.ratio * 100),
            }))}
            series={[{ key: "ratio", label: "delivered %", color: CHART_COLORS.emerald }]}
            height={120}
            yDomain={[0, 100]}
          />
          <ul className="space-y-0.5 text-[10px] text-muted-foreground">
            {points.map((p) => (
              <li key={p.sprint_id} className="flex justify-between gap-2">
                <span className="truncate">{p.sprint_name}</span>
                <span className="font-mono shrink-0">
                  {p.delivered}/{p.committed} = {Math.round(p.ratio * 100)}%
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="text-[10px] text-muted-foreground">
        Percentage of committed items delivered per sprint (commitment frozen at sprint start). A
        small commitment moves the percentage a long way, so read it with the counts.
      </p>
    </Card>
  )
}

const INVESTMENT_COLOR: Record<string, string> = {
  feature: CHART_COLORS.emerald,
  bug: CHART_COLORS.red,
  maintenance: CHART_COLORS.amber,
  docs: CHART_COLORS.blue,
  test: CHART_COLORS.violet,
  other: CHART_COLORS.slate,
}

// Investment distribution: delivered effort by merged-PR count (churn secondary), as a
// proportional bar plus legend. The unclassified ("other") share shows coverage.
function InvestmentCard({ dist }: { dist: InvestmentDistributionView }) {
  const shown = dist.buckets.filter((b) => b.count > 0)
  const other = dist.buckets.find((b) => b.category === "other")
  const otherPct =
    dist.total_count > 0 && other ? Math.round((other.count / dist.total_count) * 100) : 0
  return (
    <Card className="space-y-2">
      <div>
        <h2 className="text-sm font-medium text-foreground">
          Where effort went (investment) - {dist.total_count} merged PRs, 30-day window
        </h2>
        <p className="text-xs text-muted-foreground">
          How the period's merged work split across features, bug fixes, maintenance, docs, and
          tests.
        </p>
      </div>
      <PhaseBar
        segments={shown.map((b) => ({
          label: `${b.category} ${b.count}`,
          value: b.count,
          color: INVESTMENT_COLOR[b.category] ?? CHART_COLORS.slate,
        }))}
        height={10}
      />
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {shown.map((b) => (
          <span key={b.category} className="flex items-center gap-1.5">
            <span
              className="inline-block size-2.5 rounded-sm"
              style={{ backgroundColor: INVESTMENT_COLOR[b.category] ?? CHART_COLORS.slate }}
            />
            <span className="text-muted-foreground">{b.category}</span>
            <span className="font-mono">
              {b.count}
              {b.churn > 0 ? ` (${b.churn} churn)` : ""}
            </span>
          </span>
        ))}
      </div>
      <p className="text-[10px] text-muted-foreground">
        Merged PRs by conventional-commit type (feat/fix/docs/...) + linked issue type, by count
        with churn (lines changed) secondary. Where the effort went, never who did it.
        {otherPct > 0
          ? ` ${otherPct}% unclassified (other) - not all PRs follow the convention.`
          : ""}
      </p>
    </Card>
  )
}

// Epic progress: each Jira epic's children done/total, with an at-risk marker (remaining work,
// nothing in progress). At-risk epics lead. Shown only when the board has Jira epics.
function EpicProgressCard({ epics }: { epics: EpicProgressView[] }) {
  return (
    <Card className="space-y-2">
      <h2 className="text-sm font-medium text-muted-foreground">Epic progress</h2>
      <ul className="space-y-2">
        {epics.map((e) => {
          const pct = e.total > 0 ? Math.round((e.done / e.total) * 100) : 0
          return (
            <li key={e.key} className="space-y-1">
              <div className="flex items-center justify-between gap-2 text-sm">
                <span className="flex min-w-0 items-center gap-2">
                  <span className="truncate">
                    <span className="font-mono text-xs text-muted-foreground">{e.key}</span>{" "}
                    {e.title}
                  </span>
                  {e.at_risk && (
                    <span
                      className="shrink-0 rounded bg-red-100 px-1.5 py-0.5 text-[10px] font-semibold text-red-800"
                      title="remaining work but nothing in progress - stalled"
                    >
                      at risk
                    </span>
                  )}
                </span>
                <span className="shrink-0 font-mono text-xs text-muted-foreground">
                  {e.done}/{e.total}
                  {e.in_progress > 0 ? ` (${e.in_progress} in progress)` : ""}
                </span>
              </div>
              <Bar
                value={e.done}
                max={e.total}
                className={e.at_risk ? "bg-red-400" : "bg-emerald-500"}
              />
            </li>
          )
        })}
      </ul>
      <p className="text-[10px] text-muted-foreground">
        Jira children rolled up to their parent epic, by count (no story points). At risk =
        remaining work with nothing in progress. Never a per-person figure.
      </p>
    </Card>
  )
}

// Whole-dashboard loading placeholder, so the page holds its shape while the overview loads.
function DashboardSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3 md:grid-cols-6">
        {["a", "b", "c", "d", "e", "f"].map((k) => (
          <Skeleton key={k} className="h-14" />
        ))}
      </div>
      <SkeletonCard rows={3} />
      <SkeletonCard rows={2} />
    </div>
  )
}

const SC_TIER_CLASS: Record<string, string> = {
  gold: "border-yellow-300 bg-yellow-100 text-yellow-800",
  silver: "border-slate-300 bg-slate-100 text-slate-700",
  bronze: "border-orange-300 bg-orange-100 text-orange-800",
  none: "border-red-300 bg-red-100 text-red-800",
}

// Board scorecard: composite Bronze/Silver/Gold (the weakest repo's tier) plus per-repo tier
// badges, each expandable to the failing rules.
function ScorecardCard({ sc }: { sc: BoardScorecardView }) {
  return (
    <Card className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-medium text-foreground">
            Health scorecard (Bronze / Silver / Gold)
          </h2>
          <p className="text-xs text-muted-foreground">
            A grade per repository from fixed rules; the board's grade is its weakest repository.
          </p>
        </div>
        <span
          className={`rounded-md border px-3 py-1.5 text-sm font-bold uppercase ${SC_TIER_CLASS[sc.composite_tier] ?? SC_TIER_CLASS.none}`}
        >
          {sc.composite_tier}
        </span>
      </div>
      <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
        <span>gold {sc.gold}</span>
        <span>silver {sc.silver}</span>
        <span>bronze {sc.bronze}</span>
        <span>none {sc.none}</span>
      </div>
      {/* Only this list scrolls, so many repos do not grow the card without bound. */}
      <ul className="max-h-72 space-y-1 overflow-y-auto pr-1">
        {sc.repos.map((r) => (
          <ScorecardRow key={r.repo_id} r={r} />
        ))}
      </ul>
    </Card>
  )
}

// One repo's scorecard row: tier badge always visible, failing/unknown rules behind a native
// `<details>`. Unknown means the rule's evidence was never observed; it still caps the tier but is
// counted and listed separately from failed.
function ScorecardRow({ r }: { r: ScorecardView }) {
  const failing = r.rules.filter((rule) => rule.status === "fail")
  const unknown = r.rules.filter((rule) => rule.status === "unknown")
  const tierBadge = (
    <span
      className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${SC_TIER_CLASS[r.tier] ?? SC_TIER_CLASS.none}`}
    >
      {r.tier}
    </span>
  )

  if (failing.length === 0 && unknown.length === 0) {
    return (
      <li className="rounded-md border border-border px-2.5 py-1.5 text-sm">
        <div className="flex items-center justify-between gap-2">
          <span className="min-w-0 truncate font-medium">{repoLabel(r.full_name)}</span>
          {tierBadge}
        </div>
      </li>
    )
  }

  return (
    <li className="rounded-md border border-border text-sm">
      <details className="group">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-2.5 py-1.5 [&::-webkit-details-marker]:hidden">
          <span className="flex min-w-0 items-center gap-1.5">
            <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform group-open:rotate-90" />
            <span className="min-w-0 truncate font-medium">{repoLabel(r.full_name)}</span>
          </span>
          <span className="flex shrink-0 items-center gap-1.5">
            {failing.length > 0 && (
              <span className="text-[10px] text-muted-foreground">{failing.length} failing</span>
            )}
            {unknown.length > 0 && (
              <span className="text-[10px] text-muted-foreground">{unknown.length} unknown</span>
            )}
            {tierBadge}
          </span>
        </summary>
        <ul className="space-y-0.5 px-2.5 pb-1.5">
          {failing.map((rule) => (
            <li key={rule.rule} className="text-xs text-muted-foreground">
              <span className="text-red-600">x</span> {rule.rule}{" "}
              <span className="opacity-70">({rule.detail})</span>
            </li>
          ))}
          {unknown.map((rule) => (
            <li key={rule.rule} className="text-xs text-muted-foreground">
              <span className="text-amber-600">?</span> {rule.rule}{" "}
              <span className="opacity-70">(unknown - {rule.detail})</span>
            </li>
          ))}
        </ul>
      </details>
    </li>
  )
}

const DELTA_LABELS: Record<string, string> = {
  wip: "in flight",
  attention: "needs attention",
  stale_prs: "stale PRs",
  merged_without_review: "merged w/o review",
}

// "What changed this week" strip: week-over-week delta per metric with a direction arrow and an
// anomaly badge.
function DeltaStrip({ deltas }: { deltas: TrendDeltasView }) {
  return (
    <Card className="space-y-2">
      <div>
        <h2 className="text-sm font-medium text-foreground">What changed (last {deltas.days}d)</h2>
        <p className="text-xs text-muted-foreground">
          These count problems, so rising (red, up arrow) is worse and falling (green, down arrow)
          is better.
        </p>
      </div>
      <div className="flex flex-wrap gap-2">
        {deltas.deltas.map((d) => {
          const up = d.delta > 0
          const flat = d.delta === 0
          const Arrow = flat ? Minus : up ? ArrowUp : ArrowDown
          // For these "bad" metrics, rising is red, falling is green.
          const color = flat ? "text-muted-foreground" : up ? "text-red-600" : "text-emerald-600"
          return (
            <div
              key={d.metric}
              className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-sm"
            >
              <span className="text-muted-foreground">{DELTA_LABELS[d.metric] ?? d.metric}</span>
              <span className={`flex items-center gap-1 font-mono ${color}`}>
                <Arrow className="size-3.5" aria-hidden />
                {d.current}
                {!flat && (
                  <span className="ml-1 text-xs opacity-70">
                    ({up ? "+" : ""}
                    {d.delta})
                  </span>
                )}
              </span>
              {d.anomaly && (
                <span
                  className="rounded bg-amber-100 px-1 text-[10px] font-semibold uppercase text-amber-800"
                  title={
                    d.seasonal
                      ? "unusual for this day of the week (vs the same weekday's recent baseline)"
                      : "unusual vs this board's recent baseline"
                  }
                >
                  spike
                </span>
              )}
            </div>
          )
        })}
      </div>
    </Card>
  )
}

const TIER_CLASS: Record<string, string> = {
  elite: "border-emerald-200 bg-emerald-50 text-emerald-800",
  high: "border-lime-200 bg-lime-50 text-lime-800",
  medium: "border-amber-200 bg-amber-50 text-amber-800",
  low: "border-red-200 bg-red-50 text-red-800",
  unknown: "border-border bg-muted/40 text-muted-foreground",
}

// DORA-lite tiles: deploy frequency, lead time and change failure rate (proxies).
function DoraTiles({ dora }: { dora: DoraView }) {
  const tiles = [
    {
      label: "deploy frequency",
      tier: dora.deploy_tier,
      value:
        dora.deploy_frequency_per_week == null
          ? "-"
          : `${dora.deploy_frequency_per_week.toFixed(1)}/wk`,
      proxy: false,
    },
    {
      label: "lead time",
      tier: dora.lead_tier,
      value: formatDuration(dora.lead_time_secs),
      // Real (merge -> deploy) when releases exist; only a proxy when falling back to cycle time.
      proxy: !dora.lead_from_deploys,
    },
    {
      label: "change failure rate",
      tier: dora.cfr_tier,
      value:
        dora.change_failure_rate == null ? "-" : `${(dora.change_failure_rate * 100).toFixed(0)}%`,
      proxy: true,
    },
  ]
  return (
    <Card className="space-y-2">
      <div>
        <h2 className="text-sm font-medium text-foreground">
          Delivery health (DORA) - {dora.window_days}-day window
        </h2>
        <p className="text-xs text-muted-foreground">
          Industry-standard delivery metrics: how often you ship, how fast a change reaches
          production, and how often it goes wrong. Tiles marked "estimated" are approximated from
          available data.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {tiles.map((t) => (
          <div
            key={t.label}
            className={`rounded-md border px-3 py-2 ${TIER_CLASS[t.tier] ?? TIER_CLASS.unknown}`}
          >
            <div className="flex items-baseline justify-between text-xs">
              <span>{t.label}</span>
              {t.proxy && (
                <span
                  className="rounded bg-black/5 px-1 text-[10px] uppercase opacity-70"
                  title="approximated from repo data, not measured deployments"
                >
                  estimated
                </span>
              )}
            </div>
            <div className="font-mono text-lg">{t.value}</div>
            <div className="text-[10px] uppercase tracking-wide opacity-70">{t.tier}</div>
          </div>
        ))}
      </div>
    </Card>
  )
}

// A horizontal CSS bar scaled to `max`, with a small width floor so non-zero values stay visible.
function Bar({ value, max, className }: { value: number; max: number; className: string }) {
  const pct = max > 0 ? Math.min(100, Math.max(value > 0 ? 3 : 0, (value / max) * 100)) : 0
  return (
    <div className="h-2 w-full rounded-full bg-muted">
      <div className={`h-2 rounded-full ${className}`} style={{ width: `${pct}%` }} />
    </div>
  )
}
