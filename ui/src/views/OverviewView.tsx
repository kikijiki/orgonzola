import type { AttentionView, BoardPrView, RepoOverview } from "@/bindings"
import { commands } from "@/bindings"
import { AiSummary } from "@/components/AiSummary"
import { DataTable } from "@/components/DataTable"
import { useToast } from "@/components/Toast"
import { RepoLink, UserLink } from "@/components/links"
import {
  Card,
  EmptyState,
  ExtLink,
  IndexBadge,
  type IndexInfo,
  SkeletonCard,
  SubTabs,
  ViewHeader,
} from "@/components/primitives"
import { Badge } from "@/components/ui/badge"
import {
  actorLoginsOf,
  entityHref,
  evidenceOf,
  prSubjectOf,
  subjectKey,
  subjectOf,
} from "@/lib/attention"
import {
  attentionMeaning,
  attentionStyle,
  formatDuration,
  relativeAge,
  repoLabel,
} from "@/lib/format"
import { useBoardPref } from "@/lib/useBoardPref"
import { cn } from "@/lib/utils"
import type { ColumnDef } from "@tanstack/react-table"
import { useEffect, useState } from "react"

// The attention scope window: how far back an item can be and still matter. `null` days = all time.
const WINDOWS: { label: string; days: number | null }[] = [
  { label: "All time", days: null },
  { label: "Last year", days: 365 },
  { label: "Last quarter", days: 90 },
  { label: "Last month", days: 30 },
  { label: "Last 2 weeks", days: 14 },
  { label: "Last week", days: 7 },
  { label: "Last day", days: 1 },
]

// Attention kinds in display order (also the order the Summary breakdown lists them).
const KIND_ORDER = [
  "review_wait",
  "stale_pr",
  "risky_change",
  "aging_wip",
  "merged_without_review",
  "failing_ci",
  "flaky_ci",
  "done_not_done",
  "orphan_pr",
]

// Whether an item's timestamp is within the scope window. An item with no timestamp is kept, and
// `cutoff == null` (all time) keeps everything.
function withinWindow(ts: string | null, cutoff: string | null): boolean {
  return cutoff == null || ts == null || ts >= cutoff
}

// The outbound link for an item: its subject entity's forge URL, falling back to the
// constructible run page when the forge reported no URL for a CI run. Null means plain text.
function itemHref(a: AttentionView, webBase: string | null): string | null {
  const subject = subjectOf(a)
  if (!subject) return null
  const url = entityHref(subject)
  if (url) return url
  if (subject.type === "ci_run" && webBase && a.repo_full_name) {
    return `${webBase}/${a.repo_full_name}/actions/runs/${subject.id}`
  }
  return null
}

// The attention-first home: a control row (scope window), then subtabs: a Summary plus one per
// flagged kind, each with its item count. PR-based kinds get rich columns from each item's typed
// envelope, except the review tally, which joins the board's PR list. `teamRefs` holds the subject
// keys of items involving the board's people, shown as a "team" tag.
export function OverviewView({
  boardId,
  repos,
  teamRefs,
  indexStates,
  dataVersion,
  webBase,
  onOpenPerson,
  onManageRepos,
}: {
  boardId: string
  repos: RepoOverview[] | null
  teamRefs?: Set<string>
  indexStates?: Record<string, IndexInfo>
  // Bumped when a sync/index pass completes, so the PR join refetches.
  dataVersion?: number
  // The board's forge web root and the in-app "go to this user" nav.
  webBase: string | null
  onOpenPerson: (login: string) => void
  onManageRepos: () => void
}) {
  // Scope window: drop attention items older than the cutoff. Persisted per board.
  const [winIdx, setWinIdx] = useBoardPref<number>(boardId, "scopeWindow", 0)
  const windowDays = WINDOWS[winIdx]?.days ?? null
  const cutoff =
    windowDays == null ? null : new Date(Date.now() - windowDays * 86_400_000).toISOString()

  // Active subtab: "summary" or a kind id. Resets to Summary on a board switch.
  const [active, setActive] = useState<string>("summary")
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset only on a board switch
  useEffect(() => setActive("summary"), [boardId])

  // The board's PRs, to add author and review tally to PR-kind rows in one call. Rows render
  // un-enriched until it lands, and for any PR not in the list.
  const [prs, setPrs] = useState<BoardPrView[] | null>(null)
  const toast = useToast()
  // biome-ignore lint/correctness/useExhaustiveDependencies: dataVersion is a refetch trigger, not read
  useEffect(() => {
    let ignore = false
    void commands.boardChanges(boardId).then((res) => {
      if (!ignore) {
        if (res.status === "ok") setPrs(res.data)
        else
          toast(
            "Could not load pull request details - attention items will show limited information",
            "error",
          )
      }
    })
    return () => {
      ignore = true
    }
  }, [boardId, dataVersion, toast])
  const prById = new Map((prs ?? []).map((p) => [p.id, p]))

  // Fold observed contributor forks: they are PR-staging areas, and their CI failures are noise.
  // Kept out of the flagged list (noted on the Summary). A pinned fork stays.
  const isFolded = (r: RepoOverview) => r.is_fork && r.ownership === "observed"
  const mainRepos = (repos ?? []).filter((r) => !isFolded(r))
  const forkRepos = (repos ?? []).filter(isFolded)

  const items = mainRepos.flatMap((r) => r.attention).filter((a) => withinWindow(a.ts, cutoff))
  const byKind = new Map<string, AttentionView[]>()
  for (const a of items) {
    const list = byKind.get(a.kind)
    if (list) list.push(a)
    else byKind.set(a.kind, [a])
  }
  const presentKinds = KIND_ORDER.filter((k) => (byKind.get(k)?.length ?? 0) > 0)
  const total = items.length
  const hasItems = (r: RepoOverview) => r.attention.some((a) => withinWindow(a.ts, cutoff))
  const flaggedRepos = mainRepos.filter(hasItems)
  const upstreamRepos = mainRepos.filter((r) => r.upstream.length > 0)
  // All-clear = no in-window items and no upstream alert, so an upstream-only repo shows once.
  const clearRepos = mainRepos.filter((r) => !hasItems(r) && r.upstream.length === 0)

  // The active kind, or null (Summary) if none, or if the chosen kind emptied out.
  const activeKind = active !== "summary" && presentKinds.includes(active) ? active : null
  const subtabs = [
    { id: "summary", label: "Summary" },
    ...presentKinds.map((k) => ({
      id: k,
      label: attentionStyle(k).label,
      count: byKind.get(k)?.length,
    })),
  ]

  return (
    <div className="space-y-4">
      <ViewHeader
        subtitle={
          repos == null
            ? "loading..."
            : total === 0
              ? `nothing needs attention across ${mainRepos.length} repo(s)${windowDays == null ? "" : ` in the ${WINDOWS[winIdx].label.toLowerCase()}`}`
              : `${total} item(s) need attention across ${flaggedRepos.length} of ${mainRepos.length} repo(s)`
        }
      />

      {/* Scope window (time axis). Enabling or disabling a signal lives in Settings > Signals. */}
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5 text-muted-foreground">
          Scope
          <select
            className="h-9 rounded-md border border-border bg-background px-2 text-sm text-foreground"
            value={winIdx}
            onChange={(e) => setWinIdx(Number(e.target.value))}
          >
            {WINDOWS.map((w, i) => (
              <option key={w.label} value={i}>
                {w.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {repos == null && (
        <>
          <SkeletonCard rows={2} />
          <SkeletonCard rows={2} />
        </>
      )}

      {repos != null && mainRepos.length === 0 && forkRepos.length === 0 && (
        <EmptyState
          title="Nothing in view yet"
          hint="Add people (and optionally pin repos) under Settings, then Sync now to populate the board."
        />
      )}

      {repos != null && (mainRepos.length > 0 || forkRepos.length > 0) && (
        <>
          <SubTabs tabs={subtabs} active={activeKind ?? "summary"} onSelect={setActive} />

          {activeKind == null ? (
            <div className="space-y-4">
              <AiSummary boardId={boardId} />
              <AttentionSummary
                presentKinds={presentKinds}
                byKind={byKind}
                total={total}
                clearRepos={clearRepos}
                upstreamRepos={upstreamRepos}
                forkRepos={forkRepos}
                indexStates={indexStates}
                webBase={webBase}
                onSelectKind={setActive}
                onManageRepos={onManageRepos}
              />
            </div>
          ) : (
            <KindTable
              kind={activeKind}
              items={byKind.get(activeKind) ?? []}
              prById={prById}
              teamRefs={teamRefs}
              webBase={webBase}
              onOpenPerson={onOpenPerson}
            />
          )}
        </>
      )}
    </div>
  )
}

// The Summary subtab: per-kind breakdown chips, upstream alerts, all-clear repos, and folded forks.
function AttentionSummary({
  presentKinds,
  byKind,
  total,
  clearRepos,
  upstreamRepos,
  forkRepos,
  indexStates,
  webBase,
  onSelectKind,
  onManageRepos,
}: {
  presentKinds: string[]
  byKind: Map<string, AttentionView[]>
  total: number
  clearRepos: RepoOverview[]
  upstreamRepos: RepoOverview[]
  forkRepos: RepoOverview[]
  indexStates?: Record<string, IndexInfo>
  webBase: string | null
  onSelectKind: (kind: string) => void
  onManageRepos: () => void
}) {
  return (
    <div className="space-y-4">
      {total > 0 && (
        <Card className="space-y-2">
          <h2 className="text-sm font-medium text-muted-foreground">
            {total} item(s) need attention - by type
          </h2>
          <div className="flex flex-wrap gap-2">
            {presentKinds.map((k) => {
              const style = attentionStyle(k)
              return (
                <button
                  key={k}
                  type="button"
                  onClick={() => onSelectKind(k)}
                  title={attentionMeaning(k)}
                  className={cn(
                    "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs hover:opacity-80",
                    style.className,
                  )}
                >
                  {style.label}
                  <span className="font-mono font-semibold">{byKind.get(k)?.length}</span>
                </button>
              )
            })}
          </div>
        </Card>
      )}

      {total === 0 && (
        <Card>
          <p className="text-sm text-muted-foreground">Nothing needs attention in this window.</p>
        </Card>
      )}

      {upstreamRepos.length > 0 && (
        <Card className="space-y-2">
          <h2 className="text-sm font-medium text-muted-foreground">
            Upstream alerts ({upstreamRepos.length})
          </h2>
          <p className="text-xs text-muted-foreground">
            A pinned fork whose parent repo has flagged items - the work shows on the upstream.
          </p>
          <ul className="space-y-1 text-sm">
            {upstreamRepos.map((r) => (
              <li key={r.repo_id} className="flex items-center justify-between gap-2">
                <RepoLink webBase={webBase} fullName={r.full_name} />
                <span className="font-mono text-xs text-muted-foreground">
                  {r.upstream
                    .map((u) => `${repoLabel(u.repo_id)} (${u.attention_count})`)
                    .join(", ")}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {clearRepos.length > 0 && (
        <Card className="space-y-2">
          <h2 className="text-sm font-medium text-muted-foreground">
            All clear ({clearRepos.length})
          </h2>
          <ul className="space-y-1">
            {clearRepos.map((repo) => (
              <li key={repo.repo_id} className="flex items-center justify-between gap-2 text-sm">
                <span className="flex items-center gap-2">
                  <RepoLink webBase={webBase} fullName={repo.full_name} />
                  <IndexBadge {...indexStates?.[repo.repo_id]} />
                </span>
                <span className="font-mono text-xs text-muted-foreground">
                  {repo.digest.wip} in flight - cycle{" "}
                  {formatDuration(repo.digest.median_cycle_time_secs)}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {forkRepos.length > 0 && (
        <Card className="space-y-2">
          <h2 className="text-sm font-medium text-muted-foreground">
            Contributor forks ({forkRepos.length})
          </h2>
          <p className="text-xs text-muted-foreground">
            Personal forks used to open PRs - folded out of the flagged list (the work shows on the
            upstream, and a fork's own CI is not the project's). Pin one under Settings to track it
            as its own repo.
          </p>
          <ul className="space-y-1">
            {forkRepos.map((r) => (
              <li key={r.repo_id} className="flex items-center justify-between gap-2 text-sm">
                <RepoLink webBase={webBase} fullName={r.full_name} />
                {r.parent_full_name && (
                  <span className="shrink-0 font-mono text-xs text-muted-foreground">
                    fork of <RepoLink webBase={webBase} fullName={r.parent_full_name} />
                  </span>
                )}
              </li>
            ))}
          </ul>
        </Card>
      )}

      <button
        type="button"
        onClick={onManageRepos}
        className="text-sm text-muted-foreground underline underline-offset-2 hover:text-foreground"
      >
        Board settings
      </button>
    </div>
  )
}

// One kind's items as a table: the next-step action under the head, then the rows. PR kinds get
// rich columns joined from the board PR list; CI and aging-WIP kinds show summary, repo, and age.
function KindTable({
  kind,
  items,
  prById,
  teamRefs,
  webBase,
  onOpenPerson,
}: {
  kind: string
  items: AttentionView[]
  prById: Map<string, BoardPrView>
  teamRefs?: Set<string>
  webBase: string | null
  onOpenPerson: (login: string) => void
}) {
  const isPr = items.every((a) => prSubjectOf(a) != null)
  // Only kinds that carry supporting facts (done_not_done, risky_change) get the "why" column.
  const hasEvidence = items.some((a) => evidenceOf(a).length > 0)
  // The action is the same for every item of a kind; show it once.
  const action = items.find((a) => a.action)?.action ?? ""
  // Oldest-first surfaces the most stale; a CI run reads best newest-first.
  const sortDesc = kind === "failing_ci" || kind === "flaky_ci"
  const meaning = attentionMeaning(kind)
  return (
    <Card className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-sm font-medium text-foreground">
          {attentionStyle(kind).label} ({items.length})
        </h2>
        {action && <p className="text-xs text-muted-foreground">{action}</p>}
      </div>
      {meaning && <p className="text-xs text-muted-foreground">{meaning}</p>}
      <DataTable
        columns={
          isPr
            ? prColumns(prById, teamRefs, webBase, onOpenPerson, hasEvidence)
            : itemColumns(webBase)
        }
        data={items}
        initialSorting={[{ id: "ts", desc: sortDesc }]}
        emptyMessage="nothing here"
      />
    </Card>
  )
}

// The "why" column: the item's evidence entities. done_not_done shows the still-open work item;
// risky_change shows the dormant files the PR touches, as plain text with last-changed date.
function evidenceColumn(): ColumnDef<AttentionView, unknown> {
  return {
    id: "evidence",
    header: "why",
    cell: ({ row }) => {
      const evidence = evidenceOf(row.original)
      if (evidence.length === 0) return <span className="text-muted-foreground">-</span>
      return (
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          {evidence.map((e) =>
            e.type === "work_item" ? (
              <ExtLink key={`wi:${e.id}`} href={e.url}>
                <span className="font-mono text-xs">#{e.number}</span>{" "}
                <span className="max-w-40 truncate align-middle">{e.title}</span>
              </ExtLink>
            ) : e.type === "source_file" ? (
              <span
                key={`f:${e.path}`}
                className="font-mono text-xs text-muted-foreground"
                title={
                  e.last_changed_at
                    ? `last changed by a merged PR ${relativeAge(e.last_changed_at)}`
                    : undefined
                }
              >
                {e.path}
              </span>
            ) : null,
          )}
        </span>
      )
    },
  }
}

// Rich columns for a PR-based kind. Number, title, link and author come from the item's envelope.
// The review tally joins the board PR list (`prById`).
function prColumns(
  prById: Map<string, BoardPrView>,
  teamRefs: Set<string> | undefined,
  webBase: string | null,
  onOpenPerson: (login: string) => void,
  hasEvidence: boolean,
): ColumnDef<AttentionView, unknown>[] {
  return [
    {
      id: "number",
      accessorFn: (a) => prSubjectOf(a)?.number ?? 0,
      header: "#",
      cell: ({ row }) => {
        const pr = prSubjectOf(row.original)
        return (
          <ExtLink href={itemHref(row.original, webBase)}>
            <span className="font-mono text-xs">{pr ? `#${pr.number}` : "-"}</span>
          </ExtLink>
        )
      },
    },
    {
      id: "title",
      header: "title",
      cell: ({ row }) => {
        const a = row.original
        const pr = prSubjectOf(a)
        const key = subjectKey(a)
        const byTeam = key != null && teamRefs?.has(key)
        return (
          <span className="flex items-center gap-1.5">
            <ExtLink href={itemHref(a, webBase)}>
              <span className="block max-w-md truncate">{pr?.title ?? a.summary}</span>
            </ExtLink>
            {byTeam && (
              <Badge variant="outline" className="border-blue-200 bg-blue-50 text-blue-700">
                team
              </Badge>
            )}
          </span>
        )
      },
    },
    {
      id: "repo",
      header: "repo",
      cell: ({ row }) =>
        row.original.repo_full_name ? (
          <RepoLink webBase={webBase} fullName={row.original.repo_full_name} />
        ) : (
          <span className="text-muted-foreground">-</span>
        ),
    },
    {
      id: "author",
      accessorFn: (a) => actorLoginsOf(a)[0] ?? "",
      header: "author",
      cell: ({ row }) => {
        const login = actorLoginsOf(row.original)[0]
        return login ? (
          <UserLink login={login} webBase={webBase} onOpenPerson={onOpenPerson} />
        ) : (
          <span className="text-muted-foreground">-</span>
        )
      },
    },
    ...(hasEvidence ? [evidenceColumn()] : []),
    {
      accessorKey: "ts",
      header: "age",
      cell: ({ row }) => (
        <span className="font-mono text-xs" title={row.original.ts ?? ""}>
          {relativeAge(row.original.ts)}
        </span>
      ),
    },
    {
      id: "reviews",
      accessorFn: (a) => prById.get(prSubjectOf(a)?.id ?? "")?.review_count ?? 0,
      header: "reviews",
      cell: ({ row }) => {
        const pr = prById.get(prSubjectOf(row.original)?.id ?? "")
        return (
          <span className="font-mono text-xs">
            {pr ? pr.review_count : "-"}
            {pr?.approved ? " (approved)" : ""}
          </span>
        )
      },
    },
  ]
}

// Columns for a non-PR kind (failing CI, aging WIP): summary linked out, repo, and age.
function itemColumns(webBase: string | null): ColumnDef<AttentionView, unknown>[] {
  return [
    {
      id: "item",
      header: "item",
      cell: ({ row }) => (
        <ExtLink href={itemHref(row.original, webBase)}>
          <span className="block max-w-xl truncate">{row.original.summary}</span>
        </ExtLink>
      ),
    },
    {
      id: "repo",
      header: "repo",
      cell: ({ row }) =>
        row.original.repo_full_name ? (
          <RepoLink webBase={webBase} fullName={row.original.repo_full_name} />
        ) : (
          <span className="text-muted-foreground">-</span>
        ),
    },
    {
      accessorKey: "ts",
      header: "age",
      cell: ({ row }) => {
        const { ts, wip_percentile, wip_percentile_basis } = row.original
        const bandLabel =
          wip_percentile === "over_p90"
            ? "> P90"
            : wip_percentile === "over_p75"
              ? "> P75"
              : wip_percentile === "over_p50"
                ? "> P50"
                : null
        const bandStyle =
          wip_percentile === "over_p90"
            ? "border-red-200 bg-red-50 text-red-700"
            : wip_percentile === "over_p75"
              ? "border-orange-200 bg-orange-50 text-orange-700"
              : "border-yellow-200 bg-yellow-50 text-yellow-700"
        return (
          <span className="flex items-center gap-1.5">
            <span className="font-mono text-xs" title={ts ?? ""}>
              {relativeAge(ts)}
            </span>
            {bandLabel && (
              <Badge
                variant="outline"
                className={bandStyle}
                // The core computes the basis sentence, so the number never travels without it.
                title={wip_percentile_basis ?? undefined}
              >
                {bandLabel}
              </Badge>
            )}
          </span>
        )
      },
    },
  ]
}
