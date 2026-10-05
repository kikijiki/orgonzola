import {
  type BoardJiraView,
  type BoardPrView,
  type BoardStandupView,
  type IssueRefView,
  type LinkCoverageView,
  type PrRefView,
  commands,
} from "@/bindings"
import { AiSummary } from "@/components/AiSummary"
import { DataTable } from "@/components/DataTable"
import { useToast } from "@/components/Toast"
import { RepoLink, UserLink } from "@/components/links"
import { Markdown } from "@/components/markdown"
import { Card, EmptyState, ExtLink, SubTabs } from "@/components/primitives"
import { formatDate, relativeAge } from "@/lib/format"
import type { ColumnDef } from "@tanstack/react-table"
import { useEffect, useState } from "react"

// Standup window options as `since..until` day offsets from now (until 0 = now). "this week" is
// the last 7 days; "last week" the 7 days before that.
const WINDOWS = [
  { label: "This day", since: 1, until: 0 },
  { label: "This week", since: 7, until: 0 },
  { label: "This 2 weeks", since: 14, until: 0 },
  { label: "This month", since: 30, until: 0 },
  { label: "Last day", since: 2, until: 1 },
  { label: "Last week", since: 14, until: 7 },
  { label: "Last 2 weeks", since: 28, until: 14 },
  { label: "Last month", since: 60, until: 30 },
]

// The standup for a whole board: a retrospective of what moved this window (merged PRs, PRs in
// review, delivered and in-progress issues, Jira) split into subtabs behind a Summary. It does not
// re-list the attention queue, only points at it. Changing the window rebuilds.
export function BoardStandup({
  boardId,
  webBase,
  dataVersion,
  onOpenPerson,
  onOpenAttention,
}: {
  boardId: string
  webBase: string | null
  // Bumped by the parent on a sync/index pass so the agenda and PR join refetch.
  dataVersion?: number
  onOpenPerson: (login: string) => void
  onOpenAttention: () => void
}) {
  const [win, setWin] = useState(1) // default "This week"
  const [active, setActive] = useState<string>("summary")
  const [agenda, setAgenda] = useState<BoardStandupView | null>(null)
  const [coverage, setCoverage] = useState<LinkCoverageView | null>(null)
  const [jira, setJira] = useState<BoardJiraView | null>(null)
  const [prs, setPrs] = useState<BoardPrView[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  // Rendered markdown digest, shown and copied on demand; null = hidden.
  const [digest, setDigest] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [sendState, setSendState] = useState<"idle" | "sending" | "sent" | string>("idle")
  const toast = useToast()

  // Reset to the Summary subtab on a board switch.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset only on a board switch
  useEffect(() => setActive("summary"), [boardId])

  // Build the digest (no egress), show it, and copy it to the clipboard.
  const copyDigest = async () => {
    const w = WINDOWS[win]
    const res = await commands.boardDigest(boardId, w.since, w.until)
    if (res.status !== "ok") return
    setDigest(res.data)
    try {
      await navigator.clipboard.writeText(res.data)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard blocked; the text is still shown for manual copy.
    }
  }

  // Push the digest to the configured webhook: the only egress, on explicit click.
  const sendDigest = async () => {
    const w = WINDOWS[win]
    setSendState("sending")
    const res = await commands.sendBoardDigest(boardId, w.since, w.until)
    if (res.status === "ok") {
      setSendState("sent")
      toast("Digest sent to your Slack channel", "success")
      setTimeout(() => setSendState("idle"), 2500)
    } else {
      setSendState(res.error)
      toast("Couldn't send the digest - check the webhook under Settings", "error")
    }
  }

  useEffect(() => {
    // Ignore a late response from a previous board/window.
    let ignore = false
    const w = WINDOWS[win]
    setAgenda(null)
    setError(null)
    commands.boardStandup(boardId, w.since, w.until).then((res) => {
      if (ignore) return
      if (res.status === "ok") setAgenda(res.data)
      else setError(res.error)
    })
    return () => {
      ignore = true
    }
  }, [boardId, win])

  // Link coverage: how well the board's merged PRs tie to the work-item graph. The board PR list
  // enriches the Merged / In review rows.
  // biome-ignore lint/correctness/useExhaustiveDependencies: dataVersion is a refetch trigger, not read
  useEffect(() => {
    let ignore = false
    void commands.boardLinkCoverage(boardId).then((res) => {
      if (!ignore) {
        if (res.status === "ok") setCoverage(res.data)
        else toast("Could not load link coverage data", "error")
      }
    })
    void commands.boardChanges(boardId).then((res) => {
      if (!ignore) {
        if (res.status === "ok") setPrs(res.data)
        else toast("Could not load pull request details", "error")
      }
    })
    void commands.boardJira(boardId).then((res) => {
      if (!ignore) {
        if (res.status === "ok") setJira(res.data)
        else toast("Could not load Jira data", "error")
      }
    })
    return () => {
      ignore = true
    }
  }, [boardId, dataVersion, toast])

  const prById = new Map((prs ?? []).map((p) => [p.id, p]))
  const hasJira = jira != null && jira.projects.length > 0
  const blocked = agenda?.needs_attention.length ?? 0

  const subtabs =
    agenda == null
      ? [{ id: "summary", label: "Summary" }]
      : [
          { id: "summary", label: "Summary" },
          { id: "merged", label: "Merged", count: agenda.merged_prs.length },
          { id: "in_review", label: "In review", count: agenda.waiting_on_review.length },
          { id: "delivered", label: "Delivered", count: agenda.delivered_issues.length },
          { id: "in_progress", label: "In progress", count: agenda.in_progress_issues.length },
          ...(hasJira ? [{ id: "jira", label: "Jira", count: jira.tickets.length }] : []),
        ]
  const activeId = subtabs.some((t) => t.id === active) ? active : "summary"

  return (
    <div className="space-y-4">
      {/* Window selector and digest copy/send actions; they act on the whole window, so they sit
          above the subtabs. */}
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
        {agenda != null && (
          <p className="text-sm text-muted-foreground">
            {WINDOWS[win].label}: {formatDate(agenda.since)} - {formatDate(agenda.until)}
          </p>
        )}
        {agenda != null && (
          <button
            type="button"
            onClick={copyDigest}
            className="h-9 rounded-md border border-border bg-background px-3 text-sm text-muted-foreground hover:text-foreground"
          >
            {copied ? "Copied" : "Copy digest"}
          </button>
        )}
        {agenda != null && (
          <button
            type="button"
            onClick={sendDigest}
            disabled={sendState === "sending"}
            title="Send this digest to the Slack channel configured under Settings > Digest"
            className="h-9 rounded-md border border-border bg-background px-3 text-sm text-muted-foreground hover:text-foreground disabled:opacity-60"
          >
            {sendState === "sending"
              ? "Sending..."
              : sendState === "sent"
                ? "Sent"
                : "Send to Slack"}
          </button>
        )}
      </div>
      {typeof sendState === "string" &&
        sendState !== "idle" &&
        sendState !== "sending" &&
        sendState !== "sent" && <p className="text-xs text-red-600">{sendState}</p>}

      {digest != null && (
        <Card className="space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-muted-foreground">Digest preview</h2>
            <button
              type="button"
              onClick={() => setDigest(null)}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              hide
            </button>
          </div>
          {/* Render the markdown (Copy/Send still use the raw text). */}
          <div className="max-h-80 overflow-auto rounded-md bg-muted p-3 text-sm">
            {digest ? <Markdown>{digest}</Markdown> : "Nothing to report in this window."}
          </div>
        </Card>
      )}

      {error != null ? (
        <EmptyState title="Could not build the standup" hint={error} />
      ) : agenda == null ? (
        <EmptyState title="Building the agenda..." />
      ) : (
        <>
          <SubTabs tabs={subtabs} active={activeId} onSelect={setActive} />

          {activeId === "summary" && (
            <StandupSummary
              boardId={boardId}
              agenda={agenda}
              coverage={coverage}
              blocked={blocked}
              onOpenAttention={onOpenAttention}
              onSelect={setActive}
            />
          )}
          {activeId === "merged" && (
            <Card className="space-y-2">
              <h2 className="text-sm font-medium text-foreground">
                Merged ({agenda.merged_prs.length}) - {agenda.moved_commits} commits
              </h2>
              <DataTable
                columns={prRefColumns(prById, webBase, onOpenPerson, "merged_at")}
                data={agenda.merged_prs}
                emptyMessage="nothing merged"
              />
            </Card>
          )}
          {activeId === "in_review" && (
            <Card className="space-y-2">
              <h2 className="text-sm font-medium text-foreground">
                Waiting on review ({agenda.waiting_on_review.length})
              </h2>
              <DataTable
                columns={prRefColumns(prById, webBase, onOpenPerson, "created_at")}
                data={agenda.waiting_on_review}
                emptyMessage="nothing waiting"
              />
            </Card>
          )}
          {activeId === "delivered" && (
            <Card className="space-y-2">
              <h2 className="text-sm font-medium text-foreground">
                Delivered ({agenda.delivered_issues.length})
              </h2>
              <DataTable
                columns={issueRefColumns()}
                data={agenda.delivered_issues}
                emptyMessage="none closed"
              />
            </Card>
          )}
          {activeId === "in_progress" && (
            <Card className="space-y-2">
              <h2 className="text-sm font-medium text-foreground">
                In progress ({agenda.in_progress_issues.length})
              </h2>
              <DataTable
                columns={issueRefColumns()}
                data={agenda.in_progress_issues}
                emptyMessage="nothing in progress"
              />
            </Card>
          )}
          {activeId === "jira" && hasJira && <JiraStandupCard data={jira} />}
        </>
      )}
    </div>
  )
}

// The Summary subtab: headline counts, issue-link coverage, and a pointer to the attention queue.
function StandupSummary({
  boardId,
  agenda,
  coverage,
  blocked,
  onOpenAttention,
  onSelect,
}: {
  boardId: string
  agenda: BoardStandupView
  coverage: LinkCoverageView | null
  blocked: number
  onOpenAttention: () => void
  onSelect: (id: string) => void
}) {
  const tiles = [
    { id: "merged", label: "merged", value: agenda.merged_prs.length },
    { id: "in_review", label: "in review", value: agenda.waiting_on_review.length },
    { id: "delivered", label: "delivered", value: agenda.delivered_issues.length },
    { id: "in_progress", label: "in progress", value: agenda.in_progress_issues.length },
  ]
  return (
    <div className="space-y-4">
      <AiSummary boardId={boardId} />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {tiles.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => onSelect(t.id)}
            className="rounded-md border border-border px-3 py-2 text-left hover:bg-muted/50"
          >
            <div className="text-xs text-muted-foreground">{t.label}</div>
            <div className="font-mono text-lg">{t.value}</div>
          </button>
        ))}
        <div className="rounded-md border border-border px-3 py-2">
          <div className="text-xs text-muted-foreground">commits</div>
          <div className="font-mono text-lg">{agenda.moved_commits}</div>
        </div>
      </div>

      {/* Points at the attention queue instead of re-listing it. */}
      <Card className="flex items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          {blocked === 0
            ? "Nothing is blocked right now."
            : `${blocked} item(s) still need attention.`}
        </p>
        <button
          type="button"
          onClick={onOpenAttention}
          className="text-sm text-primary underline underline-offset-2 hover:text-foreground"
        >
          open Attention
        </button>
      </Card>

      {coverage != null && coverage.merged_total > 0 && (
        <Card className="space-y-1">
          <h2 className="text-sm font-medium text-muted-foreground">Issue-link coverage</h2>
          <p className="text-sm">
            <span className="font-mono">
              {Math.round((coverage.linked / coverage.merged_total) * 100)}%
            </span>{" "}
            of merged PRs link to a tracked issue
            <span className="text-muted-foreground">
              {" "}
              ({coverage.linked} of {coverage.merged_total}, {coverage.closes} via a closing
              reference)
            </span>
          </p>
        </Card>
      )}
    </div>
  )
}

// Columns for a standup PR-ref list (merged / in review): number and title from the ref, plus
// repo, author, age and reviews joined from the board PR list. `ageField` picks the timestamp
// (merge time or open time). A PR missing from the list still lists by its ref.
function prRefColumns(
  prById: Map<string, BoardPrView>,
  webBase: string | null,
  onOpenPerson: (login: string) => void,
  ageField: "created_at" | "merged_at",
): ColumnDef<PrRefView, unknown>[] {
  return [
    {
      accessorKey: "number",
      header: "#",
      cell: ({ row }) => (
        <ExtLink href={row.original.url}>
          <span className="font-mono text-xs">#{row.original.number}</span>
        </ExtLink>
      ),
    },
    {
      accessorKey: "title",
      header: "title",
      cell: ({ row }) => (
        <ExtLink href={row.original.url}>
          <span className="block max-w-md truncate">{row.original.title}</span>
        </ExtLink>
      ),
    },
    {
      id: "repo",
      header: "repo",
      cell: ({ row }) => {
        const full = prById.get(row.original.id)?.full_name
        return full ? (
          <RepoLink webBase={webBase} fullName={full} />
        ) : (
          <span className="text-muted-foreground">-</span>
        )
      },
    },
    {
      id: "author",
      accessorFn: (p) => prById.get(p.id)?.author_login ?? "",
      header: "author",
      cell: ({ row }) => {
        const login = prById.get(row.original.id)?.author_login
        return login ? (
          <UserLink login={login} webBase={webBase} onOpenPerson={onOpenPerson} />
        ) : (
          <span className="text-muted-foreground">-</span>
        )
      },
    },
    {
      id: "age",
      accessorFn: (p) => prById.get(p.id)?.[ageField] ?? "",
      header: "age",
      cell: ({ row }) => {
        const ts = prById.get(row.original.id)?.[ageField] ?? null
        return (
          <span className="font-mono text-xs" title={ts ?? ""}>
            {relativeAge(ts)}
          </span>
        )
      },
    },
    {
      id: "reviews",
      accessorFn: (p) => prById.get(p.id)?.review_count ?? 0,
      header: "reviews",
      cell: ({ row }) => {
        const pr = prById.get(row.original.id)
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

// Columns for a standup issue-ref list: number and title. The ref has no URL, so the title is
// plain text.
function issueRefColumns(): ColumnDef<IssueRefView, unknown>[] {
  return [
    {
      accessorKey: "number",
      header: "#",
      cell: ({ row }) => <span className="font-mono text-xs">#{row.original.number}</span>,
    },
    {
      accessorKey: "title",
      header: "title",
      cell: ({ row }) => <span className="block max-w-xl truncate">{row.original.title}</span>,
    },
  ]
}

// The board's Jira tickets: active sprint(s) and what is in progress, from data the parent
// already fetched. Rendered only when the board has a linked Jira project.
function JiraStandupCard({ data }: { data: BoardJiraView }) {
  const inProgress = data.tickets.filter((t) => t.status_category === "indeterminate")
  const active = data.sprints.filter((s) => s.state === "active")

  return (
    <Card className="space-y-2">
      <h2 className="text-sm font-medium text-foreground">
        Jira - {data.projects.map((p) => p.project_key).join(", ")} ({data.tickets.length} tickets)
      </h2>
      {active.length > 0 && (
        <p className="text-xs text-muted-foreground">
          active sprint: {active.map((s) => s.name).join(", ")}
        </p>
      )}
      <h3 className="text-xs font-medium text-muted-foreground">
        In progress ({inProgress.length})
      </h3>
      {inProgress.length === 0 ? (
        <p className="text-sm text-muted-foreground">nothing in progress</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {inProgress.slice(0, 12).map((t) => (
            <li key={t.key} className="flex items-center gap-2">
              <ExtLink href={t.url}>
                <span className="font-mono text-xs">{t.key}</span>
              </ExtLink>
              <span className="min-w-0 truncate">{t.title}</span>
              {t.assignee && (
                <span className="shrink-0 text-xs text-muted-foreground">{t.assignee}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}
