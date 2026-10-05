import { type BoardPrView, type ChangeDigestView, type ChangedFileView, commands } from "@/bindings"
import { DataTable } from "@/components/DataTable"
import { useToast } from "@/components/Toast"
import { RepoLink, UserLink } from "@/components/links"
import { Markdown } from "@/components/markdown"
import { Card, EmptyState, ExtLink, SkeletonCard, ViewHeader } from "@/components/primitives"
import { Badge } from "@/components/ui/badge"
import { prState, relativeAge } from "@/lib/format"
import { useBoardPref } from "@/lib/useBoardPref"
import { cn } from "@/lib/utils"
import type { ColumnDef } from "@tanstack/react-table"
import { useCallback, useEffect, useMemo, useState } from "react"

type StateFilter = "all" | "open" | "merged" | "closed"
const STATE_FILTERS: StateFilter[] = ["all", "open", "merged", "closed"]

// Browse every pull request across the board's repos in one sortable, filterable, paginated grid,
// and open one as a change digest (facts, linked issues, Markdown, diff).
export function ChangesView({
  boardId,
  dataVersion,
  webBase,
  onOpenPerson,
}: {
  boardId: string
  dataVersion?: number
  // Board forge web root and "go to this user" nav.
  webBase: string | null
  onOpenPerson: (login: string) => void
}) {
  const [prs, setPrs] = useState<BoardPrView[] | null>(null)
  const [prsError, setPrsError] = useState<string | null>(null)
  const [filter, setFilter] = useState("")
  // PR-state filter, persisted per board across board switches and restarts.
  const [stateFilter, setStateFilter] = useBoardPref<StateFilter>(boardId, "changesState", "all")
  const [change, setChange] = useState<ChangeDigestView | null>(null)
  const toast = useToast()

  // Reset to the skeleton only on a board switch; a dataVersion refresh updates the grid in place.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset only on board switch
  useEffect(() => {
    setChange(null)
    setPrs(null)
    setPrsError(null)
  }, [boardId])
  // biome-ignore lint/correctness/useExhaustiveDependencies: dataVersion is a refetch trigger, not read
  useEffect(() => {
    // Drop a stale response if the board switches before this fetch resolves.
    let ignore = false
    setPrsError(null)
    void commands.boardChanges(boardId).then((res) => {
      if (!ignore) {
        if (res.status === "ok") setPrs(res.data)
        else setPrsError(res.error)
      }
    })
    return () => {
      ignore = true
    }
  }, [boardId, dataVersion])

  const showChange = useCallback(
    async (prId: string) => {
      const res = await commands.changeDigest(prId)
      if (res.status === "ok") setChange(res.data)
      else toast("Could not load change details", "error")
    },
    [toast],
  )

  const columns = useMemo<ColumnDef<BoardPrView, unknown>[]>(
    () => [
      {
        accessorKey: "full_name",
        header: "repo",
        cell: ({ row }) => (
          <RepoLink
            webBase={webBase}
            fullName={row.original.full_name}
            className="font-mono text-xs text-muted-foreground"
          />
        ),
      },
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
        accessorKey: "state",
        header: "state",
        cell: ({ row }) => <StateBadge pr={row.original} />,
      },
      {
        accessorKey: "author_login",
        header: "author",
        cell: ({ row }) =>
          row.original.author_login ? (
            <UserLink
              login={row.original.author_login}
              webBase={webBase}
              onOpenPerson={onOpenPerson}
            />
          ) : (
            <span className="text-muted-foreground">-</span>
          ),
      },
      {
        accessorKey: "review_count",
        header: "reviews",
        cell: ({ row }) => (
          <span className="font-mono text-xs">
            {row.original.review_count}
            {row.original.approved ? " (approved)" : ""}
          </span>
        ),
      },
      {
        accessorKey: "created_at",
        header: "opened",
        cell: ({ row }) => (
          <span className="font-mono text-xs" title={row.original.created_at}>
            {relativeAge(row.original.created_at)}
          </span>
        ),
      },
    ],
    [webBase, onOpenPerson],
  )

  // The state filter narrows the rows first; the text filter (globalFilter) narrows further.
  const visible =
    prs == null || stateFilter === "all" ? prs : prs.filter((p) => prState(p) === stateFilter)

  if (prsError != null) {
    return <EmptyState title="Could not load pull requests" hint={prsError} />
  }
  if (prs == null || visible == null) {
    return (
      <div className="space-y-4">
        <SkeletonCard rows={8} />
      </div>
    )
  }
  if (prs.length === 0) {
    return (
      <EmptyState
        title="No pull requests in view"
        hint="Pin repos to this board (under Settings) or sync some activity first."
      />
    )
  }

  const subtitle =
    stateFilter === "all"
      ? `${prs.length} pull request(s) across ${new Set(prs.map((p) => p.repo_id)).size} repo(s)`
      : `${visible.length} ${stateFilter} of ${prs.length} pull request(s)`

  return (
    <div className="space-y-4">
      <ViewHeader
        subtitle={subtitle}
        actions={
          <input
            className="h-9 w-64 rounded-md border border-border bg-background px-2 text-sm"
            placeholder="filter (repo, title, author)..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        }
      />

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted-foreground">State:</span>
        {STATE_FILTERS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setStateFilter(s)}
            className={cn(
              "rounded-full border px-2 py-0.5",
              stateFilter === s
                ? "border-foreground bg-foreground text-background"
                : "border-border bg-muted text-muted-foreground hover:text-foreground",
            )}
          >
            {s}
          </button>
        ))}
      </div>

      <DataTable
        columns={columns}
        data={visible}
        globalFilter={filter}
        initialSorting={[{ id: "created_at", desc: true }]}
        onRowClick={(p) => void showChange(p.id)}
        emptyMessage="no pull requests match the filter"
      />

      {change && <ChangeDigest change={change} onClose={() => setChange(null)} />}
    </div>
  )
}

// A PR's state as a colored badge; merged (purple) takes precedence over open/closed.
function StateBadge({ pr }: { pr: BoardPrView }) {
  switch (prState(pr)) {
    case "merged":
      return (
        <Badge variant="outline" className="border-purple-200 bg-purple-50 text-purple-700">
          merged
        </Badge>
      )
    case "open":
      return (
        <Badge variant="outline" className="border-green-200 bg-green-50 text-green-700">
          open
        </Badge>
      )
    default:
      return (
        <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-600">
          closed
        </Badge>
      )
  }
}

function ChangeDigest({ change, onClose }: { change: ChangeDigestView; onClose: () => void }) {
  return (
    <Card className="space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <h2 className="font-semibold">
            #{change.number} {change.title}
          </h2>
          {change.approved && (
            <Badge variant="outline" className="border-green-200 bg-green-50 text-green-700">
              approved
            </Badge>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 text-sm text-muted-foreground hover:text-foreground"
        >
          close
        </button>
      </div>
      <p className="font-mono text-xs text-muted-foreground">
        {change.state}
        {change.author_login ? ` - by ${change.author_login}` : ""} - {change.review_count}{" "}
        review(s)
      </p>
      {change.linked_issues.length > 0 && (
        <ul className="flex flex-wrap gap-1.5">
          {change.linked_issues.map((l) => (
            <li key={l.id}>
              <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-600">
                {l.relation} #{l.number}
              </Badge>
            </li>
          ))}
        </ul>
      )}
      {change.linked_tickets.length > 0 && (
        <ul className="flex flex-wrap gap-1.5">
          {change.linked_tickets.map((t) => (
            <li key={t.key}>
              <Badge variant="outline" className="border-blue-200 bg-blue-50 text-blue-700">
                <ExtLink href={t.url}>{t.key}</ExtLink>
                <span className="ml-1 font-normal text-blue-700/70">{t.status}</span>
              </Badge>
            </li>
          ))}
        </ul>
      )}
      {change.body ? (
        <Markdown>{change.body}</Markdown>
      ) : (
        <p className="text-sm text-muted-foreground">(no description)</p>
      )}
      {change.files.length > 0 && (
        <div className="space-y-2 border-t border-border pt-3">
          <h3 className="text-sm font-medium">
            {change.files.length} file(s) changed{" "}
            <span className="font-mono text-xs text-green-600">+{change.additions}</span>{" "}
            <span className="font-mono text-xs text-red-600">-{change.deletions}</span>
          </h3>
          {change.files.map((f) => (
            <ChangedFile key={f.filename} file={f} />
          ))}
        </div>
      )}
    </Card>
  )
}

// One changed file: click-to-expand row with filename, +/- counts and the unified-diff patch.
function ChangedFile({ file }: { file: ChangedFileView }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-md border border-border">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-sm hover:bg-muted"
      >
        <span className="truncate font-mono text-xs">{file.filename}</span>
        <span className="shrink-0 font-mono text-xs">
          <span className="text-green-600">+{file.additions}</span>{" "}
          <span className="text-red-600">-{file.deletions}</span>
          {file.patch ? (
            <span className="ml-2 text-muted-foreground">{open ? "-" : "+"}</span>
          ) : null}
        </span>
      </button>
      {open && file.patch && (
        <pre className="overflow-x-auto border-t border-border bg-muted/40 px-3 py-2 text-xs leading-relaxed">
          {file.patch.split("\n").map((line, i) => (
            <div
              // biome-ignore lint/suspicious/noArrayIndexKey: patch lines are positional and static
              key={i}
              className={
                line.startsWith("+")
                  ? "text-green-700"
                  : line.startsWith("-")
                    ? "text-red-700"
                    : line.startsWith("@@")
                      ? "text-blue-600"
                      : "text-foreground"
              }
            >
              {line || " "}
            </div>
          ))}
        </pre>
      )}
    </div>
  )
}
