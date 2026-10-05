import {
  type BoardTrackerView,
  type BoardView,
  type ForgeView,
  type SignalPrefView,
  type TrackerView,
  commands,
} from "@/bindings"
import { useConfirm } from "@/components/Confirm"
import { EntityPicker, type SearchOutcome } from "@/components/EntityPicker"
import { Card } from "@/components/primitives"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { CircleHelp } from "lucide-react"
import { type ReactNode, useEffect, useState } from "react"

// Adapt a `searchForge*` command result into the outcome EntityPicker expects. An error is passed
// through as the reason the search could not run, not flattened into an empty result.
const searchResults = (
  p: Promise<{ status: "ok"; data: string[] } | { status: "error"; error: string }>,
): Promise<SearchOutcome> =>
  p.then((r) =>
    r.status === "ok" ? { ok: true, results: r.data } : { ok: false, reason: r.error },
  )

// A board's settings: what it watches (Scope) and what it flags (Signals), with board deletion in
// a Danger zone at the bottom. Scope inputs depend on the board kind: team (people, optionally
// narrowed by org and/or pinned repos), repo (one repo), or org (a whole org).
export function BoardSettings({
  board,
  onChanged,
  onDeleted,
  onSignalsChanged,
  onOpenSettings,
  onOpenDebug,
}: {
  board: BoardView
  onChanged: (b: BoardView) => void
  onDeleted: () => void
  // Called after a signal toggle so the parent refetches attention.
  onSignalsChanged?: () => void
  // Opens the app Settings, where forges are managed.
  onOpenSettings: () => void
  // Opens the Debug view, where a repo is forgotten from the store.
  onOpenDebug: () => void
}) {
  const [error, setError] = useState<string | null>(null)

  const apply = async (
    p: Promise<{ status: "ok"; data: BoardView } | { status: "error"; error: string }>,
  ) => {
    setError(null)
    const res = await p
    if (res.status === "ok") onChanged(res.data)
    else setError(res.error)
  }

  const confirm = useConfirm()
  const deleteBoard = async () => {
    const people = board.people.length
    const ok = await confirm({
      title: `Delete "${board.name}"?`,
      body:
        people > 0
          ? `This removes the board and its ${people} tracked ${people === 1 ? "person" : "people"}. Your synced data is kept; the board view is gone. This cannot be undone.`
          : "This removes the board. Your synced data is kept; the board view is gone. This cannot be undone.",
      confirmLabel: "Delete board",
    })
    if (!ok) return
    const res = await commands.deleteBoard(board.id)
    if (res.status === "ok") onDeleted()
  }

  return (
    <div className="space-y-4">
      {error && <p className="text-sm text-red-600">{error}</p>}

      <h2 className="text-sm font-semibold">Scope</h2>

      <ForgeInfoCard forgeId={board.forge_id} onOpenSettings={onOpenSettings} />

      <Card className="space-y-2">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-1">
              <h3 className="text-sm font-medium text-muted-foreground">Dependency scanning</h3>
              <HelpPopover label="About dependency scanning">
                Checks this board's dependencies against the OSV advisory database. Enabling it
                sends package names to osv.dev and flags packages with known advisories; it does not
                verify the exact pinned version.
              </HelpPopover>
            </div>
          </div>
          <Switch
            checked={board.scan_dependencies}
            aria-label="Dependency scanning"
            onCheckedChange={(checked) =>
              void apply(commands.setBoardScanDependencies(board.id, checked))
            }
          />
        </div>
      </Card>

      {board.kind === "team" && (
        <>
          <Card className="space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground">People</h3>
            <EntityPicker
              placeholder="github login (e.g. octocat)"
              buttonLabel="Add person"
              search={(q) => searchResults(commands.searchForgeUsers(board.forge_id ?? "", q))}
              onSelect={(v) => void apply(commands.addBoardPerson(board.id, v))}
            />
            {board.people.length > 0 && (
              <BadgeList
                items={board.people}
                onRemove={(p) => void apply(commands.removeBoardPerson(board.id, p))}
                removeTitle="remove"
              />
            )}
          </Card>

          <Card className="space-y-3">
            <h3 className="text-sm font-medium text-muted-foreground">Organization filter</h3>
            <EntityPicker
              placeholder="org login (e.g. acme-corp)"
              buttonLabel="Set org"
              search={(q) => searchResults(commands.searchForgeOrgs(board.forge_id ?? "", q))}
              onSelect={(v) => void apply(commands.setBoardOrg(board.id, v))}
            />
            {board.org ? (
              <BadgeList
                items={[board.org]}
                onRemove={() => void apply(commands.setBoardOrg(board.id, null))}
                removeTitle="clear org limit"
              />
            ) : null}
          </Card>

          <Card className="space-y-3">
            <div className="flex items-center gap-1">
              <h3 className="text-sm font-medium text-muted-foreground">Repositories</h3>
              <HelpPopover label="About repository scope">
                With no pinned repositories, Sync discovers repositories that the board's people own
                or recently touched, limited to the organization filter when set. Pinning
                repositories makes that list the board's exact scope.
              </HelpPopover>
            </div>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-1">
                <span className="text-sm font-medium">Include archived repositories</span>
                <HelpPopover label="About archived repositories">
                  Archived repositories are excluded from automatic discovery by default. A
                  repository pinned directly below is always included, even when archived.
                </HelpPopover>
              </div>
              <Switch
                checked={board.include_archived}
                aria-label="Include archived repositories"
                onCheckedChange={(checked) =>
                  void apply(commands.setBoardIncludeArchived(board.id, checked))
                }
              />
            </div>
            <EntityPicker
              placeholder="owner/repo"
              buttonLabel="Pin repo"
              search={(q) => searchResults(commands.searchForgeRepos(board.forge_id ?? "", q))}
              onSelect={(v) => void apply(commands.addBoardRepo(board.id, v))}
            />
            {board.repos.length > 0 && (
              <BadgeList
                items={board.repos}
                onRemove={(r) => void apply(commands.removeBoardRepo(board.id, r))}
                removeTitle="unpin"
              />
            )}
          </Card>
        </>
      )}

      {board.kind === "repo" && (
        <Card className="space-y-2">
          <h3 className="text-sm font-medium text-muted-foreground">Repository</h3>
          <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-700">
            {board.repos[0] ?? "(none)"}
          </Badge>
        </Card>
      )}

      {board.kind === "org" && (
        <Card className="space-y-2">
          <h3 className="text-sm font-medium text-muted-foreground">Organization</h3>
          <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-700">
            {board.org ?? "(none)"}
          </Badge>
        </Card>
      )}

      <h2 className="pt-2 text-sm font-semibold">Jira</h2>
      <JiraScopeCard boardId={board.id} onOpenSettings={onOpenSettings} />

      <h2 className="pt-2 text-sm font-semibold">Signals</h2>
      <SignalsSection boardId={board.id} onChanged={onSignalsChanged} />

      {/* Red matches the sync-error banner, error toast, and low DORA tier. */}
      <h2 className="pt-2 text-sm font-semibold text-red-800">Danger zone</h2>
      <Card className="border-red-200 bg-red-50">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-sm font-medium text-red-800">Delete this board</h3>
            <p className="text-xs text-red-700">
              Removes the board and its scope. Your synced data is kept; the board view is gone.
              This cannot be undone.
            </p>
          </div>
          <Button
            variant="destructive"
            size="sm"
            className="shrink-0"
            onClick={() => void deleteBoard()}
          >
            Delete board
          </Button>
        </div>
      </Card>

      {/* Forgetting a repo empties it from the whole database, so it is not a board action.
          This points at Debug. */}
      <Card className="border-red-200 bg-red-50">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-sm font-medium text-red-800">Forget a repo entirely</h3>
            <p className="text-xs text-red-700">
              Removing a repo from this board leaves its synced activity and its indexed code in the
              database, where Search and the assistant still find it. Forgetting removes it from
              everywhere, for every board.
            </p>
          </div>
          <Button variant="outline" size="sm" className="shrink-0" onClick={onOpenDebug}>
            Open Debug
          </Button>
        </div>
      </Card>
    </div>
  )
}

function HelpPopover({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button type="button" variant="ghost" size="icon" aria-label={label}>
          <CircleHelp aria-hidden="true" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" aria-label={label}>
        <p className="text-sm leading-relaxed">{children}</p>
      </PopoverContent>
    </Popover>
  )
}

// Link Jira project(s) to a board: pick a registered tracker and type the project key. Tickets
// sync read-only. Trackers are registered in the app Settings.
function JiraScopeCard({
  boardId,
  onOpenSettings,
}: {
  boardId: string
  onOpenSettings: () => void
}) {
  const [trackers, setTrackers] = useState<TrackerView[] | null>(null)
  const [links, setLinks] = useState<BoardTrackerView[]>([])
  const [trackerId, setTrackerId] = useState("")
  const [projectKey, setProjectKey] = useState("")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let ignore = false
    void commands.listTrackers().then((res) => {
      if (ignore || res.status !== "ok") return
      setTrackers(res.data)
      if (res.data.length === 1) setTrackerId(res.data[0].id)
    })
    void commands.boardJira(boardId).then((res) => {
      if (!ignore && res.status === "ok") setLinks(res.data.projects)
    })
    return () => {
      ignore = true
    }
  }, [boardId])

  const link = async () => {
    setError(null)
    const res = await commands.linkBoardTracker(boardId, trackerId, projectKey.trim())
    if (res.status === "ok") {
      setLinks(res.data)
      setProjectKey("")
    } else setError(res.error)
  }
  const unlink = async (l: BoardTrackerView) => {
    const res = await commands.unlinkBoardTracker(boardId, l.tracker_id, l.project_key)
    if (res.status === "ok") setLinks(res.data)
  }

  return (
    <Card className="space-y-3">
      {error && <p className="text-sm text-red-600">{error}</p>}
      <p className="text-xs text-muted-foreground">
        Watch a Jira project's tickets on this board (read-only). Tickets link to PRs by their{" "}
        <span className="font-mono">PROJ-123</span> key in the branch / PR title / commit.
      </p>
      {trackers != null && trackers.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          no Jira sites configured yet -{" "}
          <button type="button" className="text-primary underline" onClick={onOpenSettings}>
            add one in Settings
          </button>
          , then link a project here
        </p>
      ) : (
        <>
          {links.length > 0 && (
            <ul className="flex flex-wrap gap-1.5">
              {links.map((l) => (
                <li key={`${l.tracker_id}:${l.project_key}`}>
                  <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-700">
                    {l.project_key}
                    <span className="ml-1 text-muted-foreground">({l.tracker_name})</span>
                    <button
                      type="button"
                      className="ml-1.5 text-muted-foreground hover:text-red-600"
                      onClick={() => void unlink(l)}
                      title="unlink"
                    >
                      x
                    </button>
                  </Badge>
                </li>
              ))}
            </ul>
          )}
          <div className="flex items-center gap-2">
            <select
              className="h-9 rounded-md border border-border bg-background px-2 text-sm"
              value={trackerId}
              onChange={(e) => setTrackerId(e.target.value)}
            >
              <option value="">(pick a Jira site)</option>
              {trackers?.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
            <input
              className="h-9 flex-1 rounded-md border border-border bg-background px-3 text-sm"
              placeholder="project key (e.g. JENKINS)"
              value={projectKey}
              onChange={(e) => setProjectKey(e.target.value)}
            />
            <Button onClick={() => void link()} disabled={!trackerId || !projectKey.trim()}>
              Link
            </Button>
          </div>
        </>
      )}
    </Card>
  )
}

// A wrap of removable pill badges. Each badge's "x" calls `onRemove` with its value.
function BadgeList({
  items,
  onRemove,
  removeTitle,
}: {
  items: string[]
  onRemove: (value: string) => void
  removeTitle: string
}) {
  return (
    <ul className="flex flex-wrap gap-1.5">
      {items.map((it) => (
        <li key={it}>
          <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-700">
            {it}
            <button
              type="button"
              className="ml-1.5 text-muted-foreground hover:text-red-600"
              onClick={() => onRemove(it)}
              title={removeTitle}
            >
              x
            </button>
          </Badge>
        </li>
      ))}
    </ul>
  )
}

// The board's forge, read-only. It is fixed at board creation because the repos a board
// pins/discovers are namespaced by it. Connections are managed in the app Settings.
function ForgeInfoCard({
  forgeId,
  onOpenSettings,
}: {
  forgeId: string | null
  onOpenSettings: () => void
}) {
  const [forges, setForges] = useState<ForgeView[] | null>(null)

  useEffect(() => {
    let ignore = false
    void commands.listForges().then((res) => {
      if (!ignore && res.status === "ok") setForges(res.data)
    })
    return () => {
      ignore = true
    }
  }, [])

  const forge = forges?.find((f) => f.id === forgeId) ?? null

  return (
    <Card className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-medium text-muted-foreground">Forge</h3>
        <Button variant="outline" size="sm" onClick={onOpenSettings}>
          Manage
        </Button>
      </div>
      {forge ? (
        <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-700">
          {forge.name} ({forge.kind})
        </Badge>
      ) : (
        <p className="text-sm text-muted-foreground">{forgeId ?? "(none)"}</p>
      )}
    </Card>
  )
}

// A one-line description per attention signal, shown beside its toggle. Unknown ids get none.
const SIGNAL_DESCRIPTIONS: Record<string, string> = {
  stale_pr: "Open PRs left waiting past the stale-PR threshold.",
  merged_without_review: "Pull requests merged without any review.",
  failing_ci: "The most recent CI run on a repo failed.",
  flaky_ci: "A CI run went green only after a re-run on the same commit - a flaky suite.",
  upstream: "A repo this board depends on has its own attention items.",
}

// The Signals section: each attention signal with a label, description, and on/off control.
// Turning one off drops its items from this board's attention everywhere. Persisted server-side;
// `onChanged` refetches attention.
function SignalsSection({ boardId, onChanged }: { boardId: string; onChanged?: () => void }) {
  const [signals, setSignals] = useState<SignalPrefView[] | null>(null)
  useEffect(() => {
    let ignore = false
    void commands.boardSignals(boardId).then((res) => {
      if (!ignore && res.status === "ok") setSignals(res.data)
    })
    return () => {
      ignore = true
    }
  }, [boardId])

  const toggle = async (signal: string, enabled: boolean) => {
    const res = await commands.setBoardSignalEnabled(boardId, signal, enabled)
    if (res.status === "ok") {
      setSignals(res.data)
      onChanged?.()
    }
  }

  return (
    <Card className="space-y-3">
      {signals == null ? (
        <p className="text-sm text-muted-foreground">loading...</p>
      ) : (
        signals.map((s) => (
          <div key={s.signal} className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="text-sm font-medium">{s.label}</p>
              <p className="text-xs text-muted-foreground">{SIGNAL_DESCRIPTIONS[s.signal] ?? ""}</p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void toggle(s.signal, !s.enabled)}
              className={cn(
                "w-16 shrink-0",
                s.enabled
                  ? "border-foreground bg-foreground text-background hover:bg-foreground/90 hover:text-background"
                  : "text-muted-foreground",
              )}
            >
              {s.enabled ? "On" : "Off"}
            </Button>
          </div>
        ))
      )}
    </Card>
  )
}
