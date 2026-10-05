import { NewBoardDialog } from "@/components/NewBoardDialog"
import { Sidebar } from "@/components/Sidebar"
import { useToast } from "@/components/Toast"
import { AssistantPanel } from "@/components/assistant/AssistantPanel"
import { EmptyState, type IndexInfo } from "@/components/primitives"
import { COMMAND_TIMEOUT_MS, withTimeout } from "@/lib/async"
import { subjectKey } from "@/lib/attention"
import { forgeWebBase } from "@/lib/forge"
import { cn } from "@/lib/utils"
import { BoardDashboard } from "@/views/BoardDashboard"
import { BoardPeople } from "@/views/BoardPeople"
import { BoardSettings } from "@/views/BoardSettings"
import { BoardStandup } from "@/views/BoardStandup"
import { ChangesView } from "@/views/ChangesView"
import { CodeView } from "@/views/CodeView"
import { DebugView } from "@/views/DebugView"
import { OverviewView } from "@/views/OverviewView"
import { PortfolioView } from "@/views/PortfolioView"
import { SearchView } from "@/views/SearchView"
import { SettingsView } from "@/views/SettingsView"
import { useCallback, useEffect, useRef, useState } from "react"
import {
  events,
  type BoardView,
  type ForgeView,
  type RepoOverview,
  type StorageStateView,
  type SyncProgressEvent,
  commands,
} from "./bindings"

type Tab =
  | "attention"
  | "dashboard"
  | "code"
  | "standup"
  | "changes"
  | "people"
  | "search"
  | "settings"
const TABS: { id: Tab; label: string }[] = [
  { id: "attention", label: "Attention" },
  { id: "dashboard", label: "Dashboard" },
  { id: "code", label: "Code" },
  { id: "standup", label: "Standup" },
  { id: "changes", label: "Changes" },
  { id: "people", label: "People" },
  { id: "search", label: "Search" },
  { id: "settings", label: "Settings" },
]

// Tabs a board shows, by kind. Standup is team-only; Code/Search need embeddings, which org
// boards never produce. People shows on every kind.
const TABS_BY_KIND: Record<string, Tab[]> = {
  team: ["attention", "dashboard", "code", "standup", "changes", "people", "search", "settings"],
  repo: ["attention", "dashboard", "code", "changes", "people", "search", "settings"],
  org: ["attention", "dashboard", "changes", "people", "settings"],
}

// Tab descriptors visible for a board kind, in canonical order. Unknown kinds get the team set.
function tabsForKind(kind: string): { id: Tab; label: string }[] {
  const allowed = new Set(TABS_BY_KIND[kind] ?? TABS_BY_KIND.team)
  return TABS.filter((t) => allowed.has(t.id))
}

function boardSubtitle(b: BoardView): string {
  if (b.kind === "repo")
    return b.repos[0] ? `repo: ${b.repos[0]}` : "repo board - pin a repo in Settings"
  if (b.kind === "org")
    return b.org ? `org: ${b.org} (metrics rollup)` : "org board - set an org in Settings"
  const scope =
    b.repos.length > 0
      ? `${b.repos.length} pinned repo(s)`
      : b.org
        ? `discovered repos in ${b.org}`
        : "discovered repos"
  return `${b.people.length} person(s) - ${scope}`
}

// The shell: board picker on the left, one board's views in the middle. Talks to the core only
// through the generated `./bindings`.
export default function App() {
  const [boards, setBoards] = useState<BoardView[]>([])
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>("attention")
  const [newBoardOpen, setNewBoardOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [debugOpen, setDebugOpen] = useState(false)
  const [portfolioOpen, setPortfolioOpen] = useState(false)
  // Toggleable right-docked assistant panel; stays open over any board view.
  const [assistantOpen, setAssistantOpen] = useState(false)
  const [overview, setOverview] = useState<RepoOverview[] | null>(null)
  // Set when the board-overview fetch fails (rejected command), so the dashboard shows an error
  // instead of a skeleton. Cleared on a successful fetch and on board switch.
  const [overviewError, setOverviewError] = useState<string | null>(null)
  const [teamRefs, setTeamRefs] = useState<Set<string>>(new Set())
  const [healthy, setHealthy] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [status, setStatus] = useState("starting...")
  const [progress, setProgress] = useState<SyncProgressEvent | null>(null)
  // Last sync failure, shown as a dismissible banner. Separate from `status` so a failure does not
  // erase the last-successful-sync time.
  const [syncError, setSyncError] = useState<string | null>(null)
  // ISO time the last source synced successfully; survives a later failure.
  const [lastSynced, setLastSynced] = useState<string | null>(null)
  const toast = useToast()
  // Per-repo background-index state (state + embedding progress). Always a full replace from the
  // `index_status()` snapshot; merging per event lets a late event from an old pass overwrite a
  // newer one.
  const [indexStates, setIndexStates] = useState<Record<string, IndexInfo>>({})
  // Bumped when a sync or index pass completes so per-tab views (which fetch their own data)
  // refetch. Board overview refresh goes through loadOverview instead.
  const [dataVersion, setDataVersion] = useState(0)
  // Live storage state. At the ceiling the app stops syncing and indexing; this drives the banner
  // shown over every view.
  const [storage, setStorage] = useState<StorageStateView | null>(null)
  const [storageDismissed, setStorageDismissed] = useState(false)
  // Forges (for the board's web base) and the person the People tab focuses on.
  const [forges, setForges] = useState<ForgeView[]>([])
  const [personFocus, setPersonFocus] = useState<string | null>(null)

  const current = boards.find((b) => b.id === currentId) ?? null
  const webBase = current ? forgeWebBase(forges.find((f) => f.id === current.forge_id)) : null
  const openPerson = useCallback((login: string) => {
    setSettingsOpen(false)
    setDebugOpen(false)
    setPortfolioOpen(false)
    setPersonFocus(login)
    setTab("people")
  }, [])
  // Fall back to Attention (every kind has it) if the selected tab is not in this board's set.
  const visibleTabs = current ? tabsForKind(current.kind) : TABS
  const activeTab: Tab = visibleTabs.some((t) => t.id === tab) ? tab : "attention"

  // Event listeners subscribe once and read the latest board through a ref; re-subscribing on
  // board switch would drop events landing in the unlisten/relisten window.
  const currentIdRef = useRef(currentId)
  currentIdRef.current = currentId
  // Latest overview, so loadOverview can tell a first-load failure from a background one.
  const overviewRef = useRef(overview)
  overviewRef.current = overview

  const loadBoards = useCallback(async () => {
    const res = await commands.listBoards()
    if (res.status === "ok") {
      setBoards(res.data)
      // Pick the first board if none is chosen or the chosen one vanished.
      setCurrentId((prev) =>
        prev && res.data.some((b) => b.id === prev) ? prev : (res.data[0]?.id ?? null),
      )
    } else {
      setStatus(`load error: ${res.error}`)
    }
  }, [])

  const loadStorage = useCallback(async () => {
    const res = await commands.storageState()
    if (res.status === "ok") setStorage(res.data)
  }, [])

  // Replace `indexStates` wholesale from the index lane's snapshot. Repos absent from the response
  // get no entry. A failed call keeps the previous snapshot.
  const loadIndexStates = useCallback(async () => {
    const res = await commands.indexStatus()
    if (res.status !== "ok") return
    const next: Record<string, IndexInfo> = {}
    for (const entry of res.data) {
      next[entry.repo_id] = {
        state: entry.state,
        done: entry.done,
        total: entry.total,
        error: entry.error,
        pending: entry.pending,
        skipped: entry.skipped,
        note: entry.note,
      }
    }
    setIndexStates(next)
  }, [])

  // Coalesce index-status refreshes separately from `scheduleRefresh`: progress fires on every
  // percent and only the badge needs it. Same leading+trailing 1.5s shape.
  const indexRefreshAtRef = useRef(0)
  const indexRefreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const scheduleIndexRefresh = useCallback(() => {
    const run = () => {
      indexRefreshAtRef.current = Date.now()
      void loadIndexStates()
    }
    const elapsed = Date.now() - indexRefreshAtRef.current
    const MIN_MS = 1500
    if (elapsed >= MIN_MS) run()
    else if (indexRefreshTimerRef.current == null) {
      indexRefreshTimerRef.current = setTimeout(() => {
        indexRefreshTimerRef.current = null
        run()
      }, MIN_MS - elapsed)
    }
  }, [loadIndexStates])

  // Rejections are caught and `withTimeout` bounds a call that never settles, so either failure
  // reaches `setOverviewError` instead of leaving the skeleton up.
  const loadOverview = useCallback(
    async (boardId: string) => {
      try {
        // Don't null `overview` here: this runs per repo during a sync and would flicker the board.
        // The board-switch effect nulls it.
        const [ov, attn] = await Promise.all([
          withTimeout(commands.boardOverview(boardId), COMMAND_TIMEOUT_MS),
          withTimeout(commands.boardAttention(boardId), COMMAND_TIMEOUT_MS),
        ])
        // Drop a stale response if the board changed while this was in flight.
        if (boardId !== currentIdRef.current) return
        if (ov.status === "ok") {
          setOverview(ov.data)
          setOverviewError(null)
        } else {
          setStatus(`load error: ${ov.error}`)
          setOverviewError(ov.error)
        }
        // Highlight attention items flagged by_team.
        if (attn.status === "ok") {
          setTeamRefs(
            new Set(
              attn.data
                .filter((a) => a.by_team)
                .map(subjectKey)
                .filter((k): k is string => k != null),
            ),
          )
        }
      } catch (e) {
        if (boardId !== currentIdRef.current) return
        const message = e instanceof Error ? e.message : String(e)
        setStatus(`load error: ${message}`)
        setOverviewError(message)
        // Only the first load leaves `overview` null; a failing background refresh keeps the last
        // good snapshot and reports the failure.
        if (overviewRef.current != null) toast(`Could not refresh the board: ${message}`, "error")
      }
    },
    [toast],
  )

  // Coalesce the sync-driven refresh: a sync fires many per-repo events in a burst and refetching
  // on each thrashes the tables. Refresh at most once per ~1.5s (leading + trailing).
  const refreshAtRef = useRef(0)
  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const scheduleRefresh = useCallback(() => {
    const run = () => {
      refreshAtRef.current = Date.now()
      const id = currentIdRef.current
      if (id) void loadOverview(id)
      // Sync and index events move bytes, so refresh the budget state too (same coalescing).
      void loadStorage()
      setDataVersion((v) => v + 1)
    }
    const elapsed = Date.now() - refreshAtRef.current
    const MIN_MS = 1500
    if (elapsed >= MIN_MS) run()
    else if (refreshTimerRef.current == null) {
      refreshTimerRef.current = setTimeout(() => {
        refreshTimerRef.current = null
        run()
      }, MIN_MS - elapsed)
    }
  }, [loadOverview, loadStorage])

  useEffect(() => {
    void commands.health().then((h) => {
      setHealthy(h.status === "ok")
      setStatus("ready")
    })
    void commands.listForges().then((res) => {
      if (res.status === "ok") setForges(res.data)
    })
    void loadBoards()
    void loadStorage()
    void loadIndexStates()
    // The scheduler polls every source in the background; the header shows only the last sync time.
    const unlisten = events.syncProgressEvent.listen((e) => {
      const p: SyncProgressEvent = e.payload
      // Latest update for the header's progress display (shown only during a manual sync).
      setProgress(p)
      // Only a completed repo updates the status line and refreshes board data.
      if (p.finished) {
        if (p.ok) {
          // Keep the last good sync time; a later failure shows a banner without erasing it.
          setStatus(`synced ${new Date().toLocaleTimeString()}`)
          setLastSynced(new Date().toISOString())
        } else if (p.error) {
          // A per-source failure surfaces as a banner.
          setSyncError(p.error)
        }
        // A finished fetch hands the repo to the index queue; read the index lane back.
        if (p.ok && p.source_id) {
          scheduleIndexRefresh()
        }
        scheduleRefresh()
      }
    })
    // Per-repo indexing state from the background index worker. Every event requests a coalesced
    // index-status refresh.
    const unlistenIndex = events.indexProgressEvent.listen((e) => {
      scheduleIndexRefresh()
      // A finished index changes searchable content: refresh the board and per-tab views.
      if (e.payload.state === "indexed") {
        scheduleRefresh()
      }
    })
    return () => {
      void unlisten.then((f) => f())
      void unlistenIndex.then((f) => f())
    }
    // Subscribe once: handlers read the latest board via currentIdRef, so this effect must not
    // depend on currentId. The callbacks are stable.
  }, [loadBoards, scheduleRefresh, loadStorage, loadIndexStates, scheduleIndexRefresh])

  // Nothing syncs while stopped or near the ceiling, so no event announces recovery. Poll while a
  // banner is up.
  useEffect(() => {
    if (!storage || storage.pressure === "normal") return
    const t = setInterval(() => void loadStorage(), 10_000)
    return () => clearInterval(t)
  }, [storage, loadStorage])

  // A cleared warning must not stay dismissed; the next stop is new news.
  useEffect(() => {
    if (storage?.pressure === "normal") setStorageDismissed(false)
  }, [storage?.pressure])

  // Load the selected board's overview on change (clear first so a switch shows loading).
  useEffect(() => {
    if (currentId) {
      setOverview(null)
      setOverviewError(null)
      void loadOverview(currentId)
    }
  }, [currentId, loadOverview])

  const onBoardCreated = async (board: BoardView) => {
    setNewBoardOpen(false)
    await loadBoards()
    setSettingsOpen(false)
    setCurrentId(board.id)
    setTab("settings")
  }

  const onBoardChanged = (b: BoardView) => {
    setBoards((prev) => prev.map((x) => (x.id === b.id ? b : x)))
    if (currentId) void loadOverview(currentId)
  }

  const onBoardDeleted = async () => {
    await loadBoards()
    setTab("attention")
  }

  const syncNow = async () => {
    setSyncing(true)
    setProgress(null)
    setStatus("syncing all sources...")
    setSyncError(null)
    const res = await commands.syncNow()
    if (res.status === "ok") {
      // Zero sources with the store at its ceiling is the budget refusal, not an empty registry.
      const state = await commands.storageState()
      if (res.data.length === 0 && state.status === "ok" && state.data.stopped) {
        setStorage(state.data)
        setStatus("stopped at the storage budget")
        toast("At the storage budget - nothing was synced", "info")
        setSyncing(false)
        setProgress(null)
        return
      }
      const total = res.data.length
      const ok = res.data.filter((r) => r.ok).length
      setStatus(`synced ${ok}/${total} source(s)`)
      if (ok > 0) setLastSynced(new Date().toISOString())
      // A per-source failure (e.g. one unconfigured forge) carries its message.
      const failed = res.data.find((r) => !r.ok && r.error)
      if (failed?.error) setSyncError(failed.error)
      toast(
        failed ? `Synced ${ok}/${total} sources - some had problems` : `Synced ${total} source(s)`,
        failed ? "info" : "success",
      )
      if (currentId) await loadOverview(currentId)
    } else {
      // The whole sync failed: show the message as a banner and a toast.
      setSyncError(res.error)
      toast("Sync failed", "error")
    }
    setSyncing(false)
    setProgress(null)
    await loadStorage()
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar
        boards={boards}
        currentBoardId={currentId}
        settingsActive={settingsOpen}
        debugActive={debugOpen}
        portfolioActive={portfolioOpen}
        onSelectBoard={(id) => {
          setSettingsOpen(false)
          setDebugOpen(false)
          setPortfolioOpen(false)
          setCurrentId(id)
          setTab("attention")
        }}
        onNewBoard={() => setNewBoardOpen(true)}
        onOpenSettings={() => {
          setSettingsOpen(true)
          setDebugOpen(false)
          setPortfolioOpen(false)
        }}
        onOpenDebug={() => {
          setDebugOpen(true)
          setSettingsOpen(false)
          setPortfolioOpen(false)
        }}
        onOpenPortfolio={() => {
          setPortfolioOpen(true)
          setSettingsOpen(false)
          setDebugOpen(false)
        }}
        assistantActive={assistantOpen}
        onToggleAssistant={() => setAssistantOpen((open) => !open)}
        healthy={healthy}
        syncing={syncing}
        status={status}
        lastSynced={lastSynced}
        progress={progress}
        onSyncNow={() => void syncNow()}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <main className="flex-1 overflow-auto p-6">
          {storage && storage.pressure !== "normal" && !(storageDismissed && !storage.stopped) && (
            <div
              className={cn(
                "mb-4 flex items-start justify-between gap-3 rounded-md border px-4 py-3 text-sm",
                storage.stopped
                  ? "border-red-200 bg-red-50 text-red-800"
                  : "border-amber-200 bg-amber-50 text-amber-900",
              )}
            >
              <div className="space-y-1">
                <p className="font-medium">
                  {storage.stopped
                    ? "orgonzola has stopped - storage budget reached"
                    : "orgonzola is near its storage budget"}
                </p>
                <p>{storage.message}</p>
                {/* Both are one click: the storage page has the accounting and forget list; the
                    budget is a field in Settings. */}
                <div className="flex gap-3 pt-0.5">
                  <button
                    type="button"
                    onClick={() => {
                      setDebugOpen(true)
                      setSettingsOpen(false)
                      setPortfolioOpen(false)
                    }}
                    className="font-medium underline underline-offset-2"
                  >
                    Free up space
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSettingsOpen(true)
                      setDebugOpen(false)
                      setPortfolioOpen(false)
                    }}
                    className="font-medium underline underline-offset-2"
                  >
                    Change the budget
                  </button>
                </div>
              </div>
              {/* A warning can be dismissed; a stop cannot, since hiding it would hide that data is
                  not being taken in. */}
              {!storage.stopped && (
                <button
                  type="button"
                  onClick={() => setStorageDismissed(true)}
                  className="shrink-0 text-amber-600 hover:text-amber-800"
                  aria-label="Dismiss"
                >
                  x
                </button>
              )}
            </div>
          )}
          {syncError && (
            <div className="mb-4 flex items-start justify-between gap-3 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
              <div className="space-y-1">
                <p className="font-medium">Sync problem</p>
                <p>{syncError}</p>
                {/* An auth failure is fixed in Settings. Matches "Reconnect" in ForgeError::Auth's
                    user_message. */}
                {syncError.includes("Reconnect") && (
                  <button
                    type="button"
                    onClick={() => {
                      setSyncError(null)
                      setSettingsOpen(true)
                      setDebugOpen(false)
                      setPortfolioOpen(false)
                    }}
                    className="font-medium underline underline-offset-2"
                  >
                    Open Settings
                  </button>
                )}
              </div>
              <button
                type="button"
                onClick={() => setSyncError(null)}
                className="shrink-0 text-red-500 hover:text-red-700"
                aria-label="Dismiss"
              >
                x
              </button>
            </div>
          )}
          {portfolioOpen ? (
            <PortfolioView
              onSelectBoard={(id) => {
                setPortfolioOpen(false)
                setCurrentId(id)
                setTab("attention")
              }}
            />
          ) : debugOpen ? (
            <DebugView
              boardId={currentId}
              onStorageChanged={() => void loadStorage()}
              onOpenSettings={() => {
                setSettingsOpen(true)
                setDebugOpen(false)
              }}
            />
          ) : settingsOpen ? (
            <SettingsView
              storage={storage}
              onStorageChanged={() => void loadStorage()}
              onOpenStorage={() => {
                setDebugOpen(true)
                setSettingsOpen(false)
              }}
            />
          ) : current == null ? (
            forges.length === 0 ? (
              // Connect-first funnel: the first step is a GitHub connection, not a board.
              <div className="mx-auto max-w-md space-y-4 py-12 text-center">
                <h2 className="text-xl font-semibold">Welcome to orgonzola</h2>
                <p className="text-sm text-muted-foreground">
                  To get started, connect your GitHub account. orgonzola reads your team's activity
                  and keeps everything on your machine. Once connected, you will create a board (a
                  team, an org, or a set of repositories to watch).
                </p>
                <button
                  type="button"
                  onClick={() => {
                    setSettingsOpen(true)
                    setDebugOpen(false)
                    setPortfolioOpen(false)
                  }}
                  className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                >
                  Connect GitHub
                </button>
              </div>
            ) : (
              <EmptyState
                title="Create your first board"
                hint="A board is your unit of interest - a team, an org, or a set of repositories. Click + New board, add a few people, optionally pin repositories, then Sync now."
              />
            )
          ) : (
            <div className="space-y-4">
              <div>
                <h1 className="text-2xl font-semibold">{current.name}</h1>
                <p className="text-sm text-muted-foreground">{boardSubtitle(current)}</p>
              </div>
              <div className="flex gap-1 border-b border-border">
                {visibleTabs.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => setTab(t.id)}
                    className={cn(
                      "-mb-px border-b-2 px-3 py-2 text-sm transition-colors",
                      activeTab === t.id
                        ? "border-primary font-medium text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              {activeTab === "attention" && (
                <OverviewView
                  boardId={current.id}
                  repos={overview}
                  teamRefs={teamRefs}
                  indexStates={indexStates}
                  dataVersion={dataVersion}
                  webBase={webBase}
                  onOpenPerson={openPerson}
                  onManageRepos={() => setTab("settings")}
                />
              )}
              {activeTab === "dashboard" && (
                <BoardDashboard
                  boardId={current.id}
                  repos={overview}
                  reposError={overviewError}
                  people={current.people.length}
                  indexStates={indexStates}
                  dataVersion={dataVersion}
                  webBase={webBase}
                  onRepoChanged={() => {
                    const id = currentIdRef.current
                    if (id) void loadOverview(id)
                  }}
                />
              )}
              {activeTab === "code" && (
                <CodeView
                  boardId={current.id}
                  scanEnabled={current.scan_dependencies}
                  dataVersion={dataVersion}
                  webBase={webBase}
                  onOpenPerson={openPerson}
                />
              )}
              {activeTab === "standup" && (
                <BoardStandup
                  boardId={current.id}
                  webBase={webBase}
                  dataVersion={dataVersion}
                  onOpenPerson={openPerson}
                  onOpenAttention={() => setTab("attention")}
                />
              )}
              {activeTab === "changes" && (
                <ChangesView
                  boardId={current.id}
                  dataVersion={dataVersion}
                  webBase={webBase}
                  onOpenPerson={openPerson}
                />
              )}
              {activeTab === "people" && (
                <BoardPeople boardId={current.id} webBase={webBase} focusLogin={personFocus} />
              )}
              {activeTab === "search" && (
                <SearchView
                  key={current.id}
                  boardId={current.id}
                  boardLabel={current.name}
                  repoCount={overview?.length ?? null}
                  onManageRepos={() => setTab("settings")}
                />
              )}
              {activeTab === "settings" && (
                <BoardSettings
                  board={current}
                  onChanged={onBoardChanged}
                  onDeleted={onBoardDeleted}
                  onSignalsChanged={() => {
                    const id = currentIdRef.current
                    if (id) void loadOverview(id)
                  }}
                  onOpenSettings={() => {
                    setSettingsOpen(true)
                    setDebugOpen(false)
                  }}
                  onOpenDebug={() => {
                    setDebugOpen(true)
                    setSettingsOpen(false)
                    setPortfolioOpen(false)
                  }}
                />
              )}
            </div>
          )}
        </main>
      </div>
      {assistantOpen && (
        <AssistantPanel
          boardId={current?.id ?? null}
          tab={activeTab}
          onClose={() => setAssistantOpen(false)}
        />
      )}
      {newBoardOpen && (
        <NewBoardDialog
          onClose={() => setNewBoardOpen(false)}
          onCreated={(b) => void onBoardCreated(b)}
          onOpenSettings={() => {
            setNewBoardOpen(false)
            setSettingsOpen(true)
            setDebugOpen(false)
          }}
        />
      )}
    </div>
  )
}
