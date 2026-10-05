import {
  events,
  type DebugStatsView,
  type LlmStatusView,
  type StorageReportView,
  type StorageStateView,
  type StoredRepoView,
  type SyncProgressEvent,
  commands,
} from "@/bindings"
import { useConfirm } from "@/components/Confirm"
import { Card, IndexBadge, ViewHeader } from "@/components/primitives"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { formatBytes, repoLabel, storageGroupLabel } from "@/lib/format"
import { useCallback, useEffect, useState } from "react"

// One line in the live activity log: a fetch (sync) or index event, colored by outcome.
type LogLine = { at: string; repo: string; detail: string; ok: boolean }

const MAX_LOG = 80

// The Debug view: backend status, store/index counts, live queue state, an activity tail, and
// the repo-forgetting action.
export function DebugView({
  boardId,
  onStorageChanged,
  onOpenSettings,
}: {
  boardId: string | null
  // Tell the shell the bytes moved so the storage banner re-reads.
  onStorageChanged: () => void
  onOpenSettings: () => void
}) {
  const [stats, setStats] = useState<DebugStatsView | null>(null)
  const [log, setLog] = useState<LogLine[]>([])
  // Fetch lane's latest progress (transient, event-derived). The index queue comes from the
  // backend snapshot in `stats.index_active`, so it survives closing and reopening the view.
  const [fetch, setFetch] = useState<SyncProgressEvent | null>(null)
  // Live AI-inference phase, driven by `briefEvent`, plus tokens produced so far.
  const [ai, setAi] = useState<{ phase: string; tokens: number }>({ phase: "idle", tokens: 0 })
  // Storage is measured on request only: `dbstat` walks every page and per-repo sums scan the
  // payload tables, and `refresh` runs on every sync event.
  const [storage, setStorage] = useState<StorageReportView | null>(null)
  const [measuring, setMeasuring] = useState(false)
  const [storageError, setStorageError] = useState<string | null>(null)
  // Budget state, shown on the same card as the breakdown; much cheaper to compute.
  const [budget, setBudget] = useState<StorageStateView | null>(null)

  const refresh = useCallback(async () => {
    const res = await commands.debugStats()
    if (res.status === "ok") setStats(res.data)
    const b = await commands.storageState()
    if (b.status === "ok") setBudget(b.data)
  }, [])

  const measure = useCallback(async () => {
    setMeasuring(true)
    setStorageError(null)
    const res = await commands.storageReport()
    setMeasuring(false)
    if (res.status === "ok") setStorage(res.data)
    else setStorageError(res.error)
  }, [])

  const append = useCallback((line: LogLine) => {
    setLog((prev) => [line, ...prev].slice(0, MAX_LOG))
  }, [])

  useEffect(() => {
    void refresh()
    // Fetch lane: track latest progress, log per-repo completions, re-pull backend stats.
    const unlistenSync = events.syncProgressEvent.listen((e) => {
      const p = e.payload
      setFetch(p)
      if (p.finished) {
        append({
          at: new Date().toLocaleTimeString(),
          repo: repoLabel(p.source_id ?? p.item),
          detail: `fetched ${syncLine(p)}`,
          ok: p.ok,
        })
      }
      void refresh()
    })
    // Index lane: log indexed / error; queue contents come from the backend snapshot.
    const unlistenIndex = events.indexProgressEvent.listen((e) => {
      const p = e.payload
      if (p.state === "indexed" || p.state === "error") {
        append({
          at: new Date().toLocaleTimeString(),
          repo: repoLabel(p.full_name || p.repo_id),
          detail:
            p.state === "indexed"
              ? `indexed (${p.files} code file(s))`
              : `index error: ${p.error ?? "unknown"}`,
          ok: p.state !== "error",
        })
      }
      void refresh()
    })
    // AI brief lane. Lock onto one run at a time (adopt its `run_id` on start) so an overlapping
    // generation's tokens do not inflate the counter.
    let activeRun: string | null = null
    const unlistenBrief = events.briefEvent.listen((e) => {
      const p = e.payload
      switch (p.phase) {
        case "loading":
          activeRun = p.run_id
          setAi({ phase: "loading model", tokens: 0 })
          break
        case "prefilling":
          activeRun = p.run_id
          setAi({ phase: "prefilling (processing prompt)", tokens: 0 })
          break
        case "delta":
          if (p.run_id !== activeRun) break
          setAi((prev) => ({ phase: "generating", tokens: prev.tokens + 1 }))
          break
        case "done":
          if (p.run_id !== activeRun) break
          setAi((prev) => ({ phase: "done", tokens: prev.tokens }))
          break
        case "disabled":
          if (activeRun !== null && p.run_id !== activeRun) break
          setAi({ phase: "idle", tokens: 0 })
          break
        default:
          break
      }
    })
    return () => {
      void unlistenSync.then((f) => f())
      void unlistenIndex.then((f) => f())
      void unlistenBrief.then((f) => f())
    }
  }, [refresh, append])

  // The index lane's current contents, from the backend (survives reopen).
  const active = stats?.index_active ?? []
  const fetchLine =
    fetch == null
      ? "idle"
      : fetch.phase === "planning"
        ? `planning ${repoLabel(fetch.item)} (${fetch.item_done}/${fetch.item_total})`
        : `${repoLabel(fetch.item)} (${fetch.item_done}/${fetch.item_total})${
            fetch.step ? ` - ${fetch.step}` : ""
          }`

  return (
    <div className="space-y-4">
      <ViewHeader
        title="Debug"
        subtitle="What the shell is doing - backend, index, live sync, and store maintenance"
      />

      <div className="flex justify-end">
        <Button variant="outline" onClick={() => void refresh()}>
          Refresh
        </Button>
      </div>

      <Card className="space-y-2">
        <h2 className="text-sm font-semibold">Backend</h2>
        {stats == null ? (
          <p className="text-sm text-muted-foreground">loading...</p>
        ) : (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <Field label="Embedder" value={`${stats.embedder} (${stats.embedder_dims}d)`} />
            <Field label="Reranker" value={stats.reranker} />
            <Field
              label="Real-model build"
              value={stats.fastembed_built ? "yes (fastembed)" : "no (deterministic)"}
            />
            <Field label="Database" value={stats.db_path} mono />
          </dl>
        )}
      </Card>

      {stats != null && (
        <AiStatusCard
          llm={stats.llm}
          live={ai}
          canRun={boardId != null && stats.llm.feature_built && stats.llm.model_present}
          onRun={() => {
            if (boardId) void commands.startBoardBrief(boardId, crypto.randomUUID())
          }}
        />
      )}

      <Card className="space-y-2">
        <h2 className="text-sm font-semibold">Store + index</h2>
        {stats == null ? (
          <p className="text-sm text-muted-foreground">loading...</p>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <Stat label="Repos" value={stats.repos} />
            <Stat label="Commits" value={stats.commits} />
            <Stat label="Pull requests" value={stats.pull_requests} />
            <Stat label="Issues" value={stats.issues} />
            <Stat label="Sources" value={stats.sources} />
            <Stat label="Boards" value={stats.boards} />
            <Stat label="Embeddings" value={stats.embeddings_total} />
            <Stat label="Code chunks" value={stats.embeddings_code} />
            <Stat label="Activity chunks" value={stats.embeddings_activity} />
            <Stat label="Indexed code files" value={stats.code_files} />
          </div>
        )}
      </Card>

      <StorageCard
        report={storage}
        budget={budget}
        measuring={measuring}
        error={storageError}
        onMeasure={() => void measure()}
        onOpenSettings={onOpenSettings}
      />

      <ForgetRepoCard
        sizes={storage?.repos ?? null}
        onForgotten={() => {
          void refresh()
          // The measurement describes a database that no longer exists.
          setStorage(null)
          onStorageChanged()
        }}
      />

      <Card className="space-y-3">
        <h2 className="text-sm font-semibold">Queues</h2>
        <dl className="grid grid-cols-[8rem_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted-foreground">Fetch lane</dt>
          <dd className="truncate font-mono text-xs">{fetchLine}</dd>
          <dt className="text-muted-foreground">Index queue</dt>
          <dd className="font-mono text-xs">
            {stats?.index_queue_depth ?? 0} pending
            {active.length > 0 && ` - ${active.length} active`}
          </dd>
        </dl>
        {active.length > 0 && (
          <ul className="space-y-1">
            {active.map((e) => (
              <li key={e.repo_id} className="flex items-center gap-2 text-sm">
                <IndexBadge state={e.state} done={e.done} total={e.total} error={e.error} />
                <span className="truncate font-mono text-xs">{repoLabel(e.full_name)}</span>
                {e.state === "error" && e.error && (
                  <span className="truncate font-mono text-xs text-red-600" title={e.error}>
                    {e.error}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card className="space-y-2">
        <h2 className="text-sm font-semibold">Activity log</h2>
        {log.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Waiting for sync activity. Trigger Sync now, or wait for the scheduler.
          </p>
        ) : (
          <ul className="space-y-1 font-mono text-xs">
            {log.map((line, i) => (
              <li key={`${line.at}-${i}`} className="flex gap-2">
                <span className="shrink-0 text-muted-foreground">{line.at}</span>
                <span className="shrink-0">{line.repo}</span>
                <span className={line.ok ? "text-muted-foreground" : "text-red-600"}>
                  {line.detail}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  )
}

// One source's sync outcome: the non-zero per-type counts, or the error.
function syncLine(e: SyncProgressEvent): string {
  if (!e.ok) return `error: ${e.error ?? "unknown"}`
  const s = e.summary
  if (s == null) return "ok"
  const parts: string[] = []
  const add = (n: number, label: string) => {
    if (n > 0) parts.push(`${n} ${label}`)
  }
  add(s.commits, "commits")
  add(s.pull_requests, "PRs")
  add(s.issues, "issues")
  add(s.releases, "releases")
  add(s.reviews, "reviews")
  add(s.ci_runs, "CI")
  add(s.dependencies, "deps")
  add(s.code_files, "code files")
  return parts.length > 0 ? parts.join(", ") : "no changes"
}

// Local-LLM narrator status: a one-line verdict (built / configured / present / loaded / error)
// plus details, so a missing AI summary is explainable.
function AiStatusCard({
  llm,
  live,
  canRun,
  onRun,
}: {
  llm: LlmStatusView
  live: { phase: string; tokens: number }
  canRun: boolean
  onRun: () => void
}) {
  const verdict = !llm.feature_built
    ? "not built into this binary (run with the llm feature)"
    : !llm.configured
      ? "no model configured (ORGONZOLA_LLM_DIR / ORGONZOLA_LLM_GGUF)"
      : !llm.model_present
        ? "model file not found on disk"
        : llm.last_error
          ? "last run failed - falling back to the rule brief"
          : llm.loaded
            ? "loaded and ready"
            : "ready (model loads on the first summary)"
  const ok = llm.feature_built && llm.configured && llm.model_present && !llm.last_error
  return (
    <Card className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">AI inference (local LLM)</h2>
        <Button
          variant="outline"
          size="sm"
          disabled={!canRun}
          onClick={onRun}
          title={
            canRun
              ? "Generate a brief now and watch the phases below"
              : "Needs the llm feature, a present model, and a selected board"
          }
        >
          Run AI brief
        </Button>
      </div>
      <p className={`text-sm ${ok ? "text-emerald-700" : "text-amber-700"}`}>{verdict}</p>
      <p className="text-sm">
        <span className="text-muted-foreground">live: </span>
        <span
          className={`font-mono ${live.phase !== "idle" && live.phase !== "done" ? "animate-pulse text-violet-600" : ""}`}
        >
          {live.phase}
          {live.tokens > 0 ? ` - ${live.tokens} tokens` : ""}
        </span>
      </p>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <Field label="Feature built" value={llm.feature_built ? "yes" : "no"} />
        <Field label="Model loaded" value={llm.loaded ? "yes" : "no"} />
        <Field label="Model file" value={llm.model_path ?? "none"} mono />
        <Field label="File present" value={llm.model_present ? "yes" : "no"} />
      </dl>
      {llm.last_error && (
        <p className="rounded bg-red-50 px-2 py-1 font-mono text-xs text-red-700">
          {llm.last_error}
        </p>
      )}
    </Card>
  )
}

// What is on disk and what for. The database breakdown is a measurement when SQLite's `dbstat`
// is available and a payload estimate otherwise; per-repo figures are always estimates, because
// SQLite attributes pages to tables, not rows. Keep the labels that say which.
function StorageCard({
  report,
  budget,
  measuring,
  error,
  onMeasure,
  onOpenSettings,
}: {
  report: StorageReportView | null
  // Live budget state (cheap); the breakdown below is expensive and stays behind the button.
  budget: StorageStateView | null
  measuring: boolean
  error: string | null
  onMeasure: () => void
  onOpenSettings: () => void
}) {
  const [showTables, setShowTables] = useState(false)
  return (
    <Card className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">Storage</h2>
        <Button variant="outline" size="sm" disabled={measuring} onClick={onMeasure}>
          {measuring ? "Measuring..." : report ? "Measure again" : "Measure storage"}
        </Button>
      </div>
      {budget && (
        <div className="flex flex-wrap items-baseline justify-between gap-2 rounded-md border border-border px-3 py-2 text-sm">
          <span className={budget.stopped ? "text-red-700" : undefined}>{budget.message}</span>
          <button
            type="button"
            className="shrink-0 font-medium underline underline-offset-2"
            onClick={onOpenSettings}
          >
            Change the budget
          </button>
        </div>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}
      {report == null ? (
        <p className="text-sm text-muted-foreground">
          Not measured. Reading the breakdown walks every page of the database, so it runs when you
          ask rather than on every sync.
        </p>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="Database file" value={formatBytes(report.file_bytes)} />
            <Stat label="Write-ahead log" value={formatBytes(report.wal_bytes)} />
            <Stat label="Models on disk" value={formatBytes(modelBytes(report))} />
            <Stat label="Reusable free space" value={formatBytes(report.free_bytes)} />
          </div>
          <p className="text-xs text-muted-foreground">
            {report.measured
              ? "Table sizes are measured: real page counts from SQLite's dbstat."
              : "Table sizes are estimates: dbstat is not available in this build, so these are " +
                "summed value lengths and exclude every index and all page overhead."}{" "}
            Free space is inside the file and gets reused; the file only shrinks on a VACUUM, so
            forgetting a repo will not make it smaller.
          </p>

          <div className="space-y-1">
            <h3 className="text-xs font-semibold uppercase text-muted-foreground">
              What the database holds
            </h3>
            {report.groups.map((g) => (
              <Bar
                key={g.group}
                label={storageGroupLabel(g.group)}
                bytes={g.bytes}
                total={report.reserved_bytes}
                estimated={g.estimated}
              />
            ))}
            <Bar
              label="Free space (reusable, not returned to disk)"
              bytes={report.free_bytes}
              total={report.reserved_bytes}
              estimated={false}
            />
            {/* Under the estimate this is most of the file; without it the breakdown looks
                complete. */}
            <Bar
              label={
                report.measured
                  ? "Not attributed"
                  : "Not attributed (indexes and page overhead the estimate cannot see)"
              }
              bytes={report.residual_bytes}
              total={report.reserved_bytes}
              estimated={!report.measured}
            />
          </div>

          <div className="space-y-1">
            <h3 className="text-xs font-semibold uppercase text-muted-foreground">
              Models and files outside the database
            </h3>
            <ul className="space-y-1 text-sm">
              {report.assets.map((a) => (
                <li key={a.kind} className="flex items-baseline justify-between gap-2">
                  <span className="min-w-0">
                    <span>{a.name}</span>
                    {a.includes_others && (
                      <span className="text-muted-foreground">
                        {" "}
                        (holds the database and the rest)
                      </span>
                    )}
                    <span
                      className="block truncate font-mono text-xs text-muted-foreground"
                      title={a.path}
                    >
                      {a.path}
                    </span>
                  </span>
                  <span className="shrink-0 tabular-nums">
                    {a.exists ? formatBytes(a.bytes) : "not present"}
                  </span>
                </li>
              ))}
            </ul>
          </div>

          <div className="space-y-1">
            <button
              type="button"
              className="text-xs font-semibold uppercase text-muted-foreground hover:underline"
              onClick={() => setShowTables((v) => !v)}
            >
              {showTables ? "Hide" : "Show"} the per-table breakdown ({report.tables.length})
            </button>
            {showTables && (
              <ul className="max-h-72 space-y-1 overflow-y-auto text-sm">
                {report.tables.map((t) => (
                  <li key={t.name} className="flex items-baseline justify-between gap-2">
                    <span className="truncate font-mono text-xs">{t.name}</span>
                    <span className="shrink-0 tabular-nums">
                      {t.estimated ? "~" : ""}
                      {formatBytes(t.bytes)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
            <p className="text-xs text-muted-foreground">
              Each table includes its own indexes. The vector and full-text indexes are virtual
              tables, so their bytes are the shadow tables that actually hold them, reported under
              the index they belong to.
            </p>
          </div>
        </div>
      )}
    </Card>
  )
}

// Model directories only. The app-data entry is skipped because it contains them and would
// double count.
function modelBytes(report: StorageReportView): number {
  return report.assets.filter((a) => !a.includes_others).reduce((sum, a) => sum + a.bytes, 0)
}

// One breakdown line: label, proportional bar, bytes. `~` marks an estimate.
function Bar({
  label,
  bytes,
  total,
  estimated,
}: {
  label: string
  bytes: number
  total: number
  estimated: boolean
}) {
  const pct = total > 0 ? Math.max(0, Math.min(100, (bytes / total) * 100)) : 0
  return (
    <div className="space-y-0.5">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="truncate">{label}</span>
        <span className="shrink-0 tabular-nums">
          {estimated ? "~" : ""}
          {formatBytes(bytes)}
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted">
        <div className="h-1.5 rounded-full bg-primary" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

// Free up space per repo: drop a repo's search index or forget it entirely. Neither is
// board-scoped (both empty the repo for every board), so they live here next to the counts.
// Dropping the index is offered first: it only discards what the next sync rebuilds. Forgetting
// also takes the repo's daily metric history, which no re-sync restores.
function ForgetRepoCard({
  sizes,
  onForgotten,
}: {
  // Per-repo estimates from the Storage card, or null until measured. With them the list can be
  // ordered by cost.
  sizes: StorageReportView["repos"] | null
  onForgotten: () => void
}) {
  const confirm = useConfirm()
  const [repos, setRepos] = useState<StoredRepoView[] | null>(null)
  const [filter, setFilter] = useState("")
  const [busy, setBusy] = useState<string | null>(null)
  const [dropping, setDropping] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    const res = await commands.storedRepos()
    if (res.status === "ok") setRepos(res.data)
    else setError(res.error)
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const dropIndex = async (repo: StoredRepoView) => {
    const ok = await confirm({
      title: `Drop the search index for ${repo.full_name}?`,
      // Describe it as a cache eviction and name what stays.
      body:
        "This removes the repo's chunks from the code and activity search index and its per-file " +
        "records, and frees the space they take. Nothing is really lost: the next sync fetches the " +
        "files again and rebuilds the index. The repo's commits, pull requests, issues and its daily " +
        "metric history are not touched. Until it is rebuilt, this repo will not appear in Search or " +
        "in the assistant's answers.",
      confirmLabel: "Drop index",
    })
    if (!ok) return
    setDropping(repo.repo_id)
    setError(null)
    setResult(null)
    const res = await commands.dropRepoIndex(repo.repo_id)
    setDropping(null)
    if (res.status !== "ok") {
      setError(res.error)
      return
    }
    const rebuild =
      "The next sync rebuilds it. Reclaim the free space in Settings > Storage to give it back to the disk."
    setResult(
      `Dropped ${repo.full_name}'s index - ${res.data.chunks} chunks and ${res.data.code_files} file records. ${rebuild}`,
    )
    await load()
    onForgotten()
  }

  const forget = async (repo: StoredRepoView) => {
    const ok = await confirm({
      title: `Forget ${repo.full_name}?`,
      // Say what goes. A repo a board's person still owns is rediscovered on the next sync.
      body:
        "This deletes the repo's commits, pull requests, issues, code files, and everything indexed " +
        "from it, for every board. It cannot be undone - re-syncing brings back what the forge still " +
        "has, but not the daily metric history. If someone on a board still owns this repo, the next " +
        "sync discovers it again; remove the person, or pin the repos you want instead.",
      confirmLabel: "Forget repo",
    })
    if (!ok) return
    setBusy(repo.repo_id)
    setError(null)
    setResult(null)
    const res = await commands.forgetRepo(repo.repo_id)
    setBusy(null)
    if (res.status !== "ok") {
      setError(res.error)
      return
    }
    const r = res.data
    setResult(
      r.found
        ? `Forgot ${repo.full_name} - ${r.commits} commits, ${r.pull_requests} PRs, ${r.issues} issues, ` +
            `${r.code_files} code files, ${r.index_chunks} index chunks (${r.total_rows} rows).`
        : `${repo.full_name} was not in the store.`,
    )
    await load()
    onForgotten()
  }

  const bytesOf = (repoId: string) =>
    sizes?.find((s) => s.repo_id === repoId)?.estimated_bytes ?? null
  const shown = (repos ?? [])
    .filter((r) => r.full_name.toLowerCase().includes(filter.trim().toLowerCase()))
    // Unwatched repos first, then largest first once storage has been measured. Otherwise the
    // backend's name order stands.
    .sort((a, b) => {
      const watched = (r: StoredRepoView) => (r.pinned || r.discovered ? 1 : 0)
      if (watched(a) !== watched(b)) return watched(a) - watched(b)
      return (bytesOf(b.repo_id) ?? 0) - (bytesOf(a.repo_id) ?? 0)
    })
  const strays = (repos ?? []).filter((r) => !r.pinned && !r.discovered).length

  return (
    <Card className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">Free up space</h2>
        <input
          className="h-9 w-64 rounded-md border border-border bg-background px-2 text-sm"
          placeholder="filter repos..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Two ways to give space back, per repo. <strong>Drop index</strong> removes only what the
        next sync can rebuild - the search index and the per-file records - and leaves the repo's
        activity and its metric history alone. <strong>Forget</strong> removes the repo and
        everything derived from it, for every board, including the daily metric history, which does
        not come back. A repo no board watches still answers in Search and to the assistant until
        one of these is used.
        {strays > 0 && ` ${strays} repo(s) here are watched by no board.`}
        {sizes == null
          ? " Measure storage above to see what each one costs."
          : " Sizes are estimates for comparing repos against each other, not exact disk use."}
      </p>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {result && <p className="text-sm text-emerald-700">{result}</p>}
      {repos == null ? (
        <p className="text-sm text-muted-foreground">loading...</p>
      ) : shown.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {repos.length === 0 ? "No repos synced yet." : "No repo matches that filter."}
        </p>
      ) : (
        <ul className="max-h-72 space-y-1 overflow-y-auto">
          {shown.map((r) => (
            <li key={r.repo_id} className="flex items-center justify-between gap-2 text-sm">
              <span className="flex min-w-0 items-center gap-2">
                <span className="truncate font-mono text-xs">{repoLabel(r.full_name)}</span>
                {r.pinned && <Badge variant="outline">pinned</Badge>}
                {!r.pinned && r.discovered && <Badge variant="secondary">discovered</Badge>}
                {bytesOf(r.repo_id) != null && (
                  <span
                    className="shrink-0 tabular-nums text-xs text-muted-foreground"
                    title="Estimated content plus vector bytes. SQLite cannot attribute pages to rows, so this is for comparing repos, not for adding up to the file size."
                  >
                    ~{formatBytes(bytesOf(r.repo_id))}
                  </span>
                )}
              </span>
              <span className="flex shrink-0 items-center gap-2">
                {/* Available even for a pinned repo: dropping an index does not change what a board
                    watches. */}
                <Button
                  variant="outline"
                  size="sm"
                  disabled={busy != null || dropping != null}
                  title="Remove this repo's search index. The next sync rebuilds it; nothing else is touched."
                  onClick={() => void dropIndex(r)}
                >
                  {dropping === r.repo_id ? "Dropping..." : "Drop index"}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={r.pinned || busy != null || dropping != null}
                  title={
                    r.pinned
                      ? "A board pins this repo. Unpin it in that board's settings first."
                      : "Remove this repo and everything derived from it, including its metric history"
                  }
                  onClick={() => void forget(r)}
                >
                  {busy === r.repo_id ? "Forgetting..." : "Forget"}
                </Button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className={mono ? "truncate font-mono text-xs" : "truncate"} title={value}>
        {value}
      </dd>
    </>
  )
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-md border border-border p-2">
      <div className="text-lg font-semibold tabular-nums">{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </div>
  )
}
