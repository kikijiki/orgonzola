import type { BoardView, SyncProgressEvent } from "@/bindings"
import { Button } from "@/components/ui/button"
import { relativeAge, repoLabel } from "@/lib/format"
import { cn } from "@/lib/utils"
import { RefreshCw, Sparkles } from "lucide-react"
import { useEffect, useState } from "react"

const pct = (done: number, total: number) => (total > 0 ? Math.round((done / total) * 100) : 0)

function Bar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div
        className="h-full rounded-full bg-primary transition-all duration-200"
        style={{ width: `${value}%` }}
      />
    </div>
  )
}

// The left navigation: boards, board creation, the global sync control, and app settings.
export function Sidebar({
  boards,
  currentBoardId,
  settingsActive,
  debugActive,
  portfolioActive,
  onSelectBoard,
  onNewBoard,
  onOpenSettings,
  onOpenDebug,
  onOpenPortfolio,
  assistantActive,
  onToggleAssistant,
  healthy,
  syncing,
  status,
  lastSynced,
  progress,
  onSyncNow,
}: {
  boards: BoardView[]
  currentBoardId: string | null
  settingsActive: boolean
  debugActive: boolean
  portfolioActive: boolean
  onSelectBoard: (id: string) => void
  onNewBoard: () => void
  onOpenSettings: () => void
  onOpenDebug: () => void
  onOpenPortfolio: () => void
  // Whether the assistant chat panel is open. A side panel, orthogonal to the overlays.
  assistantActive: boolean
  onToggleAssistant: () => void
  healthy: boolean
  syncing: boolean
  status: string
  lastSynced: string | null
  // The latest sync-progress update, or null. The compact bar shows only during a manual sync.
  progress: SyncProgressEvent | null
  onSyncNow: () => void
}) {
  return (
    <nav className="flex w-56 shrink-0 flex-col border-r border-border bg-muted/40 p-3">
      <div className="px-2 py-3">
        <p className="text-lg font-semibold">orgonzola</p>
        <p className="text-xs text-muted-foreground">org intelligence</p>
      </div>

      <SyncControls
        healthy={healthy}
        syncing={syncing}
        status={status}
        lastSynced={lastSynced}
        progress={progress}
        onSyncNow={onSyncNow}
      />

      <p className="px-2 pb-1 pt-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Boards
      </p>
      <div className="flex flex-col gap-0.5">
        {boards.map((b) => (
          <button
            key={b.id}
            type="button"
            onClick={() => onSelectBoard(b.id)}
            className={cn(
              "flex items-center justify-between gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors",
              !settingsActive && !debugActive && !portfolioActive && currentBoardId === b.id
                ? "bg-primary text-primary-foreground"
                : "text-foreground hover:bg-muted",
            )}
          >
            <span className="truncate">{b.name}</span>
            <span className="shrink-0 text-[10px] uppercase tracking-wide opacity-60">
              {b.kind}
            </span>
          </button>
        ))}
        {boards.length === 0 && (
          <p className="px-3 py-2 text-sm text-muted-foreground">No boards yet</p>
        )}
      </div>

      <button
        type="button"
        onClick={onNewBoard}
        className="mt-1 rounded-md px-3 py-2 text-left text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
      >
        + New board
      </button>

      <div className="flex-1" />

      <Button
        variant={assistantActive ? "default" : "outline"}
        onClick={onToggleAssistant}
        className="mb-1 justify-start gap-1.5"
      >
        <Sparkles className="size-4 text-violet-500" />
        Assistant
      </Button>
      <Button
        variant={portfolioActive ? "default" : "outline"}
        onClick={onOpenPortfolio}
        className="mb-1 justify-start"
      >
        Portfolio
      </Button>
      <Button
        variant={debugActive ? "default" : "outline"}
        onClick={onOpenDebug}
        className="mb-1 justify-start"
      >
        Debug
      </Button>
      <Button
        variant={settingsActive ? "default" : "outline"}
        onClick={onOpenSettings}
        className="justify-start"
      >
        Settings
      </Button>
    </nav>
  )
}

// The global sync control: a fixed icon button, with two rows to its right, each a progress bar
// above a text line. Top is the overall phase, bottom is the current repo and its step bar.
function SyncControls({
  healthy,
  syncing,
  status,
  lastSynced,
  progress: p,
  onSyncNow,
}: {
  healthy: boolean
  syncing: boolean
  status: string
  lastSynced: string | null
  progress: SyncProgressEvent | null
  onSyncNow: () => void
}) {
  const planning = syncing && p?.phase === "planning"
  const hasProgress = syncing && p != null && p.item_total > 0
  const hasStep = hasProgress && p.step != null && p.step_total > 0
  const overall = planning
    ? `planning ${p?.item_done}/${p?.item_total}`
    : hasProgress
      ? `${p.item_done}/${p.item_total} repos`
      : "syncing..."
  const current = hasProgress && p.item ? repoLabel(p.item) : null
  return (
    <div className="flex items-stretch gap-2 px-2 pb-1">
      <Button
        variant="outline"
        onClick={onSyncNow}
        disabled={syncing}
        title="Sync now"
        className="h-auto min-h-9 w-9 shrink-0 self-stretch p-0"
      >
        <RefreshCw className={cn("size-4", syncing && "animate-spin")} />
      </Button>
      {syncing ? (
        <div className="min-w-0 flex-1 space-y-1.5">
          <ProgressRow bar={hasProgress ? pct(p.item_done, p.item_total) : null} text={overall} />
          {current && (
            <ProgressRow bar={hasStep ? pct(p.step_done, p.step_total) : null} text={current} />
          )}
        </div>
      ) : (
        <div className="flex h-9 min-w-0 flex-1 items-center gap-1.5 text-[11px] text-muted-foreground">
          <span
            className={cn(
              "h-2 w-2 shrink-0 rounded-full",
              healthy ? "bg-green-500" : "bg-slate-300",
            )}
            title={healthy ? "core healthy" : "core status unknown"}
          />
          {/* Live "synced Xm ago" after a successful sync; else the raw status. */}
          {lastSynced ? <Freshness iso={lastSynced} /> : <span className="truncate">{status}</span>}
        </div>
      )}
    </div>
  )
}

// A live "synced Xm ago" stamp; re-renders once a minute.
function Freshness({ iso }: { iso: string }) {
  const [, tick] = useState(0)
  useEffect(() => {
    const id = window.setInterval(() => tick((n) => n + 1), 60_000)
    return () => window.clearInterval(id)
  }, [])
  return <span className="truncate">synced {relativeAge(iso)}</span>
}

function ProgressRow({ bar, text }: { bar: number | null; text: string }) {
  return (
    <div className="space-y-0.5">
      {bar != null && <Bar value={bar} />}
      <p className="truncate font-mono text-[11px] text-muted-foreground">{text}</p>
    </div>
  )
}
