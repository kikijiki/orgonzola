import { type PortfolioBoardView, commands } from "@/bindings"
import { CHART_COLORS, TrendChart } from "@/components/charts"
import { Card, EmptyState, SkeletonCard, ViewHeader } from "@/components/primitives"
import { useEffect, useState } from "react"

// The Portfolio overview: one sparkline card per board, in board order. Boards are not sorted by
// metric and carry no tier badge. Clicking a card opens that board.
export function PortfolioView({ onSelectBoard }: { onSelectBoard: (id: string) => void }) {
  const [boards, setBoards] = useState<PortfolioBoardView[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let ignore = false
    void commands.portfolio().then((res) => {
      if (!ignore) {
        if (res.status === "ok") setBoards(res.data)
        else setError(res.error)
      }
    })
    return () => {
      ignore = true
    }
  }, [])

  return (
    <div className="space-y-4">
      <ViewHeader
        title="Portfolio"
        subtitle="A glance across every board's trend. Descriptive: boards are not ranked or tiered."
      />
      {error != null ? (
        <EmptyState title="Could not load the portfolio" hint={error} />
      ) : boards == null ? (
        <>
          <SkeletonCard rows={3} />
          <SkeletonCard rows={3} />
        </>
      ) : boards.length === 0 ? (
        <EmptyState
          title="No boards yet"
          hint="Create a board and sync it; its daily snapshots feed the portfolio trend."
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {boards.map((b) => (
            <PortfolioCard key={b.board_id} board={b} onOpen={() => onSelectBoard(b.board_id)} />
          ))}
        </div>
      )}
    </div>
  )
}

// One board's portfolio card: name, kind, latest WIP / needs-attention values, and a sparkline over
// the snapshot days (only with at least two days, so one point does not draw a flat line).
function PortfolioCard({ board, onOpen }: { board: PortfolioBoardView; onOpen: () => void }) {
  const last = board.trend[board.trend.length - 1]
  return (
    <Card className="space-y-2">
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full items-center justify-between gap-2"
      >
        <span className="truncate font-medium hover:underline">{board.name}</span>
        <span className="shrink-0 text-[10px] uppercase tracking-wide text-muted-foreground">
          {board.kind}
        </span>
      </button>
      {last && (
        <div className="flex gap-4 font-mono text-xs text-muted-foreground">
          <span>{last.wip} in flight</span>
          <span>{last.attention_count} needs attention</span>
        </div>
      )}
      {board.trend.length >= 2 ? (
        <TrendChart
          data={board.trend.map((t) => ({
            label: t.captured_on.slice(5),
            wip: t.wip,
            attention: t.attention_count,
          }))}
          series={[
            { key: "wip", label: "in flight", color: CHART_COLORS.blue },
            { key: "attention", label: "needs attention", color: CHART_COLORS.amber },
          ]}
        />
      ) : (
        <p className="text-xs text-muted-foreground">
          Not enough history yet - the trend accrues a point per day the app syncs.
        </p>
      )}
    </Card>
  )
}
