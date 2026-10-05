import { events, type BriefEvent, commands } from "@/bindings"
import { Markdown } from "@/components/markdown"
import { Card, Skeleton } from "@/components/primitives"
import { Loader2, Sparkles } from "lucide-react"
import { useEffect, useState } from "react"

type Phase = "idle" | "thinking" | "streaming" | "done" | "hidden"

// The local-LLM brief: starts `start_board_brief` and streams the narration token by token, with
// progress phases (loading the model, prefill, writing). Renders nothing when no local model is
// configured (or the `llm` feature is off). Re-runs on a board switch; ignores other boards.
export function AiSummary({ boardId }: { boardId: string }) {
  const [phase, setPhase] = useState<Phase>("idle")
  // Raw engine stage during "thinking" ("loading" | "prefilling"), for the progress label.
  const [stage, setStage] = useState<string>("")
  const [text, setText] = useState("")

  useEffect(() => {
    let ignore = false
    // A fresh id per run; the backend tags every event with it, so overlapping generations
    // (e.g. StrictMode double mount) do not interleave tokens.
    const runId = crypto.randomUUID()
    setPhase("thinking")
    setStage("")
    setText("")
    const unlisten = events.briefEvent.listen((e) => {
      const ev: BriefEvent = e.payload
      // Drop events for another board or another request.
      if (ignore || ev.board_id !== boardId || ev.run_id !== runId) return
      switch (ev.phase) {
        case "delta":
          setPhase("streaming")
          setText((t) => t + ev.delta)
          break
        case "done":
          setPhase("done")
          break
        case "disabled":
          setPhase("hidden")
          break
        default:
          // loading / prefilling: still thinking; keep the stage for the label.
          setPhase("thinking")
          setStage(ev.phase)
      }
    })
    void commands.startBoardBrief(boardId, runId)
    return () => {
      ignore = true
      void unlisten.then((f) => f())
    }
  }, [boardId])

  if (phase === "idle" || phase === "hidden") return null
  const loading = phase === "thinking" && text === ""
  const label =
    phase === "streaming"
      ? "writing..."
      : stage === "loading"
        ? "loading model..."
        : stage === "prefilling"
          ? "reading the board..."
          : "thinking..."
  return (
    <Card className="space-y-1.5">
      <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        <Sparkles className="size-3.5 text-violet-500" />
        AI summary
        {phase !== "done" && (
          <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <Loader2 className="size-3 animate-spin" /> {label}
          </span>
        )}
      </div>
      {loading ? (
        <div className="space-y-1.5">
          <Skeleton className="h-3.5 w-full" />
          <Skeleton className="h-3.5 w-2/3" />
        </div>
      ) : (
        // The caret sits after <Markdown> as a sibling; inside the last <li>/<p> it would distort
        // that element's layout.
        <div className="text-sm text-foreground">
          <Markdown>{text}</Markdown>
          {phase === "streaming" && (
            <span className="ml-0.5 inline-block h-3.5 w-1.5 translate-y-0.5 animate-pulse bg-foreground/70" />
          )}
        </div>
      )}
    </Card>
  )
}
