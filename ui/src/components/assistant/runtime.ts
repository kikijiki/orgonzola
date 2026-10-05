import { events, type ChatTurnView, commands } from "@/bindings"
import { type ChatModelAdapter, useLocalRuntime } from "@assistant-ui/react"
import { useMemo } from "react"

function textOf(message: { content: readonly { type: string }[] }): string {
  return message.content
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join("")
}

// LocalRuntime adapter driving the assistant through the Tauri backend: each turn calls
// `start_assistant_chat` with board context and history and accumulates the streamed
// `assistantChatEvent` deltas. Yields are cumulative (assistant-ui replaces content each yield).
function makeAdapter(boardId: string | null, tab: string): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      if (!boardId) {
        yield { content: [{ type: "text", text: "Open a board to ask the assistant about it." }] }
        return
      }
      const question = textOf(messages[messages.length - 1])
      const history: ChatTurnView[] = messages
        .slice(0, -1)
        .map((m) => ({ role: m.role, content: textOf(m) }))
        .filter((t) => t.content.length > 0)

      const runId = crypto.randomUUID()
      const pending: string[] = []
      let finished = false
      let failed = false
      // Set on an engine error (phase "error"): the message to show instead of the generic
      // "not configured" notice, which only fits the "disabled" phase.
      let errorText: string | null = null
      let wake: (() => void) | null = null
      const ping = () => {
        wake?.()
        wake = null
      }

      const unlisten = await events.assistantChatEvent.listen((e) => {
        if (e.payload.run_id !== runId) return
        switch (e.payload.phase) {
          case "delta":
            pending.push(e.payload.delta)
            ping()
            break
          case "done":
            finished = true
            ping()
            break
          case "disabled":
            failed = true
            ping()
            break
          case "error":
            errorText = e.payload.delta
            failed = true
            ping()
            break
          // "thinking" keeps the spinner; nothing to emit yet.
        }
      })

      try {
        void commands.startAssistantChat(runId, boardId, history, question, tab)
        let text = ""
        while (!finished && !failed && !abortSignal.aborted) {
          if (pending.length === 0) {
            await new Promise<void>((resolve) => {
              wake = resolve
              if (abortSignal.aborted) resolve()
            })
          }
          for (const delta of pending.splice(0)) text += delta
          if (text.length > 0) yield { content: [{ type: "text", text }] }
        }
        if (failed) {
          const msg = errorText ?? ""
          const text =
            msg.length > 0
              ? `The assistant ran into an error: ${msg}`
              : "The assistant is unavailable. Enable it and choose a model under Settings > AI."
          yield { content: [{ type: "text", text }] }
        }
      } finally {
        unlisten()
      }
    },
  }
}

// Build the assistant runtime for the current board; re-created on board change so tools and
// RAG stay scoped to it.
export function useAssistantRuntime(boardId: string | null, tab: string) {
  const adapter = useMemo(() => makeAdapter(boardId, tab), [boardId, tab])
  return useLocalRuntime(adapter)
}
