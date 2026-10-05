import { useAssistantRuntime } from "@/components/assistant/runtime"
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react"
import { Loader2, Sparkles, X } from "lucide-react"

function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-end">
      <div className="max-w-[85%] rounded-lg bg-primary px-3 py-2 text-sm text-primary-foreground">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  )
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-start">
      <div className="max-w-[90%] whitespace-pre-wrap rounded-lg bg-muted px-3 py-2 text-sm text-foreground">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  )
}

function Composer() {
  return (
    <ComposerPrimitive.Root className="flex items-end gap-2 border-t border-border p-3">
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        placeholder="Ask about this board..."
        className="flex-1 resize-none rounded-md border border-border bg-background p-2 text-sm outline-none focus:ring-1 focus:ring-ring"
      />
      <ComposerPrimitive.Send className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50">
        Ask
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  )
}

// AI assistant panel: a right-docked chat over the current board, backed by the on-device Rig
// agent through assistant-ui's LocalRuntime and `useAssistantRuntime`.
export function AssistantPanel({
  boardId,
  tab,
  onClose,
}: {
  boardId: string | null
  tab: string
  onClose: () => void
}) {
  const runtime = useAssistantRuntime(boardId, tab)
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-full w-[360px] flex-col border-l border-border bg-background">
        <div className="flex items-center gap-1.5 border-b border-border px-3 py-2 text-sm font-medium">
          <Sparkles className="size-4 text-violet-500" />
          Assistant
          <button
            type="button"
            onClick={onClose}
            className="ml-auto rounded p-1 text-muted-foreground hover:bg-muted"
            aria-label="Close assistant"
          >
            <X className="size-4" />
          </button>
        </div>
        <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col">
          <ThreadPrimitive.Viewport className="flex-1 space-y-3 overflow-y-auto p-3">
            <ThreadPrimitive.Empty>
              <p className="text-xs text-muted-foreground">
                Ask about this board - attention, stale PRs, flow, what changed. Answers are
                grounded in synced data via on-device tools.
              </p>
            </ThreadPrimitive.Empty>
            <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
            <ThreadPrimitive.If running>
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Loader2 className="size-3 animate-spin" /> thinking...
              </div>
            </ThreadPrimitive.If>
          </ThreadPrimitive.Viewport>
          <Composer />
        </ThreadPrimitive.Root>
      </div>
    </AssistantRuntimeProvider>
  )
}
