import { cn } from "@/lib/utils"
import { type ReactNode, createContext, useCallback, useContext, useState } from "react"

// A small in-house toaster for async-action feedback (sync complete, digest sent/failed): a context
// plus a fixed stack of auto-dismissing cards.

type ToastKind = "success" | "error" | "info"
type ToastFn = (message: string, kind?: ToastKind) => void
type Toast = { id: number; kind: ToastKind; message: string }

const ToastContext = createContext<ToastFn>(() => {})

// Raise a toast from under the provider: `const toast = useToast(); toast("Saved", "success")`.
export function useToast(): ToastFn {
  return useContext(ToastContext)
}

const KIND_CLASS: Record<ToastKind, string> = {
  success: "border-emerald-200 bg-emerald-50 text-emerald-800",
  error: "border-red-200 bg-red-50 text-red-800",
  info: "border-border bg-background text-foreground",
}

let nextId = 1

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const dismiss = useCallback((id: number) => {
    setToasts((t) => t.filter((x) => x.id !== id))
  }, [])
  const push = useCallback<ToastFn>(
    (message, kind = "info") => {
      const id = nextId++
      setToasts((t) => [...t, { id, kind, message }])
      // Errors linger longer so they can be read; successes are brief.
      window.setTimeout(() => dismiss(id), kind === "error" ? 7000 : 4000)
    },
    [dismiss],
  )
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
        {toasts.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => dismiss(t.id)}
            className={cn(
              "pointer-events-auto rounded-md border px-3 py-2 text-left text-sm shadow-md",
              KIND_CLASS[t.kind],
            )}
          >
            {t.message}
          </button>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
