import { Button } from "@/components/ui/button"
import { type ReactNode, createContext, useCallback, useContext, useEffect, useState } from "react"

// Reusable confirmation dialog: destructive actions route through `await confirm({...})`. A
// small modal plus a promise the caller awaits.

type ConfirmOptions = {
  title: string
  body: string
  // Confirm button label (defaults to "Delete"); destructive styling is applied.
  confirmLabel?: string
}
type ConfirmFn = (opts: ConfirmOptions) => Promise<boolean>

const ConfirmContext = createContext<ConfirmFn>(async () => false)

// `const confirm = useConfirm(); if (await confirm({ title, body })) { ...do it... }`.
export function useConfirm(): ConfirmFn {
  return useContext(ConfirmContext)
}

type Pending = ConfirmOptions & { resolve: (ok: boolean) => void }

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<Pending | null>(null)
  const confirm = useCallback<ConfirmFn>((opts) => {
    return new Promise<boolean>((resolve) => {
      // If a dialog is already open, cancel it (resolve false) so the prior caller's promise
      // resolves. Functional update to see the current pending (deps are []).
      setPending((prev) => {
        prev?.resolve(false)
        return { ...opts, resolve }
      })
    })
  }, [])
  const settle = useCallback(
    (ok: boolean) => {
      pending?.resolve(ok)
      setPending(null)
    },
    [pending],
  )
  // Escape cancels, handled at the window level so no focusable wrapper div is needed.
  useEffect(() => {
    if (!pending) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") settle(false)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [pending, settle])
  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      {pending && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* The backdrop is a button so click-to-cancel is also keyboard-reachable. */}
          <button
            type="button"
            aria-label="Cancel"
            onClick={() => settle(false)}
            className="absolute inset-0 bg-black/40"
          />
          <dialog
            open
            aria-modal="true"
            className="relative w-full max-w-sm space-y-3 rounded-lg border border-border bg-background p-5 text-foreground shadow-xl"
          >
            <h2 className="text-base font-semibold text-foreground">{pending.title}</h2>
            <p className="text-sm text-muted-foreground">{pending.body}</p>
            <div className="flex justify-end gap-2 pt-1">
              <Button variant="outline" size="sm" onClick={() => settle(false)}>
                Cancel
              </Button>
              <Button variant="destructive" size="sm" onClick={() => settle(true)}>
                {pending.confirmLabel ?? "Delete"}
              </Button>
            </div>
          </dialog>
        </div>
      )}
    </ConfirmContext.Provider>
  )
}
