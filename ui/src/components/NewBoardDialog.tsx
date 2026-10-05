import { type BoardView, type ForgeView, commands } from "@/bindings"
import { EntityPicker, type SearchOutcome } from "@/components/EntityPicker"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useEffect, useState } from "react"

// Board kinds and what each is for. The kind is fixed at creation: it decides the scope inputs,
// the visible tabs and how repos are indexed. All kinds get People (`TABS_BY_KIND` in App.tsx);
// only team boards get Standup, and org boards build no code index (no Code, no Search).
const KINDS: { id: "team" | "repo" | "org"; label: string; blurb: string }[] = [
  {
    id: "team",
    label: "Team",
    blurb:
      "Track a group of people you choose, across whatever repositories they work in. Every view, and the only kind with a standup.",
  },
  {
    id: "repo",
    label: "Repository",
    blurb:
      "Watch one repository - its activity, its code, and its people. Same views as a team board apart from the standup; the people are whoever contributed, not a roster you pick.",
  },
  {
    id: "org",
    label: "Organization",
    blurb:
      "A rollup across a whole organization's busiest repositories: attention, dashboard, changes, and people. No code is indexed, so there is no code browsing or code search.",
  },
]

// Adapt a `searchForge*` result into the EntityPicker outcome. An error is passed through as the
// reason the search could not run, not flattened into an empty result.
const searchResults = (
  p: Promise<{ status: "ok"; data: string[] } | { status: "error"; error: string }>,
): Promise<SearchOutcome> =>
  p.then((r) =>
    r.status === "ok" ? { ok: true, results: r.data } : { ok: false, reason: r.error },
  )

// Create-board modal: pick a kind and forge, then set the identifying scope. Kind, forge and
// (for repo/org) target are fixed for the board's life. A team board takes a name; a repo/org
// board picks its one repo/org, which also names it.
export function NewBoardDialog({
  onClose,
  onCreated,
  onOpenSettings,
}: {
  onClose: () => void
  onCreated: (board: BoardView) => void
  // Opens Settings (where forges are managed) when none are configured.
  onOpenSettings: () => void
}) {
  const [kind, setKind] = useState<"team" | "repo" | "org">("team")
  const [forges, setForges] = useState<ForgeView[] | null>(null)
  const [forgeId, setForgeId] = useState("")
  const [name, setName] = useState("")
  const [target, setTarget] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let ignore = false
    void commands.listForges().then((res) => {
      if (ignore || res.status !== "ok") return
      setForges(res.data)
      // Preselect the only forge.
      if (res.data.length === 1) setForgeId(res.data[0].id)
    })
    return () => {
      ignore = true
    }
  }, [])

  // Switching kind drops a target picked under the previous kind.
  const pickKind = (k: "team" | "repo" | "org") => {
    setKind(k)
    setTarget("")
  }

  const ready = forgeId !== "" && (kind === "team" ? name.trim() !== "" : target !== "")

  const create = async () => {
    if (!ready || busy) return
    setBusy(true)
    setError(null)
    // Repo/org boards are named after their target; team boards by the typed name. The host
    // ignores `name` for repo/org, but pass the right value so errors read sensibly.
    const boardName = kind === "team" ? name.trim() : target
    const res = await commands.createBoard(
      boardName,
      kind,
      forgeId,
      kind === "team" ? null : target,
    )
    setBusy(false)
    if (res.status === "ok") onCreated(res.data)
    else setError(res.error)
  }

  // Close on Escape (plain overlay, not a native modal).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  const noForges = forges != null && forges.length === 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <dialog
        open
        className="m-0 w-[28rem] max-w-[90vw] space-y-4 rounded-lg border border-border bg-background p-5 text-foreground shadow-lg"
      >
        <div>
          <h2 className="text-lg font-semibold">New board</h2>
          <p className="text-sm text-muted-foreground">Pick what this board watches.</p>
        </div>

        <div className="space-y-2">
          {KINDS.map((k) => (
            <button
              key={k.id}
              type="button"
              onClick={() => pickKind(k.id)}
              className={cn(
                "block w-full rounded-md border px-3 py-2 text-left transition-colors",
                kind === k.id ? "border-primary bg-primary/5" : "border-border hover:bg-muted",
              )}
            >
              <span className="text-sm font-medium">{k.label}</span>
              <span className="block text-xs text-muted-foreground">{k.blurb}</span>
            </button>
          ))}
          <p className="text-xs text-muted-foreground">
            The board type is fixed once created - to switch, make a new board (your synced data is
            kept).
          </p>
        </div>

        {noForges ? (
          <p className="text-sm text-muted-foreground">
            No connection set up yet -{" "}
            <button type="button" className="text-primary underline" onClick={onOpenSettings}>
              connect GitHub in Settings
            </button>{" "}
            first, then create a board.
          </p>
        ) : (
          <>
            <label className="block space-y-1">
              <span className="text-xs font-medium text-muted-foreground">Connection</span>
              <select
                className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
                value={forgeId}
                onChange={(e) => setForgeId(e.target.value)}
              >
                <option value="">(pick a connection)</option>
                {forges?.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.name} ({f.kind})
                  </option>
                ))}
              </select>
            </label>

            {kind === "team" ? (
              <input
                className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
                value={name}
                placeholder="board name (e.g. Platform team)"
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void create()
                }}
              />
            ) : target ? (
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-700">
                  {target}
                </Badge>
                <button
                  type="button"
                  className="text-xs text-primary underline"
                  onClick={() => setTarget("")}
                >
                  change
                </button>
              </div>
            ) : (
              <EntityPicker
                key={kind}
                placeholder={kind === "repo" ? "owner/repo" : "org login (e.g. acme-corp)"}
                buttonLabel={kind === "repo" ? "Set repo" : "Set org"}
                disabled={forgeId === ""}
                search={(q) =>
                  searchResults(
                    kind === "repo"
                      ? commands.searchForgeRepos(forgeId, q)
                      : commands.searchForgeOrgs(forgeId, q),
                  )
                }
                onSelect={(v) => setTarget(v)}
              />
            )}
          </>
        )}

        {error && <p className="text-sm text-red-600">{error}</p>}

        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={() => void create()} disabled={busy || !ready}>
            {busy ? "Creating..." : "Create board"}
          </Button>
        </div>
      </dialog>
    </div>
  )
}
