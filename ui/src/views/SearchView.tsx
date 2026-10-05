import { type SearchHit, commands } from "@/bindings"
import { Card, EmptyState, ExtLink, ViewHeader } from "@/components/primitives"
import { Button } from "@/components/ui/button"
import { codeRef, repoLabel } from "@/lib/format"
import { useState } from "react"

type Mode = "activity" | "code"

const MODES: { id: Mode; label: string }[] = [
  { id: "activity", label: "Activity" },
  { id: "code", label: "Code" },
]

const PLACEHOLDER: Record<Mode, string> = {
  activity: "e.g. authentication flow, flaky test, migration",
  code: "e.g. retry logic, parse the manifest, where we open the store",
}

// Semantic search over the synced index, scoped to the board in view. Activity searches commit
// messages and issue titles; Code searches indexed source files. The scope is shown to the user.
// App.tsx keys this on the board id, so switching boards remounts it and no result outlives
// its board.
export function SearchView({
  boardId,
  boardLabel,
  repoCount,
  onManageRepos,
}: {
  boardId: string
  boardLabel: string
  // Repos the board covers, or null while the overview loads. Zero means nothing to search.
  repoCount: number | null
  onManageRepos: () => void
}) {
  const [mode, setMode] = useState<Mode>("activity")
  const [query, setQuery] = useState("")
  const [hits, setHits] = useState<SearchHit[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const nothingToSearch = repoCount === 0

  const run = async () => {
    if (!query.trim() || busy || nothingToSearch) return // in-flight guard
    setBusy(true)
    setError(null)
    const res =
      mode === "code"
        ? await commands.semanticCodeSearch(boardId, query, 10)
        : await commands.semanticSearch(boardId, query, 10)
    if (res.status === "ok") setHits(res.data)
    else {
      setHits(null)
      setError(res.error)
    }
    setBusy(false)
  }

  const pick = (next: Mode) => {
    setMode(next)
    setHits(null)
    setError(null)
  }

  return (
    <div className="space-y-4">
      <ViewHeader subtitle={scopeLine(boardLabel, repoCount)} />

      <div className="inline-flex rounded-md border border-border p-0.5">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => pick(m.id)}
            className={`rounded px-3 py-1 text-sm ${
              mode === m.id ? "bg-muted font-medium" : "text-muted-foreground"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2">
        <input
          className="h-9 flex-1 rounded-md border border-border bg-background px-3 text-sm disabled:opacity-50"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void run()
          }}
          placeholder={PLACEHOLDER[mode]}
          disabled={nothingToSearch}
        />
        <Button onClick={() => void run()} disabled={busy || nothingToSearch}>
          {busy ? "Searching..." : "Search"}
        </Button>
      </div>

      {nothingToSearch && (
        <EmptyState
          title="Nothing to search"
          hint={`${boardLabel} has no repos yet, so there is no index to search. Add or discover repos on the Settings tab, then sync them.`}
        />
      )}

      {error != null && <EmptyState title="Search failed" hint={error} />}

      {!nothingToSearch && hits != null && hits.length === 0 && (
        <EmptyState title="No matches" hint={noMatchHint(mode, boardLabel, repoCount)} />
      )}

      {hits != null && hits.length > 0 && (
        <ul className="space-y-2">
          {hits.map((h, i) =>
            mode === "code" ? (
              <CodeHit key={key(h, i)} hit={h} />
            ) : (
              <ActivityHit key={key(h, i)} hit={h} />
            ),
          )}
        </ul>
      )}

      {nothingToSearch && (
        <button
          type="button"
          className="text-sm text-muted-foreground underline underline-offset-2"
          onClick={onManageRepos}
        >
          Manage this board's repos
        </button>
      )}
    </div>
  )
}

// What the search covers, in plain words. The count is shown only once known.
function scopeLine(boardLabel: string, repoCount: number | null): string {
  if (repoCount == null) return `Searches the repos on ${boardLabel}, and nothing outside it.`
  if (repoCount === 0) return `${boardLabel} has no repos, so there is nothing to search.`
  const repos = repoCount === 1 ? "1 repo" : `${repoCount} repos`
  return `Searches the ${repos} on ${boardLabel}, and nothing outside it.`
}

// An empty result has two causes: the board's repos hold nothing matching, or they are not
// indexed yet. The copy tells them apart.
function noMatchHint(mode: Mode, boardLabel: string, repoCount: number | null): string {
  const where =
    repoCount == null ? `the repos on ${boardLabel}` : `${boardLabel}'s ${repoCount} repos`
  return mode === "code"
    ? `Nothing in ${where}. Try different words, or check that these repos have finished indexing their code.`
    : `Nothing in ${where}. Try different words, or sync more activity for these repos first.`
}

// Multiple chunks of one file (code) share ref_kind:ref_id, so fold in the list index to keep
// the React key unique. The list is rebuilt per search and never reordered, so it is stable.
const key = (h: SearchHit, i: number) => `${h.ref_kind}:${h.ref_id}:${i}`

function ActivityHit({ hit }: { hit: SearchHit }) {
  return (
    <li>
      <Card className="space-y-1">
        <div className="flex items-center justify-between font-mono text-xs text-muted-foreground">
          <span>
            {hit.ref_kind} {repoLabel(hit.ref_id)}
          </span>
          <span>{hit.score.toFixed(3)}</span>
        </div>
        <p className="text-sm">{hit.chunk}</p>
      </Card>
    </li>
  )
}

function CodeHit({ hit }: { hit: SearchHit }) {
  const { repo, path } = codeRef(hit.ref_id)
  return (
    <li>
      <Card className="space-y-1">
        <div className="flex items-center justify-between font-mono text-xs text-muted-foreground">
          <span>
            {repo && <span className="text-muted-foreground/70">{repo} </span>}
            <ExtLink href={hit.url}>{path}</ExtLink>
          </span>
          <span>{hit.score.toFixed(3)}</span>
        </div>
        <pre className="overflow-x-auto rounded bg-muted p-2 font-mono text-xs">{hit.chunk}</pre>
      </Card>
    </li>
  )
}
