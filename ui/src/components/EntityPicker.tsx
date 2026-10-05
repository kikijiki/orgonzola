import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useEffect, useId, useRef, useState } from "react"

// What a source search came back with. A search that could not run is not an empty result.
export type SearchOutcome = { ok: true; results: string[] } | { ok: false; reason: string }

// A type-ahead picker for a board source: queries the board's forge as you type and lists matches.
// With `allowManual` (the default), committing a value that matched nothing still calls `onSelect`,
// so private or just-created entities and offline use remain addable.
// `search` is the (debounced) forge query; it should resolve to a [`SearchOutcome`], not reject.
// On `ok: false` the picker says why and adds free text without claiming anything about the forge.
// Selecting clears the input.
export function EntityPicker({
  search,
  onSelect,
  placeholder,
  buttonLabel = "Add",
  allowManual = true,
  disabled = false,
}: {
  search: (query: string) => Promise<SearchOutcome>
  onSelect: (value: string) => void
  placeholder?: string
  buttonLabel?: string
  allowManual?: boolean
  disabled?: boolean
}) {
  const [value, setValue] = useState("")
  const [results, setResults] = useState<string[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [highlight, setHighlight] = useState(-1)
  // Why the forge could not be searched. Distinct from "searched, no matches".
  const [unavailable, setUnavailable] = useState<string | null>(null)
  // Note shown after a free-text add: a typo warning if the forge was searched, else "not checked".
  const [addNote, setAddNote] = useState<string | null>(null)
  // Monotonic request id so a slow earlier query cannot overwrite a newer one's results.
  const latest = useRef(0)
  const listboxId = useId()

  // Debounce the forge query; keep only the latest request's results.
  useEffect(() => {
    const q = value.trim()
    if (!q) {
      setResults([])
      setLoading(false)
      return
    }
    setLoading(true)
    const id = ++latest.current
    const timer = setTimeout(async () => {
      const outcome = await search(q)
      if (id !== latest.current) return // a newer query started; discard this one
      const hits = outcome.ok ? outcome.results : []
      setResults(hits)
      setUnavailable(outcome.ok ? null : outcome.reason)
      setHighlight(hits.length > 0 ? 0 : -1)
      setLoading(false)
    }, 250)
    return () => clearTimeout(timer)
  }, [value, search])

  const commit = (v: string, matched: boolean) => {
    const trimmed = v.trim()
    if (!trimmed) return
    onSelect(trimmed)
    // A typo warning only if the forge was actually searched; otherwise say nothing was checked.
    setAddNote(
      matched
        ? null
        : unavailable
          ? `Added "${trimmed}" - not checked against the forge: ${unavailable}`
          : `Added "${trimmed}" - we could not find this on the forge, so double-check the spelling.`,
    )
    setValue("")
    setResults([])
    setOpen(false)
    setHighlight(-1)
    latest.current++ // invalidate any in-flight query for the now-cleared input
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault()
      setOpen(true)
      setHighlight((h) => Math.min(h + 1, results.length - 1))
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      setHighlight((h) => Math.max(h - 1, 0))
    } else if (e.key === "Enter") {
      e.preventDefault()
      if (highlight >= 0 && highlight < results.length) commit(results[highlight], true)
      else if (allowManual) commit(value, results.includes(value.trim()))
    } else if (e.key === "Escape") {
      setOpen(false)
    }
  }

  const showDropdown = open && value.trim().length > 0

  return (
    <div className="relative flex items-center gap-2">
      <div className="relative flex-1">
        <input
          className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
          value={value}
          disabled={disabled}
          placeholder={placeholder}
          onChange={(e) => {
            setValue(e.target.value)
            setOpen(true)
            setAddNote(null)
          }}
          onFocus={() => setOpen(true)}
          // Delay close so a click on a result lands before the dropdown unmounts.
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={onKeyDown}
          role="combobox"
          aria-expanded={showDropdown}
          aria-controls={listboxId}
          aria-autocomplete="list"
        />
        {showDropdown && (
          <ul
            id={listboxId}
            className="absolute left-0 right-0 top-full z-10 mt-1 max-h-60 overflow-auto rounded-md border border-border bg-background py-1 shadow-md"
          >
            {loading ? (
              <li className="px-3 py-1.5 text-sm text-muted-foreground">searching...</li>
            ) : unavailable ? (
              <li className="px-3 py-1.5 text-sm text-amber-700">
                cannot search: {unavailable}
                {allowManual ? " - press Enter to add as typed" : ""}
              </li>
            ) : results.length === 0 ? (
              <li className="px-3 py-1.5 text-sm text-muted-foreground">
                {allowManual ? "no matches - press Enter to add as typed" : "no matches"}
              </li>
            ) : (
              results.map((r, i) => (
                <li key={r}>
                  <button
                    type="button"
                    className={cn(
                      "block w-full px-3 py-1.5 text-left text-sm hover:bg-muted",
                      i === highlight && "bg-muted",
                    )}
                    // onMouseDown, not onClick, so it fires before onBlur closes the list.
                    onMouseDown={(e) => {
                      e.preventDefault()
                      commit(r, true)
                    }}
                    onMouseEnter={() => setHighlight(i)}
                  >
                    {r}
                  </button>
                </li>
              ))
            )}
          </ul>
        )}
      </div>
      <Button
        onClick={() => commit(value, results.includes(value.trim()))}
        disabled={disabled || !value.trim()}
      >
        {buttonLabel}
      </Button>
      {addNote && <p className="absolute left-0 top-full mt-1 text-xs text-amber-700">{addNote}</p>}
    </div>
  )
}
