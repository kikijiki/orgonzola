import { useCallback, useEffect, useRef, useState } from "react"

// A per-board view preference persisted in localStorage, keyed by board id and a stable key.
// Reads on mount and on board change, falling back to `fallback` when nothing is stored or storage
// is unavailable. Setting it writes through.
export function useBoardPref<T>(boardId: string, key: string, fallback: T): [T, (next: T) => void] {
  const storageKey = `orgonzola:board:${boardId}:${key}`
  // Held in a ref so re-reads do not depend on a possibly fresh-literal `fallback`.
  const fallbackRef = useRef(fallback)
  fallbackRef.current = fallback

  const [value, setValue] = useState<T>(() => readPref(storageKey, fallbackRef.current))

  // Re-read when the board (and thus the storage key) changes.
  useEffect(() => {
    setValue(readPref(storageKey, fallbackRef.current))
  }, [storageKey])

  const set = useCallback(
    (next: T) => {
      setValue(next)
      writePref(storageKey, next)
    },
    [storageKey],
  )

  return [value, set]
}

function readPref<T>(storageKey: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(storageKey)
    return raw == null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

function writePref<T>(storageKey: string, value: T): void {
  try {
    localStorage.setItem(storageKey, JSON.stringify(value))
  } catch {
    // storage full or disabled: keep the in-memory value, drop the persistence.
  }
}
