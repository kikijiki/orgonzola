import type { BoardPrView, RepoOverview } from "@/bindings"

export function attentionCount(r: RepoOverview): number {
  return r.attention.length + r.upstream.length
}

export function repoLabel(idOrName: string): string {
  return idOrName.replace(/^repo:/, "")
}

// A code hit's ref_id ("repo:owner/name#path/to/file") split into repo label and file path. Falls
// back to the raw id as the path when there is no "#".
export function codeRef(refId: string): { repo: string; path: string } {
  const hash = refId.indexOf("#")
  if (hash === -1) return { repo: "", path: repoLabel(refId) }
  return { repo: repoLabel(refId.slice(0, hash)), path: refId.slice(hash + 1) }
}

export function formatDuration(secs: number | null): string {
  if (secs == null) return "-"
  const days = secs / 86_400
  if (days >= 1) return `${days.toFixed(1)}d`
  const hours = secs / 3_600
  if (hours >= 1) return `${hours.toFixed(1)}h`
  return `${Math.max(1, Math.round(secs / 60))}m`
}

// A short relative age from an RFC-3339 timestamp (e.g. "5d", "3h", "just now"). Empty if
// missing or unparseable.
export function relativeAge(iso: string | null | undefined): string {
  if (!iso) return ""
  const then = Date.parse(iso)
  if (Number.isNaN(then)) return ""
  const secs = (Date.now() - then) / 1000
  if (secs < 60) return "just now"
  return formatDuration(secs)
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "-"
  const ms = Date.parse(iso)
  if (Number.isNaN(ms)) return "-"
  return new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" })
}

// A PR's display state; merged takes precedence over closed (GitHub reports merged PRs as closed).
export function prState(pr: BoardPrView): "merged" | "open" | "closed" {
  if (pr.merged_at) return "merged"
  if (pr.state === "open") return "open"
  return "closed"
}

export function attentionStyle(kind: string): { label: string; className: string } {
  switch (kind) {
    case "failing_ci":
      return { label: "failing CI", className: "border-red-200 bg-red-50 text-red-700" }
    case "flaky_ci":
      return { label: "flaky CI", className: "border-rose-200 bg-rose-50 text-rose-700" }
    case "merged_without_review":
      return {
        label: "merged w/o review",
        className: "border-amber-200 bg-amber-50 text-amber-700",
      }
    case "review_wait":
      return {
        label: "awaiting review",
        className: "border-blue-200 bg-blue-50 text-blue-700",
      }
    case "stale_pr":
      return { label: "stale PR", className: "border-yellow-200 bg-yellow-50 text-yellow-700" }
    case "risky_change":
      return {
        label: "risky change",
        className: "border-purple-200 bg-purple-50 text-purple-700",
      }
    case "aging_wip":
      return {
        label: "aging WIP",
        className: "border-amber-200 bg-amber-50 text-amber-700",
      }
    case "done_not_done":
      return {
        label: "merged, issue open",
        className: "border-orange-200 bg-orange-50 text-orange-700",
      }
    case "orphan_pr":
      return {
        label: "merged, untracked",
        className: "border-slate-200 bg-slate-50 text-slate-600",
      }
    default:
      return { label: kind, className: "border-slate-200 bg-slate-50 text-slate-600" }
  }
}

// A one-sentence definition per attention-item kind. Empty for an unknown kind.
export function attentionMeaning(kind: string): string {
  switch (kind) {
    case "failing_ci":
      return "The automated test/build checks (CI) on a recent change failed."
    case "flaky_ci":
      return "The checks passed only after a re-run on the same commit - a flaky test, not a real fix."
    case "merged_without_review":
      return "A pull request was merged without anyone reviewing it."
    case "review_wait":
      return "A pull request has been open a while with no first review yet - it is waiting for a reviewer."
    case "stale_pr":
      return "A pull request has sat open, untouched, past the staleness threshold."
    case "risky_change":
      return "An open pull request touches code that has not changed in a long time - higher risk of surprises."
    case "aging_wip":
      return "A tracked work item has been in progress longer than expected without finishing. The percentile badge compares its age against how long this board's recently finished work actually spent in progress; hover it for the window and sample size."
    case "done_not_done":
      return "A pull request that should close an issue was merged, but the issue is still open."
    case "orphan_pr":
      return "A pull request merged without being linked to any tracked issue - untracked work."
    default:
      return ""
  }
}

// A byte count as a short human string ("1.4 GB", "812 KB", "0 B"). Powers of 1024, matching `du`
// and the file manager.
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "-"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let n = Math.abs(bytes)
  let unit = 0
  while (n >= 1024 && unit < units.length - 1) {
    n /= 1024
    unit += 1
  }
  const sign = bytes < 0 ? "-" : ""
  // Whole bytes stay whole; scaled values get one decimal until large enough not to need it.
  const shown = unit === 0 ? String(Math.round(n)) : n >= 100 ? n.toFixed(0) : n.toFixed(1)
  return `${sign}${shown} ${units[unit]}`
}

// Plain-language name for a storage group. The wire values come from the Rust `StorageGroup`;
// an unknown one falls back to itself.
export function storageGroupLabel(group: string): string {
  switch (group) {
    case "search_index":
      return "Search index (vector + full text)"
    case "chunk_text":
      return "Indexed text"
    case "activity":
      return "Activity (commits, PRs, issues)"
    case "code":
      return "Code files and health"
    case "history":
      return "Metric history and digests"
    case "config":
      return "Boards, sources, and catalogue"
    case "internal":
      return "SQLite internals"
    default:
      return group
  }
}
