import { commands } from "@/bindings"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { type ReactNode, useState } from "react"

// Per-URL avatar load outcome, cached for the session. Without it, every remount of an `<img>`
// (e.g. table re-render on sync) re-attempts the load and flickers. A failed URL shows initials.
const avatarOutcome = new Map<string, "ok" | "fail">()

// A small round avatar. Initials sit underneath; the image fades in on top once it loads, so a
// forge without avatars (or offline) stays at initials.
export function Avatar({
  login,
  src,
  size = 18,
}: {
  login: string
  src: string | null
  size?: number
}) {
  // Re-render only to drop a freshly-failed image; the outcome cache holds the durable state.
  const [, bump] = useState(0)
  const outcome = src ? avatarOutcome.get(src) : "fail"
  const showImg = src != null && outcome !== "fail"
  return (
    <span
      style={{ width: size, height: size }}
      className="relative inline-flex shrink-0 items-center justify-center overflow-hidden rounded-full bg-muted text-[9px] font-medium text-muted-foreground"
      aria-hidden
    >
      {login.slice(0, 2).toLowerCase()}
      {showImg && (
        <img
          src={src}
          alt=""
          loading="lazy"
          onLoad={() => avatarOutcome.set(src, "ok")}
          onError={() => {
            avatarOutcome.set(src, "fail")
            bump((n) => n + 1)
          }}
          className="absolute inset-0 h-full w-full object-cover"
        />
      )}
    </span>
  )
}

// Avatar + login, truncating with an ellipsis. The outer span carries `min-w-0` so a shrinking
// flex/grid parent works. The login is in its own span (blockified as a flex item, so truncation
// applies). A caller in a table cell must pass an explicit `max-w-*` in `className`.
export function NameTag({
  login,
  src,
  className,
}: {
  login: string
  src: string | null
  className?: string
}) {
  return (
    <span className={cn("flex min-w-0 items-center gap-1.5", className)}>
      <Avatar login={login} src={src} />
      <span className="min-w-0 truncate" title={login}>
        {login}
      </span>
    </span>
  )
}

// A flagged item's outbound link to the forge: `children` as a link when a URL is known, else
// plain text.
export function ExtLink({ href, children }: { href: string | null; children: ReactNode }) {
  if (!href) return <span>{children}</span>
  return (
    <a
      href={href}
      // A Tauri webview does not open `target=_blank`, so hand off to the OS opener.
      // preventDefault stops in-webview navigation; stopPropagation keeps ancestor row handlers out
      onClick={(e) => {
        e.preventDefault()
        e.stopPropagation()
        void commands.openUrl(href)
      }}
      // min-w-0 lets a flex-item ExtLink shrink instead of overflowing its container.
      className="min-w-0 text-primary underline underline-offset-2 hover:text-foreground"
    >
      {children}
    </a>
  )
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn("rounded-lg border border-border bg-background p-4", className)}>
      {children}
    </div>
  )
}

// A repo's background-index state, embedding progress (done / total units) while indexing, and
// the failure reason on error.
export type IndexInfo = {
  state: string
  done?: number
  total?: number
  error?: string | null
  // Files still to index, non-zero only while "partial".
  pending?: number
  // Files this pass could not fetch, non-zero only while "partial".
  skipped?: number
  // Why a "partial" or "paused" state is what it is. Kept apart from `error`: neither is a failure.
  note?: string | null
}

// A per-repo index status badge. Renders nothing when idle, unknown, or indexed. While indexing
// with a known total the label carries a percentage. On error the reason is the hover title.
export function IndexBadge({
  state,
  done,
  total,
  error,
  pending,
  skipped,
  note,
}: Partial<IndexInfo>) {
  const style: Record<string, { label: string; className: string }> = {
    queued: { label: "queued", className: "border-slate-200 bg-slate-50 text-slate-600" },
    indexing: { label: "indexing...", className: "border-amber-200 bg-amber-50 text-amber-700" },
    indexed: { label: "indexed", className: "border-green-200 bg-green-50 text-green-700" },
    // Not green: code search only covers part of the repo, so "indexed" would mislead.
    partial: { label: "partly indexed", className: "border-amber-200 bg-amber-50 text-amber-800" },
    paused: { label: "index paused", className: "border-slate-300 bg-slate-100 text-slate-700" },
    error: { label: "index error", className: "border-red-200 bg-red-50 text-red-700" },
  }
  const s = state ? style[state] : undefined
  if (!s) return null
  let label = s.label
  if (state === "indexing" && total && total > 0) {
    label = `indexing ${Math.floor((100 * (done ?? 0)) / total)}%`
  } else if (state === "partial" && (pending || skipped)) {
    // Show how much is missing on the badge itself.
    label = pending ? `partly indexed - ${pending} to go` : `partly indexed - ${skipped} skipped`
  }
  const title = state === "error" ? (error ?? undefined) : (note ?? undefined)
  return (
    <Badge variant="outline" className={s.className}>
      <span title={title}>{label}</span>
    </Badge>
  )
}

// A shimmering placeholder block, sized by className, that holds a card's shape while loading.
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-muted", className)} />
}

// A card-shaped loading placeholder: a title line plus `rows` shimmer lines, matching card padding.
export function SkeletonCard({ rows = 3, className }: { rows?: number; className?: string }) {
  return (
    <Card className={cn("space-y-3", className)}>
      <Skeleton className="h-4 w-40" />
      <div className="space-y-2">
        {Array.from({ length: rows }, (_, i) => `row-${i}`).map((k) => (
          <Skeleton key={k} className="h-3 w-full" />
        ))}
      </div>
    </Card>
  )
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-border p-8 text-center">
      <p className="text-sm font-medium text-foreground">{title}</p>
      {hint && <p className="mt-1 text-sm text-muted-foreground">{hint}</p>}
    </div>
  )
}

// A row of pill subtabs with optional count badges, for splitting a board tab into panes.
// Controlled: the caller owns the active id and the panes.
export function SubTabs<T extends string>({
  tabs,
  active,
  onSelect,
}: {
  tabs: { id: T; label: string; count?: number }[]
  active: T
  onSelect: (id: T) => void
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {tabs.map((t) => {
        const on = active === t.id
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => onSelect(t.id)}
            aria-pressed={on}
            className={cn(
              "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors",
              on
                ? "bg-muted font-medium text-foreground"
                : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
            )}
          >
            {t.label}
            {t.count != null && (
              <span
                className={cn(
                  "rounded-full px-1.5 font-mono text-[10px] leading-5",
                  on ? "bg-foreground/10 text-foreground" : "bg-muted text-muted-foreground",
                )}
              >
                {t.count}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}

export function ViewHeader({
  title,
  subtitle,
  actions,
}: {
  // Optional: tab views pass only a subtitle, since the board tabs already label the view.
  title?: string
  subtitle?: string
  actions?: ReactNode
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div>
        {title && <h1 className="text-xl font-semibold">{title}</h1>}
        {subtitle && (
          <p className={cn("text-sm text-muted-foreground", title && "mt-0.5")}>{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  )
}
