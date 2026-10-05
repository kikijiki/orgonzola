import {
  type CodeRiskView,
  type CouplingPairView,
  type DepUsageView,
  type FileHealthView,
  type HotspotView,
  type OwnershipRiskView,
  type PackageAdvisoryView,
  commands,
} from "@/bindings"
import { useToast } from "@/components/Toast"
import { UserLink } from "@/components/links"
import { Card, EmptyState, ExtLink, SkeletonCard } from "@/components/primitives"
import { Button } from "@/components/ui/button"
import { dirUrl, fileUrl } from "@/lib/forge"
import { repoLabel } from "@/lib/format"
import { Background, type Edge, type Node, ReactFlow } from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { useEffect, useState } from "react"

// The Code tab: behavioral code analysis over the board's repos. Hotspots are files that change
// often and churn a lot. About the code, not a person.
export function CodeView({
  boardId,
  scanEnabled,
  dataVersion,
  webBase,
  onOpenPerson,
}: {
  boardId: string
  scanEnabled: boolean
  // Bumped by the parent when a sync/index pass completes so the code signals refetch.
  dataVersion?: number
  // Board forge web root and "go to this user" nav, for file / author links.
  webBase: string | null
  onOpenPerson: (login: string) => void
}) {
  const [hotspots, setHotspots] = useState<HotspotView[] | null>(null)
  const [hotspotsError, setHotspotsError] = useState<string | null>(null)
  const [risks, setRisks] = useState<OwnershipRiskView[]>([])
  const [codeRisk, setCodeRisk] = useState<CodeRiskView[]>([])
  const [coupling, setCoupling] = useState<CouplingPairView[]>([])
  const [health, setHealth] = useState<FileHealthView[]>([])
  const [deps, setDeps] = useState<DepUsageView[]>([])
  const toast = useToast()
  // Reset to the skeleton only on a board switch; a dataVersion refresh updates in place.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset only on board switch
  useEffect(() => {
    setHotspots(null)
    setHotspotsError(null)
    setRisks([])
    setCodeRisk([])
    setCoupling([])
    setHealth([])
    setDeps([])
  }, [boardId])
  // biome-ignore lint/correctness/useExhaustiveDependencies: dataVersion is a refetch trigger, not read
  useEffect(() => {
    let ignore = false
    setHotspotsError(null)
    void Promise.all([
      commands.boardHotspots(boardId),
      commands.boardOwnershipRisks(boardId),
      commands.boardCodeRisk(boardId),
      commands.boardCoupling(boardId),
      commands.boardCodeHealth(boardId),
      commands.boardDependencies(boardId),
    ]).then(([hs, rk, cr, cp, ch, dp]) => {
      if (ignore) return
      if (hs.status === "ok") setHotspots(hs.data)
      else setHotspotsError(hs.error)
      if (rk.status === "ok") setRisks(rk.data)
      else toast("Could not load knowledge risk data", "error")
      if (cr.status === "ok") setCodeRisk(cr.data)
      else toast("Could not load code risk data", "error")
      if (cp.status === "ok") setCoupling(cp.data)
      else toast("Could not load change coupling data", "error")
      if (ch.status === "ok") setHealth(ch.data)
      else toast("Could not load code health data", "error")
      if (dp.status === "ok") setDeps(dp.data)
      else toast("Could not load dependency data", "error")
    })
    return () => {
      ignore = true
    }
  }, [boardId, dataVersion, toast])

  if (hotspotsError != null) {
    return <EmptyState title="Could not load code analysis" hint={hotspotsError} />
  }
  if (hotspots == null) {
    return (
      <div className="space-y-4">
        <SkeletonCard rows={5} />
        <SkeletonCard rows={3} />
      </div>
    )
  }
  if (hotspots.length === 0) {
    return (
      <EmptyState
        title="No hotspots yet"
        hint="Hotspots come from PR file changes. Sync the board's repos (with PR diffs) to populate them."
      />
    )
  }

  const maxChurn = Math.max(...hotspots.map((h) => h.churn))
  const maxChanges = Math.max(...hotspots.map((h) => h.changes))

  return (
    <div className="space-y-4">
      <Card className="space-y-3">
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span>cooler</span>
          <span className="inline-block h-3 w-24 rounded bg-gradient-to-r from-emerald-300 via-amber-300 to-red-500" />
          <span>hotter (changed more often)</span>
          <span className="ml-auto">tile size = churn (lines changed)</span>
        </div>
        {/* Heat map of hotspots: tile size scales with churn, color with change frequency. */}
        <div className="flex flex-wrap gap-1.5">
          {hotspots.map((h) => {
            // Width grows with churn, with a floor so small tiles stay legible.
            const churnFrac = h.churn / maxChurn
            const width = 64 + Math.round(churnFrac * 200)
            const fileHref = fileUrl(webBase, h.full_name, h.path)
            // Reduced motion drops the hover lift, so an inset ring marks the tile as hoverable.
            return (
              <a
                key={`${h.repo_id}#${h.path}`}
                href={fileHref ?? undefined}
                onClick={(e) => {
                  if (fileHref) {
                    e.preventDefault()
                    void commands.openUrl(fileHref)
                  }
                }}
                className="block overflow-hidden rounded p-2 text-white shadow-sm transition-transform hover:scale-[1.02] motion-reduce:hover:scale-100 motion-reduce:hover:ring-2 motion-reduce:hover:ring-inset motion-reduce:hover:ring-white"
                style={{ width, backgroundColor: heatColor(h.changes / maxChanges) }}
                title={`${repoLabel(h.full_name)}/${h.path}\n${h.changes} changes, ${h.churn} lines churned, ${h.authors} author(s)`}
              >
                <div className="truncate text-xs font-medium">{basename(h.path)}</div>
                <div className="truncate text-[10px] opacity-80">{repoLabel(h.full_name)}</div>
                <div className="mt-1 font-mono text-[10px] opacity-90">
                  {h.changes}x - {h.churn} churn
                </div>
              </a>
            )
          })}
        </div>
      </Card>

      {codeRisk.length > 0 && <CodeRiskCard risks={codeRisk} webBase={webBase} />}

      {health.some((h) => h.score < 7) && (
        <CodeHealthCard items={health.filter((h) => h.score < 7)} webBase={webBase} />
      )}

      <Card className="space-y-2">
        <h2 className="text-sm font-medium text-muted-foreground">Top hotspots</h2>
        <ul className="space-y-1 text-sm">
          {hotspots.slice(0, 12).map((h) => (
            <li
              key={`row-${h.repo_id}#${h.path}`}
              className="flex items-center justify-between gap-3"
            >
              <span className="min-w-0 truncate font-mono text-xs">
                <span className="text-muted-foreground">{repoLabel(h.full_name)}/</span>
                <ExtLink href={fileUrl(webBase, h.full_name, h.path)}>{h.path}</ExtLink>
              </span>
              <span className="shrink-0 font-mono text-xs text-muted-foreground">
                {h.changes} changes - {h.churn} churn - {h.authors} author(s)
              </span>
            </li>
          ))}
        </ul>
      </Card>

      {risks.length > 0 && (
        <Card className="space-y-2">
          <h2 className="text-sm font-medium text-muted-foreground">
            Knowledge risk (bus factor by module)
          </h2>
          <p className="text-xs text-muted-foreground">
            Modules concentrated in few authors - a key-person / offboarding risk. About the code,
            not a ranking of people. Heuristic from PR authorship.
          </p>
          <ul className="space-y-1.5">
            {risks.map((r) => (
              <li
                key={`${r.repo_id}#${r.module}`}
                className="flex items-center justify-between gap-3 text-sm"
              >
                <span className="min-w-0 truncate font-mono text-xs">
                  <span className="text-muted-foreground">{repoLabel(r.full_name)}/</span>
                  <ExtLink href={dirUrl(webBase, r.full_name, r.module)}>{r.module}</ExtLink>
                </span>
                <span className="flex shrink-0 items-center gap-2 text-xs">
                  <span
                    className={`rounded px-1.5 py-0.5 font-semibold ${
                      r.bus_factor <= 1
                        ? "bg-red-100 text-red-800"
                        : r.bus_factor <= 2
                          ? "bg-amber-100 text-amber-800"
                          : "bg-emerald-100 text-emerald-800"
                    }`}
                    title="bus factor: the fewest authors who together made more than half the changes here - low means knowledge is concentrated"
                  >
                    {r.bus_factor} {r.bus_factor === 1 ? "dev" : "devs"} know it
                  </span>
                  <span className="flex items-center gap-1 text-muted-foreground">
                    top owner{" "}
                    <UserLink login={r.top_author} webBase={webBase} onOpenPerson={onOpenPerson} />{" "}
                    ({Math.round(r.top_share * 100)}%)
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {coupling.length > 0 && <CouplingCard pairs={coupling} webBase={webBase} />}

      {deps.length > 0 && <DependencyCard deps={deps} />}

      {scanEnabled && <VulnPanel boardId={boardId} />}
    </div>
  )
}

// Code risk: files where hotspot churn, ownership concentration and merged-unreviewed rate line
// up. Each row shows its components, the core's reasons, and recent PRs that touched it.
function CodeRiskCard({ risks, webBase }: { risks: CodeRiskView[]; webBase: string | null }) {
  return (
    <Card className="space-y-2">
      <div>
        <h2 className="text-sm font-medium text-muted-foreground">
          Code risk (hotspot x ownership x review)
        </h2>
        <p className="text-xs text-muted-foreground">
          Files that change a lot, are owned by few, and keep landing unreviewed - the changes most
          worth a careful look. Deterministic and explainable; about the code, not a ranking of
          people.
        </p>
      </div>
      <ul className="space-y-2">
        {risks.slice(0, 12).map((r) => (
          <li
            key={`${r.repo_id}#${r.path}`}
            className="space-y-1 rounded-md border border-border p-2"
          >
            <div className="flex items-start justify-between gap-3">
              <span className="min-w-0 truncate font-mono text-xs">
                <span className="text-muted-foreground">{repoLabel(r.full_name)}/</span>
                <ExtLink href={fileUrl(webBase, r.full_name, r.path)}>{r.path}</ExtLink>
              </span>
              <span className="flex shrink-0 items-center gap-2 text-xs">
                <span
                  className={`rounded px-1.5 py-0.5 font-semibold ${
                    r.bus_factor <= 1
                      ? "bg-red-100 text-red-800"
                      : r.bus_factor <= 2
                        ? "bg-amber-100 text-amber-800"
                        : "bg-emerald-100 text-emerald-800"
                  }`}
                  title="bus factor: the fewest authors who together made more than half the changes here - low means knowledge is concentrated"
                >
                  {r.bus_factor} {r.bus_factor === 1 ? "dev" : "devs"} know it
                </span>
                {r.review_gap > 0 && (
                  <span
                    className="rounded bg-red-100 px-1.5 py-0.5 font-semibold text-red-800"
                    title={`${r.unreviewed} of ${r.merged_touches} merged PRs landed unreviewed`}
                  >
                    {Math.round(r.review_gap * 100)}% unreviewed
                  </span>
                )}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">{r.reasons.join(" - ")}</p>
            {r.recent_prs.length > 0 && (
              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs">
                <span className="text-muted-foreground">PRs:</span>
                {r.recent_prs.map((p) => (
                  <ExtLink key={p.number} href={p.url}>
                    <span className="font-mono" title={p.title}>
                      #{p.number}
                    </span>
                  </ExtLink>
                ))}
              </div>
            )}
          </li>
        ))}
      </ul>
    </Card>
  )
}

// Change coupling: file pairs that often appear in the same merged PR, implying structural
// coupling not expressible as a dependency.
function CouplingCard({ pairs, webBase }: { pairs: CouplingPairView[]; webBase: string | null }) {
  return (
    <Card className="space-y-2">
      <div>
        <h2 className="text-sm font-medium text-muted-foreground">Change coupling</h2>
        <p className="text-xs text-muted-foreground">
          Files that frequently change together in the same PR - implicit coupling not visible as a
          dependency. Changing one without the other is often a mistake.
        </p>
      </div>
      <ul className="space-y-1.5">
        {pairs.slice(0, 15).map((p) => (
          <li
            key={`${p.repo_id}#${p.path_a}+${p.path_b}`}
            className="flex items-center justify-between gap-3 text-sm"
          >
            <span className="min-w-0 truncate font-mono text-xs">
              <span className="text-muted-foreground">{repoLabel(p.full_name)}/</span>
              <ExtLink href={fileUrl(webBase, p.full_name, p.path_a)}>{p.path_a}</ExtLink>
              <span className="mx-1 text-muted-foreground">+</span>
              <ExtLink href={fileUrl(webBase, p.full_name, p.path_b)}>{p.path_b}</ExtLink>
            </span>
            <span className="flex shrink-0 items-center gap-2 text-xs">
              <span
                className={`rounded px-1.5 py-0.5 font-semibold ${
                  p.coupling >= 0.8
                    ? "bg-red-100 text-red-800"
                    : p.coupling >= 0.5
                      ? "bg-amber-100 text-amber-800"
                      : "bg-slate-100 text-slate-700"
                }`}
                title={`${p.together} PR(s) changed both files together (Jaccard coupling strength)`}
              >
                {Math.round(p.coupling * 100)}% coupled
              </span>
              <span className="text-muted-foreground">{p.together}x together</span>
            </span>
          </li>
        ))}
      </ul>
    </Card>
  )
}

// Dependency vulnerability scan: on demand, queries OSV for the board's dependency packages.
// Shown only when the board has scanning enabled (opt-in egress). Package-level.
function VulnPanel({ boardId }: { boardId: string }) {
  const [hits, setHits] = useState<PackageAdvisoryView[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const scan = async () => {
    setBusy(true)
    setError(null)
    const res = await commands.scanBoardDependencies(boardId)
    setBusy(false)
    if (res.status === "ok") setHits(res.data)
    else setError(res.error)
  }

  return (
    <Card className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-medium text-muted-foreground">Dependency vulnerabilities</h2>
          <p className="text-xs text-muted-foreground">
            Packages with known advisories (OSV). Package-level - verify your pinned version.
          </p>
        </div>
        <Button size="sm" disabled={busy} onClick={() => void scan()}>
          {busy ? "Scanning..." : "Scan now"}
        </Button>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {hits != null &&
        (hits.length === 0 ? (
          <p className="text-sm text-emerald-700">No known advisories on the scanned packages.</p>
        ) : (
          <ul className="space-y-1 text-sm">
            {hits.map((h) => (
              <li
                key={`${h.ecosystem}:${h.name}`}
                className="flex items-center justify-between gap-3"
              >
                <span className="min-w-0 truncate font-mono text-xs">
                  <span className="text-muted-foreground">{h.ecosystem}:</span>
                  {h.name}
                </span>
                <span className="shrink-0 text-xs text-red-700" title={h.advisories.join(", ")}>
                  {h.advisories.length} advisory{h.advisories.length === 1 ? "" : "(s)"}
                </span>
              </li>
            ))}
          </ul>
        ))}
    </Card>
  )
}

// Dependencies: a graph of repos and their shared dependencies, plus a list. Dependencies used
// by more than one repo signal coupling / shared upstream risk.
function DependencyCard({ deps }: { deps: DepUsageView[] }) {
  const shared = deps.filter((d) => d.repos.length > 1).slice(0, 15)
  const { nodes, edges } = buildDepGraph(shared)
  return (
    <Card className="space-y-3">
      <h2 className="text-sm font-medium text-muted-foreground">
        Dependencies ({deps.length}; {shared.length} shared across repos)
      </h2>
      {nodes.length > 0 && (
        <div className="h-80 overflow-hidden rounded-md border border-border">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            proOptions={{ hideAttribution: true }}
          >
            <Background />
          </ReactFlow>
        </div>
      )}
      <ul className="grid grid-cols-1 gap-1 text-sm sm:grid-cols-2">
        {deps.slice(0, 24).map((d) => (
          <li key={`${d.ecosystem}:${d.name}`} className="flex items-center justify-between gap-2">
            <span className="min-w-0 truncate font-mono text-xs">
              <span className="text-muted-foreground">{d.ecosystem}:</span>
              {d.name}
            </span>
            {d.repos.length > 1 && (
              <span
                className="shrink-0 rounded bg-violet-100 px-1.5 py-0.5 text-[10px] font-semibold text-violet-800"
                title={d.repos.join(", ")}
              >
                shared x{d.repos.length}
              </span>
            )}
          </li>
        ))}
      </ul>
    </Card>
  )
}

// Two-column React Flow graph: repos left, shared dependencies right, edges for declarations.
function buildDepGraph(shared: DepUsageView[]): { nodes: Node[]; edges: Edge[] } {
  const repos = [...new Set(shared.flatMap((d) => d.repos))]
  const nodes: Node[] = []
  const edges: Edge[] = []
  repos.forEach((repo, i) => {
    nodes.push({
      id: `repo:${repo}`,
      position: { x: 0, y: i * 64 },
      data: { label: repoLabel(repo) },
      style: {
        background: "#eff6ff",
        border: "1px solid #bfdbfe",
        borderRadius: 6,
        fontSize: 11,
        width: 150,
      },
    })
  })
  shared.forEach((d, j) => {
    const depId = `dep:${d.ecosystem}:${d.name}`
    nodes.push({
      id: depId,
      position: { x: 320, y: j * 44 },
      data: { label: d.name },
      style: {
        background: "#f5f3ff",
        border: "1px solid #ddd6fe",
        borderRadius: 6,
        fontSize: 11,
        width: 150,
      },
    })
    for (const repo of d.repos) {
      edges.push({
        id: `${repo}->${depId}`,
        source: `repo:${repo}`,
        target: depId,
        style: { stroke: "#c4b5fd" },
      })
    }
  })
  return { nodes, edges }
}

// Green -> amber -> red heat color for a 0..1 fraction.
function heatColor(t: number): string {
  if (t >= 0.66) return "#dc2626" // red-600
  if (t >= 0.33) return "#d97706" // amber-600
  return "#059669" // emerald-600
}

function basename(path: string): string {
  return path.split("/").pop() ?? path
}

// Code health: files whose AST complexity score (from branch density and LOC, 1-10, 10 =
// simplest) is below 7. Files with no tree-sitter grammar are never parsed, carry no score and
// are not listed; the card says so. Deterministic heuristic.
function CodeHealthCard({
  items,
  webBase,
}: {
  items: FileHealthView[]
  webBase: string | null
}) {
  return (
    <Card className="space-y-2">
      <div>
        <h2 className="text-sm font-medium text-muted-foreground">Code health (AST complexity)</h2>
        <p className="text-xs text-muted-foreground">
          Files scoring below 7 on a 1-10 scale derived from branch density and size. Higher is
          simpler. Heuristic from tree-sitter analysis; about the code, not a ranking of people.
          Only parsed files are scored - a file in a language with no grammar (Kotlin, Swift, Scala,
          SQL) is not analyzed and is not listed here.
        </p>
      </div>
      <ul className="space-y-1">
        {items.slice(0, 12).map((h) => (
          <li
            key={`${h.repo_id}#${h.path}`}
            className="flex items-center justify-between gap-3 text-sm"
          >
            <span className="min-w-0 truncate font-mono text-xs">
              <span className="text-muted-foreground">{repoLabel(h.full_name)}/</span>
              <ExtLink href={fileUrl(webBase, h.full_name, h.path)}>{h.path}</ExtLink>
            </span>
            <span className="flex shrink-0 items-center gap-2 text-xs">
              <span
                className={`rounded px-1.5 py-0.5 font-semibold tabular-nums ${
                  h.score <= 3
                    ? "bg-red-100 text-red-800"
                    : h.score <= 6
                      ? "bg-amber-100 text-amber-800"
                      : "bg-emerald-100 text-emerald-800"
                }`}
                title={`Health score ${h.score}/10 - LOC: ${h.loc}, functions: ${h.functions}, branches: ${h.branches}`}
              >
                {h.score}/10
              </span>
              <span className="text-muted-foreground">{h.loc} lines</span>
            </span>
          </li>
        ))}
      </ul>
    </Card>
  )
}
