import {
  events,
  type ForgeView,
  type LlmCatalogView,
  type LlmDownloadEvent,
  type OrphanRepoReportView,
  type StorageStateView,
  type TrackerView,
  commands,
} from "@/bindings"
import { useConfirm } from "@/components/Confirm"
import { Card, ViewHeader } from "@/components/primitives"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { formatBytes } from "@/lib/format"
import { type ReactNode, useCallback, useEffect, useState } from "react"

// One labelled block of the Settings page, a subject on its own.
function Section({
  title,
  hint,
  children,
}: {
  title: string
  hint: string
  children: ReactNode
}) {
  return (
    <section className="space-y-2">
      <div>
        <h2 className="text-sm font-semibold">{title}</h2>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </div>
      {children}
    </section>
  )
}

function NumberField({
  id,
  label,
  value,
  onChange,
  placeholder,
}: {
  id: string
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="w-64 text-sm" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        type="number"
        min={1}
        className="h-9 w-28 rounded-md border border-border bg-background px-3 text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  )
}

// Application settings: scheduler cadence, stale-PR watchdog, digest delivery, local model,
// and connections.
export function SettingsView({
  storage,
  onStorageChanged,
  onOpenStorage,
}: {
  // Live storage state, owned by the shell so the banner and this card agree.
  storage: StorageStateView | null
  onStorageChanged: () => void
  onOpenStorage: () => void
}) {
  const [period, setPeriod] = useState("")
  const [staleDays, setStaleDays] = useState("")
  const [webhook, setWebhook] = useState("")
  const [scheduleHours, setScheduleHours] = useState("")
  // The budget in gigabytes. Empty means no limit.
  const [budgetGb, setBudgetGb] = useState("")
  // Which card's Save produced the message, so the result line appears there.
  const [message, setMessage] = useState<{
    card: "sync" | "digest" | "storage"
    text: string
  } | null>(null)

  const load = useCallback(async () => {
    const res = await commands.getSettings()
    if (res.status === "ok") {
      setPeriod(String(res.data.sync_period_secs))
      setStaleDays(String(res.data.stale_pr_days))
      setWebhook(res.data.digest_webhook_url ?? "")
      setScheduleHours(
        res.data.digest_schedule_hours != null ? String(res.data.digest_schedule_hours) : "",
      )
      // Stored in MB, edited in GB; two decimals so a fractional-GB budget round-trips.
      setBudgetGb(
        res.data.storage_budget_mb != null
          ? String(Number((res.data.storage_budget_mb / 1024).toFixed(2)))
          : "",
      )
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  // All values live in one settings row, so every Save writes all of them; `card` only picks where
  // the result line shows.
  const save = async (card: "sync" | "digest" | "storage") => {
    setMessage(null)
    const secs = Number.parseInt(period, 10)
    const days = Number.parseInt(staleDays, 10)
    if (!Number.isFinite(secs) || secs < 1) {
      setMessage({ card, text: "poll period must be a whole number of seconds, at least 1" })
      return
    }
    if (!Number.isFinite(days) || days < 1) {
      setMessage({ card, text: "stale-PR threshold must be a whole number of days, at least 1" })
      return
    }
    const hours = Number.parseInt(scheduleHours, 10)
    // Blank means no limit. A non-positive number is an error: reading it as no-limit would drop a
    // ceiling the user meant to set.
    const gb = Number.parseFloat(budgetGb)
    if (budgetGb.trim() !== "" && !(Number.isFinite(gb) && gb > 0)) {
      setMessage({ card, text: "storage budget must be a positive number of gigabytes, or blank" })
      return
    }
    const budgetMb = budgetGb.trim() === "" ? null : Math.max(1, Math.round(gb * 1024))
    const res = await commands.setSettings(
      secs,
      days,
      webhook.trim() || null,
      Number.isFinite(hours) && hours > 0 ? hours : null,
      budgetMb,
    )
    setMessage({
      card,
      text:
        res.status === "ok" ? "saved - applies on the next refresh / scheduler pass" : res.error,
    })
    if (res.status === "ok") onStorageChanged()
  }

  const resultLine = (card: "sync" | "digest" | "storage") =>
    message?.card === card ? <p className="text-sm text-muted-foreground">{message.text}</p> : null

  return (
    <div className="space-y-6">
      <ViewHeader title="Settings" />

      <Section
        title="Sync"
        hint="How often orgonzola polls your connections, and how long an open PR may sit before it counts as stale."
      >
        <Card className="space-y-3">
          <NumberField
            id="sync-period"
            label="scheduler poll period (seconds)"
            value={period}
            onChange={setPeriod}
          />
          <NumberField
            id="stale-days"
            label="stale open-PR threshold (days)"
            value={staleDays}
            onChange={setStaleDays}
          />
          <Button onClick={() => void save("sync")}>Save</Button>
          {resultLine("sync")}
        </Card>
      </Section>

      <Section
        title="Digest delivery"
        hint="Where the standup digest goes when you send it, and whether it is sent on a schedule."
      >
        <Card className="space-y-3">
          <div className="flex items-center gap-3">
            <label className="w-64 text-sm" htmlFor="digest-webhook">
              digest webhook URL (optional)
            </label>
            <input
              id="digest-webhook"
              type="url"
              className="h-9 min-w-[20rem] flex-1 rounded-md border border-border bg-background px-3 text-sm"
              value={webhook}
              onChange={(e) => setWebhook(e.target.value)}
              placeholder="https://hooks.slack.com/services/... (Slack-compatible)"
            />
          </div>
          <p className="text-xs text-muted-foreground">
            A Slack-compatible incoming-webhook URL. When set, the Standup tab's "Send digest" posts
            the digest to it. Empty = no egress; nothing is ever sent without this and an explicit
            click.
          </p>
          <NumberField
            id="digest-schedule"
            label="auto-send digest every (hours)"
            value={scheduleHours}
            onChange={setScheduleHours}
            placeholder="off"
          />
          <p className="text-xs text-muted-foreground">
            With a webhook set, auto-send every board's digest on this cadence (e.g. 24 = daily).
            Blank = off; auto-egress only when both a webhook and a cadence are set.
          </p>
          <Button onClick={() => void save("digest")}>Save</Button>
          {resultLine("digest")}
        </Card>
      </Section>

      <Section
        title="Storage"
        hint="How much disk orgonzola may use. It stops syncing and indexing at this number and never deletes anything to stay under it."
      >
        <StorageSection
          storage={storage}
          budgetGb={budgetGb}
          onBudgetGb={setBudgetGb}
          onSave={() => void save("storage")}
          onStorageChanged={onStorageChanged}
          onOpenStorage={onOpenStorage}
          result={resultLine("storage")}
        />
      </Section>

      <Section
        title="AI (local LLM)"
        hint="An optional on-device model that rephrases the attention brief. Off by default; the deterministic brief stands alone."
      >
        <LlmSection />
      </Section>

      <Section
        title="Connections"
        hint="The services orgonzola reads from. Read-only: nothing is ever written back. A board picks its connection under its own Settings."
      >
        <ConnectionsSection />
      </Section>
    </div>
  )
}

// GitHub's classic-token page with the needed scopes pre-selected. read:org covers team/org reads;
// repo covers private repos and statuses.
const GITHUB_TOKEN_URL =
  "https://github.com/settings/tokens/new?scopes=repo,read:org&description=orgonzola"

type DeviceAuth = { forgeId: string; userCode: string; verificationUri: string }

// The storage budget: the ceiling, current usage, and the reclaim that costs nothing. The Debug
// view keeps the per-table and per-repo breakdown.
function StorageSection({
  storage,
  budgetGb,
  onBudgetGb,
  onSave,
  onStorageChanged,
  onOpenStorage,
  result,
}: {
  storage: StorageStateView | null
  budgetGb: string
  onBudgetGb: (v: string) => void
  onSave: () => void
  onStorageChanged: () => void
  onOpenStorage: () => void
  result: ReactNode
}) {
  const confirm = useConfirm()
  const [reclaiming, setReclaiming] = useState(false)
  const [reclaimed, setReclaimed] = useState<string | null>(null)
  // Orphaned repos: pinned by no board and discovered by none. Checked on demand, since the
  // per-repo estimator scans the payload tables.
  const [orphans, setOrphans] = useState<OrphanRepoReportView | null>(null)
  const [checkingOrphans, setCheckingOrphans] = useState(false)
  const [orphansError, setOrphansError] = useState<string | null>(null)
  const [reclaimingOrphans, setReclaimingOrphans] = useState(false)
  const [orphansReclaimed, setOrphansReclaimed] = useState<string | null>(null)

  const checkOrphans = async () => {
    setCheckingOrphans(true)
    setOrphansError(null)
    setOrphansReclaimed(null)
    const res = await commands.orphanRepos()
    setCheckingOrphans(false)
    if (res.status === "ok") setOrphans(res.data)
    else setOrphansError(res.error)
  }

  const reclaimOrphans = async () => {
    if (!orphans || orphans.count === 0) return
    const ok = await confirm({
      title: `Reclaim ${orphans.count} orphaned repo(s)?`,
      // Names what goes and what does not come back, matching the per-repo forget dialog in Debug.
      body:
        "These repos are pinned by no board and discovered by none either - most likely left behind " +
        "by a board that was deleted. This removes each one and everything derived from it: commits, " +
        "pull requests, issues, releases, CI runs, code files, and everything indexed from it. It " +
        "cannot be undone - if a board ever reaches one of these repos again, a re-sync brings back " +
        "what the forge still has, but not its daily metric history.",
      confirmLabel: "Reclaim",
    })
    if (!ok) return
    setReclaimingOrphans(true)
    setOrphansError(null)
    setOrphansReclaimed(null)
    const res = await commands.reclaimOrphanRepos()
    setReclaimingOrphans(false)
    if (res.status !== "ok") {
      setOrphansError(res.error)
      return
    }
    setOrphansReclaimed(
      res.data.count === 0
        ? "No orphaned repos were left to reclaim."
        : `Reclaimed ${res.data.count} repo(s) - ${res.data.total_rows} rows removed.`,
    )
    setOrphans(null)
    onStorageChanged()
  }

  const reclaim = async () => {
    const ok = await confirm({
      title: "Reclaim free space?",
      // Say that this reclaim costs nothing, unlike the other ways of freeing space.
      body:
        "This rewrites the database without the pages that already hold nothing, so the file shrinks " +
        "on disk. It deletes no data at all - nothing you have synced or indexed is affected. It " +
        "needs room for a second copy of the file while it runs, and the app cannot write during it.",
      confirmLabel: "Reclaim",
    })
    if (!ok) return
    setReclaiming(true)
    setReclaimed(null)
    const res = await commands.reclaimFreeSpace()
    setReclaiming(false)
    setReclaimed(
      res.status === "ok"
        ? `Reclaimed ${formatBytes(res.data.freed_bytes)} - the database is now ${formatBytes(res.data.used_after)}.`
        : res.error,
    )
    onStorageChanged()
  }

  const pct =
    storage?.budget_bytes && storage.budget_bytes > 0
      ? Math.min(100, (storage.used_bytes / storage.budget_bytes) * 100)
      : null

  return (
    <Card className="space-y-3">
      {storage && (
        <div className="space-y-1">
          <div className="flex items-baseline justify-between gap-2 text-sm">
            <span>{storage.message}</span>
          </div>
          {pct != null && (
            <div className="h-2 w-full rounded-full bg-muted">
              <div
                className={
                  storage.stopped
                    ? "h-2 rounded-full bg-red-500"
                    : storage.pressure === "warning"
                      ? "h-2 rounded-full bg-amber-500"
                      : "h-2 rounded-full bg-primary"
                }
                style={{ width: `${pct}%` }}
              />
            </div>
          )}
        </div>
      )}

      <div className="flex items-center gap-3">
        <label className="w-64 text-sm" htmlFor="storage-budget">
          storage budget (GB)
        </label>
        <input
          id="storage-budget"
          type="number"
          min={0}
          step="0.5"
          className="h-9 w-28 rounded-md border border-border bg-background px-3 text-sm"
          value={budgetGb}
          onChange={(e) => onBudgetGb(e.target.value)}
          placeholder="no limit"
        />
        <Button onClick={onSave}>Save</Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Measured on the database file and its write-ahead log. Leave it blank for no limit. At the
        budget orgonzola stops syncing and indexing and says so; it never deletes anything to make
        room, and it starts again by itself once there is space.
        {storage && (
          <>
            {" "}
            Downloaded and bundled models take a further {formatBytes(storage.model_bytes)}, which
            is not counted against this budget - they are fixed downloads you chose, not something
            the app grows on its own.
          </>
        )}
      </p>

      {storage && (
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="outline" size="sm" onClick={onOpenStorage}>
            See what is using the space
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={reclaiming || storage.free_bytes <= 0}
            title={
              storage.free_bytes > 0
                ? "Rewrite the database without its empty pages. Deletes nothing."
                : "There is no free space inside the file to reclaim."
            }
            onClick={() => void reclaim()}
          >
            {reclaiming ? "Reclaiming..." : `Reclaim ${formatBytes(storage.free_bytes)} free space`}
          </Button>
        </div>
      )}
      {reclaimed && <p className="text-sm text-muted-foreground">{reclaimed}</p>}

      {storage && (
        <div className="space-y-2 border-t border-border pt-3">
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              disabled={checkingOrphans}
              onClick={() => void checkOrphans()}
            >
              {checkingOrphans ? "Checking..." : "Check for orphaned repos"}
            </Button>
            {orphans && (
              <span className="text-sm text-muted-foreground">
                {orphans.count === 0
                  ? "No orphaned repos - every repo is pinned or discovered by some board."
                  : `${orphans.count} orphaned repo(s), ~${formatBytes(orphans.estimated_bytes)} (${orphans.rows} rows)`}
              </span>
            )}
            {orphans && orphans.count > 0 && (
              <Button
                variant="outline"
                size="sm"
                disabled={reclaimingOrphans}
                title="Removes each orphaned repo and everything derived from it - the same as forgetting each one by hand."
                onClick={() => void reclaimOrphans()}
              >
                {reclaimingOrphans ? "Reclaiming..." : `Reclaim ${orphans.count} orphaned repo(s)`}
              </Button>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            A repo pinned by no board and discovered by none either - typically left behind by a
            board you deleted. Nothing here is ever removed on its own; checking costs nothing, and
            reclaiming asks first.
          </p>
          {orphansError && <p className="text-sm text-red-600">{orphansError}</p>}
          {orphansReclaimed && <p className="text-sm text-muted-foreground">{orphansReclaimed}</p>}
        </div>
      )}
      {result}
    </Card>
  )
}

// The local-LLM section: toggle on-device AI narration, and download and select a small quantized
// model. Inference is in-process, GPU-accelerated via Vulkan with a CPU fallback. A
// `--no-default-features` build compiles the engine out and this section stays inert.
function LlmSection() {
  const [cat, setCat] = useState<LlmCatalogView | null>(null)
  const [busy, setBusy] = useState<string | null>(null) // file currently downloading
  const [progress, setProgress] = useState<{ received: number; total: number } | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const load = useCallback(async () => {
    const res = await commands.llmCatalog()
    if (res.status === "ok") setCat(res.data)
  }, [])

  useEffect(() => {
    void load()
    const unlisten = events.llmDownloadEvent.listen((e) => {
      const p: LlmDownloadEvent = e.payload
      if (busy != null && p.file !== busy) return
      if (p.error) {
        setMessage(`download failed: ${p.error}`)
        setBusy(null)
        setProgress(null)
        return
      }
      if (p.done) {
        setBusy(null)
        setProgress(null)
        void load()
        return
      }
      setProgress({ received: p.received, total: p.total })
    })
    return () => {
      void unlisten.then((f) => f())
    }
  }, [load, busy])

  if (cat == null) {
    return (
      <Card>
        <p className="text-sm text-muted-foreground">loading...</p>
      </Card>
    )
  }

  const setEnabled = async (enabled: boolean) => {
    const res = await commands.setLlmSettings(enabled, cat.selected)
    if (res.status === "ok") setCat({ ...cat, enabled })
  }
  const select = async (file: string) => {
    const res = await commands.setLlmSettings(cat.enabled, file)
    if (res.status === "ok") setCat({ ...cat, selected: file })
  }
  const download = async (repo: string, file: string) => {
    setMessage(null)
    setBusy(file)
    setProgress({ received: 0, total: 0 })
    const res = await commands.downloadLlmModel(repo, file)
    if (res.status !== "ok") {
      setMessage(`download failed: ${res.error}`)
      setBusy(null)
      setProgress(null)
    }
  }
  const mb = (n: number) => `${(n / (1024 * 1024)).toFixed(0)} MB`

  return (
    <Card className="space-y-3">
      {!cat.feature_built && (
        <p className="rounded bg-amber-50 px-2 py-1 text-xs text-amber-800">
          The inference engine is not built into this binary. Launch with the AI feature (`just run`
          builds it) to enable on-device summaries.
        </p>
      )}
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          className="size-4"
          checked={cat.enabled}
          disabled={!cat.feature_built}
          onChange={(e) => void setEnabled(e.target.checked)}
        />
        Enable AI summaries (rephrase the attention brief with a local model)
      </label>
      <div className="space-y-1.5">
        {cat.models.map((m) => {
          const selected = cat.selected === m.file
          const downloading = busy === m.file
          return (
            <div
              key={m.file}
              className="flex flex-wrap items-center gap-3 rounded-md border border-border px-3 py-2 text-sm"
            >
              <span className="flex items-center gap-2">
                <input
                  type="radio"
                  name="llm-model"
                  className="size-4"
                  checked={selected}
                  disabled={!m.downloaded}
                  onChange={() => void select(m.file)}
                  title={m.downloaded ? "Use this model" : "Download it first"}
                />
                <span className="font-medium">{m.label}</span>
                <span className="font-mono text-xs text-muted-foreground">
                  {m.params} - {m.size}
                </span>
              </span>
              <span className="ml-auto flex items-center gap-3">
                {downloading ? (
                  <span className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="h-1.5 w-32 overflow-hidden rounded-full bg-muted">
                      <span
                        className="block h-full bg-primary transition-all"
                        style={{
                          width:
                            progress && progress.total > 0
                              ? `${Math.round((progress.received / progress.total) * 100)}%`
                              : "10%",
                        }}
                      />
                    </span>
                    {progress && progress.total > 0
                      ? `${mb(progress.received)} / ${mb(progress.total)}`
                      : "downloading..."}
                  </span>
                ) : m.downloaded ? (
                  <span className="text-xs text-emerald-700">downloaded</span>
                ) : (
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={busy != null}
                    onClick={() => void download(m.repo, m.file)}
                  >
                    Download
                  </Button>
                )}
              </span>
            </div>
          )
        })}
      </div>
      <p className="text-[10px] text-muted-foreground">
        Models download from Hugging Face into {cat.dir} (no key). They run in-process on CPU - no
        server, no data leaves the machine. The deterministic brief is used when AI is off or a
        model is still loading.
      </p>
      {message && <p className="text-sm text-red-600">{message}</p>}
    </Card>
  )
}

// The kinds a connection can be. Forge kinds must match `AnyForge::from_parts` on the host.
// GitLab is absent: there is no backend for it yet.
const CONNECTION_KINDS = [
  { value: "github", label: "GitHub" },
  { value: "gitea", label: "Gitea / Forgejo" },
  { value: "jira", label: "Jira" },
] as const

type ConnectionKind = (typeof CONNECTION_KINDS)[number]["value"]

const kindLabel = (kind: string) => CONNECTION_KINDS.find((k) => k.value === kind)?.label ?? kind

// Each kind's base URL, and whether the user must supply it. GitHub has one well-known endpoint,
// so the host fills it in; a blank URL for the others is an error.
const URL_FIELD: Record<ConnectionKind, { label: string; placeholder: string; required: boolean }> =
  {
    github: {
      label: "API base URL",
      placeholder: "https://api.github.com (leave blank for github.com)",
      required: false,
    },
    gitea: {
      label: "server API URL",
      placeholder: "https://git.example.com/api/v1",
      required: true,
    },
    jira: { label: "site URL", placeholder: "https://acme.atlassian.net", required: true },
  }

// Manage all connections in one list: forges (GitHub, Gitea/Forgejo) and Jira sites. Tokens are
// never shown; they are stored in the OS keychain, not the DB. Config edits take effect on the next
// app launch.
function ConnectionsSection() {
  const [forges, setForges] = useState<ForgeView[] | null>(null)
  const [trackers, setTrackers] = useState<TrackerView[] | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [kind, setKind] = useState<ConnectionKind>("github")
  const [name, setName] = useState("")
  const [baseUrl, setBaseUrl] = useState("")
  const [clientId, setClientId] = useState("")
  const [email, setEmail] = useState("")
  const [message, setMessage] = useState<string | null>(null)

  const [secret, setSecret] = useState<Record<string, string>>({})
  const [connecting, setConnecting] = useState<string | null>(null)
  const [deviceAuth, setDeviceAuth] = useState<DeviceAuth | null>(null)
  // Whether a built-in GitHub OAuth client id is baked into this build.
  const [oauthAvailable, setOauthAvailable] = useState(false)
  // Optional fields (name, self-hosted URL, custom OAuth app) are collapsed by default.
  const [advanced, setAdvanced] = useState(false)
  // Confirmation after a successful add/edit.
  const [saved, setSaved] = useState<string | null>(null)
  const confirm = useConfirm()

  const load = useCallback(async () => {
    const [f, t] = await Promise.all([commands.listForges(), commands.listTrackers()])
    if (f.status === "ok") setForges(f.data)
    if (t.status === "ok") setTrackers(t.data)
  }, [])

  useEffect(() => {
    void load()
    // Infallible command: returns the boolean directly, not a Result.
    void commands.githubOauthAvailable().then(setOauthAvailable)
  }, [load])

  // The host pushes the device code mid-flow.
  useEffect(() => {
    const unlisten = events.deviceAuthEvent.listen((e) => {
      const p = e.payload
      setDeviceAuth({
        forgeId: p.forge_id,
        userCode: p.user_code,
        verificationUri: p.verification_uri,
      })
    })
    return () => {
      void unlisten.then((f) => f())
    }
  }, [])

  const resetForm = () => {
    setEditingId(null)
    setKind("github")
    setName("")
    setBaseUrl("")
    setClientId("")
    setEmail("")
    setSaved(null)
  }

  // Only forges are editable; a tracker is add/delete.
  const startEdit = (f: ForgeView) => {
    setEditingId(f.id)
    setKind(f.kind as ConnectionKind)
    setName(f.name)
    setBaseUrl(f.base_url)
    setClientId(f.oauth_client_id ?? "")
    setEmail("")
    setMessage(null)
    setSaved(null)
    setAdvanced(true)
  }

  const save = async () => {
    setMessage(null)
    const url = baseUrl.trim()
    if (URL_FIELD[kind].required && !url) {
      setMessage(`a ${kindLabel(kind)} connection needs its ${URL_FIELD[kind].label}`)
      return
    }
    // A blank name or GitHub URL stays blank: the host derives both.
    const res =
      kind === "jira"
        ? await commands.addTracker(name.trim(), url, email.trim() || null)
        : editingId
          ? await commands.updateForge(editingId, name.trim(), kind, url, clientId.trim() || null)
          : await commands.addForge(name.trim(), kind, url, clientId.trim() || null)
    if (res.status === "ok") {
      resetForm()
      await load()
      setSaved("Saved. Connect it below to start syncing.")
    } else {
      setMessage(res.error)
    }
  }

  const removeForge = async (id: string) => {
    const fname = forges?.find((f) => f.id === id)?.name ?? "this connection"
    const ok = await confirm({
      title: `Delete the "${fname}" connection?`,
      body: "This removes the connection and its stored credentials. Boards using it will stop syncing until you reconnect. This cannot be undone.",
      confirmLabel: "Delete connection",
    })
    if (!ok) return
    setMessage(null)
    const res = await commands.deleteForge(id)
    if (res.status === "ok") {
      if (editingId === id) resetForm()
      await load()
    } else {
      setMessage(res.error)
    }
  }

  const removeTracker = async (id: string) => {
    const tname = trackers?.find((t) => t.id === id)?.name ?? "this connection"
    const ok = await confirm({
      title: `Delete the "${tname}" connection?`,
      body: "This removes the Jira connection and its stored token. Boards linked to its projects stop syncing issues. This cannot be undone.",
      confirmLabel: "Delete connection",
    })
    if (!ok) return
    setMessage(null)
    const res = await commands.deleteTracker(id)
    if (res.status === "ok") await load()
    else setMessage(res.error)
  }

  const connectDevice = async (id: string) => {
    setMessage(null)
    setDeviceAuth(null)
    setConnecting(id)
    const res = await commands.connectDeviceAuth(id)
    setConnecting(null)
    setDeviceAuth(null)
    if (res.status === "ok") {
      await load()
      setMessage("connected")
    } else {
      setMessage(res.error)
    }
  }

  const useForgePat = async (id: string) => {
    const token = (secret[id] ?? "").trim()
    if (!token) return
    setMessage(null)
    const res = await commands.setForgePat(id, token)
    if (res.status === "ok") {
      setSecret((prev) => ({ ...prev, [id]: "" }))
      await load()
      setMessage("connected")
    } else {
      setMessage(res.error)
    }
  }

  const useTrackerToken = async (id: string) => {
    setMessage(null)
    const res = await commands.setTrackerToken(id, secret[id] ?? "")
    if (res.status === "ok") {
      setSecret((prev) => ({ ...prev, [id]: "" }))
      await load()
    } else {
      setMessage(res.error)
    }
  }

  const disconnect = async (id: string) => {
    const fname = forges?.find((f) => f.id === id)?.name ?? "this connection"
    const ok = await confirm({
      title: `Disconnect "${fname}"?`,
      body: "This signs out and removes the stored credentials. Boards using it stop syncing until you reconnect.",
      confirmLabel: "Disconnect",
    })
    if (!ok) return
    setMessage(null)
    const res = await commands.disconnectForge(id)
    if (res.status === "ok") await load()
    else setMessage(res.error)
  }

  const loading = forges == null || trackers == null
  const empty = !loading && forges.length === 0 && trackers.length === 0

  return (
    <Card className="space-y-3">
      {loading ? (
        <p className="text-sm text-muted-foreground">loading...</p>
      ) : empty ? (
        <p className="text-sm text-muted-foreground">
          No connections yet - add one below, then connect it (one-click GitHub sign-in, or a
          token).
        </p>
      ) : (
        <ul className="space-y-3">
          {forges.map((f) => (
            <li
              key={f.id}
              className="space-y-2 border-b border-border pb-3 last:border-0 last:pb-0"
            >
              <ConnectionHeader
                name={f.name}
                kind={f.kind}
                detail={f.base_url}
                status={f.connected ? "connected" : "not connected"}
                good={f.connected}
                onEdit={() => startEdit(f)}
                onDelete={() => void removeForge(f.id)}
              />
              {f.connected ? (
                <Button variant="outline" size="sm" onClick={() => void disconnect(f.id)}>
                  Disconnect
                </Button>
              ) : (
                <div className="space-y-2">
                  {f.kind === "github" && (f.oauth_client_id || oauthAvailable) && (
                    <Button
                      size="sm"
                      disabled={connecting === f.id}
                      onClick={() => void connectDevice(f.id)}
                    >
                      {connecting === f.id ? "Waiting for authorization..." : "Connect with GitHub"}
                    </Button>
                  )}
                  {connecting === f.id && deviceAuth?.forgeId === f.id && (
                    <p className="text-sm">
                      Go to{" "}
                      <a
                        href={deviceAuth.verificationUri}
                        onClick={(e) => {
                          e.preventDefault()
                          void commands.openUrl(deviceAuth.verificationUri)
                        }}
                        className="font-medium underline"
                      >
                        {deviceAuth.verificationUri}
                      </a>{" "}
                      and enter{" "}
                      <span className="font-mono font-semibold">{deviceAuth.userCode}</span>
                    </p>
                  )}
                  <SecretRow
                    id={f.id}
                    placeholder="...or paste an access token"
                    action="Use token"
                    value={secret[f.id] ?? ""}
                    onChange={(v) => setSecret((prev) => ({ ...prev, [f.id]: v }))}
                    onSubmit={() => void useForgePat(f.id)}
                  />
                  {f.kind === "github" && (
                    <p className="text-xs text-muted-foreground">
                      Prefer a token?{" "}
                      <a
                        href={GITHUB_TOKEN_URL}
                        onClick={(e) => {
                          e.preventDefault()
                          void commands.openUrl(GITHUB_TOKEN_URL)
                        }}
                        className="font-medium underline underline-offset-2"
                      >
                        Create one on GitHub
                      </a>{" "}
                      with the <span className="font-mono">repo</span> and{" "}
                      <span className="font-mono">read:org</span> scopes, then paste it above.
                    </p>
                  )}
                </div>
              )}
            </li>
          ))}
          {trackers.map((t) => (
            <li
              key={t.id}
              className="space-y-2 border-b border-border pb-3 last:border-0 last:pb-0"
            >
              <ConnectionHeader
                name={t.name}
                kind={t.kind}
                detail={t.email ? `${t.base_url} - ${t.email}` : t.base_url}
                status={t.connected ? "token set" : "no token (public/anon)"}
                good={t.connected}
                onDelete={() => void removeTracker(t.id)}
              />
              <SecretRow
                id={t.id}
                placeholder="API token (leave empty for a public site)"
                action={t.connected ? "Update token" : "Connect"}
                value={secret[t.id] ?? ""}
                onChange={(v) => setSecret((prev) => ({ ...prev, [t.id]: v }))}
                onSubmit={() => void useTrackerToken(t.id)}
              />
            </li>
          ))}
        </ul>
      )}

      <div className="space-y-2 border-t border-border pt-3">
        <p className="text-sm font-medium text-muted-foreground">
          {editingId ? "Edit connection" : "Add a connection"}
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <select
            className="h-9 rounded-md border border-border bg-background px-3 text-sm"
            value={kind}
            // Changing the kind changes which fields apply, so drop the old kind's fields.
            onChange={(e) => {
              setKind(e.target.value as ConnectionKind)
              setBaseUrl("")
              setClientId("")
              setEmail("")
            }}
            disabled={editingId != null}
          >
            {CONNECTION_KINDS.map((k) => (
              <option key={k.value} value={k.value}>
                {k.label}
              </option>
            ))}
          </select>
          {URL_FIELD[kind].required && (
            <input
              className="h-9 min-w-[22rem] flex-1 rounded-md border border-border bg-background px-3 text-sm"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder={URL_FIELD[kind].placeholder}
            />
          )}
          <Button onClick={() => void save()}>{editingId ? "Save" : "Add connection"}</Button>
          {editingId && (
            <Button variant="outline" onClick={resetForm}>
              Cancel
            </Button>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {kind === "github"
            ? "That is all a github.com account needs - the name and API URL are filled in for you."
            : "Name it under Advanced if you want something other than the server's host name."}
        </p>
        <button
          type="button"
          onClick={() => setAdvanced((v) => !v)}
          className="text-xs font-medium text-muted-foreground underline-offset-2 hover:underline"
        >
          {advanced ? "Hide advanced" : "Advanced (name, self-hosted URL, custom OAuth app)"}
        </button>
        {advanced && (
          <div className="space-y-2">
            <input
              className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="display name (optional - defaults to the service or its host name)"
            />
            {!URL_FIELD[kind].required && (
              <input
                className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder={URL_FIELD[kind].placeholder}
              />
            )}
            {kind === "jira" && (
              <input
                className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Cloud email (optional; leave empty for Server/DC PAT or anonymous)"
              />
            )}
            {kind === "github" && (
              <input
                className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
                value={clientId}
                onChange={(e) => setClientId(e.target.value)}
                placeholder="OAuth client id (only for your own GitHub OAuth app)"
              />
            )}
            <p className="text-xs text-muted-foreground">
              A custom OAuth client id is only needed if you registered your own GitHub OAuth app{" "}
              {oauthAvailable ? "(orgonzola ships one, so you usually do not need this)" : ""}.
            </p>
          </div>
        )}
        {saved && (
          <div className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800">
            {saved}
          </div>
        )}
        {message && <p className="text-sm text-muted-foreground">{message}</p>}
      </div>
    </Card>
  )
}

function ConnectionHeader({
  name,
  kind,
  detail,
  status,
  good,
  onEdit,
  onDelete,
}: {
  name: string
  kind: string
  detail: string
  status: string
  good: boolean
  onEdit?: () => void
  onDelete: () => void
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">
        <p className="truncate text-sm font-medium">
          {name}{" "}
          <Badge variant="outline" className="ml-1 align-middle">
            {kindLabel(kind)}
          </Badge>{" "}
          <Badge
            variant="outline"
            className={
              good
                ? "ml-1 border-emerald-200 bg-emerald-50 align-middle text-emerald-700"
                : "ml-1 border-amber-200 bg-amber-50 align-middle text-amber-700"
            }
          >
            {status}
          </Badge>
        </p>
        <p className="truncate text-xs text-muted-foreground">{detail}</p>
      </div>
      <div className="flex shrink-0 gap-1">
        {onEdit && (
          <Button variant="outline" size="sm" onClick={onEdit}>
            Edit
          </Button>
        )}
        <Button variant="outline" size="sm" onClick={onDelete}>
          Delete
        </Button>
      </div>
    </div>
  )
}

// A masked credential field plus its submit button. The value goes straight to the host, which
// stores it in the OS keychain.
function SecretRow({
  id,
  placeholder,
  action,
  value,
  onChange,
  onSubmit,
}: {
  id: string
  placeholder: string
  action: string
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        id={`secret-${id}`}
        type="password"
        className="h-9 w-72 rounded-md border border-border bg-background px-3 text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
      <Button variant="outline" size="sm" onClick={onSubmit}>
        {action}
      </Button>
    </div>
  )
}
