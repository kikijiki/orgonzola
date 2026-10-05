//! The assistant's tools: Rig [`Tool`]s wrapping orgonzola's deterministic read-only queries
//! (`core-summary`, `core-metrics`, `core-store`). Board-scoped: constructed with the board's repo
//! ids and `now`, so the model calls them with minimal arguments. Read-only.

use crate::model::LocalModel;
use crate::retrieval::RetrievalIndex;
use core_embed::Embedder;
use core_store::{KindFilter, RepoScope, Store};
use rig::agent::{Agent, AgentBuilder};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

/// Why a tool call failed. Surfaced to the model as an error string so it can recover.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error(transparent)]
    Store(#[from] core_store::StoreError),
    #[error(transparent)]
    Summary(#[from] core_summary::SummaryError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("blocking task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// No-argument tool args (the tool is fully board-scoped). Accepts `{}` or `null` from the model.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct NoArgs {}

#[derive(Debug, Deserialize)]
pub struct QueryArgs {
    pub query: String,
    #[serde(default)]
    pub k: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct StuckArgs {
    #[serde(default)]
    pub threshold_days: Option<i64>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Args for `failing_ci`: the tool needs no lookup arguments, only an optional bound override.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct FailingCiArgs {
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct PrArgs {
    pub pr_id: String,
}

#[derive(Debug, Deserialize)]
pub struct WorkItemArgs {
    pub work_item_id: String,
}

/// Args shared by the grouped board-level tools: a required `mode`, plus optional window/limit
/// knobs that modes ignore when they do not apply.
#[derive(Debug, Deserialize)]
pub struct PeopleArgs {
    /// "stats" | "pickup".
    pub mode: String,
    #[serde(default)]
    pub since_days: Option<i64>,
    #[serde(default)]
    pub until_days: Option<i64>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct BoardHealthArgs {
    /// "scorecard" | "dora" | "investment".
    pub mode: String,
    #[serde(default)]
    pub since_days: Option<i64>,
    #[serde(default)]
    pub until_days: Option<i64>,
    #[serde(default)]
    pub window_days: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct CodeRiskArgs {
    /// "hotspots" | "ownership" | "fused" | "coupling" | "dependencies".
    pub mode: String,
    #[serde(default)]
    pub limit: Option<usize>,
}

fn obj_schema(props: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": props, "required": required })
}

/// Turn a `since_days`/`until_days` pair (days before `now`, `until_days` 0 = now) into ordered
/// RFC-3339 bounds. Relative to the host-supplied `now`, not the wall clock (the core carries no
/// clock), so a tool call is reproducible against the turn's `now`.
fn window_bounds(now: &str, since_days: i64, until_days: i64) -> (String, String) {
    use time::format_description::well_known::Rfc3339;
    use time::{Duration, OffsetDateTime};
    let lo = since_days.clamp(1, 3650);
    let hi = until_days.clamp(0, 3650).min(lo);
    let Ok(base) = OffsetDateTime::parse(now, &Rfc3339) else {
        // A malformed `now` (the host supplies it) leaves the window degenerate, not a panic.
        return (now.to_string(), now.to_string());
    };
    let since = (base - Duration::days(lo))
        .format(&Rfc3339)
        .unwrap_or_else(|_| now.to_string());
    let until = (base - Duration::days(hi))
        .format(&Rfc3339)
        .unwrap_or_else(|_| now.to_string());
    (since, until)
}

/// Default lookback for `since_days`, matching the board People panel's default window.
const DEFAULT_SINCE_DAYS: i64 = 30;

/// How many areas of familiarity to return per person in `people` pickup mode, matching the
/// host's pickup panel.
const PICKUP_AREA_LIMIT: usize = 4;

/// Default row cap for the `code_risk` tool's list-shaped modes.
const CODE_RISK_LIMIT: usize = 10;

/// Default row cap for the `people` tool's list-shaped modes. A roster is discovered from activity
/// and can reach thousands (1109 PR authors on one board); unbounded output overflowed the local
/// model's context window (`NoKvCacheSlot`).
const PEOPLE_LIMIT: usize = 20;

/// Default row cap for `board_attention`'s combined item list. Findings can run into the thousands
/// and the result must fit the model's context window.
const BOARD_ATTENTION_LIMIT: usize = 30;

/// Default row cap for `whats_stuck`'s combined per-board stale-PR list.
const STUCK_LIMIT: usize = 20;

/// Default row cap for `failing_ci`'s combined per-board failing-run list.
const FAILING_CI_LIMIT: usize = 20;

/// Upper clamp on `search_activity`/`search_code`'s `k`, which is untrusted model input.
const MAX_SEARCH_K: u64 = 20;

/// Truncate an already-ranked row set to `limit` and wrap it with its pre-truncation total, so a
/// capped list never reads as complete. `rows` must already be ordered; this only cuts the tail.
fn bounded<T: serde::Serialize>(mut rows: Vec<T>, limit: usize) -> Value {
    let total = rows.len();
    rows.truncate(limit);
    json!({ "items": rows, "total": total })
}

/// Resolve a caller-supplied `k` (default 5) clamped to `[1, MAX_SEARCH_K]`.
fn resolved_k(k: Option<u64>) -> usize {
    k.unwrap_or(5).clamp(1, MAX_SEARCH_K) as usize
}

/// Embed a query off the async runtime (ONNX), then hybrid-search a kind slice within the board's
/// repos. The repo scope keeps results from other boards out.
async fn search(
    store: &Arc<Store>,
    embedder: &Arc<dyn Embedder>,
    query: &str,
    k: usize,
    kind: KindFilter<'_>,
    repo_ids: &[String],
) -> Result<Vec<core_store::EmbeddingHit>, ToolError> {
    let embedder = embedder.clone();
    let q = query.to_string();
    let vector = tokio::task::spawn_blocking(move || embedder.embed(&q)).await?;
    Ok(store
        .hybrid_search(&vector, query, k, kind, RepoScope::Repos(repo_ids))
        .await?)
}

// ---- board-scoped tools ----------------------------------------------------

/// What needs attention across the board's repos: stale PRs, unreviewed merges, failing CI, etc.
pub struct BoardAttention {
    store: Arc<Store>,
    repo_ids: Vec<String>,
    now: String,
}

impl Tool for BoardAttention {
    const NAME: &'static str = "board_attention";
    type Error = ToolError;
    type Args = NoArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "What needs attention across this board right now (stale PRs, merged without review, \
                failing CI, aging work). Each item carries a deterministic next-step action and a typed \
                list of the entities it points at - the pull request, work item, CI run, person or file, \
                each with its forge URL where there is one - so cite those rather than restating the \
                summary sentence. Bounded to the {BOARD_ATTENTION_LIMIT} most severe items board-wide \
                (failing CI first, orphaned PRs last - see each item's `kind`); `attention_total` per \
                repo and board-wide names the true count even when more items were found. No arguments."
            ),
            parameters: obj_schema(json!({}), &[]),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Board-level WIP aging bands, computed once so every repo's items share a baseline.
        let bands =
            core_summary::board_wip_aging_bands(&self.store, &self.repo_ids, &self.now).await?;
        let mut per_repo: Vec<(String, Vec<core_summary::AttentionItem>)> = Vec::new();
        for repo in &self.repo_ids {
            let items =
                core_summary::attention_enriched(&self.store, repo, &self.now, &bands).await?;
            per_repo.push((repo.clone(), items));
        }
        // True per-repo counts, taken before bounding.
        let repo_totals: Vec<usize> = per_repo.iter().map(|(_, items)| items.len()).collect();
        let board_total: usize = repo_totals.iter().sum();
        if board_total > BOARD_ATTENTION_LIMIT {
            // Keep the most severe items board-wide (core_summary::attention_severity_rank), not a
            // per-repo slice, which could crowd out the repo with the serious findings.
            let mut ranked: Vec<(usize, usize, usize)> = Vec::with_capacity(board_total);
            for (ri, (_, items)) in per_repo.iter().enumerate() {
                for (ii, item) in items.iter().enumerate() {
                    ranked.push((ri, ii, core_summary::attention_severity_rank(&item.kind)));
                }
            }
            ranked.sort_by_key(|&(_, _, rank)| rank);
            let keep: std::collections::HashSet<(usize, usize)> = ranked
                .into_iter()
                .take(BOARD_ATTENTION_LIMIT)
                .map(|(ri, ii, _)| (ri, ii))
                .collect();
            for (ri, (_, items)) in per_repo.iter_mut().enumerate() {
                let mut i = 0usize;
                items.retain(|_| {
                    let keep_this = keep.contains(&(ri, i));
                    i += 1;
                    keep_this
                });
            }
        }
        let repos: Vec<Value> = per_repo
            .into_iter()
            .zip(repo_totals)
            .map(|((repo, items), attention_total)| {
                json!({ "repo": repo, "items": items, "attention_total": attention_total })
            })
            .collect();
        Ok(json!({ "repos": repos, "attention_total": board_total }))
    }
}

/// Per-repo flow: median PR cycle time (seconds) and current WIP.
pub struct FlowMetrics {
    store: Arc<Store>,
    repo_ids: Vec<String>,
}

impl Tool for FlowMetrics {
    const NAME: &'static str = "flow_metrics";
    type Error = ToolError;
    type Args = NoArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Per-repo flow metrics for this board: median PR cycle time (seconds) and WIP count. No arguments.".to_string(),
            parameters: obj_schema(json!({}), &[]),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut out = Vec::new();
        for repo in &self.repo_ids {
            let prs = self.store.pull_requests(repo).await?;
            out.push(json!({
                "repo": repo,
                "median_cycle_time_secs": core_metrics::median_cycle_time_secs(&prs),
                "wip": core_metrics::wip_count(&prs),
            }));
        }
        Ok(Value::Array(out))
    }
}

/// Open PRs across the board older than a day threshold (default 7).
pub struct WhatsStuck {
    store: Arc<Store>,
    repo_ids: Vec<String>,
    now: String,
}

impl Tool for WhatsStuck {
    const NAME: &'static str = "whats_stuck";
    type Error = ToolError;
    type Args = StuckArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Open PRs across the board older than `threshold_days` (default 7). Bounded to the \
                {STUCK_LIMIT} oldest (default, override with `limit`) across the whole board - the \
                longest-waiting PRs are the most actionable to see in a capped list; `stuck_total` names \
                the true count."
            ),
            parameters: obj_schema(
                json!({
                    "threshold_days": { "type": "integer", "description": "age in days, default 7" },
                    "limit": { "type": "integer", "description": "max PRs across the board, default 20" },
                }),
                &[],
            ),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let days = args.threshold_days.unwrap_or(7);
        let limit = args.limit.unwrap_or(STUCK_LIMIT);
        let mut all: Vec<(String, core_store::PullRequest)> = Vec::new();
        for repo in &self.repo_ids {
            let prs = self.store.pull_requests(repo).await?;
            for pr in core_metrics::stale_open_prs(&prs, &self.now, days) {
                all.push((repo.clone(), pr.clone()));
            }
        }
        let stuck_total = all.len();
        // Oldest first: the longest-stuck PRs matter most in a bounded slice.
        all.sort_by(|a, b| a.1.created_at.cmp(&b.1.created_at));
        all.truncate(limit);
        let mut grouped: Vec<(String, Vec<Value>)> = Vec::new();
        for (repo, pr) in all {
            let row = json!({
                "id": pr.id, "number": pr.number, "title": pr.title, "created_at": pr.created_at
            });
            match grouped.iter_mut().find(|(r, _)| *r == repo) {
                Some((_, v)) => v.push(row),
                None => grouped.push((repo, vec![row])),
            }
        }
        let repos: Vec<Value> = grouped
            .into_iter()
            .map(|(repo, stuck)| json!({ "repo": repo, "stuck": stuck }))
            .collect();
        Ok(json!({ "repos": repos, "stuck_total": stuck_total }))
    }
}

/// CI runs across the board that completed with a failure.
pub struct FailingCi {
    store: Arc<Store>,
    repo_ids: Vec<String>,
}

impl Tool for FailingCi {
    const NAME: &'static str = "failing_ci";
    type Error = ToolError;
    type Args = FailingCiArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "CI runs across this board that completed with a failure. Bounded to the \
                {FAILING_CI_LIMIT} most recently completed (default, override with `limit`) across the \
                whole board; `failing_total` names the true count."
            ),
            parameters: obj_schema(
                json!({ "limit": { "type": "integer", "description": "max runs across the board, default 20" } }),
                &[],
            ),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let limit = args.limit.unwrap_or(FAILING_CI_LIMIT);
        let mut all: Vec<(String, core_store::CiRun)> = Vec::new();
        for repo in &self.repo_ids {
            let runs = self.store.ci_runs_for_repo(repo).await?;
            for run in core_metrics::failing_ci_runs(&runs) {
                all.push((repo.clone(), run.clone()));
            }
        }
        let failing_total = all.len();
        // Most recently completed first: old failures on since-fixed branches matter less.
        all.sort_by(|a, b| b.1.completed_at.cmp(&a.1.completed_at));
        all.truncate(limit);
        let mut grouped: Vec<(String, Vec<Value>)> = Vec::new();
        for (repo, run) in all {
            let row = serde_json::to_value(&run)?;
            match grouped.iter_mut().find(|(r, _)| *r == repo) {
                Some((_, v)) => v.push(row),
                None => grouped.push((repo, vec![row])),
            }
        }
        let repos: Vec<Value> = grouped
            .into_iter()
            .map(|(repo, failing)| json!({ "repo": repo, "failing": failing }))
            .collect();
        Ok(json!({ "repos": repos, "failing_total": failing_total }))
    }
}

/// Semantic search over indexed activity (commits, issues).
pub struct SearchActivity {
    store: Arc<Store>,
    embedder: Arc<dyn Embedder>,
    repo_ids: Vec<String>,
}

impl Tool for SearchActivity {
    const NAME: &'static str = "search_activity";
    type Error = ToolError;
    type Args = QueryArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Semantic search over indexed activity (commits, issues). Returns the most relevant \
                snippets, capped at {MAX_SEARCH_K} regardless of `k`."
            ),
            parameters: obj_schema(
                json!({
                    "query": { "type": "string" },
                    "k": { "type": "integer", "description": "max results, default 5, capped at 20" }
                }),
                &["query"],
            ),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let k = resolved_k(args.k);
        let hits = search(
            &self.store,
            &self.embedder,
            &args.query,
            k,
            KindFilter::Not("code"),
            &self.repo_ids,
        )
        .await?;
        Ok(serde_json::to_value(hits)?)
    }
}

/// Semantic search over indexed source code.
pub struct SearchCode {
    store: Arc<Store>,
    embedder: Arc<dyn Embedder>,
    repo_ids: Vec<String>,
}

impl Tool for SearchCode {
    const NAME: &'static str = "search_code";
    type Error = ToolError;
    type Args = QueryArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Semantic search over indexed source code. Returns the most relevant file snippets, \
                capped at {MAX_SEARCH_K} regardless of `k`."
            ),
            parameters: obj_schema(
                json!({
                    "query": { "type": "string" },
                    "k": { "type": "integer", "description": "max results, default 5, capped at 20" }
                }),
                &["query"],
            ),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let k = resolved_k(args.k);
        let hits = search(
            &self.store,
            &self.embedder,
            &args.query,
            k,
            KindFilter::Is("code"),
            &self.repo_ids,
        )
        .await?;
        Ok(serde_json::to_value(hits)?)
    }
}

/// All teams in the knowledge base.
pub struct ListTeams {
    store: Arc<Store>,
}

impl Tool for ListTeams {
    const NAME: &'static str = "list_teams";
    type Error = ToolError;
    type Args = NoArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List all teams in the knowledge base. No arguments.".to_string(),
            parameters: obj_schema(json!({}), &[]),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(serde_json::to_value(self.store.list_teams().await?)?)
    }
}

/// A per-PR change digest (facts, body, review state, linked issues).
pub struct ChangeDigest {
    store: Arc<Store>,
}

impl Tool for ChangeDigest {
    const NAME: &'static str = "change_digest";
    type Error = ToolError;
    type Args = PrArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Given a pull-request id, return its digest: facts, body, review state, linked issues.".to_string(),
            parameters: obj_schema(json!({ "pr_id": { "type": "string" } }), &["pr_id"]),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(serde_json::to_value(
            core_summary::change_digest(&self.store, &args.pr_id).await?,
        )?)
    }
}

/// A work item plus its outgoing links, or null if unknown.
pub struct WorkItemStatus {
    store: Arc<Store>,
}

impl Tool for WorkItemStatus {
    const NAME: &'static str = "work_item_status";
    type Error = ToolError;
    type Args = WorkItemArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Given a work-item id, return the item and its outgoing links, or null if unknown."
                    .to_string(),
            parameters: obj_schema(
                json!({ "work_item_id": { "type": "string" } }),
                &["work_item_id"],
            ),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match self.store.work_item(&args.work_item_id).await? {
            Some(item) => {
                let links = self.store.links_from("work_item", &item.id).await?;
                Ok(json!({ "work_item": item, "links": links }))
            }
            None => Ok(json!({ "work_item": Value::Null, "links": [] })),
        }
    }
}

/// Per-person activity on this board, grouped behind one tool with a `mode`. Descriptive only,
/// never a ranking or performance score. The roster is discovered from board activity in the
/// window, as the host does for repo/org boards.
pub struct People {
    store: Arc<Store>,
    repo_ids: Vec<String>,
    now: String,
}

impl Tool for People {
    const NAME: &'static str = "people";
    type Error = ToolError;
    type Args = PeopleArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Per-person activity on this board. Descriptive only, never a ranking or score. \
                mode \"stats\": commits, PRs opened/merged, reviews, approvals, self-merges, median cycle \
                time, review latency, active days, CI failures, off-hours share, for everyone active in \
                the window (default last 30 days), busiest first (most timestamped work events). Use for \
                \"who is working the most\" / \"who reviews the most\". mode \"pickup\": current open-PR \
                load, last-active time, and codebase areas of context, most-loaded first (most open PRs), \
                for \"who is free\" / \"who should pick this up\". Both modes are bounded to the \
                {PEOPLE_LIMIT} rows the ordering names above (default, override with `limit`) - a real \
                board's roster can run into the hundreds or thousands of drive-by contributors; \
                `people_total` names the true roster size even when more people were found."
            ),
            parameters: obj_schema(
                json!({
                    "mode": { "type": "string", "enum": ["stats", "pickup"] },
                    "since_days": { "type": "integer", "description": "window start, days before now (default 30)" },
                    "until_days": { "type": "integer", "description": "window end, days before now (default 0 = now)" },
                    "limit": { "type": "integer", "description": "max people returned, default 20" },
                }),
                &["mode"],
            ),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let since_days = args.since_days.unwrap_or(DEFAULT_SINCE_DAYS);
        let until_days = args.until_days.unwrap_or(0);
        let limit = args.limit.unwrap_or(PEOPLE_LIMIT);
        let (since, until) = window_bounds(&self.now, since_days, until_days);
        let logins =
            core_summary::contributors(&self.store, &self.repo_ids, &since, &until).await?;
        match args.mode.as_str() {
            "pickup" => {
                let mut rows = core_summary::board_pickup(
                    &self.store,
                    &self.repo_ids,
                    &logins,
                    PICKUP_AREA_LIMIT,
                )
                .await?;
                let people_total = rows.len();
                // Most loaded first (open-PR count), ties by most recent activity.
                rows.sort_by(|a, b| {
                    b.open_prs
                        .cmp(&a.open_prs)
                        .then_with(|| b.last_active.cmp(&a.last_active))
                        .then_with(|| a.login.cmp(&b.login))
                });
                rows.truncate(limit);
                Ok(json!({ "people": rows, "people_total": people_total }))
            }
            _ => {
                let mut stats = core_summary::board_people_stats(
                    &self.store,
                    &self.repo_ids,
                    &logins,
                    &since,
                    &until,
                )
                .await?;
                let people_total = stats.people.len();
                // Busiest first (most timestamped work events).
                stats.people.sort_by(|a, b| {
                    b.total_events
                        .cmp(&a.total_events)
                        .then_with(|| a.login.cmp(&b.login))
                });
                stats.people.truncate(limit);
                let mut out = serde_json::to_value(&stats)?;
                out["people_total"] = json!(people_total);
                Ok(out)
            }
        }
    }
}

/// Board-level health/delivery reads, grouped behind one tool with a `mode`.
/// Board-level only, never per person.
pub struct BoardHealth {
    store: Arc<Store>,
    repo_ids: Vec<String>,
    now: String,
}

impl Tool for BoardHealth {
    const NAME: &'static str = "board_health";
    type Error = ToolError;
    type Args = BoardHealthArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Board-level health and delivery reads, never per-person. mode \"scorecard\": \
                each repo's Bronze/Silver/Gold tier from flow+CI+ownership facts, plus the board composite \
                (weakest repo). mode \"dora\": DORA-lite deploy frequency/lead time/change-failure-rate, \
                tiered elite..low, pooled over window_days (default 30); deploy frequency is real, the \
                others are documented proxies without deploy data. mode \"investment\": merged effort by \
                category (feature/bug/maintenance/docs/test/other) over a window (default last 30 days), \
                by PR count and churn - \"what did we spend this period on\"."
                .to_string(),
            parameters: obj_schema(
                json!({
                    "mode": { "type": "string", "enum": ["scorecard", "dora", "investment"] },
                    "window_days": { "type": "integer", "description": "dora mode: lookback window in days (default 30)" },
                    "since_days": { "type": "integer", "description": "investment mode: window start, days before now (default 30)" },
                    "until_days": { "type": "integer", "description": "investment mode: window end, days before now (default 0 = now)" },
                }),
                &["mode"],
            ),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.mode.as_str() {
            "dora" => {
                let window_days = args.window_days.unwrap_or(core_summary::DORA_WINDOW_DAYS);
                let metrics =
                    core_summary::board_dora(&self.store, &self.repo_ids, &self.now, window_days)
                        .await?;
                Ok(serde_json::to_value(metrics)?)
            }
            "investment" => {
                let since_days = args.since_days.unwrap_or(DEFAULT_SINCE_DAYS);
                let until_days = args.until_days.unwrap_or(0);
                let (since, until) = window_bounds(&self.now, since_days, until_days);
                let dist =
                    core_summary::board_investment(&self.store, &self.repo_ids, &since, &until)
                        .await?;
                Ok(serde_json::to_value(dist)?)
            }
            _ => {
                let scorecard =
                    core_summary::board_scorecard(&self.store, &self.repo_ids, &self.now).await?;
                Ok(serde_json::to_value(scorecard)?)
            }
        }
    }
}

/// Board-level code-risk signals, grouped behind one tool with a `mode`. Every mode is about code
/// (file/module/dependency), not a ranking of people; a module may name its sole owner.
pub struct CodeRisk {
    store: Arc<Store>,
    repo_ids: Vec<String>,
}

impl Tool for CodeRisk {
    const NAME: &'static str = "code_risk";
    type Error = ToolError;
    type Args = CodeRiskArgs;
    type Output = Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Board-level code-risk signals: the subject is always code (a file/module/dependency), \
                never a ranking of people. mode \"hotspots\": files that change often and \
                churn a lot. mode \"ownership\": modules with a low bus factor or one dominant author - \
                key-person/offboarding risk. mode \"fused\": combined hotspot x ownership x review-gap \
                score with reasons and recent touching PRs - the best single \"riskiest code\" answer. \
                mode \"coupling\": file pairs that frequently change together in the same merged PR. mode \
                \"dependencies\": third-party packages and which repos share them, most-shared first. \
                Every mode is bounded to `limit` rows (default {CODE_RISK_LIMIT}), riskiest/most-shared \
                first; the result's `total` names the true row count even when more rows exist than were \
                returned."
            ),
            parameters: obj_schema(
                json!({
                    "mode": { "type": "string", "enum": ["hotspots", "ownership", "fused", "coupling", "dependencies"] },
                    "limit": { "type": "integer", "description": "max rows returned, default 10" },
                }),
                &["mode"],
            ),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let limit = args.limit.unwrap_or(CODE_RISK_LIMIT);
        match args.mode.as_str() {
            // The ranked modes sort the full list before truncating to their own `limit`, so
            // usize::MAX returns the complete ordered list and the true total is captured here.
            "ownership" => {
                let rows =
                    core_summary::board_ownership_risks(&self.store, &self.repo_ids, usize::MAX)
                        .await?;
                Ok(bounded(rows, limit))
            }
            "fused" => {
                let rows =
                    core_summary::board_code_risk(&self.store, &self.repo_ids, usize::MAX).await?;
                Ok(bounded(rows, limit))
            }
            "coupling" => {
                let rows =
                    core_summary::board_coupling(&self.store, &self.repo_ids, usize::MAX).await?;
                Ok(bounded(rows, limit))
            }
            "dependencies" => {
                // board_dependencies does not truncate internally, so the limit applies here.
                let rows = core_summary::board_dependencies(&self.store, &self.repo_ids).await?;
                Ok(bounded(rows, limit))
            }
            _ => {
                let rows =
                    core_summary::board_hotspots(&self.store, &self.repo_ids, usize::MAX).await?;
                Ok(bounded(rows, limit))
            }
        }
    }
}

/// Assemble the board-scoped agent: the local model, the read-only tools, and RAG
/// `dynamic_context` over hybrid retrieval, with `preamble` setting the role and board context.
pub fn build_agent(
    model: LocalModel,
    store: Arc<Store>,
    embedder: Arc<dyn Embedder>,
    repo_ids: Vec<String>,
    now: String,
    preamble: String,
) -> Agent<LocalModel> {
    let activity_index = RetrievalIndex {
        store: store.clone(),
        embedder: embedder.clone(),
        repo_ids: repo_ids.clone(),
        code: false,
    };
    // Cap tool-call rounds so a confused model cannot loop forever.
    const MAX_TURNS: usize = 6;
    AgentBuilder::new(model)
        .preamble(&preamble)
        .temperature(0.2)
        .default_max_turns(MAX_TURNS)
        // RAG grounding: inject the most relevant activity snippets for each turn.
        .dynamic_context(4, activity_index)
        .tool(BoardAttention {
            store: store.clone(),
            repo_ids: repo_ids.clone(),
            now: now.clone(),
        })
        .tool(FlowMetrics {
            store: store.clone(),
            repo_ids: repo_ids.clone(),
        })
        .tool(WhatsStuck {
            store: store.clone(),
            repo_ids: repo_ids.clone(),
            now: now.clone(),
        })
        .tool(FailingCi {
            store: store.clone(),
            repo_ids: repo_ids.clone(),
        })
        .tool(SearchActivity {
            store: store.clone(),
            embedder: embedder.clone(),
            repo_ids: repo_ids.clone(),
        })
        .tool(SearchCode {
            store: store.clone(),
            embedder,
            repo_ids: repo_ids.clone(),
        })
        .tool(ListTeams {
            store: store.clone(),
        })
        .tool(ChangeDigest {
            store: store.clone(),
        })
        .tool(WorkItemStatus {
            store: store.clone(),
        })
        .tool(People {
            store: store.clone(),
            repo_ids: repo_ids.clone(),
            now: now.clone(),
        })
        .tool(BoardHealth {
            store: store.clone(),
            repo_ids: repo_ids.clone(),
            now,
        })
        .tool(CodeRisk { store, repo_ids })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_embed::HashEmbedder;
    use core_store::{DependencyRow, PrFile, PullRequest, Release, Repo};

    fn repo(id: &str) -> Repo {
        Repo {
            id: id.into(),
            owner: "acme".into(),
            name: "widget".into(),
            full_name: "acme/widget".into(),
            ownership: "owned".into(),
        }
    }

    fn pr(
        id: &str,
        repo_id: &str,
        author: &str,
        state: &str,
        created_at: &str,
        merged_at: Option<&str>,
    ) -> PullRequest {
        PullRequest {
            id: id.into(),
            repo_id: repo_id.into(),
            number: 1,
            title: format!("PR {id}"),
            state: state.into(),
            author_login: Some(author.into()),
            body: None,
            created_at: created_at.into(),
            merged_at: merged_at.map(Into::into),
            html_url: None,
        }
    }

    fn file(filename: &str, additions: i64, deletions: i64) -> PrFile {
        PrFile {
            pr_id: String::new(),
            filename: filename.into(),
            status: "modified".into(),
            additions,
            deletions,
            patch: None,
        }
    }

    async fn empty_store() -> Arc<Store> {
        Arc::new(Store::open_in_memory().await.unwrap())
    }

    fn ci_run(id: &str, repo_id: &str, sha: &str, completed_at: &str) -> core_store::CiRun {
        core_store::CiRun {
            id: id.into(),
            repo_id: repo_id.into(),
            commit_sha: Some(sha.into()),
            status: "completed".into(),
            conclusion: Some("failure".into()),
            completed_at: Some(completed_at.into()),
            html_url: None,
            run_attempt: Some(1),
        }
    }

    // ---- board_attention / whats_stuck / failing_ci ------------

    /// `board_attention` is bounded: two repos, each over the cap alone, with different severities
    /// so the board-wide (not per-repo) ranking is provable: failing_ci outranks stale_pr in
    /// `core_summary::ATTENTION_ORDER`.
    #[tokio::test]
    async fn board_attention_bounded_board_wide_by_severity_with_honest_totals() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:ci")).await.unwrap();
        store.upsert_repo(&repo("repo:stale")).await.unwrap();
        const FAILING: usize = 25;
        const STALE: usize = 25;
        for i in 0..FAILING {
            store
                .upsert_ci_run(&ci_run(
                    &format!("run{i}"),
                    "repo:ci",
                    &format!("sha{i}"),
                    "2026-06-19T00:00:00Z",
                ))
                .await
                .unwrap();
        }
        for i in 0..STALE {
            let id = format!("stale{i}");
            store
                .upsert_pull_request(&pr(
                    &id,
                    "repo:stale",
                    "alice",
                    "open",
                    "2026-04-01T00:00:00Z",
                    None,
                ))
                .await
                .unwrap();
            // A review already happened, so this fires `stale_pr` only, not also `review_wait`.
            store
                .upsert_review(&core_store::Review {
                    id: format!("rev{i}"),
                    pr_id: id,
                    reviewer_login: Some("bob".into()),
                    state: "commented".into(),
                    submitted_at: Some("2026-04-02T00:00:00Z".into()),
                })
                .await
                .unwrap();
        }

        let tool = BoardAttention {
            store: store.clone(),
            repo_ids: vec!["repo:ci".into(), "repo:stale".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool.call(NoArgs {}).await.unwrap();
        assert_eq!(out["attention_total"], (FAILING + STALE) as u64);

        let repos = out["repos"].as_array().unwrap();
        let ci_repo = repos.iter().find(|r| r["repo"] == "repo:ci").unwrap();
        let stale_repo = repos.iter().find(|r| r["repo"] == "repo:stale").unwrap();
        // True per-repo totals survive the bound even though the combined list was truncated.
        assert_eq!(ci_repo["attention_total"], FAILING as u64);
        assert_eq!(stale_repo["attention_total"], STALE as u64);

        let mut kinds: Vec<String> = repos
            .iter()
            .flat_map(|r| r["items"].as_array().unwrap())
            .map(|i| i["kind"].as_str().unwrap().to_string())
            .collect();
        kinds.sort();
        assert_eq!(
            kinds.len(),
            BOARD_ATTENTION_LIMIT,
            "bounded board-wide, not per repo"
        );
        let failing_kept = kinds.iter().filter(|k| *k == "failing_ci").count();
        let stale_kept = kinds.iter().filter(|k| *k == "stale_pr").count();
        assert_eq!(
            failing_kept, FAILING,
            "the more severe kind is kept in full"
        );
        assert_eq!(
            stale_kept,
            BOARD_ATTENTION_LIMIT - FAILING,
            "the rest fills the remaining slots"
        );
    }

    #[tokio::test]
    async fn board_attention_small_board_returns_everything_with_matching_total() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        store
            .upsert_ci_run(&ci_run("run1", "repo:a", "sha1", "2026-06-19T00:00:00Z"))
            .await
            .unwrap();

        let tool = BoardAttention {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool.call(NoArgs {}).await.unwrap();
        assert_eq!(out["attention_total"], 1);
        let repos = out["repos"].as_array().unwrap();
        assert_eq!(repos[0]["attention_total"], 1);
        assert_eq!(repos[0]["items"].as_array().unwrap().len(), 1);
    }

    /// `whats_stuck` is bounded and oldest-first: the longest-stuck PR is the most actionable.
    #[tokio::test]
    async fn whats_stuck_bounded_oldest_first_with_honest_total() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        const N: usize = STUCK_LIMIT + 10;
        for i in 0..N {
            // Older i = older PR: i=0 opened first, so it is the most stale.
            let created = format!("2026-01-01T00:{:02}:{:02}Z", i / 60, i % 60);
            store
                .upsert_pull_request(&pr(
                    &format!("p{i:03}"),
                    "repo:a",
                    "alice",
                    "open",
                    &created,
                    None,
                ))
                .await
                .unwrap();
        }

        let tool = WhatsStuck {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(StuckArgs {
                threshold_days: None,
                limit: None,
            })
            .await
            .unwrap();
        assert_eq!(out["stuck_total"], N as u64);
        let repos = out["repos"].as_array().unwrap();
        let stuck = repos[0]["stuck"].as_array().unwrap();
        assert_eq!(stuck.len(), STUCK_LIMIT);
        assert_eq!(stuck[0]["id"], "p000");
    }

    /// `failing_ci` is bounded and most-recent-first.
    #[tokio::test]
    async fn failing_ci_bounded_most_recent_first_with_honest_total() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        const N: usize = FAILING_CI_LIMIT + 10;
        for i in 0..N {
            // Higher i = more recent completion, so i=N-1 is the most recent.
            let completed = format!("2026-06-01T00:{:02}:{:02}Z", i / 60, i % 60);
            store
                .upsert_ci_run(&ci_run(
                    &format!("run{i:03}"),
                    "repo:a",
                    &format!("sha{i}"),
                    &completed,
                ))
                .await
                .unwrap();
        }

        let tool = FailingCi {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
        };
        let out = tool.call(FailingCiArgs { limit: None }).await.unwrap();
        assert_eq!(out["failing_total"], N as u64);
        let repos = out["repos"].as_array().unwrap();
        let failing = repos[0]["failing"].as_array().unwrap();
        assert_eq!(failing.len(), FAILING_CI_LIMIT);
        assert_eq!(failing[0]["id"], format!("run{:03}", N - 1));
    }

    // ---- people -----------------------------------------------------------

    #[tokio::test]
    async fn people_stats_mode_returns_per_person_throughput() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        store
            .upsert_pull_request(&pr(
                "p1",
                "repo:a",
                "alice",
                "closed",
                "2026-06-10T10:00:00Z",
                Some("2026-06-11T10:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .upsert_commit(&core_store::Commit {
                sha: "c1".into(),
                repo_id: "repo:a".into(),
                author_login: Some("alice".into()),
                message: "m".into(),
                committed_at: "2026-06-10T09:00:00Z".into(),
            })
            .await
            .unwrap();

        let tool = People {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(PeopleArgs {
                mode: "stats".into(),
                since_days: None,
                until_days: None,
                limit: None,
            })
            .await
            .unwrap();
        let people = out["people"].as_array().unwrap();
        let alice = people
            .iter()
            .find(|p| p["login"] == "alice")
            .expect("alice has a row (discovered from window activity)");
        assert_eq!(alice["commits"], 1);
        assert_eq!(alice["prs_merged"], 1);
        assert_eq!(out["people_total"], 1);
    }

    #[tokio::test]
    async fn people_pickup_mode_returns_load_and_areas() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        store
            .upsert_pull_request(&pr(
                "p1",
                "repo:a",
                "alice",
                "open",
                "2026-06-15T10:00:00Z",
                None,
            ))
            .await
            .unwrap();
        store
            .replace_pr_files("p1", &[file("src/foo.rs", 5, 1)])
            .await
            .unwrap();

        let tool = People {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(PeopleArgs {
                mode: "pickup".into(),
                since_days: None,
                until_days: None,
                limit: None,
            })
            .await
            .unwrap();
        let rows = out["people"].as_array().unwrap();
        let alice = rows
            .iter()
            .find(|p| p["login"] == "alice")
            .expect("alice has a row (open PR = current load)");
        assert_eq!(alice["open_prs"], 1);
        let areas = alice["areas"].as_array().unwrap();
        assert!(areas
            .iter()
            .any(|a| a["area"] == "src" && a["changes"] == 1));
        assert_eq!(out["people_total"], 1);
    }

    #[tokio::test]
    async fn people_honest_empty_on_empty_board() {
        let store = empty_store().await;
        let tool = People {
            store: store.clone(),
            repo_ids: vec![],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let stats = tool
            .call(PeopleArgs {
                mode: "stats".into(),
                since_days: None,
                until_days: None,
                limit: None,
            })
            .await
            .unwrap();
        assert_eq!(stats["people"], serde_json::json!([]));
        assert_eq!(stats["people_total"], 0);
        assert_eq!(stats["total_events"], 0);
        assert_eq!(stats["off_hours_events"], 0);

        let pickup = tool
            .call(PeopleArgs {
                mode: "pickup".into(),
                since_days: None,
                until_days: None,
                limit: None,
            })
            .await
            .unwrap();
        assert_eq!(pickup["people"], serde_json::json!([]));
        assert_eq!(pickup["people_total"], 0);
    }

    /// The `people` roster is bounded: one discovered from activity can reach the hundreds. Proves
    /// the bound engages, the total is reported, and the kept rows are the busiest.
    #[tokio::test]
    async fn people_stats_mode_bounded_with_honest_total_for_a_large_roster() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        const N: usize = 300;
        for i in 0..N {
            let login = format!("user{i:04}");
            // Descending activity (user0000 has N commits, user0001 N-1, ...) fixes the order.
            let commits = N - i;
            for c in 0..commits {
                store
                    .upsert_commit(&core_store::Commit {
                        sha: format!("{login}-c{c}"),
                        repo_id: "repo:a".into(),
                        author_login: Some(login.clone()),
                        message: "m".into(),
                        committed_at: "2026-06-10T09:00:00Z".into(),
                    })
                    .await
                    .unwrap();
            }
        }

        let tool = People {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(PeopleArgs {
                mode: "stats".into(),
                since_days: None,
                until_days: None,
                limit: None,
            })
            .await
            .unwrap();
        let people = out["people"].as_array().unwrap();
        assert_eq!(people.len(), PEOPLE_LIMIT, "bounded to the default limit");
        assert_eq!(
            out["people_total"], N as u64,
            "the true roster size, not the capped count"
        );
        // Busiest first: the top PEOPLE_LIMIT rows are user0000..user{PEOPLE_LIMIT-1}, in order.
        for (i, person) in people.iter().enumerate() {
            assert_eq!(person["login"], format!("user{i:04}"));
            assert_eq!(person["commits"], (N - i) as u64);
        }
    }

    #[tokio::test]
    async fn people_pickup_mode_bounded_with_honest_total_for_a_large_roster() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        const N: usize = 300;
        for i in 0..N {
            let login = format!("user{i:04}");
            // Descending open-PR count: user0000 carries N open PRs, user0001 carries N-1, ...
            let open_prs = N - i;
            for p in 0..open_prs {
                store
                    .upsert_pull_request(&pr(
                        &format!("{login}-p{p}"),
                        "repo:a",
                        &login,
                        "open",
                        "2026-06-15T10:00:00Z",
                        None,
                    ))
                    .await
                    .unwrap();
            }
        }

        let tool = People {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(PeopleArgs {
                mode: "pickup".into(),
                since_days: None,
                until_days: None,
                limit: None,
            })
            .await
            .unwrap();
        let people = out["people"].as_array().unwrap();
        assert_eq!(people.len(), PEOPLE_LIMIT, "bounded to the default limit");
        assert_eq!(
            out["people_total"], N as u64,
            "the true roster size, not the capped count"
        );
        for (i, person) in people.iter().enumerate() {
            assert_eq!(person["login"], format!("user{i:04}"));
            assert_eq!(person["open_prs"], (N - i) as u64);
        }
    }

    /// Below the limit, the bound is a no-op and the total matches what was returned.
    #[tokio::test]
    async fn people_small_board_returns_everything_with_matching_total() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        for login in ["alice", "bob", "carol"] {
            store
                .upsert_commit(&core_store::Commit {
                    sha: format!("{login}-c1"),
                    repo_id: "repo:a".into(),
                    author_login: Some(login.into()),
                    message: "m".into(),
                    committed_at: "2026-06-10T09:00:00Z".into(),
                })
                .await
                .unwrap();
        }

        let tool = People {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(PeopleArgs {
                mode: "stats".into(),
                since_days: None,
                until_days: None,
                limit: None,
            })
            .await
            .unwrap();
        let people = out["people"].as_array().unwrap();
        assert_eq!(people.len(), 3);
        assert_eq!(out["people_total"], 3);
    }

    /// Not a correctness test: measures the serialized size of `people` mode `stats` for a roster
    /// of ~1109 authors, unbounded (`board_people_stats` serialized whole) vs the bounded tool.
    /// Uses a ~4 chars/token estimate. Run with `--nocapture` to see the numbers.
    #[tokio::test]
    async fn people_stats_result_size_before_and_after_bounding() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        const N: usize = 1100;
        let mut logins = Vec::with_capacity(N);
        for i in 0..N {
            let login = format!("user{i:04}");
            store
                .upsert_commit(&core_store::Commit {
                    sha: format!("{login}-c1"),
                    repo_id: "repo:a".into(),
                    author_login: Some(login.clone()),
                    message: "m".into(),
                    committed_at: "2026-06-10T09:00:00Z".into(),
                })
                .await
                .unwrap();
            logins.push(login);
        }

        // Before: every row, no limit.
        let repo_ids = vec!["repo:a".to_string()];
        let unbounded = core_summary::board_people_stats(
            &store,
            &repo_ids,
            &logins,
            "2026-05-21T00:00:00Z",
            "2026-06-20T00:00:00Z",
        )
        .await
        .unwrap();
        let before = serde_json::to_string(&unbounded).unwrap();

        // After: the bounded tool, same data, default limit.
        let tool = People {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(PeopleArgs {
                mode: "stats".into(),
                since_days: None,
                until_days: None,
                limit: None,
            })
            .await
            .unwrap();
        let after = serde_json::to_string(&out).unwrap();

        println!(
            "people stats result size, {N} contributors: before = {} chars (~{} tok); \
             after = {} chars (~{} tok)",
            before.len(),
            before.len() / 4,
            after.len(),
            after.len() / 4,
        );
        assert!(
            after.len() < before.len() / 10,
            "the bound should cut the payload by at least an order of magnitude"
        );
    }

    // ---- board_health -------------------------------------------------------

    #[tokio::test]
    async fn board_health_scorecard_mode_returns_repo_tiers() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        store
            .upsert_pull_request(&pr(
                "p1",
                "repo:a",
                "alice",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-02T00:00:00Z"),
            ))
            .await
            .unwrap();

        let tool = BoardHealth {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(BoardHealthArgs {
                mode: "scorecard".into(),
                since_days: None,
                until_days: None,
                window_days: None,
            })
            .await
            .unwrap();
        assert!(out["composite_tier"].is_string());
        assert_eq!(out["repos"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn board_health_dora_mode_reports_real_deploy_frequency() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        store
            .upsert_release(&Release {
                id: "r1".into(),
                repo_id: "repo:a".into(),
                tag: "v1".into(),
                name: Some("v1".into()),
                published_at: Some("2026-06-15T00:00:00Z".into()),
            })
            .await
            .unwrap();

        let tool = BoardHealth {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(BoardHealthArgs {
                mode: "dora".into(),
                since_days: None,
                until_days: None,
                window_days: None,
            })
            .await
            .unwrap();
        assert_eq!(out["window_days"], 30);
        assert!(
            out["deploy_frequency_per_week"].as_f64().unwrap() > 0.0,
            "the release in the window is real deploy data, not a proxy"
        );
    }

    #[tokio::test]
    async fn board_health_investment_mode_classifies_conventional_titles() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        let mut merged = pr(
            "p1",
            "repo:a",
            "alice",
            "closed",
            "2026-06-10T00:00:00Z",
            Some("2026-06-11T00:00:00Z"),
        );
        merged.title = "feat: add widgets".into();
        store.upsert_pull_request(&merged).await.unwrap();

        let tool = BoardHealth {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let out = tool
            .call(BoardHealthArgs {
                mode: "investment".into(),
                since_days: None,
                until_days: None,
                window_days: None,
            })
            .await
            .unwrap();
        let buckets = out["buckets"].as_array().unwrap();
        let feature = buckets
            .iter()
            .find(|b| b["category"] == "feature")
            .expect("a feat: PR classifies as feature");
        assert_eq!(feature["count"], 1);
        assert_eq!(out["total_count"], 1);
    }

    #[tokio::test]
    async fn board_health_honest_empty_on_empty_board() {
        let store = empty_store().await;
        let tool = BoardHealth {
            store: store.clone(),
            repo_ids: vec![],
            now: "2026-06-20T00:00:00Z".into(),
        };
        let args = |mode: &str| BoardHealthArgs {
            mode: mode.into(),
            since_days: None,
            until_days: None,
            window_days: None,
        };

        let scorecard = tool.call(args("scorecard")).await.unwrap();
        assert_eq!(scorecard["composite_tier"], "none");
        assert_eq!(scorecard["repos"], serde_json::json!([]));

        // No releases/PRs/commits at all: the DORA proxies are absent, not a fabricated tier.
        let dora = tool.call(args("dora")).await.unwrap();
        assert_eq!(dora["lead_tier"], "unknown");
        assert_eq!(dora["cfr_tier"], "unknown");
        assert_eq!(dora["lead_time_secs"], serde_json::Value::Null);

        let investment = tool.call(args("investment")).await.unwrap();
        assert_eq!(investment["total_count"], 0);
        assert_eq!(investment["total_churn"], 0);
    }

    // ---- code_risk ----------------------------------------------------------

    /// One file (`src/foo.rs`) touched by one author across 3 merged PRs: enough to clear the
    /// `board_ownership_risks` minimum-changes gate and score as a hotspot/fused-risk row.
    async fn seeded_code_risk_store() -> Arc<Store> {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        for i in 1..=3 {
            let id = format!("p{i}");
            let created = format!("2026-06-0{i}T00:00:00Z");
            let merged = format!("2026-06-0{i}T12:00:00Z");
            store
                .upsert_pull_request(&pr(
                    &id,
                    "repo:a",
                    "alice",
                    "closed",
                    &created,
                    Some(&merged),
                ))
                .await
                .unwrap();
            store
                .replace_pr_files(&id, &[file("src/foo.rs", 5, 1)])
                .await
                .unwrap();
        }
        store
    }

    #[tokio::test]
    async fn code_risk_hotspots_mode_ranks_by_change_x_churn() {
        let store = seeded_code_risk_store().await;
        let tool = CodeRisk {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
        };
        let out = tool
            .call(CodeRiskArgs {
                mode: "hotspots".into(),
                limit: None,
            })
            .await
            .unwrap();
        let rows = out["items"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["path"], "src/foo.rs");
        assert_eq!(rows[0]["changes"], 3);
        assert_eq!(rows[0]["churn"], 18); // 3 PRs x (5 additions + 1 deletion)
        assert_eq!(
            out["total"], 1,
            "below the limit: total matches the returned rows"
        );
    }

    #[tokio::test]
    async fn code_risk_ownership_mode_flags_single_author_module() {
        let store = seeded_code_risk_store().await;
        let tool = CodeRisk {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
        };
        let out = tool
            .call(CodeRiskArgs {
                mode: "ownership".into(),
                limit: None,
            })
            .await
            .unwrap();
        let rows = out["items"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["module"], "src");
        assert_eq!(rows[0]["top_author"], "alice");
        assert_eq!(rows[0]["bus_factor"], 1);
        assert_eq!(out["total"], 1);
    }

    #[tokio::test]
    async fn code_risk_fused_mode_scores_and_explains() {
        let store = seeded_code_risk_store().await;
        let tool = CodeRisk {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
        };
        let out = tool
            .call(CodeRiskArgs {
                mode: "fused".into(),
                limit: None,
            })
            .await
            .unwrap();
        let rows = out["items"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["path"], "src/foo.rs");
        assert!(rows[0]["score"].as_f64().unwrap() > 0.0);
        assert!(!rows[0]["reasons"].as_array().unwrap().is_empty());
        assert_eq!(out["total"], 1);
    }

    #[tokio::test]
    async fn code_risk_coupling_mode_finds_co_changing_files() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        for id in ["p1", "p2"] {
            store
                .upsert_pull_request(&pr(
                    id,
                    "repo:a",
                    "alice",
                    "closed",
                    "2026-06-01T00:00:00Z",
                    Some("2026-06-02T00:00:00Z"),
                ))
                .await
                .unwrap();
            store
                .replace_pr_files(id, &[file("src/a.rs", 3, 0), file("src/b.rs", 2, 0)])
                .await
                .unwrap();
        }

        let tool = CodeRisk {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
        };
        let out = tool
            .call(CodeRiskArgs {
                mode: "coupling".into(),
                limit: None,
            })
            .await
            .unwrap();
        let rows = out["items"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["path_a"], "src/a.rs");
        assert_eq!(rows[0]["path_b"], "src/b.rs");
        assert_eq!(rows[0]["together"], 2);
        assert_eq!(out["total"], 1);
    }

    #[tokio::test]
    async fn code_risk_dependencies_mode_lists_shared_packages() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        store
            .upsert_dependency(&DependencyRow {
                repo_id: "repo:a".into(),
                ecosystem: "cargo".into(),
                name: "serde".into(),
                version_req: Some("1".into()),
                kind: "normal".into(),
                source: "Cargo.toml".into(),
            })
            .await
            .unwrap();

        let tool = CodeRisk {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
        };
        let out = tool
            .call(CodeRiskArgs {
                mode: "dependencies".into(),
                limit: None,
            })
            .await
            .unwrap();
        let rows = out["items"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["name"], "serde");
        assert_eq!(rows[0]["ecosystem"], "cargo");
        assert_eq!(out["total"], 1);
    }

    /// `dependencies` mode applies the tool's `limit` argument.
    #[tokio::test]
    async fn code_risk_dependencies_mode_bounded_with_honest_total() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        const N: usize = 40;
        for i in 0..N {
            store
                .upsert_dependency(&DependencyRow {
                    repo_id: "repo:a".into(),
                    ecosystem: "cargo".into(),
                    name: format!("pkg{i:03}"),
                    version_req: Some("1".into()),
                    kind: "normal".into(),
                    source: "Cargo.toml".into(),
                })
                .await
                .unwrap();
        }

        let tool = CodeRisk {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
        };
        let out = tool
            .call(CodeRiskArgs {
                mode: "dependencies".into(),
                limit: None,
            })
            .await
            .unwrap();
        let rows = out["items"].as_array().unwrap();
        assert_eq!(
            rows.len(),
            CODE_RISK_LIMIT,
            "bounded to the default limit, not all {N}"
        );
        assert_eq!(out["total"], N as u64, "the true count, not the capped one");
    }

    /// A hotspot count above the limit reports the true total rather than silently dropping rows.
    #[tokio::test]
    async fn code_risk_hotspots_mode_reports_total_above_the_limit() {
        let store = empty_store().await;
        store.upsert_repo(&repo("repo:a")).await.unwrap();
        const N: usize = CODE_RISK_LIMIT + 5;
        for i in 0..N {
            let id = format!("p{i}");
            store
                .upsert_pull_request(&pr(
                    &id,
                    "repo:a",
                    "alice",
                    "closed",
                    "2026-06-01T00:00:00Z",
                    Some("2026-06-02T00:00:00Z"),
                ))
                .await
                .unwrap();
            store
                .replace_pr_files(&id, &[file(&format!("src/file{i:03}.rs"), 5, 1)])
                .await
                .unwrap();
        }

        let tool = CodeRisk {
            store: store.clone(),
            repo_ids: vec!["repo:a".into()],
        };
        let out = tool
            .call(CodeRiskArgs {
                mode: "hotspots".into(),
                limit: None,
            })
            .await
            .unwrap();
        let rows = out["items"].as_array().unwrap();
        assert_eq!(rows.len(), CODE_RISK_LIMIT);
        assert_eq!(
            out["total"], N as u64,
            "more hotspots exist than were returned"
        );
    }

    #[tokio::test]
    async fn code_risk_honest_empty_on_empty_board() {
        let store = empty_store().await;
        let tool = CodeRisk {
            store: store.clone(),
            repo_ids: vec![],
        };
        for mode in ["hotspots", "ownership", "fused", "coupling", "dependencies"] {
            let out = tool
                .call(CodeRiskArgs {
                    mode: mode.into(),
                    limit: None,
                })
                .await
                .unwrap();
            assert_eq!(
                out["items"],
                serde_json::json!([]),
                "mode {mode} should be honestly empty"
            );
            assert_eq!(
                out["total"], 0,
                "mode {mode} should report a true zero total"
            );
        }
    }

    // ---- tool-schema prompt-size report ---------------------------

    /// Not a correctness test: measures the prompt size of each tool's definition rendered as in
    /// `prompt::tools_instruction` (name + description + JSON schema), original 9 tools vs the 3
    /// added, with a ~4 chars/token estimate. Run with `--nocapture` to see the numbers.
    #[tokio::test]
    async fn tool_schema_block_char_and_token_size() {
        let store = empty_store().await;
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default());
        let repo_ids = vec!["repo:a".to_string()];
        let now = "2026-06-20T00:00:00Z".to_string();

        let original = vec![
            BoardAttention {
                store: store.clone(),
                repo_ids: repo_ids.clone(),
                now: now.clone(),
            }
            .definition(String::new())
            .await,
            FlowMetrics {
                store: store.clone(),
                repo_ids: repo_ids.clone(),
            }
            .definition(String::new())
            .await,
            WhatsStuck {
                store: store.clone(),
                repo_ids: repo_ids.clone(),
                now: now.clone(),
            }
            .definition(String::new())
            .await,
            FailingCi {
                store: store.clone(),
                repo_ids: repo_ids.clone(),
            }
            .definition(String::new())
            .await,
            SearchActivity {
                store: store.clone(),
                embedder: embedder.clone(),
                repo_ids: repo_ids.clone(),
            }
            .definition(String::new())
            .await,
            SearchCode {
                store: store.clone(),
                embedder: embedder.clone(),
                repo_ids: repo_ids.clone(),
            }
            .definition(String::new())
            .await,
            ListTeams {
                store: store.clone(),
            }
            .definition(String::new())
            .await,
            ChangeDigest {
                store: store.clone(),
            }
            .definition(String::new())
            .await,
            WorkItemStatus {
                store: store.clone(),
            }
            .definition(String::new())
            .await,
        ];
        let added = vec![
            People {
                store: store.clone(),
                repo_ids: repo_ids.clone(),
                now: now.clone(),
            }
            .definition(String::new())
            .await,
            BoardHealth {
                store: store.clone(),
                repo_ids: repo_ids.clone(),
                now: now.clone(),
            }
            .definition(String::new())
            .await,
            CodeRisk {
                store: store.clone(),
                repo_ids: repo_ids.clone(),
            }
            .definition(String::new())
            .await,
        ];

        let block = |defs: &[rig::completion::ToolDefinition]| -> String {
            defs.iter()
                .map(|t| {
                    format!(
                        "- {}: {}\n  arguments JSON schema: {}\n",
                        t.name, t.description, t.parameters
                    )
                })
                .collect()
        };
        let before = block(&original);
        let added_only = block(&added);
        let after = format!("{before}{added_only}");

        println!(
            "tool schema block: before = {} chars (~{} tok, 9 tools); added = {} chars \
             (~{} tok, 3 tools); after = {} chars (~{} tok, 12 tools)",
            before.len(),
            before.len() / 4,
            added_only.len(),
            added_only.len() / 4,
            after.len(),
            after.len() / 4,
        );
        assert!(!before.is_empty());
        assert!(after.len() > before.len());
    }
}
