//! orgonzola knowledge-base store.
//! A SQLite database behind embedded migrations and a typed API. Host-agnostic: no display, no
//! network; the tests run against an in-memory database.

use std::str::FromStr;

mod storage;
pub use storage::{
    table_group, BudgetState, DroppedIndex, GroupStorage, ReclaimReport, RepoStorage, StorageGroup,
    StorageMethod, StoragePressure, StorageReport, StorageUsage, TableStorage, MB, WARN_FRACTION,
};

use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
    #[error(transparent)]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error("failed to (de)serialize a stored JSON field: {0}")]
    Json(#[from] serde_json::Error),
    /// A repo cannot be forgotten while something still watches it. `by` names the holder in the
    /// user's terms.
    #[error("{repo_id} is still watched by {by}. Remove it there first, then forget the repo.")]
    StillWatched { repo_id: String, by: String },
}

/// Register sqlite-vec on SQLite's global auto-extension list, once per process, so every
/// connection (including pool connections) has the `vec0` module. Must run before any connection
/// is opened.
fn register_sqlite_vec() {
    use std::sync::Once;
    static REGISTER: Once = Once::new();
    // SQLite's auto-extension entry point: the generic signature `sqlite3_auto_extension` stores.
    type ExtInit = unsafe extern "C" fn(
        *mut libsqlite3_sys::sqlite3,
        *mut *mut std::os::raw::c_char,
        *const libsqlite3_sys::sqlite3_api_routines,
    ) -> std::os::raw::c_int;
    REGISTER.call_once(|| {
        // SAFETY: the documented sqlite-vec registration. `sqlite3_auto_extension` stores the
        // entry point in SQLite's global list; the cast adapts sqlite-vec's typed init fn to the
        // entry-point signature SQLite expects. Idempotent under `Once`.
        unsafe {
            let init: ExtInit = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
            libsqlite3_sys::sqlite3_auto_extension(Some(init));
        }
    });
}

// ---- row types -------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Repo {
    pub id: String,
    pub owner: String,
    pub name: String,
    pub full_name: String,
    /// "owned" | "observed".
    pub ownership: String,
}

/// A repo's fork relationship, read on demand: whether the forge reports a fork and, if so, its
/// upstream parent's repo id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct RepoFork {
    pub repo_id: String,
    pub is_fork: bool,
    pub parent_repo_id: Option<String>,
}

/// A configured watch source (persisted `core_model::Source`) that the scheduler polls. `filters`
/// is stored as a JSON array in a TEXT column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRow {
    pub id: String,
    /// "org" | "repo" | "user".
    pub kind: String,
    /// "owner/name" for a repo, the login for an org or user.
    pub name: String,
    /// "owned" | "observed".
    pub ownership: String,
    pub filters: Vec<String>,
    pub stale_pr_days: u32,
    /// The forge this source belongs to. `None` = unrouted; the scheduler skips it.
    pub forge_id: Option<String>,
}

/// A configured forge. `id` is the routing key and appears in repo ids
/// (`repo:<forge_id>/<owner>/<name>`). The access token lives in the OS keychain keyed by `id`,
/// not here. `oauth_client_id` is the public OAuth client id for the device flow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct ForgeRow {
    pub id: String,
    pub name: String,
    /// "github" | "gitea".
    pub kind: String,
    /// The API root, e.g. `https://api.github.com` or `https://gitea.example.com/api/v1`.
    pub base_url: String,
    /// Public OAuth client id for the device flow. `None` = PAT-only.
    pub oauth_client_id: Option<String>,
}

/// Global application settings, stored as the single row of the `settings` table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Settings {
    /// How often the scheduler polls each watched source, in seconds.
    pub sync_period_secs: i64,
    /// Days after which an open PR is flagged "stale" by the watchdog.
    pub stale_pr_days: i64,
    /// Optional Slack-compatible incoming-webhook URL for the digest push. `None` = no egress.
    pub digest_webhook_url: Option<String>,
    /// Optional cadence (hours) for the scheduled digest push. `None` = off (no auto-egress).
    pub digest_schedule_hours: Option<i64>,
    /// Runtime on/off switch for local-LLM narration. Also needs the `llm` feature and a model.
    pub llm_enabled: bool,
    /// The selected GGUF model filename in the models dir. `None` = fall back to the env default.
    pub llm_model: Option<String>,
    /// Disk budget for the database file and its WAL, in megabytes. `None` = no limit. Growth
    /// pauses near the limit and stops at it; nothing is deleted to fit.
    pub storage_budget_mb: Option<i64>,
}

/// What a board watches, fixed at creation. The kind decides the scope inputs, which rollups
/// apply, and how repos are indexed:
/// - `Team`: people-first, optionally narrowed by an org and/or pinned repos. Discovers the repos
///   its people touch; code and activity are embedded.
/// - `Repo`: one repo, no people, no discovery; indexed like a team's repo.
/// - `Org`: a whole org, metrics rollup. Syncs active repos' activity; no people, no embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoardKind {
    Team,
    Repo,
    Org,
}

impl BoardKind {
    /// The lowercase tag stored in `boards.kind`.
    pub fn as_str(self) -> &'static str {
        match self {
            BoardKind::Team => "team",
            BoardKind::Repo => "repo",
            BoardKind::Org => "org",
        }
    }

    /// Parse the stored tag; an unknown value falls back to `Team`.
    pub fn from_tag(s: &str) -> Self {
        match s {
            "repo" => BoardKind::Repo,
            "org" => BoardKind::Org,
            _ => BoardKind::Team,
        }
    }
}

/// A board: the unit of scope. Its `kind` decides what it watches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Board {
    pub id: String,
    pub name: String,
    /// What the board watches; set at creation.
    pub kind: BoardKind,
    pub created_at: String,
    /// Restrict discovery to repos in this org. `None` = no restriction.
    pub org: Option<String>,
    /// The forge this board syncs from. `None` = unassigned, so it cannot be synced.
    pub forge_id: Option<String>,
    /// Opt-in dependency vulnerability scanning. When on, a scan queries the OSV advisory database
    /// (network egress).
    pub scan_dependencies: bool,
    /// Whether discovery may include archived repos. Pinned repos are unaffected.
    pub include_archived: bool,
    pub people: Vec<String>,
    /// Repos pinned to the board. Empty = discover from people.
    pub repos: Vec<String>,
}

/// A file changed by a pull request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct PrFile {
    pub pr_id: String,
    pub filename: String,
    /// "added" | "modified" | "removed" | "renamed" | ...
    pub status: String,
    pub additions: i64,
    pub deletions: i64,
    /// The unified-diff hunk; `None` when GitHub omits it (binary or too large).
    pub patch: Option<String>,
}

/// Per-file PR change activity for a repo, aggregated from `pr_files` and `pull_requests`. Input
/// to hotspot and ownership analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct FileStat {
    pub path: String,
    /// Number of PRs that touched this file.
    pub changes: i64,
    pub additions: i64,
    pub deletions: i64,
    /// Distinct PR-author logins that touched it.
    pub authors: i64,
}

/// One (file, author) PR-change count for a repo: the input to ownership / bus-factor risk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct FileAuthor {
    pub path: String,
    pub author: String,
    pub changes: i64,
}

/// Aggregate store and index row counts for the Debug view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreCounts {
    pub repos: i64,
    pub commits: i64,
    pub pull_requests: i64,
    pub issues: i64,
    pub sources: i64,
    pub boards: i64,
    pub embeddings_total: i64,
    pub embeddings_code: i64,
    pub embeddings_activity: i64,
    pub code_files: i64,
}

/// What [`Store::forget_repo`] removed. `found` is false when the repo id was not in the store.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgottenRepo {
    pub repo_id: String,
    pub found: bool,
    pub commits: i64,
    pub pull_requests: i64,
    pub issues: i64,
    pub code_files: i64,
    /// Chunks removed from the search index, and with them their vector + full-text rows.
    pub index_chunks: i64,
    /// Every row deleted across every table, including the counts above.
    pub total_rows: i64,
}

/// Repos no board can reach: not pinned in `board_repos` and not in `board_discovered_repos`. What
/// `reclaim_orphan_repos` may remove.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrphanRepoReport {
    /// Every orphaned repo, sized as `storage()` sizes a repo, largest first.
    pub repos: Vec<RepoStorage>,
    /// `repos.len()`.
    pub count: i64,
    /// Sum of each repo's `estimated_bytes` (an estimate, see `RepoStorage`).
    pub estimated_bytes: i64,
    /// Sum of each repo's `rows`.
    pub rows: i64,
}

/// What [`Store::reclaim_orphan_repos`] removed: every orphaned repo.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrphanReclaimReport {
    /// The repo ids removed.
    pub repo_ids: Vec<String>,
    /// `repo_ids.len()`.
    pub count: i64,
    /// Every row deleted across all repos and tables.
    pub total_rows: i64,
}

/// A daily metric snapshot for a scope, for trends. `captured_on` is a date `YYYY-MM-DD`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct MetricSnapshot {
    pub scope_kind: String,
    pub scope_id: String,
    pub captured_on: String,
    pub wip: i64,
    pub stale_open_prs: i64,
    pub merged_without_review: i64,
    pub attention_count: i64,
    pub median_cycle_time_secs: Option<i64>,
    /// Cycle-time phase medians: pickup = created -> first review, review = first review ->
    /// merge. Nullable when there is nothing to measure that day.
    pub median_pickup_secs: Option<i64>,
    pub median_review_secs: Option<i64>,
}

/// Decompose a snapshot into the `(metric_key, value)` rows of the long-format table. Nullable
/// medians add a row only when present.
fn snapshot_metrics(s: &MetricSnapshot) -> Vec<(&'static str, i64)> {
    let mut kv = vec![
        ("wip", s.wip),
        ("stale_open_prs", s.stale_open_prs),
        ("merged_without_review", s.merged_without_review),
        ("attention_count", s.attention_count),
    ];
    for (key, value) in [
        ("median_cycle_time_secs", s.median_cycle_time_secs),
        ("median_pickup_secs", s.median_pickup_secs),
        ("median_review_secs", s.median_review_secs),
    ] {
        if let Some(value) = value {
            kv.push((key, value));
        }
    }
    kv
}

// The raw row as stored: `filters` is JSON text, decoded into `SourceRow::filters` on read.
#[derive(sqlx::FromRow)]
struct SourceRecord {
    id: String,
    kind: String,
    name: String,
    ownership: String,
    filters: String,
    stale_pr_days: i64,
    forge_id: Option<String>,
}

// The raw board row, without people/repos (loaded separately).
#[derive(sqlx::FromRow)]
struct BoardRecord {
    id: String,
    name: String,
    kind: String,
    created_at: String,
    org: Option<String>,
    forge_id: Option<String>,
    scan_dependencies: bool,
    include_archived: bool,
}

impl SourceRecord {
    fn into_row(self) -> Result<SourceRow, StoreError> {
        Ok(SourceRow {
            id: self.id,
            kind: self.kind,
            name: self.name,
            ownership: self.ownership,
            filters: serde_json::from_str(&self.filters)?,
            stale_pr_days: self.stale_pr_days as u32,
            forge_id: self.forge_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Team {
    pub id: String,
    pub name: String,
    /// "github" | "manual".
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Sprint {
    pub id: String,
    pub team_id: Option<String>,
    pub name: String,
    pub starts_on: Option<String>,
    pub ends_on: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct WorkItem {
    pub id: String,
    pub title: String,
    pub kind: String,
    pub state: String,
    pub source: String,
    /// Normalized lifecycle bucket: 'new' | 'indeterminate' | 'done'. `None` until the source
    /// reports a status.
    pub status_category: Option<String>,
}

/// A registered Jira site connection. Read-only. The API token lives in the OS keychain, not here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Tracker {
    pub id: String,
    pub name: String,
    /// "jira".
    pub kind: String,
    pub base_url: String,
    /// Jira Cloud basic-auth email; `None` for a bearer PAT (Server/DC).
    pub email: Option<String>,
}

/// A board's link to a tracker project: which Jira project(s) a board watches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct BoardTracker {
    pub board_id: String,
    pub tracker_id: String,
    pub project_key: String,
}

/// A Jira issue: base work_item facts plus Jira detail from `jira_issues`. `work_item_id` is
/// `jira:<tracker>:<KEY>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct JiraIssueRow {
    pub work_item_id: String,
    pub tracker_id: String,
    pub project: String,
    pub issue_key: String,
    pub title: String,
    /// The Jira issue type (stored as the work_item `kind`): Bug / Task / Story / Epic / ...
    pub issue_type: String,
    /// The Jira status name (stored as the work_item `state`): Open / In Progress / Done / ...
    pub status: String,
    /// The workflow-agnostic status category: "new" | "indeterminate" | "done".
    pub status_category: Option<String>,
    pub assignee: Option<String>,
    pub url: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub resolved_at: Option<String>,
    /// Parent/epic issue key (`fields.parent.key`). `None` for a top-level issue.
    pub parent_key: Option<String>,
}

/// A Jira sprint: base sprint facts plus its Jira state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct JiraSprintRow {
    pub sprint_id: String,
    pub tracker_id: String,
    pub project: String,
    pub name: String,
    /// "active" | "closed" | "future".
    pub state: Option<String>,
    pub starts_on: Option<String>,
    pub ends_on: Option<String>,
    /// Date the start-of-sprint commitment was frozen; `None` if not yet taken.
    pub committed_at: Option<String>,
}

/// A directed, typed link between two entities (issue->PR, PR->commit, ...).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Link {
    pub id: i64,
    pub src_kind: String,
    pub src_id: String,
    pub dst_kind: String,
    pub dst_id: String,
    pub relation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Commit {
    pub sha: String,
    pub repo_id: String,
    pub author_login: Option<String>,
    pub message: String,
    pub committed_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct PullRequest {
    pub id: String,
    pub repo_id: String,
    pub number: i64,
    pub title: String,
    /// "open" | "closed".
    pub state: String,
    pub author_login: Option<String>,
    /// The PR description (GitHub Markdown). `None` if the PR has no body.
    pub body: Option<String>,
    pub created_at: String,
    pub merged_at: Option<String>,
    /// The forge web URL of the PR. `None` if the forge did not report one.
    pub html_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Review {
    pub id: String,
    pub pr_id: String,
    pub reviewer_login: Option<String>,
    pub state: String,
    pub submitted_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Issue {
    pub id: String,
    pub repo_id: String,
    pub number: i64,
    pub title: String,
    /// "open" | "closed".
    pub state: String,
    pub author_login: Option<String>,
    /// The issue description (GitHub Markdown). `None` if the issue has no body.
    pub body: Option<String>,
    pub created_at: String,
    pub closed_at: Option<String>,
    /// The issue's label names, newline-joined; empty when none.
    pub labels: String,
    /// The issue's forge web URL. `None` if the forge reported none.
    pub html_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct Release {
    pub id: String,
    pub repo_id: String,
    pub tag: String,
    pub name: Option<String>,
    pub published_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct CiRun {
    pub id: String,
    pub repo_id: String,
    pub commit_sha: Option<String>,
    pub status: String,
    pub conclusion: Option<String>,
    pub completed_at: Option<String>,
    /// The forge web URL of the run. `None` if the forge did not report one.
    pub html_url: Option<String>,
    /// Latest attempt number: 1 on the first try, incremented on each re-run of the same run `id`.
    /// `None` when not reported. Success on an attempt past 1 is the flaky signal.
    pub run_attempt: Option<i64>,
}

/// A declared direct dependency of a repo, keyed by (repo, source manifest, name).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, sqlx::FromRow)]
pub struct DependencyRow {
    pub repo_id: String,
    /// "cargo" | "npm" | "pip" | "go".
    pub ecosystem: String,
    pub name: String,
    pub version_req: Option<String>,
    /// "normal" | "dev" | "build" | "indirect".
    pub kind: String,
    /// The manifest the dependency was declared in (e.g. "Cargo.toml"), so a changed manifest can
    /// replace only its own deps.
    pub source: String,
}

/// One semantic search result: a stored chunk and its cosine similarity to the query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingHit {
    pub ref_kind: String,
    pub ref_id: String,
    pub chunk: String,
    pub score: f32,
}

/// Encode a vector as little-endian f32 bytes, the binary form sqlite-vec accepts for `float[N]`
/// columns and KNN query vectors.
fn vector_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

/// How a search restricts results by `ref_kind`: `Is` keeps one kind, `Not` excludes one, `Any`
/// matches all. Maps to `=` / `!=`, which vec0 metadata filters and the FTS join both support.
#[derive(Clone, Copy)]
pub enum KindFilter<'a> {
    Any,
    Is(&'a str),
    Not(&'a str),
}

impl<'a> KindFilter<'a> {
    /// The `AND <col> = ?` / `AND <col> != ?` fragment (empty for `Any`); bind `value()` when
    /// non-None.
    fn clause(&self, col: &str) -> String {
        match self {
            KindFilter::Any => String::new(),
            KindFilter::Is(_) => format!(" AND {col} = ?"),
            KindFilter::Not(_) => format!(" AND {col} != ?"),
        }
    }

    /// The kind to bind for the clause, or `None` for `Any` (no bind).
    fn value(&self) -> Option<&'a str> {
        match self {
            KindFilter::Any => None,
            KindFilter::Is(k) | KindFilter::Not(k) => Some(k),
        }
    }
}

/// How a search restricts results by repo. `Repos` matches only chunks of the given repo ids, so a
/// board-scoped search cannot see another board's code. An empty `Repos` matches nothing; it does
/// not widen to the whole store.
#[derive(Clone, Copy)]
pub enum RepoScope<'a> {
    All,
    Repos(&'a [String]),
}

impl<'a> RepoScope<'a> {
    /// True when the scope cannot match anything, so the caller can skip the query entirely.
    fn is_empty(&self) -> bool {
        matches!(self, RepoScope::Repos(ids) if ids.is_empty())
    }

    /// The ` AND rowid IN (SELECT id FROM embeddings WHERE ...)` fragment (empty for `All`), using
    /// `embeddings`' unqualified columns. Both search arms key on `embeddings.id`.
    /// Covers the three `ref_kind`s `core-sync` writes. An unlisted kind is excluded from a scoped
    /// search (fails closed but loses recall), so a new indexed `ref_kind` must be added here.
    /// Code chunks are keyed `<repo_id>#<path>`. Matching cuts at the first `#` and compares whole
    /// ids instead of using `LIKE`, because `_` in a repo id is a `LIKE` wildcard (`my_repo` would
    /// match `my-repo`). A repo id cannot contain `#` (`repo:{forge_id}/{full_name}`, forge_id is
    /// alphanumerics and `-`). A `ref_id` with no `#` yields the empty string, which matches no
    /// repo id.
    fn clause(&self, rowid_col: &str) -> String {
        match self {
            RepoScope::All => String::new(),
            RepoScope::Repos(ids) => {
                let holes = std::iter::repeat_n("?", ids.len())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    " AND {rowid_col} IN (SELECT id FROM embeddings WHERE \
                     (ref_kind = 'code' AND substr(ref_id, 1, instr(ref_id, '#') - 1) IN ({holes})) \
                     OR (ref_kind = 'commit' AND ref_id IN (SELECT sha FROM commits WHERE repo_id IN ({holes}))) \
                     OR (ref_kind = 'issue' AND ref_id IN (SELECT id FROM issues WHERE repo_id IN ({holes}))))"
                )
            }
        }
    }

    /// The repo ids to bind for `clause`, in order. The fragment names the set three times, so the
    /// list is bound three times.
    fn binds(&self) -> impl Iterator<Item = &'a String> {
        let ids: &'a [String] = match self {
            RepoScope::All => &[],
            RepoScope::Repos(ids) => ids,
        };
        std::iter::repeat_n(ids, 3).flatten()
    }
}

/// Turn free text into a safe FTS5 MATCH query: lowercase alphanumeric terms OR-ed together,
/// favoring recall. `None` when there are no usable terms.
fn fts_query(text: &str) -> Option<String> {
    let mut terms: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect();
    terms.dedup();
    if terms.is_empty() {
        return None;
    }
    // Quote each term so a stray FTS keyword (e.g. "or", "near") cannot be parsed as syntax.
    Some(
        terms
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

// ---- store -----------------------------------------------------------------

/// Owns the SQLite pool and the typed access methods. `Clone` is cheap; clones share one pool.
#[derive(Clone)]
pub struct Store {
    pool: SqlitePool,
}

/// The `sync_cursors` entity suffix for a repo's history-backfill resume token, e.g.
/// `("repo:gh/acme/widget", "commits:backfill")` beside the forward `"commits"` row.
pub const BACKFILL_SUFFIX: &str = ":backfill";

/// Value stored once the backfill has reached the end of history. No forge mints a resume token
/// equal to it.
pub const BACKFILL_DONE: &str = "done";

/// The `sync_cursors` entity name that holds `entity`'s backfill resume token.
pub fn backfill_entity(entity: &str) -> String {
    format!("{entity}{BACKFILL_SUFFIX}")
}

/// Where one repo entity's history backfill stands, from [`Store::backfill_states`]. `resume` is
/// the opaque forge token to continue from; `complete` means the end of history was reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackfillState {
    pub repo_id: String,
    /// The forward entity name (`"commits"`, `"pulls"`), not the suffixed row name.
    pub entity: String,
    pub complete: bool,
    pub resume: Option<String>,
}

impl Store {
    /// Open a file-backed database, creating it if missing, and run migrations.
    /// If the on-disk schema is incompatible with the migrations (e.g. a shipped migration's
    /// checksum changed), the file is copied to `.bak`, the database is rebuilt, and the user-
    /// authored config (`sources`, `boards` with people/repos, `settings`) is copied forward best-
    /// effort. Cache tables re-sync from the forge.
    pub async fn open(path: &str) -> Result<Self, StoreError> {
        match Self::open_at(path).await {
            Err(StoreError::Migrate(e)) => {
                eprintln!(
                    "store: schema at {path} is incompatible ({e}); preserving it as a .bak and rebuilding"
                );
                let backup = Self::set_aside_incompatible_db(path);
                let store = Self::open_at(path).await?;
                // Copy user-authored config from the old file. A table whose schema changed fails
                // its copy and is skipped; it stays in the .bak. Cache tables are not restored;
                // they re-sync.
                if let Some(backup) = backup {
                    store.restore_config_from(&backup).await;
                }
                Ok(store)
            }
            other => other,
        }
    }

    /// Copy the user-authored config tables from the `.bak` into this fresh database, best-effort.
    /// Parents are restored before children; `settings` (the seeded row) is replaced, the rest are
    /// insert-or-ignore. A table that fails to copy is logged and skipped.
    async fn restore_config_from(&self, backup_path: &str) {
        let escaped = backup_path.replace('\'', "''");
        let mut conn = match self.pool.acquire().await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("store: could not acquire a connection to restore config ({e})");
                return;
            }
        };
        if let Err(e) = sqlx::query(&format!("ATTACH DATABASE '{escaped}' AS bak"))
            .execute(&mut *conn)
            .await
        {
            eprintln!("store: could not open the backup to restore config ({e})");
            return;
        }
        // (table, copy statement). settings is the seeded id=1 row, so REPLACE it; the rest IGNORE
        // duplicates. Order respects the board_people/board_repos -> boards foreign keys.
        let copies = [
            (
                // Column-explicit (not SELECT *): an older .bak may still have the dropped `token`
                // column.
                "forges",
                "INSERT OR IGNORE INTO main.forges (id, name, kind, base_url) \
                 SELECT id, name, kind, base_url FROM bak.forges",
            ),
            (
                "sources",
                "INSERT OR IGNORE INTO main.sources SELECT * FROM bak.sources",
            ),
            (
                // Column-explicit (not SELECT *): an older .bak lacks the digest_webhook_url
                // column (the webhook resets to off on a rebuild).
                "settings",
                "INSERT OR REPLACE INTO main.settings (id, sync_period_secs, stale_pr_days) \
                 SELECT id, sync_period_secs, stale_pr_days FROM bak.settings",
            ),
            (
                // Column-explicit (not SELECT *): an older .bak has no `kind` column; a backup
                // lacking columns is skipped with a logged error.
                "boards",
                "INSERT OR IGNORE INTO main.boards (id, name, kind, created_at, org, forge_id, scan_dependencies) \
                 SELECT id, name, kind, created_at, org, forge_id, scan_dependencies FROM bak.boards",
            ),
            (
                // Separate from the base board copy so a backup without `include_archived` still
                // restores its boards.
                "board_archive_settings",
                "UPDATE main.boards SET include_archived = ( \
                   SELECT include_archived FROM bak.boards WHERE bak.boards.id = main.boards.id \
                 ) WHERE EXISTS (SELECT 1 FROM bak.boards WHERE bak.boards.id = main.boards.id)",
            ),
            (
                "board_people",
                "INSERT OR IGNORE INTO main.board_people SELECT * FROM bak.board_people",
            ),
            (
                "board_repos",
                "INSERT OR IGNORE INTO main.board_repos SELECT * FROM bak.board_repos",
            ),
            (
                "repo_index_prefs",
                "INSERT OR IGNORE INTO main.repo_index_prefs SELECT * FROM bak.repo_index_prefs",
            ),
            (
                "board_disabled_signals",
                "INSERT OR IGNORE INTO main.board_disabled_signals SELECT * FROM bak.board_disabled_signals",
            ),
        ];
        let mut restored = Vec::new();
        for (name, sql) in copies {
            match sqlx::query(sql).execute(&mut *conn).await {
                Ok(r) if r.rows_affected() > 0 => {
                    restored.push(format!("{name}={}", r.rows_affected()))
                }
                Ok(_) => {}
                Err(e) => eprintln!("store: could not restore {name} from backup ({e})"),
            }
        }
        let _ = sqlx::query("DETACH DATABASE bak").execute(&mut *conn).await;
        if restored.is_empty() {
            eprintln!("store: no user config restored from the backup");
        } else {
            eprintln!(
                "store: restored user config from the backup ({})",
                restored.join(", ")
            );
        }
    }

    /// Open and migrate a file-backed database at `path`, creating it if missing. No recovery;
    /// `open` wraps this.
    async fn open_at(path: &str) -> Result<Self, StoreError> {
        register_sqlite_vec();
        // SQLite allows one writer at a time. With a multi-connection pool the fetch lane and the
        // index worker collide, and one can fail with SQLITE_BUSY ("database is locked"). A single
        // pooled connection serializes all DB access in-process. Operations are short, so a read
        // waits only for the current statement. WAL and a busy timeout stay for durability and
        // out-of-process readers.
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(std::time::Duration::from_secs(5));

        // Run migrations first, on a throwaway connection with foreign keys OFF. Some migrations
        // recreate a table that FK children reference (0019 rebuilds `repos`, 0020 rebuilds
        // `forges`). SQLite only allows dropping a referenced parent with foreign keys off, and
        // `defer_foreign_keys` does not cover it: the DROP bumps the deferred-violation counter
        // and the RENAME never clears it, so COMMIT fails. sqlx-sqlite runs each migration in its
        // own transaction and `PRAGMA foreign_keys` is a no-op inside one, so the toggle must
        // happen on the connection before the migrator runs.
        // The app pool below keeps foreign keys ON so ON DELETE cascades stay enforced at runtime.
        let migrate_pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts.clone().foreign_keys(false))
            .await?;
        sqlx::migrate!().run(&migrate_pool).await?;
        migrate_pool.close().await;

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await?;
        // `with_pool` re-runs the migrator as a no-op that confirms the schema is current.
        Self::with_pool(pool).await
    }

    /// Set aside an incompatible database before recreating it: copy the main file to
    /// `{path}.bak`, then remove the original and its `-wal` / `-shm` sidecars. Returns the backup
    /// path on a successful copy, `None` otherwise. Best-effort: a missing file or failed copy
    /// does not abort the rebuild. The copy is a snapshot of the committed file (WAL was
    /// checkpointed when the failed open's connection dropped).
    fn set_aside_incompatible_db(path: &str) -> Option<String> {
        let backup = format!("{path}.bak");
        let copied = match std::fs::copy(path, &backup) {
            Ok(_) => {
                eprintln!("store: preserved the old database as {backup}");
                Some(backup)
            }
            Err(e) => {
                eprintln!("store: could not back up {path} before rebuild ({e})");
                None
            }
        };
        for p in [
            path.to_string(),
            format!("{path}-wal"),
            format!("{path}-shm"),
        ] {
            let _ = std::fs::remove_file(&p);
        }
        copied
    }

    /// Open an in-memory database (for tests) and run migrations. One connection is kept so the
    /// schema persists.
    pub async fn open_in_memory() -> Result<Self, StoreError> {
        register_sqlite_vec();
        let opts = SqliteConnectOptions::from_str("sqlite::memory:")?;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .min_connections(1)
            .connect_with(opts)
            .await?;
        Self::with_pool(pool).await
    }

    async fn with_pool(pool: SqlitePool) -> Result<Self, StoreError> {
        sqlx::migrate!().run(&pool).await?;
        Ok(Self { pool })
    }

    /// Names of the schema's tables (excludes sqlite internals + the migration ledger).
    pub async fn table_names(&self) -> Result<Vec<String>, StoreError> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master \
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name <> '_sqlx_migrations' \
             ORDER BY name",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|(n,)| n).collect())
    }

    /// Every row in every table of the schema, as one number. Lets a test assert that an operation
    /// deleted nothing without listing tables. Table names come from the schema and are quoted.
    pub async fn total_row_count(&self) -> Result<i64, StoreError> {
        let mut total = 0;
        for table in self.table_names().await? {
            let (n,): (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM \"{table}\""))
                .fetch_one(&self.pool)
                .await?;
            total += n;
        }
        Ok(total)
    }

    /// Upsert a repo by primary key.
    pub async fn upsert_repo(&self, repo: &Repo) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO repos (id, owner, name, full_name, ownership) VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
               owner = excluded.owner, name = excluded.name, \
               full_name = excluded.full_name, ownership = excluded.ownership",
        )
        .bind(&repo.id)
        .bind(&repo.owner)
        .bind(&repo.name)
        .bind(&repo.full_name)
        .bind(&repo.ownership)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get_repo(&self, id: &str) -> Result<Option<Repo>, StoreError> {
        let repo = sqlx::query_as::<_, Repo>(
            "SELECT id, owner, name, full_name, ownership FROM repos WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(repo)
    }

    pub async fn count_repos(&self) -> Result<i64, StoreError> {
        let (n,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM repos")
            .fetch_one(&self.pool)
            .await?;
        Ok(n)
    }

    /// All synced repos, ordered by id. Used by cross-repo passes (e.g. dependency resolution).
    pub async fn repos(&self) -> Result<Vec<Repo>, StoreError> {
        let repos = sqlx::query_as::<_, Repo>(
            "SELECT id, owner, name, full_name, ownership FROM repos ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(repos)
    }

    /// Record a repo's fork relationship: whether the forge reports a fork and its upstream
    /// parent's repo id.
    pub async fn set_repo_fork(
        &self,
        repo_id: &str,
        is_fork: bool,
        parent_repo_id: Option<&str>,
    ) -> Result<(), StoreError> {
        sqlx::query("UPDATE repos SET is_fork = ?, parent_repo_id = ? WHERE id = ?")
            .bind(is_fork)
            .bind(parent_repo_id)
            .bind(repo_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Every repo the forge has flagged as a fork, with its upstream parent.
    pub async fn repo_forks(&self) -> Result<Vec<RepoFork>, StoreError> {
        let forks = sqlx::query_as::<_, RepoFork>(
            "SELECT id AS repo_id, is_fork, parent_repo_id FROM repos WHERE is_fork = 1",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(forks)
    }

    /// The repo ids pinned by any board, distinct. User configuration, unlike the discovered set.
    pub async fn pinned_repo_ids(&self) -> Result<Vec<String>, StoreError> {
        let ids = sqlx::query_scalar("SELECT DISTINCT repo_id FROM board_repos ORDER BY repo_id")
            .fetch_all(&self.pool)
            .await?;
        Ok(ids)
    }

    /// The repo ids some board currently discovers, distinct. Derived: discovery rewrites this set
    /// every pass.
    pub async fn discovered_repo_ids(&self) -> Result<Vec<String>, StoreError> {
        let ids = sqlx::query_scalar(
            "SELECT DISTINCT repo_id FROM board_discovered_repos ORDER BY repo_id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(ids)
    }

    /// Remove a repo and everything derived from it, in one transaction.
    /// Refuses while the repo is still configured: pinned to a board, or carrying a `sources` row
    /// that would sync it back. `board_discovered_repos` rows are derived and go with the rest.
    /// Not a block list: if a person on a board still owns the repo, the next discovery pass finds
    /// it again.
    /// Idempotent: an unknown id returns `found: false` and changes nothing.
    pub async fn forget_repo(&self, repo_id: &str) -> Result<ForgottenRepo, StoreError> {
        // Configuration that a sync does not rewrite. Forgetting under it would strand a board or
        // be re-fetched at once.
        let boards: Vec<String> = sqlx::query_scalar(
            "SELECT b.name FROM board_repos br JOIN boards b ON b.id = br.board_id \
             WHERE br.repo_id = ? ORDER BY b.name",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        let (sources,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM sources WHERE id = ?")
            .bind(repo_id)
            .fetch_one(&self.pool)
            .await?;
        if !boards.is_empty() || sources > 0 {
            let mut by = Vec::new();
            if !boards.is_empty() {
                by.push(format!("board {}", boards.join(", ")));
            }
            if sources > 0 {
                by.push("a watch source".to_string());
            }
            return Err(StoreError::StillWatched {
                repo_id: repo_id.to_string(),
                by: by.join(" and "),
            });
        }

        let mut tx = self.pool.begin().await?;
        let forgotten = self.delete_repo_everything(&mut tx, repo_id).await?;
        tx.commit().await?;
        Ok(forgotten)
    }

    /// Delete a repo and everything derived from it, in the caller's transaction. Single list of
    /// owned tables, shared by `forget_repo` and `reclaim_orphan_repos`.
    /// The final `sources` delete is a no-op under `forget_repo` (which already refused if a
    /// source existed). Under `reclaim_orphan_repos` it removes a source with no board behind it,
    /// which `prune_orphan_sources` (run only on a board/pin change) may not have caught.
    async fn delete_repo_everything(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        repo_id: &str,
    ) -> Result<ForgottenRepo, StoreError> {
        // Code chunks are keyed `<repo_id>#<path>`. Match the prefix with `substr`, not `LIKE`:
        // `_` in a repo id is a `LIKE` wildcard, so `my_repo` would also match `my-repo#...`.
        let prefix = format!("{repo_id}#");
        // The repo's index rows across the three chunk kinds. Used verbatim by four statements
        // (vector rows, full-text rows, chunks, hash markers); each binds prefix, prefix, repo_id,
        // repo_id in that order. Both subqueries read `commits` / `issues`, so all four must run
        // before those tables are emptied.
        const INDEX_ROWS: &str = "(ref_kind = 'code' AND substr(ref_id, 1, length(?)) = ?) \
             OR (ref_kind = 'commit' AND ref_id IN (SELECT sha FROM commits WHERE repo_id = ?)) \
             OR (ref_kind = 'issue' AND ref_id IN (SELECT id FROM issues WHERE repo_id = ?))";

        // Counts for the report, read before anything is deleted.
        let (found,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM repos WHERE id = ?")
            .bind(repo_id)
            .fetch_one(&mut **tx)
            .await?;
        let (commits,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM commits WHERE repo_id = ?")
            .bind(repo_id)
            .fetch_one(&mut **tx)
            .await?;
        let (pull_requests,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM pull_requests WHERE repo_id = ?")
                .bind(repo_id)
                .fetch_one(&mut **tx)
                .await?;
        let (issues,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM issues WHERE repo_id = ?")
            .bind(repo_id)
            .fetch_one(&mut **tx)
            .await?;
        let (code_files,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM code_files WHERE repo_id = ?")
                .bind(repo_id)
                .fetch_one(&mut **tx)
                .await?;

        let mut total = 0i64;
        let mut index_chunks = 0i64;

        // 1. The index, first, while `commits` and `issues` still exist for the subqueries.
        // Deleting the vector and full-text rows removes the repo from search results (both arms
        // start from them); deleting `embeddings` removes the chunk text. Those rows have no
        // foreign key or trigger to `embeddings`, so deleting the chunk alone leaves them behind.
        // They cannot surface as hits (the arms join back to `embeddings` by rowid) but they
        // occupy KNN candidate slots. So they go first, by rowid.
        for sql in [
            format!("DELETE FROM vec_embeddings WHERE rowid IN (SELECT id FROM embeddings WHERE {INDEX_ROWS})"),
            format!("DELETE FROM fts_embeddings WHERE rowid IN (SELECT id FROM embeddings WHERE {INDEX_ROWS})"),
            format!("DELETE FROM embeddings WHERE {INDEX_ROWS}"),
            format!("DELETE FROM indexed_text WHERE {INDEX_ROWS}"),
        ] {
            let n = sqlx::query(&sql)
                .bind(&prefix)
                .bind(&prefix)
                .bind(repo_id)
                .bind(repo_id)
                .execute(&mut **tx)
                .await?
                .rows_affected() as i64;
            total += n;
            // The chunk count is the `embeddings` delete: the vector and full-text rows mirror it,
            // and `indexed_text` counts entities rather than chunks.
            if sql.starts_with("DELETE FROM embeddings") {
                index_chunks = n;
            }
        }

        // 2. The cross-reference edges, while their endpoints still exist. Covers the repo itself
        // (`depends_on`), its PRs and commits (`closes` / `mentions`), and its issues under both
        // the `issue` and `work_item` kinds the linker uses.
        let links = sqlx::query(
            "DELETE FROM links WHERE \
               (src_kind = 'repo' AND src_id = ?) OR (dst_kind = 'repo' AND dst_id = ?) \
               OR (src_kind = 'pull_request' AND src_id IN (SELECT id FROM pull_requests WHERE repo_id = ?)) \
               OR (dst_kind = 'pull_request' AND dst_id IN (SELECT id FROM pull_requests WHERE repo_id = ?)) \
               OR (src_kind = 'commit' AND src_id IN (SELECT sha FROM commits WHERE repo_id = ?)) \
               OR (dst_kind = 'commit' AND dst_id IN (SELECT sha FROM commits WHERE repo_id = ?)) \
               OR (src_kind IN ('issue', 'work_item') AND src_id IN (SELECT id FROM issues WHERE repo_id = ?)) \
               OR (dst_kind IN ('issue', 'work_item') AND dst_id IN (SELECT id FROM issues WHERE repo_id = ?))",
        );
        let mut links = links;
        for _ in 0..8 {
            links = links.bind(repo_id);
        }
        total += links.execute(&mut **tx).await?.rows_affected() as i64;

        // 3. Everything else, children before parents: the app pool has foreign keys on, so order
        // matters. Each statement binds the repo id once.
        for sql in [
            "DELETE FROM reviews WHERE pr_id IN (SELECT id FROM pull_requests WHERE repo_id = ?)",
            "DELETE FROM pr_files WHERE pr_id IN (SELECT id FROM pull_requests WHERE repo_id = ?)",
            "DELETE FROM pr_closing_issues WHERE pr_id IN (SELECT id FROM pull_requests WHERE repo_id = ?)",
            // An issue becomes a work item with the issue's own id, so its spine rows and history
            // belong to the repo. `source = 'github'` keeps a Jira item with the same id out.
            "DELETE FROM work_item_status_history WHERE work_item_id IN (SELECT id FROM issues WHERE repo_id = ?)",
            "DELETE FROM sprint_work_items WHERE work_item_id IN (SELECT id FROM issues WHERE repo_id = ?)",
            "DELETE FROM work_items WHERE source = 'github' AND id IN (SELECT id FROM issues WHERE repo_id = ?)",
            "DELETE FROM issues WHERE repo_id = ?",
            "DELETE FROM pull_requests WHERE repo_id = ?",
            "DELETE FROM commits WHERE repo_id = ?",
            "DELETE FROM releases WHERE repo_id = ?",
            "DELETE FROM ci_runs WHERE repo_id = ?",
            "DELETE FROM dependencies WHERE repo_id = ?",
            "DELETE FROM code_files WHERE repo_id = ?",
            // `file_health` has `ON DELETE CASCADE`, so the `repos` delete would take it anyway.
            // Deleted explicitly so the report counts it (a cascade does not report
            // `rows_affected`) and cleanup does not depend on one table's FK clause.
            "DELETE FROM file_health WHERE repo_id = ?",
            "DELETE FROM sync_cursors WHERE repo_id = ?",
            "DELETE FROM repo_index_prefs WHERE repo_id = ?",
            "DELETE FROM ownership WHERE repo_id = ?",
            // Derived, not configuration: discovery rewrites this set every pass.
            "DELETE FROM board_discovered_repos WHERE repo_id = ?",
            "DELETE FROM digests WHERE scope_kind = 'repo' AND scope_id = ?",
            "DELETE FROM metric_snapshots WHERE scope_kind = 'repo' AND scope_id = ?",
            // See the doc comment: a no-op under `forget_repo`, real cleanup under orphan reclaim.
            "DELETE FROM sources WHERE id = ?",
            "DELETE FROM repos WHERE id = ?",
        ] {
            total += sqlx::query(sql)
                .bind(repo_id)
                .execute(&mut **tx)
                .await?
                .rows_affected() as i64;
        }

        Ok(ForgottenRepo {
            repo_id: repo_id.to_string(),
            found: found > 0,
            commits,
            pull_requests,
            issues,
            code_files,
            index_chunks,
            total_rows: total,
        })
    }

    /// Repo ids no board references: not pinned in `board_repos` and not in
    /// `board_discovered_repos`.
    async fn orphan_repo_ids(&self) -> Result<Vec<String>, StoreError> {
        let ids = sqlx::query_scalar(
            "SELECT id FROM repos \
             WHERE id NOT IN (SELECT repo_id FROM board_repos) \
               AND id NOT IN (SELECT repo_id FROM board_discovered_repos) \
             ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(ids)
    }

    /// What no board can currently reach, sized like every other repo. Read-only. Reuses
    /// `repo_storage`'s estimator so this number matches the Debug view's.
    pub async fn orphan_repos(&self) -> Result<OrphanRepoReport, StoreError> {
        let orphans: std::collections::HashSet<String> =
            self.orphan_repo_ids().await?.into_iter().collect();
        let repos: Vec<RepoStorage> = self
            .repo_storage()
            .await?
            .into_iter()
            .filter(|r| orphans.contains(&r.repo_id))
            .collect();
        let estimated_bytes = repos.iter().map(|r| r.estimated_bytes).sum();
        let rows = repos.iter().map(|r| r.rows).sum();
        Ok(OrphanRepoReport {
            count: repos.len() as i64,
            estimated_bytes,
            rows,
            repos,
        })
    }

    /// Remove every orphaned repo and everything derived from each, in a single transaction. Same
    /// removal as `forget_repo`.
    /// Does not refuse: every selected id is already unpinned and undiscovered, which is all
    /// `forget_repo`'s refusal checks. This is the explicit user action; nothing else removes an
    /// orphan.
    pub async fn reclaim_orphan_repos(&self) -> Result<OrphanReclaimReport, StoreError> {
        let repo_ids = self.orphan_repo_ids().await?;
        let mut tx = self.pool.begin().await?;
        let mut total_rows = 0i64;
        for repo_id in &repo_ids {
            total_rows += self
                .delete_repo_everything(&mut tx, repo_id)
                .await?
                .total_rows;
        }
        tx.commit().await?;
        Ok(OrphanReclaimReport {
            count: repo_ids.len() as i64,
            total_rows,
            repo_ids,
        })
    }

    /// Upsert a configured watch source, idempotent by `id`. `filters` is stored as a JSON array.
    pub async fn upsert_source(&self, source: &SourceRow) -> Result<(), StoreError> {
        let filters = serde_json::to_string(&source.filters)?;
        sqlx::query(
            "INSERT INTO sources (id, kind, name, ownership, filters, stale_pr_days, forge_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
               kind = excluded.kind, name = excluded.name, ownership = excluded.ownership, \
               filters = excluded.filters, stale_pr_days = excluded.stale_pr_days, \
               forge_id = excluded.forge_id",
        )
        .bind(&source.id)
        .bind(&source.kind)
        .bind(&source.name)
        .bind(&source.ownership)
        .bind(&filters)
        .bind(i64::from(source.stale_pr_days))
        .bind(&source.forge_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// All configured sources, ordered by id. Seeds the scheduler.
    pub async fn sources(&self) -> Result<Vec<SourceRow>, StoreError> {
        let records = sqlx::query_as::<_, SourceRecord>(
            "SELECT id, kind, name, ownership, filters, stale_pr_days, forge_id FROM sources ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;
        records.into_iter().map(SourceRecord::into_row).collect()
    }

    /// Remove a configured source by id. Removing an absent id is a no-op.
    pub async fn remove_source(&self, id: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM sources WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Drop `sources` rows no board can reach, pinned or discovered. A source whose repo is no
    /// longer pinned may still be watched by an org board that discovers it (discovery never
    /// writes a `sources` row), so it is kept while any board finds the repo. Called after a board
    /// or pin is removed.
    /// Governs the `sources` table only, not synced repo data; that is `orphan_repos` /
    /// `reclaim_orphan_repos`.
    pub async fn prune_orphan_sources(&self) -> Result<(), StoreError> {
        sqlx::query(
            "DELETE FROM sources \
             WHERE id NOT IN (SELECT repo_id FROM board_repos) \
               AND id NOT IN (SELECT repo_id FROM board_discovered_repos)",
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Upsert a configured forge's non-secret config, idempotent by `id`. The access token is in
    /// the OS keychain.
    pub async fn upsert_forge(&self, forge: &ForgeRow) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO forges (id, name, kind, base_url, oauth_client_id) VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
               name = excluded.name, kind = excluded.kind, \
               base_url = excluded.base_url, oauth_client_id = excluded.oauth_client_id",
        )
        .bind(&forge.id)
        .bind(&forge.name)
        .bind(&forge.kind)
        .bind(&forge.base_url)
        .bind(&forge.oauth_client_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// All configured forges, ordered by name then id.
    pub async fn forges(&self) -> Result<Vec<ForgeRow>, StoreError> {
        let forges = sqlx::query_as::<_, ForgeRow>(
            "SELECT id, name, kind, base_url, oauth_client_id FROM forges ORDER BY name, id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(forges)
    }

    /// One forge by id, or `None`.
    pub async fn forge(&self, id: &str) -> Result<Option<ForgeRow>, StoreError> {
        let forge = sqlx::query_as::<_, ForgeRow>(
            "SELECT id, name, kind, base_url, oauth_client_id FROM forges WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(forge)
    }

    /// Remove a forge by id. Removing an absent id is a no-op. A board still pointing at it must
    /// be reassigned first (the boards foreign key blocks the delete otherwise).
    pub async fn remove_forge(&self, id: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM forges WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// The current application settings (the single seeded row).
    pub async fn settings(&self) -> Result<Settings, StoreError> {
        let settings = sqlx::query_as::<_, Settings>(
            "SELECT sync_period_secs, stale_pr_days, digest_webhook_url, digest_schedule_hours, \
             llm_enabled, llm_model, storage_budget_mb \
             FROM settings WHERE id = 1",
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(settings)
    }

    /// Replace the application settings (updates the single row in place).
    pub async fn update_settings(&self, settings: &Settings) -> Result<(), StoreError> {
        sqlx::query(
            "UPDATE settings SET sync_period_secs = ?, stale_pr_days = ?, digest_webhook_url = ?, \
             digest_schedule_hours = ?, llm_enabled = ?, llm_model = ?, storage_budget_mb = ? \
             WHERE id = 1",
        )
        .bind(settings.sync_period_secs)
        .bind(settings.stale_pr_days)
        .bind(&settings.digest_webhook_url)
        .bind(settings.digest_schedule_hours)
        .bind(settings.llm_enabled)
        .bind(&settings.llm_model)
        .bind(settings.storage_budget_mb)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ---- metric snapshots: daily points for trends --------------

    /// Record (or overwrite) a scope's metric snapshot for its `captured_on` day. Replaces the
    /// day's rows (so a metric that became null drops its row), then inserts the present metric
    /// keys.
    pub async fn upsert_metric_snapshot(&self, s: &MetricSnapshot) -> Result<(), StoreError> {
        sqlx::query(
            "DELETE FROM metric_snapshots WHERE scope_kind = ? AND scope_id = ? AND captured_on = ?",
        )
        .bind(&s.scope_kind)
        .bind(&s.scope_id)
        .bind(&s.captured_on)
        .execute(&self.pool)
        .await?;
        for (key, value) in snapshot_metrics(s) {
            self.record_metric(&s.scope_kind, &s.scope_id, &s.captured_on, key, value)
                .await?;
        }
        Ok(())
    }

    /// Record one arbitrary trended metric for a scope on a day; new flow metrics write here with
    /// no schema change. Idempotent per `(scope_kind, scope_id, captured_on, metric_key)`.
    pub async fn record_metric(
        &self,
        scope_kind: &str,
        scope_id: &str,
        captured_on: &str,
        metric_key: &str,
        value: i64,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO metric_snapshots (scope_kind, scope_id, captured_on, metric_key, value) \
             VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(scope_kind, scope_id, captured_on, metric_key) DO UPDATE SET \
               value = excluded.value",
        )
        .bind(scope_kind)
        .bind(scope_id)
        .bind(captured_on)
        .bind(metric_key)
        .bind(value)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// One metric's series for a scope, oldest first: `(captured_on, value)` per recorded day.
    pub async fn metric_series(
        &self,
        scope_kind: &str,
        scope_id: &str,
        metric_key: &str,
    ) -> Result<Vec<(String, i64)>, StoreError> {
        let rows = sqlx::query_as::<_, (String, i64)>(
            "SELECT captured_on, value FROM metric_snapshots \
             WHERE scope_kind = ? AND scope_id = ? AND metric_key = ? ORDER BY captured_on",
        )
        .bind(scope_kind)
        .bind(scope_id)
        .bind(metric_key)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Keep only a scope's most recent `keep` distinct days of snapshots. Count-based so the core
    /// stays clock-free. Returns rows deleted (several per day, since snapshots are long-format).
    pub async fn trim_metric_snapshots(
        &self,
        scope_kind: &str,
        scope_id: &str,
        keep: i64,
    ) -> Result<u64, StoreError> {
        let result = sqlx::query(
            "DELETE FROM metric_snapshots \
             WHERE scope_kind = ? AND scope_id = ? AND captured_on NOT IN ( \
               SELECT DISTINCT captured_on FROM metric_snapshots \
               WHERE scope_kind = ? AND scope_id = ? ORDER BY captured_on DESC LIMIT ?)",
        )
        .bind(scope_kind)
        .bind(scope_id)
        .bind(scope_kind)
        .bind(scope_id)
        .bind(keep)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// A scope's metric snapshots as the typed daily aggregate, oldest first, recomposed from the
    /// long key/value rows.
    pub async fn metric_snapshots(
        &self,
        scope_kind: &str,
        scope_id: &str,
    ) -> Result<Vec<MetricSnapshot>, StoreError> {
        let rows = sqlx::query_as::<_, (String, String, i64)>(
            "SELECT captured_on, metric_key, value FROM metric_snapshots \
             WHERE scope_kind = ? AND scope_id = ? ORDER BY captured_on, metric_key",
        )
        .bind(scope_kind)
        .bind(scope_id)
        .fetch_all(&self.pool)
        .await?;
        let mut out: Vec<MetricSnapshot> = Vec::new();
        for (captured_on, key, value) in rows {
            if out
                .last()
                .map(|s| s.captured_on != captured_on)
                .unwrap_or(true)
            {
                out.push(MetricSnapshot {
                    scope_kind: scope_kind.to_string(),
                    scope_id: scope_id.to_string(),
                    captured_on,
                    wip: 0,
                    stale_open_prs: 0,
                    merged_without_review: 0,
                    attention_count: 0,
                    median_cycle_time_secs: None,
                    median_pickup_secs: None,
                    median_review_secs: None,
                });
            }
            let s = out.last_mut().expect("just pushed");
            match key.as_str() {
                "wip" => s.wip = value,
                "stale_open_prs" => s.stale_open_prs = value,
                "merged_without_review" => s.merged_without_review = value,
                "attention_count" => s.attention_count = value,
                "median_cycle_time_secs" => s.median_cycle_time_secs = Some(value),
                "median_pickup_secs" => s.median_pickup_secs = Some(value),
                "median_review_secs" => s.median_review_secs = Some(value),
                _ => {} // a future metric key not part of the typed daily aggregate
            }
        }
        Ok(out)
    }

    // ---- PR changed files: the diff -----------------------------

    /// Replace a PR's changed-file set (delete-then-insert, so a re-sync is idempotent).
    pub async fn replace_pr_files(&self, pr_id: &str, files: &[PrFile]) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("DELETE FROM pr_files WHERE pr_id = ?")
            .bind(pr_id)
            .execute(&mut *tx)
            .await?;
        for f in files {
            sqlx::query(
                "INSERT OR REPLACE INTO pr_files (pr_id, filename, status, additions, deletions, patch) \
                 VALUES (?, ?, ?, ?, ?, ?)",
            )
            .bind(pr_id)
            .bind(&f.filename)
            .bind(&f.status)
            .bind(f.additions)
            .bind(f.deletions)
            .bind(&f.patch)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// A PR's changed files, ordered by filename.
    pub async fn pr_files(&self, pr_id: &str) -> Result<Vec<PrFile>, StoreError> {
        let files = sqlx::query_as::<_, PrFile>(
            "SELECT pr_id, filename, status, additions, deletions, patch FROM pr_files \
             WHERE pr_id = ? ORDER BY filename",
        )
        .bind(pr_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(files)
    }

    /// Per (file, author) counts of PRs the author touched the file in. Authors with no login are
    /// excluded. Input to ownership and bus-factor analysis.
    pub async fn repo_file_authorship(&self, repo_id: &str) -> Result<Vec<FileAuthor>, StoreError> {
        let rows = sqlx::query_as::<_, FileAuthor>(
            "SELECT f.filename AS path, p.author_login AS author, \
                    COUNT(DISTINCT f.pr_id) AS changes \
             FROM pr_files f JOIN pull_requests p ON f.pr_id = p.id \
             WHERE p.repo_id = ? AND p.author_login IS NOT NULL \
             GROUP BY f.filename, p.author_login",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Per-file change activity for a repo from `pr_files` joined to its PRs: PR count, total
    /// additions/deletions, distinct PR authors. Basis for hotspot and ownership analysis.
    pub async fn repo_file_stats(&self, repo_id: &str) -> Result<Vec<FileStat>, StoreError> {
        let stats = sqlx::query_as::<_, FileStat>(
            "SELECT f.filename AS path, \
                    COUNT(DISTINCT f.pr_id) AS changes, \
                    COALESCE(SUM(f.additions), 0) AS additions, \
                    COALESCE(SUM(f.deletions), 0) AS deletions, \
                    COUNT(DISTINCT p.author_login) AS authors \
             FROM pr_files f JOIN pull_requests p ON f.pr_id = p.id \
             WHERE p.repo_id = ? \
             GROUP BY f.filename",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(stats)
    }

    /// The latest `merged_at` per filename among merged PRs that touched it. Basis for the risky-
    /// change signal: an open PR touching a long-stable file is riskier. Open PRs are excluded;
    /// they are what is being judged.
    pub async fn repo_file_last_changed(
        &self,
        repo_id: &str,
    ) -> Result<Vec<(String, String)>, StoreError> {
        let rows = sqlx::query_as::<_, (String, String)>(
            "SELECT f.filename, MAX(p.merged_at) AS last_changed \
             FROM pr_files f JOIN pull_requests p ON f.pr_id = p.id \
             WHERE p.repo_id = ? AND p.merged_at IS NOT NULL \
             GROUP BY f.filename",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Per-file merged-PR review coverage: how many merged PRs touched each file and how many had
    /// no review. Unreviewed changes in a hot, single-owner file are the worst case.
    pub async fn repo_file_review_gaps(
        &self,
        repo_id: &str,
    ) -> Result<Vec<(String, i64, i64)>, StoreError> {
        // unreviewed = merged PRs touching the file with no row in `reviews`. The LEFT JOIN keeps
        // a PR with zero reviews (its `reviews.pr_id` is NULL) so the CASE can count it once.
        let rows = sqlx::query_as::<_, (String, i64, i64)>(
            "SELECT f.filename, \
                    COUNT(DISTINCT p.id) AS merged_touches, \
                    COUNT(DISTINCT CASE WHEN r.pr_id IS NULL THEN p.id END) AS unreviewed \
             FROM pr_files f \
             JOIN pull_requests p ON f.pr_id = p.id AND p.repo_id = ? AND p.merged_at IS NOT NULL \
             LEFT JOIN reviews r ON r.pr_id = p.id \
             GROUP BY f.filename",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Change-coupling pairs: files that co-appear in the same merged PR, with each file's merged-
    /// PR count. Raw input for `core_codehealth::coupling_pairs`. Capped at 200 rows (by co-
    /// occurrence) so the caller can filter generated paths and low-count pairs.
    pub async fn repo_file_coupling(
        &self,
        repo_id: &str,
    ) -> Result<Vec<(String, String, i64, i64, i64)>, StoreError> {
        let rows = sqlx::query_as::<_, (String, String, i64, i64, i64)>(
            "WITH file_counts AS ( \
                 SELECT f.filename, COUNT(DISTINCT f.pr_id) AS total_prs \
                 FROM pr_files f JOIN pull_requests p ON f.pr_id = p.id \
                 WHERE p.repo_id = ? AND p.merged_at IS NOT NULL \
                 GROUP BY f.filename \
             ) \
             SELECT f1.filename AS path_a, f2.filename AS path_b, \
                    COUNT(DISTINCT f1.pr_id) AS together, \
                    fc1.total_prs AS prs_a, \
                    fc2.total_prs AS prs_b \
             FROM pr_files f1 \
             JOIN pr_files f2 ON f1.pr_id = f2.pr_id AND f1.filename < f2.filename \
             JOIN pull_requests p ON f1.pr_id = p.id AND p.repo_id = ? AND p.merged_at IS NOT NULL \
             JOIN file_counts fc1 ON fc1.filename = f1.filename \
             JOIN file_counts fc2 ON fc2.filename = f2.filename \
             GROUP BY f1.filename, f2.filename \
             ORDER BY together DESC \
             LIMIT 200",
        )
        .bind(repo_id)
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Per-PR churn (additions + deletions over its files), the secondary weight for the
    /// investment distribution. A PR with no synced files is absent (0 churn).
    pub async fn repo_pr_churn(&self, repo_id: &str) -> Result<Vec<(String, i64)>, StoreError> {
        let rows = sqlx::query_as::<_, (String, i64)>(
            "SELECT f.pr_id, COALESCE(SUM(f.additions + f.deletions), 0) AS churn \
             FROM pr_files f JOIN pull_requests p ON f.pr_id = p.id \
             WHERE p.repo_id = ? \
             GROUP BY f.pr_id",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// The PRs that touched each file, newest first: filename, PR number, title, URL, created_at.
    /// The drill-down behind a code-risk row; the caller caps how many to show per file.
    pub async fn repo_file_prs(
        &self,
        repo_id: &str,
    ) -> Result<Vec<(String, i64, String, Option<String>, String)>, StoreError> {
        let rows = sqlx::query_as::<_, (String, i64, String, Option<String>, String)>(
            "SELECT f.filename, p.number, p.title, p.html_url, p.created_at \
             FROM pr_files f JOIN pull_requests p ON f.pr_id = p.id \
             WHERE p.repo_id = ? \
             ORDER BY p.created_at DESC",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    // ---- boards: a manager's unit of interest -------------------

    /// Create a board. `created_at` is supplied by the host (the core has no clock). Idempotent by
    /// id: re-creating updates the name.
    pub async fn create_board(
        &self,
        id: &str,
        name: &str,
        kind: BoardKind,
        created_at: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO boards (id, name, kind, created_at) VALUES (?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET name = excluded.name",
        )
        .bind(id)
        .bind(name)
        .bind(kind.as_str())
        .bind(created_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Rename a board. Renaming an absent board is a no-op.
    pub async fn rename_board(&self, id: &str, name: &str) -> Result<(), StoreError> {
        sqlx::query("UPDATE boards SET name = ? WHERE id = ?")
            .bind(name)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Delete a board and its members (cascade), then drop sources the board's pins leave
    /// orphaned. The cascade clears `board_repos` first, so the prune sees the post-delete state.
    pub async fn delete_board(&self, id: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM board_trackers WHERE board_id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM boards WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        self.prune_orphan_sources().await?;
        Ok(())
    }

    /// All boards, each with its people (logins) and repo ids, ordered by name.
    pub async fn boards(&self) -> Result<Vec<Board>, StoreError> {
        let rows = sqlx::query_as::<_, BoardRecord>(
            "SELECT id, name, kind, created_at, org, forge_id, scan_dependencies, include_archived FROM boards ORDER BY name, id",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut boards = Vec::with_capacity(rows.len());
        for r in rows {
            boards.push(Board {
                people: self.board_people(&r.id).await?,
                repos: self.board_repos(&r.id).await?,
                id: r.id,
                name: r.name,
                kind: BoardKind::from_tag(&r.kind),
                created_at: r.created_at,
                org: r.org,
                forge_id: r.forge_id,
                scan_dependencies: r.scan_dependencies,
                include_archived: r.include_archived,
            });
        }
        Ok(boards)
    }

    /// One board by id, with its people and repos, or `None` if there is no such board.
    pub async fn board(&self, id: &str) -> Result<Option<Board>, StoreError> {
        let row = sqlx::query_as::<_, BoardRecord>(
            "SELECT id, name, kind, created_at, org, forge_id, scan_dependencies, include_archived FROM boards WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        let Some(r) = row else {
            return Ok(None);
        };
        Ok(Some(Board {
            people: self.board_people(&r.id).await?,
            repos: self.board_repos(&r.id).await?,
            id: r.id,
            name: r.name,
            kind: BoardKind::from_tag(&r.kind),
            created_at: r.created_at,
            org: r.org,
            forge_id: r.forge_id,
            scan_dependencies: r.scan_dependencies,
            include_archived: r.include_archived,
        }))
    }

    /// Set or clear a board's org restriction (`None` clears it).
    pub async fn set_board_org(&self, board_id: &str, org: Option<&str>) -> Result<(), StoreError> {
        sqlx::query("UPDATE boards SET org = ? WHERE id = ?")
            .bind(org)
            .bind(board_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Set or clear the forge a board syncs from (`None` clears it). The forge must exist (the
    /// foreign key rejects a dangling id).
    pub async fn set_board_forge(
        &self,
        board_id: &str,
        forge_id: Option<&str>,
    ) -> Result<(), StoreError> {
        sqlx::query("UPDATE boards SET forge_id = ? WHERE id = ?")
            .bind(forge_id)
            .bind(board_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Turn dependency vulnerability scanning on/off for a board. On means a scan may query OSV
    /// (network egress).
    pub async fn set_board_scan_dependencies(
        &self,
        board_id: &str,
        enabled: bool,
    ) -> Result<(), StoreError> {
        sqlx::query("UPDATE boards SET scan_dependencies = ? WHERE id = ?")
            .bind(enabled)
            .bind(board_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Allow or exclude archived repos during discovery. Pinned repos are unaffected.
    pub async fn set_board_include_archived(
        &self,
        board_id: &str,
        include_archived: bool,
    ) -> Result<(), StoreError> {
        sqlx::query("UPDATE boards SET include_archived = ? WHERE id = ?")
            .bind(include_archived)
            .bind(board_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// The repos discovered for a board's people (recorded by the sync engine), sorted.
    pub async fn board_discovered_repos(&self, board_id: &str) -> Result<Vec<String>, StoreError> {
        let repos = sqlx::query_scalar(
            "SELECT repo_id FROM board_discovered_repos WHERE board_id = ? ORDER BY repo_id",
        )
        .bind(board_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(repos)
    }

    /// Replace the discovered-repo set for a board (delete-then-insert, so re-discovery is
    /// idempotent).
    pub async fn replace_board_discovered_repos(
        &self,
        board_id: &str,
        repo_ids: &[String],
    ) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("DELETE FROM board_discovered_repos WHERE board_id = ?")
            .bind(board_id)
            .execute(&mut *tx)
            .await?;
        for repo_id in repo_ids {
            sqlx::query(
                "INSERT OR IGNORE INTO board_discovered_repos (board_id, repo_id) VALUES (?, ?)",
            )
            .bind(board_id)
            .bind(repo_id)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Add one repo to a board's discovered set, idempotent. Lets a sync surface a repo on its
    /// board as soon as it is fetched instead of writing the whole set at the end.
    pub async fn add_board_discovered_repo(
        &self,
        board_id: &str,
        repo_id: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR IGNORE INTO board_discovered_repos (board_id, repo_id) VALUES (?, ?)",
        )
        .bind(board_id)
        .bind(repo_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// The logins on a board, sorted.
    pub async fn board_people(&self, board_id: &str) -> Result<Vec<String>, StoreError> {
        let logins =
            sqlx::query_scalar("SELECT login FROM board_people WHERE board_id = ? ORDER BY login")
                .bind(board_id)
                .fetch_all(&self.pool)
                .await?;
        Ok(logins)
    }

    /// The repo ids pinned to a board, sorted.
    pub async fn board_repos(&self, board_id: &str) -> Result<Vec<String>, StoreError> {
        let repos = sqlx::query_scalar(
            "SELECT repo_id FROM board_repos WHERE board_id = ? ORDER BY repo_id",
        )
        .bind(board_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(repos)
    }

    /// Add a person (login) to a board. Idempotent.
    pub async fn add_board_person(&self, board_id: &str, login: &str) -> Result<(), StoreError> {
        sqlx::query("INSERT OR IGNORE INTO board_people (board_id, login) VALUES (?, ?)")
            .bind(board_id)
            .bind(login)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Remove a person from a board. Removing an absent person is a no-op.
    pub async fn remove_board_person(&self, board_id: &str, login: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM board_people WHERE board_id = ? AND login = ?")
            .bind(board_id)
            .bind(login)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Pin a repo to a board. Idempotent.
    pub async fn add_board_repo(&self, board_id: &str, repo_id: &str) -> Result<(), StoreError> {
        sqlx::query("INSERT OR IGNORE INTO board_repos (board_id, repo_id) VALUES (?, ?)")
            .bind(board_id)
            .bind(repo_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Unpin a repo from a board, then drop its source if no other board pins it. Removing an
    /// absent repo is a no-op.
    pub async fn remove_board_repo(&self, board_id: &str, repo_id: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM board_repos WHERE board_id = ? AND repo_id = ?")
            .bind(board_id)
            .bind(repo_id)
            .execute(&self.pool)
            .await?;
        self.prune_orphan_sources().await?;
        Ok(())
    }

    /// A board's effective repo ids: its pinned repos if any, otherwise the repos discovered for
    /// its people. Empty until discovery runs.
    pub async fn board_effective_repo_ids(
        &self,
        board_id: &str,
    ) -> Result<Vec<String>, StoreError> {
        let pinned = self.board_repos(board_id).await?;
        if !pinned.is_empty() {
            return Ok(pinned);
        }
        self.board_discovered_repos(board_id).await
    }

    /// Upsert a declared dependency, idempotent by (repo, source, name).
    pub async fn upsert_dependency(&self, dep: &DependencyRow) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO dependencies (repo_id, ecosystem, name, version_req, kind, source) \
             VALUES (?, ?, ?, ?, ?, ?) \
             ON CONFLICT(repo_id, source, name) DO UPDATE SET \
               ecosystem = excluded.ecosystem, version_req = excluded.version_req, kind = excluded.kind",
        )
        .bind(&dep.repo_id)
        .bind(&dep.ecosystem)
        .bind(&dep.name)
        .bind(&dep.version_req)
        .bind(&dep.kind)
        .bind(&dep.source)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Replace a repo's dependencies from one manifest (`source`) in a transaction. Prunes
    /// dependencies removed from the manifest, and is keyed per manifest so one does not disturb
    /// another (e.g. pip's two).
    pub async fn replace_dependencies_from_source(
        &self,
        repo_id: &str,
        source: &str,
        deps: &[DependencyRow],
    ) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("DELETE FROM dependencies WHERE repo_id = ? AND source = ?")
            .bind(repo_id)
            .bind(source)
            .execute(&mut *tx)
            .await?;
        for dep in deps {
            sqlx::query(
                "INSERT INTO dependencies (repo_id, ecosystem, name, version_req, kind, source) \
                 VALUES (?, ?, ?, ?, ?, ?) \
                 ON CONFLICT(repo_id, source, name) DO UPDATE SET \
                   ecosystem = excluded.ecosystem, version_req = excluded.version_req, \
                   kind = excluded.kind",
            )
            .bind(&dep.repo_id)
            .bind(&dep.ecosystem)
            .bind(&dep.name)
            .bind(&dep.version_req)
            .bind(&dep.kind)
            .bind(source)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// All declared dependencies of a repo, ordered by ecosystem then name. A dependency declared
    /// in more than one manifest appears once per manifest (distinct `source`).
    pub async fn dependencies_for_repo(
        &self,
        repo_id: &str,
    ) -> Result<Vec<DependencyRow>, StoreError> {
        let deps = sqlx::query_as::<_, DependencyRow>(
            "SELECT repo_id, ecosystem, name, version_req, kind, source FROM dependencies \
             WHERE repo_id = ? ORDER BY ecosystem, name, source",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(deps)
    }

    /// Replace all embeddings for an entity: delete existing chunks for `(ref_kind, ref_id)` from
    /// the metadata and vec0 tables, then insert one row per `(chunk, vector)` into each under the
    /// same id. Each vector must have `core_embed::EMBED_DIM` elements.
    pub async fn replace_embeddings(
        &self,
        ref_kind: &str,
        ref_id: &str,
        chunks: &[(String, Vec<f32>)],
    ) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await?;
        // The vec0 table has no metadata-keyed delete, so find the ids first and delete by rowid.
        let ids: Vec<(i64,)> =
            sqlx::query_as("SELECT id FROM embeddings WHERE ref_kind = ? AND ref_id = ?")
                .bind(ref_kind)
                .bind(ref_id)
                .fetch_all(&mut *tx)
                .await?;
        for (id,) in &ids {
            sqlx::query("DELETE FROM vec_embeddings WHERE rowid = ?")
                .bind(id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM fts_embeddings WHERE rowid = ?")
                .bind(id)
                .execute(&mut *tx)
                .await?;
        }
        sqlx::query("DELETE FROM embeddings WHERE ref_kind = ? AND ref_id = ?")
            .bind(ref_kind)
            .bind(ref_id)
            .execute(&mut *tx)
            .await?;
        for (chunk, vector) in chunks {
            let id: i64 = sqlx::query_scalar(
                "INSERT INTO embeddings (ref_kind, ref_id, chunk) VALUES (?, ?, ?) RETURNING id",
            )
            .bind(ref_kind)
            .bind(ref_id)
            .bind(chunk)
            .fetch_one(&mut *tx)
            .await?;
            sqlx::query("INSERT INTO vec_embeddings (rowid, embedding, ref_kind) VALUES (?, ?, ?)")
                .bind(id)
                .bind(vector_to_bytes(vector))
                .bind(ref_kind)
                .execute(&mut *tx)
                .await?;
            sqlx::query("INSERT INTO fts_embeddings (rowid, chunk) VALUES (?, ?)")
                .bind(id)
                .bind(chunk)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// The `k` chunks nearest to `query` (vector arm only), nearest first, via the sqlite-vec KNN
    /// index. `filter` restricts by `ref_kind` and `scope` by repo, both inside the KNN. `score`
    /// is cosine similarity (`1 - distance`). An empty index returns empty.
    pub async fn vector_search(
        &self,
        query: &[f32],
        k: usize,
        filter: KindFilter<'_>,
        scope: RepoScope<'_>,
    ) -> Result<Vec<EmbeddingHit>, StoreError> {
        Ok(self
            .vector_ranked(query, k, filter, scope)
            .await?
            .into_iter()
            .map(|(_, hit)| hit)
            .collect())
    }

    /// Vector KNN, returning each hit's `embeddings.id` for fusion. `ref_kind` is a vec0 metadata
    /// column, so the kind filter runs inside the KNN; the join adds chunk text and ref.
    /// The repo scope is a `rowid IN (...)` constraint that vec0 also applies inside the search.
    /// This placement matters: the KNN returns the `k` globally nearest rows, so a join `WHERE`
    /// would filter what was already picked and return nothing when the scope is a small slice of
    /// the index. Measured on sqlite-vec 0.1.9 with 45 near out-of-scope chunks, 5 far in-scope
    /// ones and `k = 5`: this form returns the 5 in-scope chunks, a join predicate returns none. A
    /// test pins this.
    async fn vector_ranked(
        &self,
        query: &[f32],
        k: usize,
        filter: KindFilter<'_>,
        scope: RepoScope<'_>,
    ) -> Result<Vec<(i64, EmbeddingHit)>, StoreError> {
        if scope.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!(
            "SELECT e.id, e.ref_kind, e.ref_id, e.chunk, v.distance \
             FROM vec_embeddings v JOIN embeddings e ON e.id = v.rowid \
             WHERE v.embedding MATCH ? AND k = ?{}{} \
             ORDER BY v.distance",
            filter.clause("v.ref_kind"),
            scope.clause("v.rowid")
        );
        let mut q = sqlx::query_as::<_, (i64, String, String, String, f64)>(&sql)
            .bind(vector_to_bytes(query))
            .bind(k as i64);
        if let Some(kind) = filter.value() {
            q = q.bind(kind);
        }
        for repo_id in scope.binds() {
            q = q.bind(repo_id);
        }
        Ok(q.fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(|(id, ref_kind, ref_id, chunk, distance)| {
                (
                    id,
                    EmbeddingHit {
                        ref_kind,
                        ref_id,
                        chunk,
                        score: 1.0 - distance as f32,
                    },
                )
            })
            .collect())
    }

    /// FTS5 keyword search over chunk text, best BM25 match first. `query_text` is reduced to
    /// alphanumeric terms OR-ed together, so `parse_manifest` matches a chunk containing `parse`
    /// or `manifest`. Empty query -> no results.
    async fn keyword_ranked(
        &self,
        query_text: &str,
        k: usize,
        filter: KindFilter<'_>,
        scope: RepoScope<'_>,
    ) -> Result<Vec<(i64, EmbeddingHit)>, StoreError> {
        if scope.is_empty() {
            return Ok(Vec::new());
        }
        let Some(match_query) = fts_query(query_text) else {
            return Ok(Vec::new());
        };
        let sql = format!(
            "SELECT e.id, e.ref_kind, e.ref_id, e.chunk \
             FROM fts_embeddings f JOIN embeddings e ON e.id = f.rowid \
             WHERE fts_embeddings MATCH ?{}{} \
             ORDER BY f.rank LIMIT ?",
            filter.clause("e.ref_kind"),
            scope.clause("f.rowid")
        );
        let mut q = sqlx::query_as::<_, (i64, String, String, String)>(&sql).bind(match_query);
        if let Some(kind) = filter.value() {
            q = q.bind(kind);
        }
        for repo_id in scope.binds() {
            q = q.bind(repo_id);
        }
        q = q.bind(k as i64);
        Ok(q.fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(|(id, ref_kind, ref_id, chunk)| {
                (
                    id,
                    EmbeddingHit {
                        ref_kind,
                        ref_id,
                        chunk,
                        score: 0.0, // replaced by the fused score
                    },
                )
            })
            .collect())
    }

    /// Hybrid search: fuse the vector KNN and FTS5 keyword arms with reciprocal rank fusion, so a
    /// chunk ranking well in either arm surfaces. `score` is the fused RRF score. Returns `k`
    /// results; each arm fetches more candidates to give fusion room.
    /// `filter` selects code vs activity, `scope` the repos the answer may come from. Every board
    /// surface uses this entry point.
    pub async fn hybrid_search(
        &self,
        query_vec: &[f32],
        query_text: &str,
        k: usize,
        filter: KindFilter<'_>,
        scope: RepoScope<'_>,
    ) -> Result<Vec<EmbeddingHit>, StoreError> {
        // Reciprocal rank fusion constant: dampens how much a top rank dominates (the standard
        // 60).
        const RRF_K: f32 = 60.0;
        let candidates = (k * 4).max(20);
        // Both arms carry the scope. Restricting only one would still leak out-of-scope chunks,
        // because fusion keeps whatever either arm found.
        let vector = self
            .vector_ranked(query_vec, candidates, filter, scope)
            .await?;
        let keyword = self
            .keyword_ranked(query_text, candidates, filter, scope)
            .await?;

        let mut fused: std::collections::HashMap<i64, (f32, EmbeddingHit)> =
            std::collections::HashMap::new();
        for arm in [vector, keyword] {
            for (rank, (id, hit)) in arm.into_iter().enumerate() {
                let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
                fused
                    .entry(id)
                    .and_modify(|(score, _)| *score += contribution)
                    .or_insert((contribution, hit));
            }
        }
        let mut hits: Vec<EmbeddingHit> = fused
            .into_values()
            .map(|(score, mut hit)| {
                hit.score = score;
                hit
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(k);
        Ok(hits)
    }

    /// The `k` chunks nearest to `query` across all kinds and repos, vector arm only. Unscoped: a
    /// diagnostic/sync-side entry point. Board surfaces use `hybrid_search` with a `RepoScope`.
    pub async fn nearest_embeddings(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<EmbeddingHit>, StoreError> {
        self.vector_search(query, k, KindFilter::Any, RepoScope::All)
            .await
    }

    /// The `k` chunks nearest to `query` within one `ref_kind`, vector arm only. Keeps code search
    /// and activity search separate.
    pub async fn nearest_embeddings_of_kind(
        &self,
        query: &[f32],
        k: usize,
        ref_kind: &str,
    ) -> Result<Vec<EmbeddingHit>, StoreError> {
        self.vector_search(query, k, KindFilter::Is(ref_kind), RepoScope::All)
            .await
    }

    /// The indexed source files of a repo as a `path -> blob_sha` map. The sync engine diffs it
    /// against the current tree to decide which files to embed or prune.
    pub async fn code_file_shas(
        &self,
        repo_id: &str,
    ) -> Result<std::collections::HashMap<String, String>, StoreError> {
        let rows: Vec<(String, String)> =
            sqlx::query_as("SELECT path, blob_sha FROM code_files WHERE repo_id = ?")
                .bind(repo_id)
                .fetch_all(&self.pool)
                .await?;
        Ok(rows.into_iter().collect())
    }

    /// Record the indexed blob SHA for one source file. Idempotent by `(repo_id, path)`.
    pub async fn upsert_code_file(
        &self,
        repo_id: &str,
        path: &str,
        blob_sha: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO code_files (repo_id, path, blob_sha) VALUES (?, ?, ?) \
             ON CONFLICT(repo_id, path) DO UPDATE SET blob_sha = excluded.blob_sha",
        )
        .bind(repo_id)
        .bind(path)
        .bind(blob_sha)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Drop the tracking row for one source file. The caller deletes its embeddings separately
    /// (`replace_embeddings` with no chunks); this makes the file count as un-indexed if it
    /// reappears.
    pub async fn delete_code_file(&self, repo_id: &str, path: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM code_files WHERE repo_id = ? AND path = ?")
            .bind(repo_id)
            .bind(path)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ---- file health -------------------------------------------

    /// Store or update per-file AST health metrics. Idempotent by `(repo_id, path)`.
    /// `score` is `None` for a file that could not be parsed (no tree-sitter grammar for its
    /// language). The row is still written (its LOC is real) but marked not-analyzed with no
    /// score. `analyzed` is derived here so the two cannot drift.
    pub async fn upsert_file_health(
        &self,
        repo_id: &str,
        path: &str,
        loc: u32,
        functions: u32,
        branches: u32,
        score: Option<u8>,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO file_health (repo_id, path, loc, analyzed, functions, branches, score) \
             VALUES (?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(repo_id, path) DO UPDATE SET \
               loc = excluded.loc, analyzed = excluded.analyzed, \
               functions = excluded.functions, \
               branches = excluded.branches, score = excluded.score",
        )
        .bind(repo_id)
        .bind(path)
        .bind(loc as i64)
        .bind(i64::from(score.is_some()))
        .bind(functions as i64)
        .bind(branches as i64)
        .bind(score.map(i64::from))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Delete the health record for one file. Called when the file is removed from the index.
    pub async fn delete_file_health_path(
        &self,
        repo_id: &str,
        path: &str,
    ) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM file_health WHERE repo_id = ? AND path = ?")
            .bind(repo_id)
            .bind(path)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Delete all health records for a repo. Called when the code index for a repo is cleared.
    pub async fn delete_file_health_repo(&self, repo_id: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM file_health WHERE repo_id = ?")
            .bind(repo_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Per-file health metrics for a repo, weakest score first. Only parsed files are returned; a
    /// file with no grammar has no score.
    pub async fn repo_file_health(
        &self,
        repo_id: &str,
    ) -> Result<Vec<(String, i64, i64, i64, i64)>, StoreError> {
        // (path, loc, functions, branches, score)
        let rows: Vec<(String, i64, i64, i64, i64)> = sqlx::query_as(
            "SELECT path, loc, functions, branches, score FROM file_health \
             WHERE repo_id = ? AND analyzed = 1 ORDER BY score ASC, path",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Upsert a commit by sha (idempotent: re-syncing the same commit leaves one row).
    pub async fn upsert_commit(&self, commit: &Commit) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO commits (sha, repo_id, author_login, message, committed_at) \
             VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(sha) DO UPDATE SET \
               repo_id = excluded.repo_id, author_login = excluded.author_login, \
               message = excluded.message, committed_at = excluded.committed_at",
        )
        .bind(&commit.sha)
        .bind(&commit.repo_id)
        .bind(&commit.author_login)
        .bind(&commit.message)
        .bind(&commit.committed_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn count_commits(&self) -> Result<i64, StoreError> {
        let (n,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM commits")
            .fetch_one(&self.pool)
            .await?;
        Ok(n)
    }

    /// Aggregate row counts across the store for the Debug view. One round trip per table;
    /// `embeddings_code` counts the source-code chunks and `embeddings_activity` the rest.
    pub async fn counts(&self) -> Result<StoreCounts, StoreError> {
        let one = |sql: &'static str| async move {
            let (n,): (i64,) = sqlx::query_as(sql).fetch_one(&self.pool).await?;
            Ok::<i64, StoreError>(n)
        };
        Ok(StoreCounts {
            repos: one("SELECT COUNT(*) FROM repos").await?,
            commits: one("SELECT COUNT(*) FROM commits").await?,
            pull_requests: one("SELECT COUNT(*) FROM pull_requests").await?,
            issues: one("SELECT COUNT(*) FROM issues").await?,
            sources: one("SELECT COUNT(*) FROM sources").await?,
            boards: one("SELECT COUNT(*) FROM boards").await?,
            embeddings_total: one("SELECT COUNT(*) FROM embeddings").await?,
            embeddings_code: one("SELECT COUNT(*) FROM embeddings WHERE ref_kind = 'code'").await?,
            embeddings_activity: one("SELECT COUNT(*) FROM embeddings WHERE ref_kind != 'code'")
                .await?,
            code_files: one("SELECT COUNT(*) FROM code_files").await?,
        })
    }

    pub async fn commits_for_repo(&self, repo_id: &str) -> Result<Vec<Commit>, StoreError> {
        let commits = sqlx::query_as::<_, Commit>(
            "SELECT sha, repo_id, author_login, message, committed_at FROM commits \
             WHERE repo_id = ? ORDER BY committed_at DESC, sha",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(commits)
    }

    /// Upsert a pull request by id (idempotent re-sync).
    pub async fn upsert_pull_request(&self, pr: &PullRequest) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO pull_requests (id, repo_id, number, title, state, author_login, body, created_at, merged_at, html_url) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
               repo_id = excluded.repo_id, number = excluded.number, title = excluded.title, \
               state = excluded.state, author_login = excluded.author_login, body = excluded.body, \
               created_at = excluded.created_at, merged_at = excluded.merged_at, html_url = excluded.html_url",
        )
        .bind(&pr.id)
        .bind(&pr.repo_id)
        .bind(pr.number)
        .bind(&pr.title)
        .bind(&pr.state)
        .bind(&pr.author_login)
        .bind(&pr.body)
        .bind(&pr.created_at)
        .bind(&pr.merged_at)
        .bind(&pr.html_url)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// All pull requests for a repo, newest first.
    pub async fn pull_requests(&self, repo_id: &str) -> Result<Vec<PullRequest>, StoreError> {
        let prs = sqlx::query_as::<_, PullRequest>(
            "SELECT id, repo_id, number, title, state, author_login, body, created_at, merged_at, html_url \
             FROM pull_requests WHERE repo_id = ? ORDER BY created_at DESC, number",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(prs)
    }

    /// Merged PRs in a repo with no stored changed-files yet, newest first, capped at `limit`. The
    /// index lane back-fills diffs in bounded batches; a merged PR's files never change, so they
    /// are fetched once, and the cap keeps a big repo's first sync from spending the whole rate
    /// budget on history.
    pub async fn merged_pull_requests_missing_files(
        &self,
        repo_id: &str,
        limit: i64,
    ) -> Result<Vec<PullRequest>, StoreError> {
        let prs = sqlx::query_as::<_, PullRequest>(
            "SELECT id, repo_id, number, title, state, author_login, body, created_at, merged_at, html_url \
             FROM pull_requests p WHERE p.repo_id = ? AND p.merged_at IS NOT NULL \
               AND p.files_synced = 0 \
               AND NOT EXISTS (SELECT 1 FROM pr_files f WHERE f.pr_id = p.id) \
             ORDER BY p.created_at DESC, p.number LIMIT ?",
        )
        .bind(repo_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(prs)
    }

    /// Mark a PR's changed files as fetched, even when zero files came back, so the PR is not re-
    /// listed by `merged_pull_requests_missing_files` every sync.
    pub async fn mark_pr_files_synced(&self, pr_id: &str) -> Result<(), StoreError> {
        sqlx::query("UPDATE pull_requests SET files_synced = 1 WHERE id = ?")
            .bind(pr_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// The attention signals a board has turned OFF, e.g. `["merged_without_review"]`. Empty means
    /// all are enabled. Filters the board's attention rollup.
    pub async fn board_disabled_signals(&self, board_id: &str) -> Result<Vec<String>, StoreError> {
        let rows: Vec<(String,)> =
            sqlx::query_as("SELECT signal FROM board_disabled_signals WHERE board_id = ?")
                .bind(board_id)
                .fetch_all(&self.pool)
                .await?;
        Ok(rows.into_iter().map(|(s,)| s).collect())
    }

    /// Enable or disable one attention signal for a board. Disabling records a row; enabling
    /// removes it (no row = enabled). Idempotent.
    pub async fn set_board_signal_enabled(
        &self,
        board_id: &str,
        signal: &str,
        enabled: bool,
    ) -> Result<(), StoreError> {
        if enabled {
            sqlx::query("DELETE FROM board_disabled_signals WHERE board_id = ? AND signal = ?")
                .bind(board_id)
                .bind(signal)
                .execute(&self.pool)
                .await?;
        } else {
            sqlx::query(
                "INSERT OR IGNORE INTO board_disabled_signals (board_id, signal) VALUES (?, ?)",
            )
            .bind(board_id)
            .bind(signal)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    /// Whether a repo's source code should be indexed. Defaults to `true` when there is no
    /// preference row; only an explicit opt-out (or re-enable) is recorded.
    pub async fn repo_index_code(&self, repo_id: &str) -> Result<bool, StoreError> {
        let v: Option<i64> =
            sqlx::query_scalar("SELECT index_code FROM repo_index_prefs WHERE repo_id = ?")
                .bind(repo_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(v.map(|n| n != 0).unwrap_or(true))
    }

    /// Set a repo's "index code" preference. Idempotent by `repo_id`.
    pub async fn set_repo_index_code(
        &self,
        repo_id: &str,
        index_code: bool,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO repo_index_prefs (repo_id, index_code) VALUES (?, ?) \
             ON CONFLICT(repo_id) DO UPDATE SET index_code = excluded.index_code",
        )
        .bind(repo_id)
        .bind(i64::from(index_code))
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn upsert_review(&self, review: &Review) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO reviews (id, pr_id, reviewer_login, state, submitted_at) \
             VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
               pr_id = excluded.pr_id, reviewer_login = excluded.reviewer_login, \
               state = excluded.state, submitted_at = excluded.submitted_at",
        )
        .bind(&review.id)
        .bind(&review.pr_id)
        .bind(&review.reviewer_login)
        .bind(&review.state)
        .bind(&review.submitted_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn reviews_for_pr(&self, pr_id: &str) -> Result<Vec<Review>, StoreError> {
        let reviews = sqlx::query_as::<_, Review>(
            "SELECT id, pr_id, reviewer_login, state, submitted_at FROM reviews \
             WHERE pr_id = ? ORDER BY submitted_at, id",
        )
        .bind(pr_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(reviews)
    }

    /// All reviews across a repo's pull requests (for repo-scoped metrics like review
    /// concentration).
    pub async fn reviews_for_repo(&self, repo_id: &str) -> Result<Vec<Review>, StoreError> {
        let reviews = sqlx::query_as::<_, Review>(
            "SELECT r.id, r.pr_id, r.reviewer_login, r.state, r.submitted_at \
             FROM reviews r JOIN pull_requests p ON r.pr_id = p.id \
             WHERE p.repo_id = ? ORDER BY r.submitted_at, r.id",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(reviews)
    }

    /// Upsert an issue by id (idempotent re-sync).
    pub async fn upsert_issue(&self, issue: &Issue) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO issues (id, repo_id, number, title, state, author_login, body, created_at, closed_at, labels, html_url) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
               repo_id = excluded.repo_id, number = excluded.number, title = excluded.title, \
               state = excluded.state, author_login = excluded.author_login, body = excluded.body, \
               created_at = excluded.created_at, closed_at = excluded.closed_at, labels = excluded.labels, \
               html_url = excluded.html_url",
        )
        .bind(&issue.id)
        .bind(&issue.repo_id)
        .bind(issue.number)
        .bind(&issue.title)
        .bind(&issue.state)
        .bind(&issue.author_login)
        .bind(&issue.body)
        .bind(&issue.created_at)
        .bind(&issue.closed_at)
        .bind(&issue.labels)
        .bind(&issue.html_url)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn issues_for_repo(&self, repo_id: &str) -> Result<Vec<Issue>, StoreError> {
        let issues = sqlx::query_as::<_, Issue>(
            "SELECT id, repo_id, number, title, state, author_login, body, created_at, closed_at, labels, html_url \
             FROM issues WHERE repo_id = ? ORDER BY created_at DESC, number",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(issues)
    }

    /// Upsert a release by id (idempotent re-sync).
    pub async fn upsert_release(&self, release: &Release) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO releases (id, repo_id, tag, name, published_at) VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
               repo_id = excluded.repo_id, tag = excluded.tag, name = excluded.name, \
               published_at = excluded.published_at",
        )
        .bind(&release.id)
        .bind(&release.repo_id)
        .bind(&release.tag)
        .bind(&release.name)
        .bind(&release.published_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn releases_for_repo(&self, repo_id: &str) -> Result<Vec<Release>, StoreError> {
        let releases = sqlx::query_as::<_, Release>(
            "SELECT id, repo_id, tag, name, published_at FROM releases \
             WHERE repo_id = ? ORDER BY published_at DESC, id",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(releases)
    }

    /// Upsert a CI run by id (idempotent re-sync).
    pub async fn upsert_ci_run(&self, run: &CiRun) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO ci_runs (id, repo_id, commit_sha, status, conclusion, completed_at, html_url, run_attempt) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
               repo_id = excluded.repo_id, commit_sha = excluded.commit_sha, \
               status = excluded.status, conclusion = excluded.conclusion, \
               completed_at = excluded.completed_at, html_url = excluded.html_url, \
               run_attempt = excluded.run_attempt",
        )
        .bind(&run.id)
        .bind(&run.repo_id)
        .bind(&run.commit_sha)
        .bind(&run.status)
        .bind(&run.conclusion)
        .bind(&run.completed_at)
        .bind(&run.html_url)
        .bind(run.run_attempt)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn ci_runs_for_repo(&self, repo_id: &str) -> Result<Vec<CiRun>, StoreError> {
        let runs = sqlx::query_as::<_, CiRun>(
            "SELECT id, repo_id, commit_sha, status, conclusion, completed_at, html_url, run_attempt FROM ci_runs \
             WHERE repo_id = ? ORDER BY completed_at DESC, id",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(runs)
    }

    /// Save (upsert) a precomputed digest body for a scope.
    pub async fn save_digest(
        &self,
        scope_kind: &str,
        scope_id: &str,
        generated_at: &str,
        body: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO digests (scope_kind, scope_id, generated_at, body) VALUES (?, ?, ?, ?) \
             ON CONFLICT(scope_kind, scope_id) DO UPDATE SET \
               generated_at = excluded.generated_at, body = excluded.body",
        )
        .bind(scope_kind)
        .bind(scope_id)
        .bind(generated_at)
        .bind(body)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Read back a stored digest body for a scope.
    pub async fn get_digest(
        &self,
        scope_kind: &str,
        scope_id: &str,
    ) -> Result<Option<String>, StoreError> {
        let body: Option<String> =
            sqlx::query_scalar("SELECT body FROM digests WHERE scope_kind = ? AND scope_id = ?")
                .bind(scope_kind)
                .bind(scope_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(body)
    }

    pub async fn add_team(&self, team: &Team) -> Result<(), StoreError> {
        sqlx::query("INSERT INTO teams (id, name, source) VALUES (?, ?, ?)")
            .bind(&team.id)
            .bind(&team.name)
            .bind(&team.source)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn list_teams(&self) -> Result<Vec<Team>, StoreError> {
        let teams = sqlx::query_as::<_, Team>("SELECT id, name, source FROM teams ORDER BY id")
            .fetch_all(&self.pool)
            .await?;
        Ok(teams)
    }

    pub async fn add_sprint(&self, sprint: &Sprint) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO sprints (id, team_id, name, starts_on, ends_on, source) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&sprint.id)
        .bind(&sprint.team_id)
        .bind(&sprint.name)
        .bind(&sprint.starts_on)
        .bind(&sprint.ends_on)
        .bind(&sprint.source)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn add_work_item(&self, item: &WorkItem) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO work_items (id, title, kind, state, source, status_category) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&item.id)
        .bind(&item.title)
        .bind(&item.kind)
        .bind(&item.state)
        .bind(&item.source)
        .bind(&item.status_category)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn work_item(&self, id: &str) -> Result<Option<WorkItem>, StoreError> {
        let item = sqlx::query_as::<_, WorkItem>(
            "SELECT id, title, kind, state, source, status_category FROM work_items WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(item)
    }

    /// All work items, ordered by id.
    pub async fn work_items(&self) -> Result<Vec<WorkItem>, StoreError> {
        let items = sqlx::query_as::<_, WorkItem>(
            "SELECT id, title, kind, state, source, status_category FROM work_items ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(items)
    }

    /// Project a GitHub issue into the work_items spine: `source = 'github'`, id reused as the
    /// issue id (so a `pull_request -> work_item` link points at it), with the derived
    /// `status_category`. Idempotent.
    pub async fn upsert_github_work_item(
        &self,
        id: &str,
        title: &str,
        kind: &str,
        state: &str,
        status_category: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO work_items (id, title, kind, state, source, status_category) \
             VALUES (?, ?, ?, ?, 'github', ?) \
             ON CONFLICT(id) DO UPDATE SET title = excluded.title, kind = excluded.kind, \
               state = excluded.state, status_category = excluded.status_category",
        )
        .bind(id)
        .bind(title)
        .bind(kind)
        .bind(state)
        .bind(status_category)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Register a person (identity). Required before adding them to a team (foreign key).
    pub async fn add_identity(&self, id: &str, canonical_login: &str) -> Result<(), StoreError> {
        sqlx::query("INSERT OR IGNORE INTO identities (id, canonical_login) VALUES (?, ?)")
            .bind(id)
            .bind(canonical_login)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Add a person to a team (idempotent membership).
    pub async fn add_team_member(
        &self,
        team_id: &str,
        identity_id: &str,
    ) -> Result<(), StoreError> {
        sqlx::query("INSERT OR IGNORE INTO team_members (team_id, identity_id) VALUES (?, ?)")
            .bind(team_id)
            .bind(identity_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// The identity ids of a team's members.
    pub async fn team_members(&self, team_id: &str) -> Result<Vec<String>, StoreError> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT identity_id FROM team_members WHERE team_id = ? ORDER BY identity_id",
        )
        .bind(team_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|(id,)| id).collect())
    }

    /// Add a work item to a sprint (idempotent membership).
    pub async fn add_sprint_work_item(
        &self,
        sprint_id: &str,
        work_item_id: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR IGNORE INTO sprint_work_items (sprint_id, work_item_id) VALUES (?, ?)",
        )
        .bind(sprint_id)
        .bind(work_item_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// The work items that belong to a sprint.
    pub async fn work_items_for_sprint(
        &self,
        sprint_id: &str,
    ) -> Result<Vec<WorkItem>, StoreError> {
        let items = sqlx::query_as::<_, WorkItem>(
            "SELECT w.id, w.title, w.kind, w.state, w.source, w.status_category \
             FROM work_items w JOIN sprint_work_items s ON w.id = s.work_item_id \
             WHERE s.sprint_id = ? ORDER BY w.id",
        )
        .bind(sprint_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(items)
    }

    /// Replace a sprint's live membership wholesale, so an item pulled from the sprint stops being
    /// a member (`add_sprint_work_item` would let it linger). The commitment snapshot is separate
    /// and frozen.
    pub async fn replace_sprint_work_items(
        &self,
        sprint_id: &str,
        work_item_ids: &[String],
    ) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("DELETE FROM sprint_work_items WHERE sprint_id = ?")
            .bind(sprint_id)
            .execute(&mut *tx)
            .await?;
        for id in work_item_ids {
            sqlx::query(
                "INSERT OR IGNORE INTO sprint_work_items (sprint_id, work_item_id) VALUES (?, ?)",
            )
            .bind(sprint_id)
            .bind(id)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Whether a sprint's start-of-sprint commitment snapshot has been frozen
    /// (`jira_sprints.committed_at` is set).
    pub async fn sprint_commitment_taken(&self, sprint_id: &str) -> Result<bool, StoreError> {
        let at: Option<Option<String>> =
            sqlx::query_scalar("SELECT committed_at FROM jira_sprints WHERE sprint_id = ?")
                .bind(sprint_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(matches!(at, Some(Some(_))))
    }

    /// Freeze a sprint's commitment snapshot: record the work items as the committed set and stamp
    /// `committed_at`. The stamp is left alone if already set.
    pub async fn record_sprint_commitment(
        &self,
        sprint_id: &str,
        work_item_ids: &[String],
        captured_on: &str,
    ) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await?;
        for id in work_item_ids {
            sqlx::query(
                "INSERT OR IGNORE INTO sprint_commitments (sprint_id, work_item_id, captured_on) \
                 VALUES (?, ?, ?)",
            )
            .bind(sprint_id)
            .bind(id)
            .bind(captured_on)
            .execute(&mut *tx)
            .await?;
        }
        sqlx::query(
            "UPDATE jira_sprints SET committed_at = ? WHERE sprint_id = ? AND committed_at IS NULL",
        )
        .bind(captured_on)
        .bind(sprint_id)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    /// A sprint's committed work items (the frozen start-of-sprint set), each with its current
    /// status category coalesced from the Jira detail (the `work_items` row carries none for a
    /// Jira issue). Used for say-do (delivered = committed items now `done`).
    pub async fn sprint_committed_items(
        &self,
        sprint_id: &str,
    ) -> Result<Vec<WorkItem>, StoreError> {
        let items = sqlx::query_as::<_, WorkItem>(
            "SELECT w.id, w.title, w.kind, w.state, w.source, \
                    COALESCE(w.status_category, j.status_category) AS status_category \
             FROM sprint_commitments c JOIN work_items w ON w.id = c.work_item_id \
             LEFT JOIN jira_issues j ON j.work_item_id = w.id \
             WHERE c.sprint_id = ? ORDER BY w.id",
        )
        .bind(sprint_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(items)
    }

    pub async fn pull_request(&self, id: &str) -> Result<Option<PullRequest>, StoreError> {
        let pr = sqlx::query_as::<_, PullRequest>(
            "SELECT id, repo_id, number, title, state, author_login, body, created_at, merged_at, html_url \
             FROM pull_requests WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(pr)
    }

    /// Replace the authoritative closing-issue references for a PR (the issue numbers it closes
    /// per the forge's link data), wholesale so a removed reference does not linger. Empty
    /// `numbers` clears them.
    pub async fn replace_pr_closing_issues(
        &self,
        pr_id: &str,
        numbers: &[i64],
    ) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM pr_closing_issues WHERE pr_id = ?")
            .bind(pr_id)
            .execute(&self.pool)
            .await?;
        for n in numbers {
            sqlx::query(
                "INSERT OR IGNORE INTO pr_closing_issues (pr_id, issue_number) VALUES (?, ?)",
            )
            .bind(pr_id)
            .bind(n)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    /// The issue numbers a PR authoritatively closes, empty if none/unknown.
    pub async fn pr_closing_issues(&self, pr_id: &str) -> Result<Vec<i64>, StoreError> {
        let rows: Vec<(i64,)> =
            sqlx::query_as("SELECT issue_number FROM pr_closing_issues WHERE pr_id = ?")
                .bind(pr_id)
                .fetch_all(&self.pool)
                .await?;
        Ok(rows.into_iter().map(|(n,)| n).collect())
    }

    /// Every `(pr_id, issue_number)` authoritative closing link for the PRs in a repo, in one
    /// query. The per-repo batch behind per-person issues resolved by merged PRs.
    pub async fn repo_pr_closing_issues(
        &self,
        repo_id: &str,
    ) -> Result<Vec<(String, i64)>, StoreError> {
        let rows = sqlx::query_as::<_, (String, i64)>(
            "SELECT c.pr_id, c.issue_number \
             FROM pr_closing_issues c JOIN pull_requests p ON c.pr_id = p.id \
             WHERE p.repo_id = ?",
        )
        .bind(repo_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Replace a work item's status history: the `(status_category, entered_at)` pairs of when it
    /// entered each lifecycle bucket. Wholesale per work item, re-derived each sync.
    pub async fn replace_work_item_status_history(
        &self,
        work_item_id: &str,
        history: &[(&str, String)],
    ) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM work_item_status_history WHERE work_item_id = ?")
            .bind(work_item_id)
            .execute(&self.pool)
            .await?;
        for (status_category, entered_at) in history {
            sqlx::query(
                "INSERT OR IGNORE INTO work_item_status_history \
                   (work_item_id, status_category, entered_at) VALUES (?, ?, ?)",
            )
            .bind(work_item_id)
            .bind(status_category)
            .bind(entered_at)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    /// A work item's status history in entry order: `(status_category, entered_at)` per bucket.
    pub async fn work_item_status_history(
        &self,
        work_item_id: &str,
    ) -> Result<Vec<(String, String)>, StoreError> {
        let rows = sqlx::query_as::<_, (String, String)>(
            "SELECT status_category, entered_at FROM work_item_status_history \
             WHERE work_item_id = ? ORDER BY entered_at, status_category",
        )
        .bind(work_item_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Record a directed link. Idempotent: re-recording the same edge is a no-op.
    pub async fn add_link(
        &self,
        src_kind: &str,
        src_id: &str,
        dst_kind: &str,
        dst_id: &str,
        relation: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR IGNORE INTO links (src_kind, src_id, dst_kind, dst_id, relation) \
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(src_kind)
        .bind(src_id)
        .bind(dst_kind)
        .bind(dst_id)
        .bind(relation)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// All links originating at `(src_kind, src_id)`.
    pub async fn links_from(&self, src_kind: &str, src_id: &str) -> Result<Vec<Link>, StoreError> {
        let links = sqlx::query_as::<_, Link>(
            "SELECT id, src_kind, src_id, dst_kind, dst_id, relation FROM links \
             WHERE src_kind = ? AND src_id = ? ORDER BY id",
        )
        .bind(src_kind)
        .bind(src_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(links)
    }

    /// Whether this repo has at least one `pull_request -> work_item` link, of any relation. The
    /// linker (`core_sync::link_references`) runs as one pass over the whole repo, so once it has
    /// run every PR was judged. A repo it never touched (an org board skips the index lane) has
    /// zero rows, which must not be read as "checked, none linked".
    pub async fn repo_has_work_item_links(&self, repo_id: &str) -> Result<bool, StoreError> {
        let row: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM links l \
             JOIN pull_requests p ON p.id = l.src_id AND l.src_kind = 'pull_request' \
             WHERE l.dst_kind = 'work_item' AND p.repo_id = ? LIMIT 1",
        )
        .bind(repo_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.is_some())
    }

    /// All links pointing at `(dst_kind, dst_id)`, the reverse of [`links_from`]. Finds the work
    /// items that link to a PR.
    pub async fn links_to(&self, dst_kind: &str, dst_id: &str) -> Result<Vec<Link>, StoreError> {
        let links = sqlx::query_as::<_, Link>(
            "SELECT id, src_kind, src_id, dst_kind, dst_id, relation FROM links \
             WHERE dst_kind = ? AND dst_id = ? ORDER BY id",
        )
        .bind(dst_kind)
        .bind(dst_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(links)
    }

    // ---- trackers (Jira) -------------------------------

    /// Register or update a tracker connection (Jira site). Idempotent by id.
    pub async fn upsert_tracker(&self, t: &Tracker) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO trackers (id, name, kind, base_url, email) VALUES (?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET name = excluded.name, kind = excluded.kind, \
               base_url = excluded.base_url, email = excluded.email",
        )
        .bind(&t.id)
        .bind(&t.name)
        .bind(&t.kind)
        .bind(&t.base_url)
        .bind(&t.email)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn trackers(&self) -> Result<Vec<Tracker>, StoreError> {
        let rows = sqlx::query_as::<_, Tracker>(
            "SELECT id, name, kind, base_url, email FROM trackers ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    pub async fn tracker(&self, id: &str) -> Result<Option<Tracker>, StoreError> {
        let row = sqlx::query_as::<_, Tracker>(
            "SELECT id, name, kind, base_url, email FROM trackers WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    /// Delete a tracker + its board links (the keychain token is removed by the host).
    pub async fn delete_tracker(&self, id: &str) -> Result<(), StoreError> {
        sqlx::query("DELETE FROM board_trackers WHERE tracker_id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM trackers WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Link a Jira project to a board (idempotent).
    pub async fn link_board_tracker(
        &self,
        board_id: &str,
        tracker_id: &str,
        project_key: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT OR IGNORE INTO board_trackers (board_id, tracker_id, project_key) \
             VALUES (?, ?, ?)",
        )
        .bind(board_id)
        .bind(tracker_id)
        .bind(project_key)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn unlink_board_tracker(
        &self,
        board_id: &str,
        tracker_id: &str,
        project_key: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "DELETE FROM board_trackers WHERE board_id = ? AND tracker_id = ? AND project_key = ?",
        )
        .bind(board_id)
        .bind(tracker_id)
        .bind(project_key)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn board_trackers(&self, board_id: &str) -> Result<Vec<BoardTracker>, StoreError> {
        let rows = sqlx::query_as::<_, BoardTracker>(
            "SELECT board_id, tracker_id, project_key FROM board_trackers WHERE board_id = ? \
             ORDER BY tracker_id, project_key",
        )
        .bind(board_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Every board-tracker link across all boards (the sync planner uses this to know what to
    /// fetch).
    pub async fn all_board_trackers(&self) -> Result<Vec<BoardTracker>, StoreError> {
        let rows = sqlx::query_as::<_, BoardTracker>(
            "SELECT board_id, tracker_id, project_key FROM board_trackers ORDER BY tracker_id, project_key",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Upsert a Jira issue: its base work_item row (source = "jira") + the jira_issues detail row.
    pub async fn upsert_jira_issue(&self, j: &JiraIssueRow) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO work_items (id, title, kind, state, source) VALUES (?, ?, ?, ?, 'jira') \
             ON CONFLICT(id) DO UPDATE SET title = excluded.title, kind = excluded.kind, \
               state = excluded.state",
        )
        .bind(&j.work_item_id)
        .bind(&j.title)
        .bind(&j.issue_type)
        .bind(&j.status)
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "INSERT INTO jira_issues (work_item_id, tracker_id, project, issue_key, assignee, \
               status_category, url, created_at, updated_at, resolved_at, parent_key) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(work_item_id) DO UPDATE SET tracker_id = excluded.tracker_id, \
               project = excluded.project, issue_key = excluded.issue_key, assignee = excluded.assignee, \
               status_category = excluded.status_category, url = excluded.url, \
               created_at = excluded.created_at, updated_at = excluded.updated_at, \
               resolved_at = excluded.resolved_at, parent_key = excluded.parent_key",
        )
        .bind(&j.work_item_id)
        .bind(&j.tracker_id)
        .bind(&j.project)
        .bind(&j.issue_key)
        .bind(&j.assignee)
        .bind(&j.status_category)
        .bind(&j.url)
        .bind(&j.created_at)
        .bind(&j.updated_at)
        .bind(&j.resolved_at)
        .bind(&j.parent_key)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// The select shared by the Jira-issue queries (work_item base joined to its jira_issues
    /// detail).
    const JIRA_ISSUE_SELECT: &'static str = "SELECT j.work_item_id, j.tracker_id, j.project, \
        j.issue_key, w.title, w.kind AS issue_type, w.state AS status, j.status_category, j.assignee, \
        j.url, j.created_at, j.updated_at, j.resolved_at, j.parent_key \
        FROM jira_issues j JOIN work_items w ON w.id = j.work_item_id";

    /// A Jira issue by its key within a tracker (for linking a `PROJ-123` reference).
    pub async fn jira_issue_by_key(
        &self,
        tracker_id: &str,
        key: &str,
    ) -> Result<Option<JiraIssueRow>, StoreError> {
        let sql = format!(
            "{} WHERE j.tracker_id = ? AND j.issue_key = ?",
            Self::JIRA_ISSUE_SELECT
        );
        let row = sqlx::query_as::<_, JiraIssueRow>(&sql)
            .bind(tracker_id)
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row)
    }

    /// All Jira issues a board watches (across its linked projects), newest-updated first.
    pub async fn jira_issues_for_board(
        &self,
        board_id: &str,
    ) -> Result<Vec<JiraIssueRow>, StoreError> {
        let sql = format!(
            "{} JOIN board_trackers bt ON bt.tracker_id = j.tracker_id AND bt.project_key = j.project \
             WHERE bt.board_id = ? ORDER BY j.updated_at DESC",
            Self::JIRA_ISSUE_SELECT
        );
        let rows = sqlx::query_as::<_, JiraIssueRow>(&sql)
            .bind(board_id)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows)
    }

    /// The Jira issues linked to a PR (via the `links` table: work_item -> pull_request).
    pub async fn jira_issues_for_pr(&self, pr_id: &str) -> Result<Vec<JiraIssueRow>, StoreError> {
        let sql = format!(
            "{} JOIN links l ON l.src_kind = 'work_item' AND l.src_id = j.work_item_id \
             WHERE l.dst_kind = 'pull_request' AND l.dst_id = ? ORDER BY j.issue_key",
            Self::JIRA_ISSUE_SELECT
        );
        let rows = sqlx::query_as::<_, JiraIssueRow>(&sql)
            .bind(pr_id)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows)
    }

    /// Upsert a Jira sprint: its base sprint row (source = "jira") + the jira_sprints detail.
    pub async fn upsert_jira_sprint(&self, s: &JiraSprintRow) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO sprints (id, team_id, name, starts_on, ends_on, source) \
             VALUES (?, NULL, ?, ?, ?, 'jira') \
             ON CONFLICT(id) DO UPDATE SET name = excluded.name, starts_on = excluded.starts_on, \
               ends_on = excluded.ends_on",
        )
        .bind(&s.sprint_id)
        .bind(&s.name)
        .bind(&s.starts_on)
        .bind(&s.ends_on)
        .execute(&self.pool)
        .await?;
        sqlx::query(
            "INSERT INTO jira_sprints (sprint_id, tracker_id, project, state) VALUES (?, ?, ?, ?) \
             ON CONFLICT(sprint_id) DO UPDATE SET tracker_id = excluded.tracker_id, \
               project = excluded.project, state = excluded.state",
        )
        .bind(&s.sprint_id)
        .bind(&s.tracker_id)
        .bind(&s.project)
        .bind(&s.state)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// The Jira sprints a board watches.
    pub async fn jira_sprints_for_board(
        &self,
        board_id: &str,
    ) -> Result<Vec<JiraSprintRow>, StoreError> {
        let rows = sqlx::query_as::<_, JiraSprintRow>(
            "SELECT js.sprint_id, js.tracker_id, js.project, s.name, js.state, s.starts_on, s.ends_on, \
                    js.committed_at \
             FROM jira_sprints js JOIN sprints s ON s.id = js.sprint_id \
             JOIN board_trackers bt ON bt.tracker_id = js.tracker_id AND bt.project_key = js.project \
             WHERE bt.board_id = ? ORDER BY s.starts_on DESC",
        )
        .bind(board_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// The content hash last embedded for an activity entity, or `None` if never embedded.
    /// The index lane compares it to the current text's hash and re-embeds only on a difference.
    pub async fn indexed_text_hash(
        &self,
        ref_kind: &str,
        ref_id: &str,
    ) -> Result<Option<String>, StoreError> {
        let hash: Option<String> =
            sqlx::query_scalar("SELECT hash FROM indexed_text WHERE ref_kind = ? AND ref_id = ?")
                .bind(ref_kind)
                .bind(ref_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(hash)
    }

    /// Record the content hash just embedded for an activity entity. Idempotent by
    /// `(ref_kind, ref_id)`.
    pub async fn set_indexed_text_hash(
        &self,
        ref_kind: &str,
        ref_id: &str,
        hash: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO indexed_text (ref_kind, ref_id, hash) VALUES (?, ?, ?) \
             ON CONFLICT(ref_kind, ref_id) DO UPDATE SET hash = excluded.hash",
        )
        .bind(ref_kind)
        .bind(ref_id)
        .bind(hash)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Read an app-level scalar from the `meta` kv table, or `None` if unset.
    pub async fn meta_get(&self, key: &str) -> Result<Option<String>, StoreError> {
        let value: Option<String> = sqlx::query_scalar("SELECT value FROM meta WHERE key = ?")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        Ok(value)
    }

    /// Write an app-level scalar to the `meta` kv table. Idempotent by `key`.
    pub async fn meta_set(&self, key: &str, value: &str) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO meta (key, value) VALUES (?, ?) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        )
        .bind(key)
        .bind(value)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Drop the entire embedding index: the vectors (`embeddings` and the `vec0`/FTS sidecars) and
    /// the incremental markers (`indexed_text` hashes, `code_files` blob SHAs, the `code` fetch
    /// cursors), so the next sync re-embeds everything. Used when the bundled model or chunking
    /// scheme changes. Activity re-embeds from the stored commit/issue rows; code blobs are not
    /// stored, so the `code` cursors are cleared too, otherwise `fetch_changed_code` would see the
    /// cursor at `pushed_at` and skip the refetch. Source activity/code rows are untouched.
    pub async fn clear_index(&self) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await?;
        for stmt in [
            "DELETE FROM vec_embeddings",
            "DELETE FROM fts_embeddings",
            "DELETE FROM embeddings",
            "DELETE FROM indexed_text",
            "DELETE FROM code_files",
            "DELETE FROM file_health",
            // Both index-state cursors: the per-file code watermark and the activity-index
            // checkpoint. Dropping them forces a full re-fetch, re-embed and re-link.
            "DELETE FROM sync_cursors WHERE entity IN ('code', 'activity_index')",
        ] {
            sqlx::query(stmt).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Drop a single repo's code index: the `code` chunk embeddings (and vec0 + FTS sidecars),
    /// per-file blob-SHA markers, and the `code` fetch cursor. Used when "index code" is turned
    /// off for a repo. Clearing the cursor makes a later re-enable re-fetch the tree. Activity
    /// embeddings are untouched.
    pub async fn clear_repo_code_index(&self, repo_id: &str) -> Result<(), StoreError> {
        // Anchored with `substr`, not `LIKE`: `_` in a repo id is a `LIKE` wildcard, so `my_repo`
        // would also match `my-repo#...` and take its code index.
        let prefix = format!("{repo_id}#");
        let mut tx = self.pool.begin().await?;
        // The vec0 + FTS sidecars are keyed by the embeddings rowid, so find this repo's code rows
        // first.
        let ids: Vec<(i64,)> = sqlx::query_as(
            "SELECT id FROM embeddings \
             WHERE ref_kind = 'code' AND substr(ref_id, 1, length(?)) = ?",
        )
        .bind(&prefix)
        .bind(&prefix)
        .fetch_all(&mut *tx)
        .await?;
        for (id,) in &ids {
            sqlx::query("DELETE FROM vec_embeddings WHERE rowid = ?")
                .bind(id)
                .execute(&mut *tx)
                .await?;
            sqlx::query("DELETE FROM fts_embeddings WHERE rowid = ?")
                .bind(id)
                .execute(&mut *tx)
                .await?;
        }
        sqlx::query(
            "DELETE FROM embeddings \
             WHERE ref_kind = 'code' AND substr(ref_id, 1, length(?)) = ?",
        )
        .bind(&prefix)
        .bind(&prefix)
        .execute(&mut *tx)
        .await?;
        sqlx::query("DELETE FROM code_files WHERE repo_id = ?")
            .bind(repo_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM file_health WHERE repo_id = ?")
            .bind(repo_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM sync_cursors WHERE repo_id = ? AND entity = 'code'")
            .bind(repo_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// The persisted incremental-fetch cursor for a repo's entity type, e.g. `(repo, "issues")` ->
    /// the max `updated_at` seen. `None` before the first sync of that entity.
    pub async fn sync_cursor(
        &self,
        repo_id: &str,
        entity: &str,
    ) -> Result<Option<String>, StoreError> {
        let cursor: Option<String> =
            sqlx::query_scalar("SELECT cursor FROM sync_cursors WHERE repo_id = ? AND entity = ?")
                .bind(repo_id)
                .bind(entity)
                .fetch_optional(&self.pool)
                .await?;
        Ok(cursor)
    }

    /// Advance (or set) the incremental-fetch cursor for a repo's entity type. Idempotent by
    /// `(repo_id, entity)`.
    pub async fn set_sync_cursor(
        &self,
        repo_id: &str,
        entity: &str,
        cursor: &str,
    ) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO sync_cursors (repo_id, entity, cursor) VALUES (?, ?, ?) \
             ON CONFLICT(repo_id, entity) DO UPDATE SET cursor = excluded.cursor",
        )
        .bind(repo_id)
        .bind(entity)
        .bind(cursor)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Every repo entity with a history backfill on the books, in progress or complete, newest-
    /// named first. Answers whether a repo is still backfilling or sync is broken.
    pub async fn backfill_states(&self) -> Result<Vec<BackfillState>, StoreError> {
        let rows: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT repo_id, entity, cursor FROM sync_cursors WHERE entity LIKE ? \
             ORDER BY repo_id, entity",
        )
        .bind(format!("%{BACKFILL_SUFFIX}"))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|(repo_id, entity, cursor)| BackfillState {
                repo_id,
                entity: entity
                    .strip_suffix(BACKFILL_SUFFIX)
                    .unwrap_or(&entity)
                    .to_string(),
                complete: cursor == BACKFILL_DONE,
                resume: (cursor != BACKFILL_DONE).then_some(cursor),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_aside_incompatible_db_backs_up_then_removes() {
        // An incompatible cache is preserved as a .bak before the rebuild.
        let path =
            std::env::temp_dir().join(format!("orgonzola_setaside_{}.db", std::process::id()));
        let path_s = path.to_str().unwrap();
        let wal = format!("{path_s}-wal");
        std::fs::write(&path, b"OLDCONFIG").unwrap();
        std::fs::write(&wal, b"wal").unwrap();

        let backup = Store::set_aside_incompatible_db(path_s);
        assert_eq!(
            backup.as_deref(),
            Some(format!("{path_s}.bak").as_str()),
            "the backup path is returned so config can be restored from it"
        );

        assert!(
            !path.exists(),
            "the original db is removed so a fresh one is created"
        );
        assert!(
            !std::path::Path::new(&wal).exists(),
            "the -wal sidecar is removed too"
        );
        let bak = format!("{path_s}.bak");
        assert_eq!(
            std::fs::read(&bak).unwrap(),
            b"OLDCONFIG",
            "the old db is preserved as a .bak for recovery"
        );
        std::fs::remove_file(&bak).ok();
    }

    fn repo(full_name: &str) -> Repo {
        Repo {
            id: "repo:acme/widget".into(),
            owner: "acme".into(),
            name: "widget".into(),
            full_name: full_name.into(),
            ownership: "owned".into(),
        }
    }

    // ---- forgetting a repo -------------------------------------------------------------

    /// Fill every table a repo's sync writes to, so a forget test can prove each is cleaned. `n`
    /// keys the entity ids apart so two seeded repos never collide.
    async fn seed_repo(store: &Store, repo_id: &str, full_name: &str, n: &str) {
        use core_embed::{Embedder, HashEmbedder};
        let (owner, name) = full_name.split_once('/').unwrap();
        store
            .upsert_repo(&Repo {
                id: repo_id.into(),
                owner: owner.into(),
                name: name.into(),
                full_name: full_name.into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        store
            .upsert_commit(&Commit {
                sha: format!("sha{n}"),
                repo_id: repo_id.into(),
                author_login: Some("dev".into()),
                message: format!("commit in {full_name}"),
                committed_at: "2026-08-01T00:00:00Z".into(),
            })
            .await
            .unwrap();
        store
            .upsert_pull_request(&PullRequest {
                id: format!("pr{n}"),
                repo_id: repo_id.into(),
                number: 1,
                title: "a change".into(),
                state: "open".into(),
                author_login: Some("dev".into()),
                body: None,
                created_at: "2026-08-01T00:00:00Z".into(),
                merged_at: None,
                html_url: None,
            })
            .await
            .unwrap();
        store
            .upsert_review(&Review {
                id: format!("rv{n}"),
                pr_id: format!("pr{n}"),
                reviewer_login: Some("other".into()),
                state: "APPROVED".into(),
                submitted_at: Some("2026-08-02T00:00:00Z".into()),
            })
            .await
            .unwrap();
        store
            .replace_pr_files(
                &format!("pr{n}"),
                &[PrFile {
                    pr_id: format!("pr{n}"),
                    filename: "src/lib.rs".into(),
                    status: "modified".into(),
                    additions: 3,
                    deletions: 1,
                    patch: None,
                }],
            )
            .await
            .unwrap();
        store
            .replace_pr_closing_issues(&format!("pr{n}"), &[1])
            .await
            .unwrap();
        store
            .upsert_issue(&Issue {
                id: format!("is{n}"),
                repo_id: repo_id.into(),
                number: 1,
                title: format!("an issue in {full_name}"),
                state: "open".into(),
                author_login: Some("dev".into()),
                body: None,
                created_at: "2026-08-01T00:00:00Z".into(),
                closed_at: None,
                labels: String::new(),
                html_url: None,
            })
            .await
            .unwrap();
        // An issue becomes a work item under its own id, with status history and sprint membership.
        store
            .upsert_github_work_item(&format!("is{n}"), "an issue", "issue", "open", "new")
            .await
            .unwrap();
        store
            .replace_work_item_status_history(
                &format!("is{n}"),
                &[("new", "2026-08-01T00:00:00Z".to_string())],
            )
            .await
            .unwrap();
        store
            .add_sprint(&Sprint {
                id: format!("sp{n}"),
                team_id: None,
                name: "S1".into(),
                starts_on: None,
                ends_on: None,
                source: "manual".into(),
            })
            .await
            .unwrap();
        store
            .add_sprint_work_item(&format!("sp{n}"), &format!("is{n}"))
            .await
            .unwrap();
        store
            .upsert_release(&Release {
                id: format!("rl{n}"),
                repo_id: repo_id.into(),
                tag: "v1".into(),
                name: None,
                published_at: None,
            })
            .await
            .unwrap();
        store
            .upsert_ci_run(&CiRun {
                id: format!("ci{n}"),
                repo_id: repo_id.into(),
                commit_sha: Some(format!("sha{n}")),
                status: "completed".into(),
                conclusion: Some("success".into()),
                completed_at: Some("2026-08-02T00:00:00Z".into()),
                html_url: None,
                run_attempt: Some(1),
            })
            .await
            .unwrap();
        store
            .upsert_dependency(&DependencyRow {
                repo_id: repo_id.into(),
                ecosystem: "cargo".into(),
                name: "serde".into(),
                version_req: Some("1".into()),
                kind: "normal".into(),
                source: "Cargo.toml".into(),
            })
            .await
            .unwrap();
        store
            .upsert_code_file(repo_id, "src/lib.rs", "blob1")
            .await
            .unwrap();
        store
            .upsert_file_health(repo_id, "src/lib.rs", 100, 4, 6, Some(7))
            .await
            .unwrap();
        store
            .set_sync_cursor(repo_id, "commits", "2026-08-01")
            .await
            .unwrap();
        store.set_repo_index_code(repo_id, false).await.unwrap();
        store
            .record_metric("repo", repo_id, "2026-08-01", "wip", 3)
            .await
            .unwrap();
        store
            .save_digest("repo", repo_id, "2026-08-02T00:00:00Z", "{}")
            .await
            .unwrap();
        // The cross-reference edges a sync writes, on both kinds a repo contributes.
        store
            .add_link(
                "pull_request",
                &format!("pr{n}"),
                "work_item",
                &format!("is{n}"),
                "closes",
            )
            .await
            .unwrap();
        store
            .add_link("repo", repo_id, "repo", "repo:other/dep", "depends_on")
            .await
            .unwrap();
        // Rows with no public writer: a team's ownership of the repo, and a board that discovered
        // it.
        sqlx::query("INSERT OR IGNORE INTO teams (id, name, source) VALUES (?, 'T', 'manual')")
            .bind(format!("team{n}"))
            .execute(&store.pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO ownership (repo_id, team_id) VALUES (?, ?)")
            .bind(repo_id)
            .bind(format!("team{n}"))
            .execute(&store.pool)
            .await
            .unwrap();
        store
            .create_board(
                &format!("board{n}"),
                "B",
                BoardKind::Team,
                "2026-08-01T00:00:00Z",
            )
            .await
            .unwrap();
        store
            .replace_board_discovered_repos(&format!("board{n}"), &[repo_id.to_string()])
            .await
            .unwrap();
        // The index: a code chunk, a commit chunk, and an issue chunk, each with its hash marker.
        let e = HashEmbedder::default();
        let chunk = |t: &str| vec![(t.to_string(), e.embed(t))];
        store
            .replace_embeddings(
                "code",
                &format!("{repo_id}#src/lib.rs"),
                &chunk(&format!("fn render_{n}() {{}}")),
            )
            .await
            .unwrap();
        store
            .set_indexed_text_hash("code", &format!("{repo_id}#src/lib.rs"), "h1")
            .await
            .unwrap();
        store
            .replace_embeddings(
                "commit",
                &format!("sha{n}"),
                &chunk(&format!("commit in {full_name}")),
            )
            .await
            .unwrap();
        store
            .set_indexed_text_hash("commit", &format!("sha{n}"), "h2")
            .await
            .unwrap();
        store
            .replace_embeddings(
                "issue",
                &format!("is{n}"),
                &chunk(&format!("issue in {full_name}")),
            )
            .await
            .unwrap();
        store
            .set_indexed_text_hash("issue", &format!("is{n}"), "h3")
            .await
            .unwrap();
    }

    /// What is still in the store for a seeded repo, table by table. Entity-keyed tables (reviews,
    /// PR files, work items, links, the index) are counted by seeded id since they have no repo
    /// column.
    async fn leftovers(store: &Store, repo_id: &str, n: &str) -> Vec<(String, i64)> {
        let mut out: Vec<(String, i64)> = Vec::new();
        let one = |sql: &'static str, bind: String| async move {
            let (c,): (i64,) = sqlx::query_as(sql)
                .bind(bind)
                .fetch_one(&store.pool)
                .await
                .unwrap();
            c
        };
        for (label, sql) in [
            ("repos", "SELECT COUNT(*) FROM repos WHERE id = ?"),
            ("commits", "SELECT COUNT(*) FROM commits WHERE repo_id = ?"),
            (
                "pull_requests",
                "SELECT COUNT(*) FROM pull_requests WHERE repo_id = ?",
            ),
            ("issues", "SELECT COUNT(*) FROM issues WHERE repo_id = ?"),
            (
                "releases",
                "SELECT COUNT(*) FROM releases WHERE repo_id = ?",
            ),
            ("ci_runs", "SELECT COUNT(*) FROM ci_runs WHERE repo_id = ?"),
            (
                "dependencies",
                "SELECT COUNT(*) FROM dependencies WHERE repo_id = ?",
            ),
            (
                "code_files",
                "SELECT COUNT(*) FROM code_files WHERE repo_id = ?",
            ),
            (
                "file_health",
                "SELECT COUNT(*) FROM file_health WHERE repo_id = ?",
            ),
            (
                "sync_cursors",
                "SELECT COUNT(*) FROM sync_cursors WHERE repo_id = ?",
            ),
            (
                "repo_index_prefs",
                "SELECT COUNT(*) FROM repo_index_prefs WHERE repo_id = ?",
            ),
            (
                "ownership",
                "SELECT COUNT(*) FROM ownership WHERE repo_id = ?",
            ),
            (
                "board_discovered_repos",
                "SELECT COUNT(*) FROM board_discovered_repos WHERE repo_id = ?",
            ),
            (
                "digests",
                "SELECT COUNT(*) FROM digests WHERE scope_kind = 'repo' AND scope_id = ?",
            ),
            (
                "metric_snapshots",
                "SELECT COUNT(*) FROM metric_snapshots WHERE scope_kind = 'repo' AND scope_id = ?",
            ),
        ] {
            out.push((label.to_string(), one(sql, repo_id.to_string()).await));
        }
        for (label, sql, id) in [
            (
                "reviews",
                "SELECT COUNT(*) FROM reviews WHERE pr_id = ?",
                format!("pr{n}"),
            ),
            (
                "pr_files",
                "SELECT COUNT(*) FROM pr_files WHERE pr_id = ?",
                format!("pr{n}"),
            ),
            (
                "pr_closing_issues",
                "SELECT COUNT(*) FROM pr_closing_issues WHERE pr_id = ?",
                format!("pr{n}"),
            ),
            (
                "work_items",
                "SELECT COUNT(*) FROM work_items WHERE id = ?",
                format!("is{n}"),
            ),
            (
                "work_item_status_history",
                "SELECT COUNT(*) FROM work_item_status_history WHERE work_item_id = ?",
                format!("is{n}"),
            ),
            (
                "sprint_work_items",
                "SELECT COUNT(*) FROM sprint_work_items WHERE work_item_id = ?",
                format!("is{n}"),
            ),
        ] {
            out.push((label.to_string(), one(sql, id).await));
        }
        let ids = [
            repo_id.to_string(),
            format!("pr{n}"),
            format!("is{n}"),
            format!("sha{n}"),
        ];
        let mut q = sqlx::query_as::<_, (i64,)>(
            "SELECT COUNT(*) FROM links WHERE src_id IN (?, ?, ?, ?) OR dst_id IN (?, ?, ?, ?)",
        );
        for id in ids.iter().chain(ids.iter()) {
            q = q.bind(id.clone());
        }
        out.push((
            "links".to_string(),
            q.fetch_one(&store.pool).await.unwrap().0,
        ));

        let refs = [
            format!("{repo_id}#src/lib.rs"),
            format!("sha{n}"),
            format!("is{n}"),
        ];
        for (label, table) in [
            ("embeddings", "embeddings"),
            ("indexed_text", "indexed_text"),
        ] {
            let sql = format!("SELECT COUNT(*) FROM {table} WHERE ref_id IN (?, ?, ?)");
            let mut q = sqlx::query_as::<_, (i64,)>(&sql);
            for r in &refs {
                q = q.bind(r.clone());
            }
            out.push((label.to_string(), q.fetch_one(&store.pool).await.unwrap().0));
        }
        out
    }

    /// Rows in the two index tables with no foreign key back to `embeddings`: the sqlite-vec
    /// vector table and the FTS5 table. Counted directly.
    async fn vector_and_fts_rows(store: &Store) -> (i64, i64) {
        let (v,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM vec_embeddings")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        let (f,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM fts_embeddings")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        (v, f)
    }

    #[tokio::test]
    async fn forget_repo_removes_every_row_it_claims_to() {
        // Two seeded repos; forgetting one empties every table for it and touches none of the
        // other's.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/gone", "acme/gone", "A").await;
        seed_repo(&store, "repo:gh/acme/kept", "acme/kept", "B").await;

        let report = store.forget_repo("repo:gh/acme/gone").await.unwrap();
        assert!(report.found);
        assert_eq!(
            (report.commits, report.pull_requests, report.issues),
            (1, 1, 1)
        );
        assert_eq!(report.code_files, 1);
        assert_eq!(report.index_chunks, 3); // code + commit + issue
        assert!(report.total_rows >= 25, "total was {}", report.total_rows);

        let left: Vec<(String, i64)> = leftovers(&store, "repo:gh/acme/gone", "A")
            .await
            .into_iter()
            .filter(|(_, c)| *c > 0)
            .collect();
        assert!(left.is_empty(), "rows left behind: {left:?}");

        let kept: Vec<(String, i64)> = leftovers(&store, "repo:gh/acme/kept", "B")
            .await
            .into_iter()
            .filter(|(_, c)| *c == 0)
            .collect();
        assert!(kept.is_empty(), "the neighbour lost rows: {kept:?}");
    }

    #[tokio::test]
    async fn forget_repo_clears_the_vector_and_full_text_rows() {
        // `vec_embeddings` (vec0) and `fts_embeddings` (FTS5) have no foreign key to `embeddings`
        // and no trigger, so deleting the chunk alone leaves both. Counted directly because a
        // search would not show them (both arms join back to `embeddings`).
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/gone", "acme/gone", "A").await;
        seed_repo(&store, "repo:gh/acme/kept", "acme/kept", "B").await;
        assert_eq!(vector_and_fts_rows(&store).await, (6, 6)); // 3 chunks each

        store.forget_repo("repo:gh/acme/gone").await.unwrap();
        assert_eq!(
            vector_and_fts_rows(&store).await,
            (3, 3),
            "the forgotten repo's vector / full-text rows outlived its chunks"
        );
    }

    #[tokio::test]
    async fn forget_repo_takes_the_repo_out_of_search() {
        // Search is global (filtered by kind, not board), so a forgotten repo must stop appearing
        // in Search and in assistant grounding.
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/gone", "acme/gone", "A").await;
        seed_repo(&store, "repo:gh/acme/kept", "acme/kept", "B").await;
        let e = HashEmbedder::default();

        store.forget_repo("repo:gh/acme/gone").await.unwrap();

        for query in [
            "fn render_A() {}",
            "commit in acme/gone",
            "issue in acme/gone",
        ] {
            let vec_hits = store
                .vector_search(&e.embed(query), 10, KindFilter::Any, RepoScope::All)
                .await
                .unwrap();
            assert!(
                vec_hits
                    .iter()
                    .all(|h| !h.ref_id.contains("gone") && h.ref_id != "shaA" && h.ref_id != "isA"),
                "vector search still returns the forgotten repo for {query:?}: {vec_hits:?}"
            );
            let hybrid = store
                .hybrid_search(&e.embed(query), query, 10, KindFilter::Any, RepoScope::All)
                .await
                .unwrap();
            assert!(
                hybrid
                    .iter()
                    .all(|h| !h.ref_id.contains("gone") && h.ref_id != "shaA" && h.ref_id != "isA"),
                "hybrid search still returns the forgotten repo for {query:?}: {hybrid:?}"
            );
        }
        // The other repo is still searchable.
        let kept = store
            .hybrid_search(
                &e.embed("commit in acme/kept"),
                "commit in acme/kept",
                10,
                KindFilter::Any,
                RepoScope::All,
            )
            .await
            .unwrap();
        assert!(
            kept.iter().any(|h| h.ref_id == "shaB"),
            "the neighbour left the index too: {kept:?}"
        );
    }

    #[tokio::test]
    async fn forget_repo_refuses_while_a_board_pins_it() {
        // A pin is configuration: forgetting under it would leave a board pointing at a missing
        // repo, so the call refuses and names the board.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/gone", "acme/gone", "A").await;
        store
            .create_board(
                "board:pin",
                "Platform",
                BoardKind::Team,
                "2026-08-01T00:00:00Z",
            )
            .await
            .unwrap();
        store
            .add_board_repo("board:pin", "repo:gh/acme/gone")
            .await
            .unwrap();

        let err = store.forget_repo("repo:gh/acme/gone").await.unwrap_err();
        assert!(
            matches!(&err, StoreError::StillWatched { by, .. } if by.contains("Platform")),
            "expected a refusal naming the board, got {err}"
        );
        let left: Vec<(String, i64)> = leftovers(&store, "repo:gh/acme/gone", "A")
            .await
            .into_iter()
            .filter(|(_, c)| *c == 0)
            .collect();
        assert!(
            left.is_empty(),
            "a refused forget still deleted rows: {left:?}"
        );
    }

    #[tokio::test]
    async fn forget_repo_refuses_while_a_source_schedules_it() {
        // A source row means the scheduler still syncs the repo, so forgetting would be undone.
        // Refuse.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/gone", "acme/gone", "A").await;
        store
            .upsert_source(&SourceRow {
                id: "repo:gh/acme/gone".into(),
                kind: "repo".into(),
                name: "acme/gone".into(),
                ownership: "owned".into(),
                filters: Vec::new(),
                stale_pr_days: 7,
                forge_id: None,
            })
            .await
            .unwrap();

        let err = store.forget_repo("repo:gh/acme/gone").await.unwrap_err();
        assert!(
            matches!(&err, StoreError::StillWatched { by, .. } if by.contains("watch source")),
            "expected a refusal naming the source, got {err}"
        );
        assert_eq!(store.count_repos().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn forget_repo_is_idempotent_and_safe_on_an_unknown_id() {
        // Re-running on an unknown id is not an error and reports no removals.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/gone", "acme/gone", "A").await;

        let first = store.forget_repo("repo:gh/acme/gone").await.unwrap();
        assert!(first.found && first.total_rows > 0);
        let second = store.forget_repo("repo:gh/acme/gone").await.unwrap();
        assert!(!second.found);
        assert_eq!(second.total_rows, 0);

        let never = store
            .forget_repo("repo:gh/acme/never-synced")
            .await
            .unwrap();
        assert!(!never.found);
        assert_eq!(never.total_rows, 0);
    }

    #[tokio::test]
    async fn forget_repo_does_not_catch_a_lookalike_repo() {
        // Code chunks are keyed `<repo_id>#<path>` and a repo id can contain `_`, a `LIKE`
        // wildcard. A `LIKE` prefix match would take `my-repo`'s index down with `my_repo`'s; the
        // match is a `substr` equality.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/my_repo", "acme/my_repo", "A").await;
        seed_repo(&store, "repo:gh/acme/my-repo", "acme/my-repo", "B").await;

        store.forget_repo("repo:gh/acme/my_repo").await.unwrap();

        let (chunks,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM embeddings WHERE ref_id = ?")
            .bind("repo:gh/acme/my-repo#src/lib.rs")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        assert_eq!(chunks, 1, "the lookalike repo's code chunk was deleted too");
        let (marks,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM indexed_text WHERE ref_id = ?")
            .bind("repo:gh/acme/my-repo#src/lib.rs")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        assert_eq!(marks, 1, "the lookalike repo's hash marker was deleted too");
        assert_eq!(vector_and_fts_rows(&store).await, (3, 3));
    }

    // ---- orphan repo reclaim -----------------------------------------------------------

    #[tokio::test]
    async fn orphan_repos_excludes_a_pinned_or_a_discovered_repo() {
        // `seed_repo` discovers its repo onto its own board; only a repo whose discovered set is
        // cleared and which nothing pins is a true orphan.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/pinned", "acme/pinned", "A").await;
        seed_repo(&store, "repo:gh/acme/discovered", "acme/discovered", "B").await;
        seed_repo(&store, "repo:gh/acme/orphan", "acme/orphan", "C").await;

        // Pinned: replace its discovery with an explicit pin instead, so it is watched only that
        // way.
        store
            .replace_board_discovered_repos("boardA", &[])
            .await
            .unwrap();
        store
            .add_board_repo("boardA", "repo:gh/acme/pinned")
            .await
            .unwrap();
        // Discovered: left as `seed_repo` set it up (boardB discovers it).
        // Orphan: no board references it at all.
        store
            .replace_board_discovered_repos("boardC", &[])
            .await
            .unwrap();

        let report = store.orphan_repos().await.unwrap();
        let ids: Vec<&str> = report.repos.iter().map(|r| r.repo_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["repo:gh/acme/orphan"],
            "expected only the unwatched repo, got {ids:?}"
        );
        assert_eq!(report.count, 1);
        assert!(report.rows > 0, "the orphan's row count was not summed");
    }

    #[tokio::test]
    async fn orphan_repo_survives_when_a_board_that_also_pinned_it_is_unpinned() {
        // Pinned to boardA and also discovered by boardB (an org-style board). Unpinning from
        // boardA must not orphan it.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/shared", "acme/shared", "A").await;
        store
            .create_board("board:org", "Org", BoardKind::Org, "2026-08-01T00:00:00Z")
            .await
            .unwrap();
        store
            .replace_board_discovered_repos("board:org", &["repo:gh/acme/shared".to_string()])
            .await
            .unwrap();
        store
            .add_board_repo("boardA", "repo:gh/acme/shared")
            .await
            .unwrap();

        store
            .remove_board_repo("boardA", "repo:gh/acme/shared")
            .await
            .unwrap();

        let report = store.orphan_repos().await.unwrap();
        assert!(
            report.repos.is_empty(),
            "a repo board:org still discovers was reported orphaned: {:?}",
            report.repos
        );
        let reclaimed = store.reclaim_orphan_repos().await.unwrap();
        assert_eq!(reclaimed.count, 0);
        assert_eq!(store.count_repos().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn orphan_repo_survives_when_an_unrelated_org_board_that_discovered_it_is_deleted() {
        // Pinned to boardA and also discovered by an org board. Deleting the org board must not
        // orphan it.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/shared", "acme/shared", "A").await;
        store
            .replace_board_discovered_repos("boardA", &[])
            .await
            .unwrap();
        store
            .add_board_repo("boardA", "repo:gh/acme/shared")
            .await
            .unwrap();
        store
            .create_board("board:org", "Org", BoardKind::Org, "2026-08-01T00:00:00Z")
            .await
            .unwrap();
        store
            .replace_board_discovered_repos("board:org", &["repo:gh/acme/shared".to_string()])
            .await
            .unwrap();

        store.delete_board("board:org").await.unwrap();

        let report = store.orphan_repos().await.unwrap();
        assert!(
            report.repos.is_empty(),
            "a repo boardA still pins was reported orphaned: {:?}",
            report.repos
        );
        let reclaimed = store.reclaim_orphan_repos().await.unwrap();
        assert_eq!(reclaimed.count, 0);
        assert_eq!(store.count_repos().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn reclaim_orphan_repos_removes_every_row_of_an_orphan_and_nothing_else() {
        // An org board's repo: boardB discovers it, then the board is deleted, cascading
        // `board_discovered_repos` and leaving every other table behind. `prune_orphan_sources`
        // never touched those.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/gone", "acme/gone", "A").await;
        seed_repo(&store, "repo:gh/acme/kept", "acme/kept", "B").await;

        store.delete_board("boardA").await.unwrap();
        // The kept repo stays discovered by its own board.
        assert!(store
            .orphan_repos()
            .await
            .unwrap()
            .repos
            .iter()
            .any(|r| r.repo_id == "repo:gh/acme/gone"));

        let report = store.reclaim_orphan_repos().await.unwrap();
        assert_eq!(report.repo_ids, vec!["repo:gh/acme/gone".to_string()]);
        assert_eq!(report.count, 1);
        assert!(report.total_rows >= 25, "total was {}", report.total_rows);

        let left: Vec<(String, i64)> = leftovers(&store, "repo:gh/acme/gone", "A")
            .await
            .into_iter()
            .filter(|(_, c)| *c > 0)
            .collect();
        assert!(left.is_empty(), "rows left behind: {left:?}");

        let kept: Vec<(String, i64)> = leftovers(&store, "repo:gh/acme/kept", "B")
            .await
            .into_iter()
            .filter(|(_, c)| *c == 0)
            .collect();
        assert!(kept.is_empty(), "the neighbour lost rows: {kept:?}");

        assert!(
            store.orphan_repos().await.unwrap().repos.is_empty(),
            "the orphan report is not empty after reclaim"
        );
    }

    #[tokio::test]
    async fn reclaim_orphan_repos_reports_zero_when_nothing_is_orphaned() {
        // A clean store reports zero rather than erroring.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/kept", "acme/kept", "A").await;

        let report = store.reclaim_orphan_repos().await.unwrap();
        assert_eq!(report.count, 0);
        assert_eq!(report.total_rows, 0);
        assert!(report.repo_ids.is_empty());
        assert_eq!(store.count_repos().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn prune_orphan_sources_keeps_a_source_whose_repo_is_still_discovered() {
        // A source that survives only because an org board discovers the same repo must not be
        // dropped when the board that pinned it goes away.
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        store
            .upsert_source(&source("repo:acme/widget", "owned"))
            .await
            .unwrap();
        store
            .create_board("board:org", "Org", BoardKind::Org, "2026-08-01T00:00:00Z")
            .await
            .unwrap();
        store
            .replace_board_discovered_repos("board:org", &["repo:acme/widget".to_string()])
            .await
            .unwrap();

        store.prune_orphan_sources().await.unwrap();
        assert_eq!(
            store.sources().await.unwrap().len(),
            1,
            "a source still discovered by a live board was pruned"
        );

        store
            .replace_board_discovered_repos("board:org", &[])
            .await
            .unwrap();
        store.prune_orphan_sources().await.unwrap();
        assert!(
            store.sources().await.unwrap().is_empty(),
            "a source reachable by no board survived the prune"
        );
    }

    #[tokio::test]
    async fn open_rebuilds_cache_on_incompatible_schema_and_restores_user_config() {
        // A file-backed store whose recorded migration checksum no longer matches is rebuilt, not
        // an error. The cache is dropped and re-syncs; user config (watch list, boards with
        // people/repos, settings) is copied forward from the .bak.
        let path = std::env::temp_dir().join(format!("orgonzola-open-{}.db", std::process::id()));
        let path = path.to_string_lossy().to_string();
        let bak = format!("{path}.bak");
        let cleanup = || {
            for p in [
                path.clone(),
                format!("{path}-wal"),
                format!("{path}-shm"),
                bak.clone(),
            ] {
                let _ = std::fs::remove_file(&p);
            }
        };
        cleanup();

        let store = Store::open(&path).await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap(); // cache (re-syncable)
        store
            .upsert_source(&SourceRow {
                id: "org:acme".into(),
                kind: "org".into(),
                name: "acme".into(),
                ownership: "owned".into(),
                filters: Vec::new(),
                stale_pr_days: 9,
                forge_id: None,
            })
            .await
            .unwrap();
        store
            .upsert_forge(&ForgeRow {
                id: "gh".into(),
                name: "GitHub".into(),
                kind: "github".into(),
                base_url: "https://api.github.com".into(),
                oauth_client_id: None,
            })
            .await
            .unwrap();
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store.set_board_forge("board:1", Some("gh")).await.unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        store
            .add_board_repo("board:1", "repo:acme/widget")
            .await
            .unwrap();
        store
            .update_settings(&Settings {
                sync_period_secs: 111,
                stale_pr_days: 9,
                digest_webhook_url: None,
                digest_schedule_hours: None,
                llm_enabled: true,
                llm_model: None,
                storage_budget_mb: None,
            })
            .await
            .unwrap();
        assert_eq!(store.count_repos().await.unwrap(), 1);
        drop(store);

        // Tamper with a stored migration checksum so the next migrate sees a mismatch.
        let opts = SqliteConnectOptions::new().filename(&path);
        let pool = SqlitePoolOptions::new().connect_with(opts).await.unwrap();
        sqlx::query("UPDATE _sqlx_migrations SET checksum = X'00' WHERE version = (SELECT MIN(version) FROM _sqlx_migrations)")
            .execute(&pool)
            .await
            .unwrap();
        pool.close().await;

        // open() rebuilds: the cache is empty (re-syncs), the old file is preserved as a .bak, and
        // user config is restored from it.
        let store = Store::open(&path).await.unwrap();
        assert_eq!(
            store.count_repos().await.unwrap(),
            0,
            "cache is dropped, re-syncs"
        );
        assert!(
            std::path::Path::new(&bak).exists(),
            "the incompatible db is preserved as a .bak, not silently destroyed"
        );
        let boards = store.boards().await.unwrap();
        assert_eq!(boards.len(), 1, "the board is restored");
        assert_eq!(
            store.forges().await.unwrap().len(),
            1,
            "the forge is restored"
        );
        assert_eq!(
            boards[0].forge_id,
            Some("gh".to_string()),
            "board forge assignment restored"
        );
        assert_eq!(
            boards[0].people,
            vec!["alice".to_string()],
            "board people restored"
        );
        assert_eq!(
            boards[0].repos,
            vec!["repo:acme/widget".to_string()],
            "board repos restored"
        );
        assert_eq!(
            store.sources().await.unwrap().len(),
            1,
            "the watch source is restored"
        );
        let settings = store.settings().await.unwrap();
        assert_eq!(settings.sync_period_secs, 111, "settings restored");
        assert_eq!(settings.stale_pr_days, 9, "settings restored");
        cleanup();
    }

    #[tokio::test]
    async fn open_creates_the_full_schema() {
        let store = Store::open_in_memory().await.unwrap();
        let tables = store.table_names().await.unwrap();
        for expected in [
            "repos",
            "commits",
            "pull_requests",
            "issues",
            "releases",
            "reviews",
            "ci_runs",
            "identities",
            "identity_accounts",
            "teams",
            "team_members",
            "ownership",
            "sprints",
            "work_items",
            "sprint_work_items",
            "links",
            "sync_cursors",
            "sources",
            "dependencies",
            "settings",
            "boards",
            "board_people",
            "board_repos",
            "board_discovered_repos",
            "metric_snapshots",
            "pr_files",
            "code_files",
            "indexed_text",
            "embeddings",
        ] {
            assert!(
                tables.contains(&expected.to_string()),
                "missing table {expected}"
            );
        }
    }

    #[tokio::test]
    async fn upsert_repo_is_idempotent_by_id() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        store
            .upsert_repo(&repo("acme/widget-renamed"))
            .await
            .unwrap();
        assert_eq!(store.count_repos().await.unwrap(), 1);
        assert_eq!(
            store
                .get_repo("repo:acme/widget")
                .await
                .unwrap()
                .unwrap()
                .full_name,
            "acme/widget-renamed"
        );
    }

    fn source(id: &str, ownership: &str) -> SourceRow {
        SourceRow {
            id: id.into(),
            kind: "repo".into(),
            name: "acme/widget".into(),
            ownership: ownership.into(),
            filters: vec!["label:bug".into(), "path:src/".into()],
            stale_pr_days: 7,
            forge_id: None,
        }
    }

    #[tokio::test]
    async fn sources_round_trip_with_filters_and_upsert_by_id() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_source(&source("repo:a", "owned"))
            .await
            .unwrap();
        store
            .upsert_source(&source("repo:b", "observed"))
            .await
            .unwrap();
        // Upsert by id: re-adding "repo:a" with changed fields updates, does not duplicate.
        let mut a2 = source("repo:a", "observed");
        a2.filters = vec!["label:urgent".into()];
        a2.stale_pr_days = 3;
        store.upsert_source(&a2).await.unwrap();

        let sources = store.sources().await.unwrap();
        assert_eq!(sources.len(), 2);
        let a = sources.iter().find(|s| s.id == "repo:a").unwrap();
        assert_eq!(a.ownership, "observed");
        assert_eq!(a.stale_pr_days, 3);
        assert_eq!(a.filters, vec!["label:urgent".to_string()]);
        let b = sources.iter().find(|s| s.id == "repo:b").unwrap();
        assert_eq!(
            b.filters,
            vec!["label:bug".to_string(), "path:src/".to_string()]
        );
    }

    #[tokio::test]
    async fn remove_source_deletes_only_that_id() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_source(&source("repo:a", "owned"))
            .await
            .unwrap();
        store
            .upsert_source(&source("repo:b", "owned"))
            .await
            .unwrap();
        store.remove_source("repo:a").await.unwrap();
        store.remove_source("repo:absent").await.unwrap(); // no-op, no error
        let sources = store.sources().await.unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].id, "repo:b");
    }

    #[tokio::test]
    async fn settings_default_then_update_in_place() {
        let store = Store::open_in_memory().await.unwrap();
        // Seeded defaults.
        let s = store.settings().await.unwrap();
        assert_eq!(s.sync_period_secs, 300);
        assert_eq!(s.stale_pr_days, 7);
        assert_eq!(s.digest_webhook_url, None);
        assert_eq!(s.digest_schedule_hours, None);
        store
            .update_settings(&Settings {
                sync_period_secs: 60,
                stale_pr_days: 30,
                digest_webhook_url: Some("https://hooks.slack.com/x".into()),
                digest_schedule_hours: Some(24),
                llm_enabled: true,
                llm_model: None,
                storage_budget_mb: None,
            })
            .await
            .unwrap();
        let s = store.settings().await.unwrap();
        assert_eq!(s.sync_period_secs, 60);
        assert_eq!(s.stale_pr_days, 30);
        assert_eq!(
            s.digest_webhook_url.as_deref(),
            Some("https://hooks.slack.com/x")
        );
        assert_eq!(s.digest_schedule_hours, Some(24));
    }

    #[tokio::test]
    async fn metric_snapshots_one_row_per_day_in_order() {
        let store = Store::open_in_memory().await.unwrap();
        let snap = |day: &str, wip: i64| MetricSnapshot {
            scope_kind: "repo".into(),
            scope_id: "repo:a".into(),
            captured_on: day.into(),
            wip,
            stale_open_prs: 0,
            merged_without_review: 0,
            attention_count: 0,
            median_cycle_time_secs: None,
            median_pickup_secs: None,
            median_review_secs: None,
        };
        store
            .upsert_metric_snapshot(&snap("2026-06-10", 2))
            .await
            .unwrap();
        // Same day overwrites.
        store
            .upsert_metric_snapshot(&snap("2026-06-10", 5))
            .await
            .unwrap();
        store
            .upsert_metric_snapshot(&snap("2026-06-11", 3))
            .await
            .unwrap();

        let history = store.metric_snapshots("repo", "repo:a").await.unwrap();
        assert_eq!(history.len(), 2); // two distinct days
        assert_eq!(history[0].captured_on, "2026-06-10");
        assert_eq!(history[0].wip, 5); // later write won
        assert_eq!(history[1].captured_on, "2026-06-11");
        assert!(store
            .metric_snapshots("repo", "nope")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn record_metric_and_series_roundtrip() {
        let store = Store::open_in_memory().await.unwrap();
        // An arbitrary metric via the generic key/value API.
        store
            .record_metric("board", "b1", "2026-06-10", "flow_efficiency_pct", 42)
            .await
            .unwrap();
        store
            .record_metric("board", "b1", "2026-06-11", "flow_efficiency_pct", 55)
            .await
            .unwrap();
        // Idempotent per (scope, day, key): a re-record overwrites.
        store
            .record_metric("board", "b1", "2026-06-11", "flow_efficiency_pct", 60)
            .await
            .unwrap();

        let series = store
            .metric_series("board", "b1", "flow_efficiency_pct")
            .await
            .unwrap();
        assert_eq!(
            series,
            vec![
                ("2026-06-10".to_string(), 42),
                ("2026-06-11".to_string(), 60),
            ]
        );
        // An unrelated key is empty; the typed snapshot recompose ignores the unknown key.
        assert!(store
            .metric_series("board", "b1", "wip")
            .await
            .unwrap()
            .is_empty());
        assert!(store.metric_snapshots("board", "b1").await.unwrap().len() == 2);
    }

    #[tokio::test]
    async fn trim_metric_snapshots_keeps_latest_per_scope() {
        let store = Store::open_in_memory().await.unwrap();
        let snap = |scope: &str, day: &str| MetricSnapshot {
            scope_kind: "repo".into(),
            scope_id: scope.into(),
            captured_on: day.into(),
            wip: 0,
            stale_open_prs: 0,
            merged_without_review: 0,
            attention_count: 0,
            median_cycle_time_secs: None,
            median_pickup_secs: None,
            median_review_secs: None,
        };
        for day in ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"] {
            store
                .upsert_metric_snapshot(&snap("repo:a", day))
                .await
                .unwrap();
        }
        store
            .upsert_metric_snapshot(&snap("repo:b", "2026-06-01"))
            .await
            .unwrap();

        // Keep the latest 2 days for repo:a; repo:b is a different scope and untouched.
        let deleted = store
            .trim_metric_snapshots("repo", "repo:a", 2)
            .await
            .unwrap();
        // Long format: 2 dropped days x 4 non-null metric rows each (the medians are null here).
        assert_eq!(deleted, 8);
        let a = store.metric_snapshots("repo", "repo:a").await.unwrap();
        assert_eq!(
            a.iter().map(|s| s.captured_on.as_str()).collect::<Vec<_>>(),
            vec!["2026-06-03", "2026-06-04"]
        );
        assert_eq!(
            store
                .metric_snapshots("repo", "repo:b")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn pr_files_replace_is_one_set_per_pr() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        store
            .upsert_pull_request(&PullRequest {
                id: "p1".into(),
                repo_id: "repo:acme/widget".into(),
                number: 1,
                title: "PR".into(),
                state: "open".into(),
                author_login: None,
                body: None,
                created_at: "2026-06-01T00:00:00Z".into(),
                merged_at: None,
                html_url: None,
            })
            .await
            .unwrap();
        let f = |name: &str, a: i64, d: i64| PrFile {
            pr_id: "p1".into(),
            filename: name.into(),
            status: "modified".into(),
            additions: a,
            deletions: d,
            patch: Some(format!("@@ {name} @@")),
        };
        store
            .replace_pr_files("p1", &[f("a.rs", 6, 1), f("b.rs", 4, 2)])
            .await
            .unwrap();
        let files = store.pr_files("p1").await.unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].filename, "a.rs"); // ordered by filename
        assert_eq!(files[0].additions, 6);

        // Replace with a smaller set: the old rows are gone.
        store
            .replace_pr_files("p1", &[f("a.rs", 1, 0)])
            .await
            .unwrap();
        let files = store.pr_files("p1").await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].additions, 1);
    }

    #[tokio::test]
    async fn repo_file_stats_aggregate_across_prs() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        let mkpr = |id: &str, author: &str| PullRequest {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number: 1,
            title: "PR".into(),
            state: "closed".into(),
            author_login: Some(author.into()),
            body: None,
            created_at: "2026-06-01T00:00:00Z".into(),
            merged_at: Some("2026-06-02T00:00:00Z".into()),
            html_url: None,
        };
        store
            .upsert_pull_request(&mkpr("p1", "alice"))
            .await
            .unwrap();
        store.upsert_pull_request(&mkpr("p2", "bob")).await.unwrap();
        let f = |pr: &str, name: &str, a: i64, d: i64| PrFile {
            pr_id: pr.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: a,
            deletions: d,
            patch: None,
        };
        // a.rs touched by both PRs (2 authors); b.rs only by p1.
        store
            .replace_pr_files("p1", &[f("p1", "a.rs", 5, 2), f("p1", "b.rs", 3, 0)])
            .await
            .unwrap();
        store
            .replace_pr_files("p2", &[f("p2", "a.rs", 4, 1)])
            .await
            .unwrap();

        let stats = store.repo_file_stats("repo:acme/widget").await.unwrap();
        let a = stats.iter().find(|s| s.path == "a.rs").unwrap();
        assert_eq!(a.changes, 2);
        assert_eq!(a.additions, 9);
        assert_eq!(a.deletions, 3);
        assert_eq!(a.authors, 2);
        let b = stats.iter().find(|s| s.path == "b.rs").unwrap();
        assert_eq!(b.changes, 1);
        assert_eq!(b.authors, 1);

        let authorship = store
            .repo_file_authorship("repo:acme/widget")
            .await
            .unwrap();
        let a_alice = authorship
            .iter()
            .find(|r| r.path == "a.rs" && r.author == "alice")
            .unwrap();
        assert_eq!(a_alice.changes, 1);
        assert!(authorship
            .iter()
            .any(|r| r.path == "a.rs" && r.author == "bob"));
        assert!(authorship
            .iter()
            .all(|r| r.path != "b.rs" || r.author == "alice"));
    }

    #[tokio::test]
    async fn repo_file_last_changed_takes_latest_merged_only() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        let mkpr = |id: &str, state: &str, merged: Option<&str>| PullRequest {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number: 1,
            title: "PR".into(),
            state: state.into(),
            author_login: Some("alice".into()),
            body: None,
            created_at: "2026-06-01T00:00:00Z".into(),
            merged_at: merged.map(Into::into),
            html_url: None,
        };
        let f = |pr: &str, name: &str| PrFile {
            pr_id: pr.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: 1,
            deletions: 0,
            patch: None,
        };
        // Two merged PRs touched core.rs (later merge wins), plus an open PR on core.rs and a
        // merged PR on util.rs only.
        store
            .upsert_pull_request(&mkpr("m1", "closed", Some("2026-01-01T00:00:00Z")))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr("m2", "closed", Some("2026-03-01T00:00:00Z")))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr("open", "open", None))
            .await
            .unwrap();
        store
            .replace_pr_files("m1", &[f("m1", "core.rs")])
            .await
            .unwrap();
        store
            .replace_pr_files("m2", &[f("m2", "core.rs"), f("m2", "util.rs")])
            .await
            .unwrap();
        store
            .replace_pr_files("open", &[f("open", "core.rs")])
            .await
            .unwrap();

        let map: std::collections::HashMap<String, String> = store
            .repo_file_last_changed("repo:acme/widget")
            .await
            .unwrap()
            .into_iter()
            .collect();
        // core.rs takes the later of the two merges; the open PR does not contribute.
        assert_eq!(
            map.get("core.rs").map(String::as_str),
            Some("2026-03-01T00:00:00Z")
        );
        assert_eq!(
            map.get("util.rs").map(String::as_str),
            Some("2026-03-01T00:00:00Z")
        );
    }

    #[tokio::test]
    async fn repo_file_review_gaps_and_prs() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        let mkpr =
            |id: &str, num: i64, state: &str, merged: Option<&str>, created: &str| PullRequest {
                id: id.into(),
                repo_id: "repo:acme/widget".into(),
                number: num,
                title: format!("PR {num}"),
                state: state.into(),
                author_login: Some("alice".into()),
                body: None,
                created_at: created.into(),
                merged_at: merged.map(Into::into),
                html_url: Some(format!("https://h/pr/{num}")),
            };
        let f = |pr: &str, name: &str| PrFile {
            pr_id: pr.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: 1,
            deletions: 0,
            patch: None,
        };
        // core.rs: touched by two merged PRs (m1 reviewed, m2 unreviewed) -> gap 1/2. util.rs: one
        // merged reviewed PR -> gap 0. open.rs: an open PR only -> not counted (no merged touch).
        store
            .upsert_pull_request(&mkpr(
                "m1",
                1,
                "closed",
                Some("2026-02-01T00:00:00Z"),
                "2026-01-01T00:00:00Z",
            ))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr(
                "m2",
                2,
                "closed",
                Some("2026-03-01T00:00:00Z"),
                "2026-02-15T00:00:00Z",
            ))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr("open", 3, "open", None, "2026-04-01T00:00:00Z"))
            .await
            .unwrap();
        store
            .replace_pr_files("m1", &[f("m1", "core.rs"), f("m1", "util.rs")])
            .await
            .unwrap();
        store
            .replace_pr_files("m2", &[f("m2", "core.rs")])
            .await
            .unwrap();
        store
            .replace_pr_files("open", &[f("open", "core.rs")])
            .await
            .unwrap();
        // m1 has a review; m2 has none.
        store
            .upsert_review(&Review {
                id: "rv1".into(),
                pr_id: "m1".into(),
                reviewer_login: Some("bob".into()),
                state: "APPROVED".into(),
                submitted_at: Some("2026-01-15T00:00:00Z".into()),
            })
            .await
            .unwrap();

        let gaps: std::collections::HashMap<String, (i64, i64)> = store
            .repo_file_review_gaps("repo:acme/widget")
            .await
            .unwrap()
            .into_iter()
            .map(|(name, merged, unrev)| (name, (merged, unrev)))
            .collect();
        assert_eq!(
            gaps.get("core.rs"),
            Some(&(2, 1)),
            "2 merged touches, 1 unreviewed"
        );
        assert_eq!(
            gaps.get("util.rs"),
            Some(&(1, 0)),
            "1 merged touch, fully reviewed"
        );

        // Drill: core.rs was touched by the open PR (newest), then m2, then m1.
        let prs = store.repo_file_prs("repo:acme/widget").await.unwrap();
        let core_prs: Vec<i64> = prs
            .iter()
            .filter(|(name, ..)| name == "core.rs")
            .map(|(_, number, ..)| *number)
            .collect();
        assert_eq!(core_prs, vec![3, 2, 1], "newest-first by creation");

        // Per-PR churn: each file here is +1/-0, so m1 (2 files) = 2, m2/open (1 file) = 1.
        let churn: std::collections::HashMap<String, i64> = store
            .repo_pr_churn("repo:acme/widget")
            .await
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(churn.get("m1"), Some(&2));
        assert_eq!(churn.get("m2"), Some(&1));
    }

    #[tokio::test]
    async fn repo_has_work_item_links_is_false_until_a_link_is_recorded() {
        // Org boards skip the index lane, so their repos never run the linker: this must read
        // false for a repo with PRs but no recorded link, even once a different repo has links.
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        store.upsert_repo(&repo("acme/other")).await.unwrap();
        store
            .upsert_pull_request(&PullRequest {
                id: "pr1".into(),
                repo_id: "repo:acme/widget".into(),
                number: 1,
                title: "PR 1".into(),
                state: "closed".into(),
                author_login: Some("alice".into()),
                body: None,
                created_at: "2026-01-01T00:00:00Z".into(),
                merged_at: Some("2026-01-02T00:00:00Z".into()),
                html_url: None,
            })
            .await
            .unwrap();
        assert!(!store
            .repo_has_work_item_links("repo:acme/widget")
            .await
            .unwrap());

        store
            .add_link("pull_request", "pr1", "work_item", "wi1", "closes")
            .await
            .unwrap();
        assert!(store
            .repo_has_work_item_links("repo:acme/widget")
            .await
            .unwrap());
        // A neighbouring repo with zero links of its own must not read true off pr1's link.
        assert!(!store
            .repo_has_work_item_links("repo:acme/other")
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn repo_file_coupling_counts_co_occurrences_in_merged_prs() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        let mkpr = |id: &str, merged: Option<&str>| PullRequest {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number: 1,
            title: "PR".into(),
            state: "closed".into(),
            author_login: Some("alice".into()),
            body: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            merged_at: merged.map(Into::into),
            html_url: None,
        };
        let f = |pr: &str, name: &str| PrFile {
            pr_id: pr.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: 1,
            deletions: 0,
            patch: None,
        };
        // m1 and m2 are merged; open is not. a.rs + b.rs co-appear in both merged PRs ->
        // together=2. b.rs + c.rs only appear together in m1 -> together=1. Open PRs are not
        // counted.
        store
            .upsert_pull_request(&mkpr("m1", Some("2026-02-01T00:00:00Z")))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr("m2", Some("2026-03-01T00:00:00Z")))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr("open", None))
            .await
            .unwrap();
        store
            .replace_pr_files("m1", &[f("m1", "a.rs"), f("m1", "b.rs"), f("m1", "c.rs")])
            .await
            .unwrap();
        store
            .replace_pr_files("m2", &[f("m2", "a.rs"), f("m2", "b.rs")])
            .await
            .unwrap();
        store
            .replace_pr_files("open", &[f("open", "a.rs"), f("open", "b.rs")])
            .await
            .unwrap();

        let pairs: std::collections::HashMap<(String, String), (i64, i64, i64)> = store
            .repo_file_coupling("repo:acme/widget")
            .await
            .unwrap()
            .into_iter()
            .map(|(a, b, tog, pa, pb)| ((a, b), (tog, pa, pb)))
            .collect();
        // a.rs + b.rs: together=2, prs_a=2, prs_b=2.
        let ab = pairs
            .get(&("a.rs".into(), "b.rs".into()))
            .expect("a+b pair");
        assert_eq!(ab.0, 2, "together");
        assert_eq!(ab.1, 2, "prs_a");
        assert_eq!(ab.2, 2, "prs_b");
        // a.rs + c.rs: together=1.
        let ac = pairs
            .get(&("a.rs".into(), "c.rs".into()))
            .expect("a+c pair");
        assert_eq!(ac.0, 1, "together");
        // Open PR does not count.
        let total_together: i64 = pairs.values().map(|(t, _, _)| t).sum();
        assert_eq!(total_together, 1 + 1 + 2, "3 merged pairs total");
    }

    #[tokio::test]
    async fn boards_round_trip_members_and_effective_repos() {
        let store = Store::open_in_memory().await.unwrap();
        let mk = |name: &str| Repo {
            id: format!("repo:acme/{name}"),
            owner: "acme".into(),
            name: name.into(),
            full_name: format!("acme/{name}"),
            ownership: "owned".into(),
        };
        store.upsert_repo(&mk("api")).await.unwrap();
        store.upsert_repo(&mk("web")).await.unwrap();

        store
            .create_board(
                "board:platform",
                "Platform",
                BoardKind::Team,
                "2026-06-10T00:00:00Z",
            )
            .await
            .unwrap();
        store
            .add_board_person("board:platform", "alice")
            .await
            .unwrap();
        store
            .add_board_person("board:platform", "bob")
            .await
            .unwrap();
        store
            .add_board_person("board:platform", "alice")
            .await
            .unwrap(); // idempotent
        store
            .add_board_repo("board:platform", "repo:acme/api")
            .await
            .unwrap();

        let board = store.board("board:platform").await.unwrap().unwrap();
        assert_eq!(board.name, "Platform");
        assert_eq!(board.people, vec!["alice", "bob"]);
        assert_eq!(board.repos, vec!["repo:acme/api"]);

        // Pinned repos -> effective set is those repos.
        assert_eq!(
            store
                .board_effective_repo_ids("board:platform")
                .await
                .unwrap(),
            vec!["repo:acme/api"]
        );

        // Unpin all -> effective set is the discovered repos (empty until discovery runs), not all
        // synced repos.
        store
            .remove_board_repo("board:platform", "repo:acme/api")
            .await
            .unwrap();
        assert!(store
            .board_effective_repo_ids("board:platform")
            .await
            .unwrap()
            .is_empty());
        store
            .replace_board_discovered_repos("board:platform", &["repo:acme/web".to_string()])
            .await
            .unwrap();
        assert_eq!(
            store
                .board_effective_repo_ids("board:platform")
                .await
                .unwrap(),
            vec!["repo:acme/web"]
        );

        // An org restriction round-trips.
        store
            .set_board_org("board:platform", Some("acme"))
            .await
            .unwrap();
        assert_eq!(
            store
                .board("board:platform")
                .await
                .unwrap()
                .unwrap()
                .org
                .as_deref(),
            Some("acme")
        );

        // Remove a person; delete cascades members.
        store
            .remove_board_person("board:platform", "bob")
            .await
            .unwrap();
        assert_eq!(
            store.board("board:platform").await.unwrap().unwrap().people,
            vec!["alice"]
        );
        store.delete_board("board:platform").await.unwrap();
        assert!(store.board("board:platform").await.unwrap().is_none());
        assert!(store
            .board_people("board:platform")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn deleting_or_unpinning_prunes_orphaned_sources() {
        // A pinned repo on two boards is one source. Removing it from one board keeps the source
        // (the other still pins it); deleting the last board that pins it drops the source so it
        // stops syncing.
        let store = Store::open_in_memory().await.unwrap();
        store
            .create_board("board:a", "A", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .create_board("board:b", "B", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .upsert_source(&source("repo:acme/widget", "owned"))
            .await
            .unwrap();
        store
            .add_board_repo("board:a", "repo:acme/widget")
            .await
            .unwrap();
        store
            .add_board_repo("board:b", "repo:acme/widget")
            .await
            .unwrap();

        // Unpin from one board: still pinned by the other, so the source survives.
        store
            .remove_board_repo("board:a", "repo:acme/widget")
            .await
            .unwrap();
        assert_eq!(
            store.sources().await.unwrap().len(),
            1,
            "still pinned by board:b"
        );

        // Delete the last board pinning it: the source is now orphaned and pruned.
        store.delete_board("board:b").await.unwrap();
        assert!(
            store.sources().await.unwrap().is_empty(),
            "no board pins it, so it must stop syncing"
        );
    }

    #[tokio::test]
    async fn forges_round_trip_and_board_assignment() {
        let store = Store::open_in_memory().await.unwrap();
        let gh = ForgeRow {
            id: "gh".into(),
            name: "GitHub".into(),
            kind: "github".into(),
            base_url: "https://api.github.com".into(),
            oauth_client_id: Some("client-xyz".into()),
        };
        let gitea = ForgeRow {
            id: "gitea".into(),
            name: "Self-hosted".into(),
            kind: "gitea".into(),
            base_url: "https://gitea.example.com/api/v1".into(),
            oauth_client_id: None,
        };
        store.upsert_forge(&gh).await.unwrap();
        store.upsert_forge(&gitea).await.unwrap();
        let forges = store.forges().await.unwrap();
        assert_eq!(forges.len(), 2);
        assert_eq!(store.forge("gh").await.unwrap().as_ref(), Some(&gh));

        // Re-upsert edits in place.
        let mut edited = gh.clone();
        edited.base_url = "https://ghe.example.com/api/v3".into();
        edited.oauth_client_id = Some("client-rotated".into());
        store.upsert_forge(&edited).await.unwrap();
        assert_eq!(store.forges().await.unwrap().len(), 2, "no duplicate by id");
        let stored = store.forge("gh").await.unwrap().unwrap();
        assert_eq!(stored.base_url, "https://ghe.example.com/api/v3");
        assert_eq!(stored.oauth_client_id.as_deref(), Some("client-rotated"));

        // A board records which forge it syncs from; setting it round-trips, clearing it nulls it
        // out.
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        assert_eq!(
            store.board("board:1").await.unwrap().unwrap().forge_id,
            None
        );
        store
            .set_board_forge("board:1", Some("gitea"))
            .await
            .unwrap();
        assert_eq!(
            store.board("board:1").await.unwrap().unwrap().forge_id,
            Some("gitea".into())
        );
        store.set_board_forge("board:1", None).await.unwrap();
        assert_eq!(
            store.board("board:1").await.unwrap().unwrap().forge_id,
            None
        );

        // Dependency-scanning flag: off by default, toggles, round-trips.
        assert!(
            !store
                .board("board:1")
                .await
                .unwrap()
                .unwrap()
                .scan_dependencies
        );
        store
            .set_board_scan_dependencies("board:1", true)
            .await
            .unwrap();
        assert!(
            store
                .board("board:1")
                .await
                .unwrap()
                .unwrap()
                .scan_dependencies
        );

        // Archived-repo discovery is opt-in: boards default off and the preference round-trips
        // independently of other board settings.
        assert!(
            !store
                .board("board:1")
                .await
                .unwrap()
                .unwrap()
                .include_archived
        );
        store
            .set_board_include_archived("board:1", true)
            .await
            .unwrap();
        assert!(
            store
                .board("board:1")
                .await
                .unwrap()
                .unwrap()
                .include_archived
        );

        // A forge no board references can be removed; removing an absent id is a no-op.
        store.remove_forge("gh").await.unwrap();
        store.remove_forge("nope").await.unwrap();
        assert_eq!(store.forges().await.unwrap().len(), 1);
        assert!(store.forge("gh").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn dependencies_round_trip_and_upsert_by_repo_ecosystem_name() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        let dep = |name: &str, version: &str, kind: &str| DependencyRow {
            repo_id: "repo:acme/widget".into(),
            ecosystem: "cargo".into(),
            name: name.into(),
            version_req: Some(version.into()),
            kind: kind.into(),
            source: "Cargo.toml".into(),
        };
        store
            .upsert_dependency(&dep("serde", "1", "normal"))
            .await
            .unwrap();
        store
            .upsert_dependency(&dep("tokio", "1", "dev"))
            .await
            .unwrap();
        // Re-upsert serde with a new version + kind: updates in place, no duplicate.
        store
            .upsert_dependency(&dep("serde", "2", "normal"))
            .await
            .unwrap();

        let deps = store
            .dependencies_for_repo("repo:acme/widget")
            .await
            .unwrap();
        assert_eq!(deps.len(), 2);
        let serde = deps.iter().find(|d| d.name == "serde").unwrap();
        assert_eq!(serde.version_req.as_deref(), Some("2"));
        let tokio = deps.iter().find(|d| d.name == "tokio").unwrap();
        assert_eq!(tokio.kind, "dev");
    }

    #[tokio::test]
    async fn board_disabled_signals_round_trip() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        // Default: nothing disabled.
        assert!(store
            .board_disabled_signals("board:1")
            .await
            .unwrap()
            .is_empty());
        store
            .set_board_signal_enabled("board:1", "merged_without_review", false)
            .await
            .unwrap();
        assert_eq!(
            store.board_disabled_signals("board:1").await.unwrap(),
            vec!["merged_without_review".to_string()]
        );
        // Re-enabling removes it.
        store
            .set_board_signal_enabled("board:1", "merged_without_review", true)
            .await
            .unwrap();
        assert!(store
            .board_disabled_signals("board:1")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn repo_index_code_defaults_on_and_toggles() {
        let store = Store::open_in_memory().await.unwrap();
        // No row -> default true (index code).
        assert!(store.repo_index_code("repo:acme/widget").await.unwrap());
        store
            .set_repo_index_code("repo:acme/widget", false)
            .await
            .unwrap();
        assert!(!store.repo_index_code("repo:acme/widget").await.unwrap());
        store
            .set_repo_index_code("repo:acme/widget", true)
            .await
            .unwrap();
        assert!(store.repo_index_code("repo:acme/widget").await.unwrap());
        // A different repo is unaffected.
        assert!(store.repo_index_code("repo:acme/other").await.unwrap());
    }

    #[tokio::test]
    async fn clear_repo_code_index_drops_only_that_repos_code() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap(); // code_files FKs repos
        let embedder = HashEmbedder::default();
        let chunk = |t: &str| (t.to_string(), embedder.embed(t));
        // Code chunks for two repos + an activity chunk that must survive.
        store
            .replace_embeddings("code", "repo:acme/widget#a.rs", &[chunk("fn render() {}")])
            .await
            .unwrap();
        store
            .upsert_code_file("repo:acme/widget", "a.rs", "sha1")
            .await
            .unwrap();
        store
            .set_sync_cursor("repo:acme/widget", "code", "2026-06-10")
            .await
            .unwrap();
        store
            .replace_embeddings("code", "repo:acme/other#b.rs", &[chunk("fn keep() {}")])
            .await
            .unwrap();
        store
            .replace_embeddings("commit", "c1", &[chunk("activity stays")])
            .await
            .unwrap();

        store
            .clear_repo_code_index("repo:acme/widget")
            .await
            .unwrap();

        // widget's code is gone (embeddings + marker + cursor); the other repo's code + activity
        // remain.
        let hits = store
            .nearest_embeddings_of_kind(&embedder.embed("render"), 5, "code")
            .await
            .unwrap();
        assert!(hits.iter().all(|h| h.ref_id != "repo:acme/widget#a.rs"));
        assert!(hits.iter().any(|h| h.ref_id == "repo:acme/other#b.rs"));
        assert!(store
            .code_file_shas("repo:acme/widget")
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            store.sync_cursor("repo:acme/widget", "code").await.unwrap(),
            None
        );
        // The activity embedding is untouched.
        let act = store
            .nearest_embeddings_of_kind(&embedder.embed("activity"), 5, "commit")
            .await
            .unwrap();
        assert!(act.iter().any(|h| h.ref_id == "c1"));
    }

    #[tokio::test]
    async fn embeddings_persist_and_search_nearest_first() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        let embedder = HashEmbedder::default();
        let index = |text: &str| (text.to_string(), embedder.embed(text));

        store
            .replace_embeddings("commit", "c1", &[index("user authentication login flow")])
            .await
            .unwrap();
        store
            .replace_embeddings(
                "commit",
                "c2",
                &[index("database schema migration rollback")],
            )
            .await
            .unwrap();

        let query = embedder.embed("login authentication for a user");
        let hits = store.nearest_embeddings(&query, 2).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].ref_id, "c1"); // the auth chunk ranks first
        assert!(hits[0].score >= hits[1].score);

        // Re-indexing c1 replaces, does not duplicate.
        store
            .replace_embeddings("commit", "c1", &[index("auth flow rewritten")])
            .await
            .unwrap();
        let all = store.nearest_embeddings(&query, 10).await.unwrap();
        assert_eq!(all.len(), 2); // still two entities, not three
    }

    #[tokio::test]
    async fn nearest_embeddings_on_empty_index_is_empty() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        let query = HashEmbedder::default().embed("anything");
        let hits = store.nearest_embeddings(&query, 5).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn nearest_embeddings_of_kind_excludes_other_kinds() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        let embedder = HashEmbedder::default();
        let index = |text: &str| (text.to_string(), embedder.embed(text));
        store
            .replace_embeddings("commit", "c1", &[index("parse the dependency manifest")])
            .await
            .unwrap();
        store
            .replace_embeddings(
                "code",
                "repo:acme/widget#src/graph.rs",
                &[index("fn parse_manifest(text: &str)")],
            )
            .await
            .unwrap();

        let query = embedder.embed("parse manifest");
        // A code-only search returns the code chunk and never the commit.
        let code = store
            .nearest_embeddings_of_kind(&query, 5, "code")
            .await
            .unwrap();
        assert_eq!(code.len(), 1);
        assert_eq!(code[0].ref_kind, "code");
        assert_eq!(code[0].ref_id, "repo:acme/widget#src/graph.rs");
    }

    #[tokio::test]
    async fn hybrid_search_fuses_arms_and_respects_kind_filter() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        let embedder = HashEmbedder::default();
        let index = |text: &str| (text.to_string(), embedder.embed(text));
        store
            .replace_embeddings(
                "code",
                "repo:acme/widget#src/graph.rs",
                &[index("fn parse_manifest(text: &str) -> Vec<Dep>")],
            )
            .await
            .unwrap();
        store
            .replace_embeddings("commit", "c1", &[index("fix the login flow for users")])
            .await
            .unwrap();

        // Code-only hybrid search for an identifier returns the code chunk (keyword + vector
        // agree).
        let code = store
            .hybrid_search(
                &embedder.embed("parse manifest"),
                "parse manifest",
                5,
                KindFilter::Is("code"),
                RepoScope::All,
            )
            .await
            .unwrap();
        assert_eq!(code.len(), 1);
        assert_eq!(code[0].ref_kind, "code");

        // Activity search (Not code) for "login" returns the commit, never the code chunk.
        let activity = store
            .hybrid_search(
                &embedder.embed("login"),
                "login",
                5,
                KindFilter::Not("code"),
                RepoScope::All,
            )
            .await
            .unwrap();
        assert!(activity.iter().all(|h| h.ref_kind != "code"));
        assert!(activity.iter().any(|h| h.ref_id == "c1"));
    }

    #[test]
    fn fts_query_builds_or_of_quoted_terms_or_none() {
        assert_eq!(
            fts_query("parse_manifest now").as_deref(),
            Some("\"parse\" OR \"manifest\" OR \"now\"")
        );
        assert!(fts_query("   ").is_none());
    }

    #[tokio::test]
    async fn counts_aggregate_store_and_index() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        let embedder = HashEmbedder::default();
        let index = |t: &str| (t.to_string(), embedder.embed(t));
        store
            .replace_embeddings("code", "repo:acme/widget#a.rs", &[index("fn a()")])
            .await
            .unwrap();
        store
            .replace_embeddings("commit", "c1", &[index("fix bug")])
            .await
            .unwrap();
        store
            .upsert_code_file("repo:acme/widget", "a.rs", "sha1")
            .await
            .unwrap();

        let c = store.counts().await.unwrap();
        assert_eq!(c.repos, 1);
        assert_eq!(c.embeddings_total, 2);
        assert_eq!(c.embeddings_code, 1);
        assert_eq!(c.embeddings_activity, 1);
        assert_eq!(c.code_files, 1);
    }

    #[tokio::test]
    async fn meta_roundtrip_and_clear_index() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();

        assert_eq!(store.meta_get("embedder_id").await.unwrap(), None);
        store.meta_set("embedder_id", "old:768").await.unwrap();
        assert_eq!(
            store.meta_get("embedder_id").await.unwrap(),
            Some("old:768".to_string())
        );
        store.meta_set("embedder_id", "new:768").await.unwrap();
        assert_eq!(
            store.meta_get("embedder_id").await.unwrap(),
            Some("new:768".to_string())
        );

        // Seed source rows + the whole index (vectors + both marker tables).
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        let embedder = HashEmbedder::default();
        store
            .replace_embeddings(
                "code",
                "repo:acme/widget#a.rs",
                &[("fn a()".to_string(), embedder.embed("fn a()"))],
            )
            .await
            .unwrap();
        store
            .set_indexed_text_hash("commit", "c1", "h")
            .await
            .unwrap();
        store
            .upsert_code_file("repo:acme/widget", "a.rs", "sha1")
            .await
            .unwrap();
        assert_eq!(store.counts().await.unwrap().embeddings_total, 1);

        // clear_index drops the vectors + markers, but leaves the source repo row.
        store.clear_index().await.unwrap();
        let c = store.counts().await.unwrap();
        assert_eq!(c.embeddings_total, 0);
        assert_eq!(c.repos, 1);
        assert_eq!(store.indexed_text_hash("commit", "c1").await.unwrap(), None);
        assert!(store
            .code_file_shas("repo:acme/widget")
            .await
            .unwrap()
            .is_empty());
        assert!(store.get_repo("repo:acme/widget").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn file_health_stores_unanalyzed_rows_but_never_reports_them_as_scored() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap(); // file_health FKs repos
        store
            .upsert_file_health("repo:acme/widget", "src/a.rs", 120, 4, 9, Some(8))
            .await
            .unwrap();
        store
            .upsert_file_health("repo:acme/widget", "src/B.kt", 900, 0, 0, None)
            .await
            .unwrap();

        let rows = store.repo_file_health("repo:acme/widget").await.unwrap();
        assert_eq!(
            rows,
            vec![("src/a.rs".to_string(), 120, 4, 9, 8)],
            "the unanalyzed Kotlin file has no score, so it is not reported at all"
        );
        // The row is kept with its real LOC but no score.
        let unanalyzed: Vec<(String, i64, Option<i64>)> = sqlx::query_as(
            "SELECT path, loc, score FROM file_health WHERE repo_id = ? AND analyzed = 0",
        )
        .bind("repo:acme/widget")
        .fetch_all(&store.pool)
        .await
        .unwrap();
        assert_eq!(unanalyzed, vec![("src/B.kt".to_string(), 900, None)]);

        // Re-indexing the same path overwrites in place, including flipping analyzed back off.
        store
            .upsert_file_health("repo:acme/widget", "src/a.rs", 130, 5, 40, Some(6))
            .await
            .unwrap();
        assert_eq!(
            store.repo_file_health("repo:acme/widget").await.unwrap(),
            vec![("src/a.rs".to_string(), 130, 5, 40, 6)]
        );
        store
            .delete_file_health_path("repo:acme/widget", "src/a.rs")
            .await
            .unwrap();
        assert!(store
            .repo_file_health("repo:acme/widget")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn both_index_cleanup_paths_drop_file_health_rows() {
        async fn health_rows(store: &Store, repo_id: &str) -> i64 {
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM file_health WHERE repo_id = ?")
                .bind(repo_id)
                .fetch_one(&store.pool)
                .await
                .unwrap()
        }
        // clear_repo_code_index drops only the named repo's rows.
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        // The `repo` helper hardcodes one id, so the second repo is built by hand.
        store
            .upsert_repo(&Repo {
                id: "repo:acme/other".into(),
                owner: "acme".into(),
                name: "other".into(),
                full_name: "acme/other".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        store
            .upsert_file_health("repo:acme/widget", "a.rs", 10, 1, 1, Some(9))
            .await
            .unwrap();
        store
            .upsert_file_health("repo:acme/widget", "B.kt", 10, 0, 0, None)
            .await
            .unwrap();
        store
            .upsert_file_health("repo:acme/other", "b.rs", 10, 1, 1, Some(9))
            .await
            .unwrap();
        store
            .clear_repo_code_index("repo:acme/widget")
            .await
            .unwrap();
        assert_eq!(
            health_rows(&store, "repo:acme/widget").await,
            0,
            "both the scored and the unanalyzed row are dropped"
        );
        assert_eq!(
            health_rows(&store, "repo:acme/other").await,
            1,
            "the other repo's health rows survive"
        );

        // clear_index drops every repo's rows.
        store
            .upsert_file_health("repo:acme/widget", "a.rs", 10, 1, 1, Some(9))
            .await
            .unwrap();
        store.clear_index().await.unwrap();
        assert_eq!(health_rows(&store, "repo:acme/widget").await, 0);
        assert_eq!(health_rows(&store, "repo:acme/other").await, 0);
        assert!(
            store.get_repo("repo:acme/widget").await.unwrap().is_some(),
            "the source repo row is untouched"
        );
    }

    #[tokio::test]
    async fn code_file_shas_track_upsert_and_delete() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo("acme/widget")).await.unwrap();
        let rid = "repo:acme/widget";
        store
            .upsert_code_file(rid, "src/lib.rs", "sha1")
            .await
            .unwrap();
        store
            .upsert_code_file(rid, "src/main.rs", "sha2")
            .await
            .unwrap();
        let shas = store.code_file_shas(rid).await.unwrap();
        assert_eq!(shas.get("src/lib.rs"), Some(&"sha1".to_string()));
        assert_eq!(shas.len(), 2);

        // Upsert overwrites the tracked SHA in place (no duplicate row).
        store
            .upsert_code_file(rid, "src/lib.rs", "sha1b")
            .await
            .unwrap();
        let shas = store.code_file_shas(rid).await.unwrap();
        assert_eq!(shas.get("src/lib.rs"), Some(&"sha1b".to_string()));
        assert_eq!(shas.len(), 2);

        store.delete_code_file(rid, "src/main.rs").await.unwrap();
        let shas = store.code_file_shas(rid).await.unwrap();
        assert!(!shas.contains_key("src/main.rs"));
        assert_eq!(shas.len(), 1);
    }

    #[tokio::test]
    async fn teams_store_their_source_verbatim() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .add_team(&Team {
                id: "t1".into(),
                name: "Platform".into(),
                source: "manual".into(),
            })
            .await
            .unwrap();
        store
            .add_team(&Team {
                id: "t2".into(),
                name: "Imported".into(),
                source: "github".into(),
            })
            .await
            .unwrap();
        let teams = store.list_teams().await.unwrap();
        assert_eq!(teams[0].source, "manual");
        assert_eq!(teams[1].source, "github");
    }

    #[tokio::test]
    async fn work_item_status_history_replace_is_idempotent() {
        let store = Store::open_in_memory().await.unwrap();
        let h = vec![
            ("new", "2026-06-01T00:00:00Z".to_string()),
            ("done", "2026-06-05T00:00:00Z".to_string()),
        ];
        store
            .replace_work_item_status_history("wi1", &h)
            .await
            .unwrap();
        store
            .replace_work_item_status_history("wi1", &h)
            .await
            .unwrap(); // idempotent
        let got = store.work_item_status_history("wi1").await.unwrap();
        assert_eq!(
            got,
            vec![
                ("new".to_string(), "2026-06-01T00:00:00Z".to_string()),
                ("done".to_string(), "2026-06-05T00:00:00Z".to_string()),
            ]
        );
        // Re-deriving with a changed timestamp replaces wholesale.
        store
            .replace_work_item_status_history("wi1", &[("new", "2026-06-02T00:00:00Z".to_string())])
            .await
            .unwrap();
        assert_eq!(
            store.work_item_status_history("wi1").await.unwrap(),
            vec![("new".to_string(), "2026-06-02T00:00:00Z".to_string())]
        );
    }

    #[tokio::test]
    async fn work_item_links_are_queryable() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .add_work_item(&WorkItem {
                id: "wi1".into(),
                title: "Ship it".into(),
                kind: "story".into(),
                state: "open".into(),
                source: "manual".into(),
                status_category: None,
            })
            .await
            .unwrap();
        store
            .add_link("issue", "I1", "pull_request", "P1", "closes")
            .await
            .unwrap();
        // Idempotent: recording the same edge again does not duplicate it.
        store
            .add_link("issue", "I1", "pull_request", "P1", "closes")
            .await
            .unwrap();

        let links = store.links_from("issue", "I1").await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].dst_id, "P1");
        assert_eq!(links[0].relation, "closes");
    }

    #[tokio::test]
    async fn backfill_states_lists_pending_and_complete_walks() {
        // The observability query behind "is this repo still backfilling, or is sync broken?".
        let store = Store::open_in_memory().await.unwrap();
        let rid = "repo:default/acme/widget";
        store
            .set_sync_cursor(rid, "commits", "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_sync_cursor(
                rid,
                &backfill_entity("commits"),
                "bf:c:7:2019-01-01T00:00:00Z",
            )
            .await
            .unwrap();
        store
            .set_sync_cursor(rid, &backfill_entity("pulls"), BACKFILL_DONE)
            .await
            .unwrap();

        let states = store.backfill_states().await.unwrap();
        // The plain forward cursor is not a backfill row and must not show up here.
        assert_eq!(states.len(), 2);
        let commits = states.iter().find(|s| s.entity == "commits").unwrap();
        assert_eq!(commits.repo_id, rid);
        assert!(!commits.complete);
        assert_eq!(
            commits.resume.as_deref(),
            Some("bf:c:7:2019-01-01T00:00:00Z")
        );
        let pulls = states.iter().find(|s| s.entity == "pulls").unwrap();
        assert!(pulls.complete);
        assert_eq!(pulls.resume, None);
    }

    #[tokio::test]
    async fn sync_cursors_are_per_entity_and_advance() {
        let store = Store::open_in_memory().await.unwrap();
        let rid = "repo:acme/widget";
        assert_eq!(store.sync_cursor(rid, "issues").await.unwrap(), None);
        store.set_sync_cursor(rid, "issues", "T1").await.unwrap();
        store.set_sync_cursor(rid, "commits", "C1").await.unwrap();
        // Advancing one entity does not touch the other.
        store.set_sync_cursor(rid, "issues", "T2").await.unwrap();
        assert_eq!(
            store.sync_cursor(rid, "issues").await.unwrap(),
            Some("T2".to_string())
        );
        assert_eq!(
            store.sync_cursor(rid, "commits").await.unwrap(),
            Some("C1".to_string())
        );
        // A different repo has its own cursor space.
        assert_eq!(
            store.sync_cursor("repo:other", "issues").await.unwrap(),
            None
        );
    }

    // Sprint membership and a PR lookup by id.
    #[tokio::test]
    async fn sprint_membership_and_pr_lookup() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .add_sprint(&Sprint {
                id: "s1".into(),
                team_id: None,
                name: "Sprint 1".into(),
                starts_on: None,
                ends_on: None,
                source: "manual".into(),
            })
            .await
            .unwrap();
        store
            .add_work_item(&WorkItem {
                id: "wi1".into(),
                title: "Ship".into(),
                kind: "story".into(),
                state: "open".into(),
                source: "manual".into(),
                status_category: None,
            })
            .await
            .unwrap();
        store.add_sprint_work_item("s1", "wi1").await.unwrap();
        // Idempotent membership.
        store.add_sprint_work_item("s1", "wi1").await.unwrap();
        let items = store.work_items_for_sprint("s1").await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "wi1");

        // The repo must exist first: pull_requests.repo_id references repos(id) and sqlx enables
        // SQLite foreign keys.
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        store
            .upsert_pull_request(&PullRequest {
                id: "p1".into(),
                repo_id: "repo:acme/widget".into(),
                number: 1,
                title: "PR".into(),
                state: "closed".into(),
                author_login: None,
                body: None,
                created_at: "2026-06-01T00:00:00Z".into(),
                merged_at: Some("2026-06-02T00:00:00Z".into()),
                html_url: None,
            })
            .await
            .unwrap();
        assert_eq!(store.pull_request("p1").await.unwrap().unwrap().number, 1);
        assert!(store.pull_request("nope").await.unwrap().is_none());
    }

    // ---- board-scoped search ------------------------------------------------------------

    /// A repo with one code chunk, one commit and one issue, all carrying `text` so one query
    /// matches every kind. The `commits` / `issues` rows are how a scoped search attributes an
    /// activity chunk to a repo.
    async fn seed_searchable_repo(
        store: &Store,
        repo_id: &str,
        full_name: &str,
        n: &str,
        text: &str,
    ) {
        use core_embed::{Embedder, HashEmbedder};
        let embedder = HashEmbedder::default();
        let chunk = |t: &str| (t.to_string(), embedder.embed(t));
        let (owner, name) = full_name.split_once('/').unwrap();
        store
            .upsert_repo(&Repo {
                id: repo_id.into(),
                owner: owner.into(),
                name: name.into(),
                full_name: full_name.into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        store
            .upsert_commit(&Commit {
                sha: format!("sha{n}"),
                repo_id: repo_id.into(),
                author_login: Some("dev".into()),
                message: text.into(),
                committed_at: "2026-08-01T00:00:00Z".into(),
            })
            .await
            .unwrap();
        store
            .upsert_issue(&Issue {
                id: format!("is{n}"),
                repo_id: repo_id.into(),
                number: 1,
                title: text.into(),
                state: "open".into(),
                author_login: Some("dev".into()),
                body: None,
                created_at: "2026-08-01T00:00:00Z".into(),
                closed_at: None,
                labels: String::new(),
                html_url: None,
            })
            .await
            .unwrap();
        store
            .replace_embeddings("code", &format!("{repo_id}#src/lib.rs"), &[chunk(text)])
            .await
            .unwrap();
        store
            .replace_embeddings("commit", &format!("sha{n}"), &[chunk(text)])
            .await
            .unwrap();
        store
            .replace_embeddings("issue", &format!("is{n}"), &[chunk(text)])
            .await
            .unwrap();
    }

    /// Board scope: both repos hold an identical chunk of every indexed kind, so only the scope
    /// can separate them.
    #[tokio::test]
    async fn a_scoped_search_returns_only_the_scoped_repos_chunks() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        seed_searchable_repo(
            &store,
            "repo:gh/acme/inside",
            "acme/inside",
            "1",
            "parse the manifest",
        )
        .await;
        seed_searchable_repo(
            &store,
            "repo:gh/other/outside",
            "other/outside",
            "2",
            "parse the manifest",
        )
        .await;

        let q = HashEmbedder::default().embed("parse the manifest");
        let scope = ["repo:gh/acme/inside".to_string()];
        let scope = RepoScope::Repos(&scope);

        // Code: the in-scope file comes back, the out-of-scope one cannot.
        let code = store
            .hybrid_search(&q, "parse the manifest", 10, KindFilter::Is("code"), scope)
            .await
            .unwrap();
        assert!(
            code.iter()
                .any(|h| h.ref_id == "repo:gh/acme/inside#src/lib.rs"),
            "the board's own code must still be findable: {code:?}"
        );
        assert!(
            code.iter()
                .all(|h| !h.ref_id.starts_with("repo:gh/other/outside")),
            "a repo outside the board must not surface: {code:?}"
        );

        // Activity: commit and issue chunks are attributed through `commits` / `issues`, not the
        // ref_id.
        let activity = store
            .hybrid_search(&q, "parse the manifest", 10, KindFilter::Not("code"), scope)
            .await
            .unwrap();
        let ids: Vec<&str> = activity.iter().map(|h| h.ref_id.as_str()).collect();
        assert!(
            ids.contains(&"sha1") && ids.contains(&"is1"),
            "in-scope activity: {ids:?}"
        );
        assert!(
            !ids.contains(&"sha2") && !ids.contains(&"is2"),
            "out-of-scope activity leaked: {ids:?}"
        );

        // Unscoped: `RepoScope::All` still sees both repos (the diagnostic path).
        let all = store
            .hybrid_search(
                &q,
                "parse the manifest",
                10,
                KindFilter::Is("code"),
                RepoScope::All,
            )
            .await
            .unwrap();
        assert!(all
            .iter()
            .any(|h| h.ref_id.starts_with("repo:gh/other/outside")));
    }

    /// The scope goes in a `rowid IN (...)` constraint, not the join's `WHERE`. The out-of-scope
    /// repo owns the 45 nearest chunks; the board owns 5 far ones. vec0 applies `rowid IN` inside
    /// the KNN, so asking for 5 returns the board's 5. A post-filter over the globally nearest 5
    /// would return nothing.
    #[tokio::test]
    async fn scoped_search_reaches_past_the_global_k() {
        let store = Store::open_in_memory().await.unwrap();
        // Unit vectors on two axes: `near` sits on the query's own axis, `far` is orthogonal to it.
        let axis = |a: usize| {
            let mut v = vec![0f32; 768];
            v[a] = 1.0;
            v
        };
        for n in 0..45 {
            store
                .replace_embeddings(
                    "code",
                    &format!("repo:gh/other/outside#f{n}.rs"),
                    &[(format!("near {n}"), axis(0))],
                )
                .await
                .unwrap();
        }
        for n in 0..5 {
            store
                .replace_embeddings(
                    "code",
                    &format!("repo:gh/acme/inside#f{n}.rs"),
                    &[(format!("far {n}"), axis(1))],
                )
                .await
                .unwrap();
        }

        let scope = ["repo:gh/acme/inside".to_string()];
        let hits = store
            .vector_search(
                &axis(0),
                5,
                KindFilter::Is("code"),
                RepoScope::Repos(&scope),
            )
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            5,
            "the scope must be applied inside the KNN, not after it: {hits:?}"
        );
        assert!(hits
            .iter()
            .all(|h| h.ref_id.starts_with("repo:gh/acme/inside#")));
    }

    /// Code chunks are keyed `<repo_id>#<path>`, so scoping to `my_repo` must not admit `my-repo`
    /// (`LIKE` would, since `_` matches any character).
    #[tokio::test]
    async fn scoping_tells_apart_repo_ids_that_differ_by_an_underscore() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        let embedder = HashEmbedder::default();
        let chunk = |t: &str| (t.to_string(), embedder.embed(t));
        for id in ["repo:gh/acme/my_repo", "repo:gh/acme/my-repo"] {
            store
                .replace_embeddings(
                    "code",
                    &format!("{id}#src/lib.rs"),
                    &[chunk("open the store")],
                )
                .await
                .unwrap();
        }

        let scope = ["repo:gh/acme/my_repo".to_string()];
        let hits = store
            .hybrid_search(
                &embedder.embed("open the store"),
                "open the store",
                10,
                KindFilter::Is("code"),
                RepoScope::Repos(&scope),
            )
            .await
            .unwrap();
        assert_eq!(
            hits.iter().map(|h| h.ref_id.as_str()).collect::<Vec<_>>(),
            vec!["repo:gh/acme/my_repo#src/lib.rs"],
            "a neighbour that differs only by an underscore must not be swept in"
        );
    }

    /// A board with no pinned and no discovered repos has nothing to search: the empty scope must
    /// return empty even though the index is full of matches.
    #[tokio::test]
    async fn an_empty_scope_searches_nothing_rather_than_everything() {
        use core_embed::{Embedder, HashEmbedder};
        let store = Store::open_in_memory().await.unwrap();
        seed_searchable_repo(
            &store,
            "repo:gh/acme/inside",
            "acme/inside",
            "1",
            "retry the request",
        )
        .await;
        let q = HashEmbedder::default().embed("retry the request");

        let empty: [String; 0] = [];
        assert!(store
            .hybrid_search(
                &q,
                "retry the request",
                10,
                KindFilter::Any,
                RepoScope::Repos(&empty)
            )
            .await
            .unwrap()
            .is_empty());
        assert!(store
            .vector_search(&q, 10, KindFilter::Any, RepoScope::Repos(&empty))
            .await
            .unwrap()
            .is_empty());
        // The same query against the whole store finds a hit, so the emptiness is the scope.
        assert!(!store
            .hybrid_search(&q, "retry the request", 10, KindFilter::Any, RepoScope::All)
            .await
            .unwrap()
            .is_empty());
    }

    // ---- storage accounting ------------------------------------------------------------

    /// Push a large body into a repo's index: `n` code chunks of roughly `bytes` each, plus their
    /// `code_files` rows. Bulky because `dbstat` reports whole pages, so a few short rows would
    /// not move the measurement.
    async fn seed_bulk_chunks(store: &Store, repo_id: &str, n: usize, bytes: usize) {
        use core_embed::{Embedder, HashEmbedder};
        let e = HashEmbedder::default();
        for i in 0..n {
            let text = format!("fn f{i:04}() {{ /* {} */ }}", "x".repeat(bytes));
            let path = format!("src/f{i}.rs");
            store
                .upsert_code_file(repo_id, &path, "blob")
                .await
                .unwrap();
            store
                .replace_embeddings(
                    "code",
                    &format!("{repo_id}#{path}"),
                    &[(text.clone(), e.embed(&text))],
                )
                .await
                .unwrap();
        }
    }

    /// One table's reported bytes in a storage report, or 0 when it is not listed.
    fn table_bytes(report: &StorageReport, name: &str) -> i64 {
        report
            .tables
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.bytes)
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn storage_tracks_data_arriving_and_leaving() {
        // The numbers follow reality in both directions: measure empty, insert a known body,
        // measure again, forget the repo, measure a third time. Constants or cached figures cannot
        // rise on insert and fall on delete.
        let store = Store::open_in_memory().await.unwrap();
        let empty = store.storage().await.unwrap();

        seed_repo(&store, "repo:gh/acme/big", "acme/big", "A").await;
        seed_bulk_chunks(&store, "repo:gh/acme/big", 200, 900).await;
        let full = store.storage().await.unwrap();
        assert!(
            full.reserved_bytes > empty.reserved_bytes,
            "the database did not grow: {} -> {}",
            empty.reserved_bytes,
            full.reserved_bytes
        );
        assert!(
            full.attributed_bytes > empty.attributed_bytes + 500_000,
            "200 chunks of ~900 bytes plus their vectors should be well over half a megabyte, got {}",
            full.attributed_bytes - empty.attributed_bytes
        );
        assert!(
            table_bytes(&full, "embeddings") > table_bytes(&empty, "embeddings"),
            "the chunk text is not attributed to `embeddings`"
        );

        store.forget_repo("repo:gh/acme/big").await.unwrap();
        let after = store.storage().await.unwrap();
        assert!(
            after.attributed_bytes < full.attributed_bytes,
            "forgetting the repo did not reduce the attributed bytes: {} -> {}",
            full.attributed_bytes,
            after.attributed_bytes
        );
        assert!(
            after.free_bytes > full.free_bytes,
            "the released pages did not land on the free list: {} -> {}",
            full.free_bytes,
            after.free_bytes
        );
        // The file does not shrink on a delete: SQLite keeps the pages and reuses them.
        assert_eq!(
            after.reserved_bytes, full.reserved_bytes,
            "reserved bytes should not fall without a VACUUM"
        );
    }

    #[tokio::test]
    async fn storage_attributes_the_vector_and_full_text_indexes() {
        // `vec_embeddings` (vec0) and `fts_embeddings` (FTS5) report nothing useful when queried
        // directly; their bytes live in shadow tables (`vec_embeddings_vector_chunks00`,
        // `fts_embeddings_data`, ...). Reading the virtual tables by name, or leaving the shadow
        // tables out of the roll-up, reports 0.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/big", "acme/big", "A").await;
        seed_bulk_chunks(&store, "repo:gh/acme/big", 200, 900).await;

        let full = store.storage().await.unwrap();
        let vec_bytes = table_bytes(&full, "vec_embeddings");
        let fts_bytes = table_bytes(&full, "fts_embeddings");
        assert!(vec_bytes > 0, "the vector index was attributed nothing");
        assert!(fts_bytes > 0, "the full-text index was attributed nothing");
        // 203 chunks of 768 f32s is ~623 KB of raw vector before any vec0 overhead, so a figure
        // far below that would mean the vector chunk shadow table was missed.
        assert!(
            vec_bytes > 600_000,
            "the vector index looks too small to include its vector chunks: {vec_bytes}"
        );
        // No shadow table appears on its own; each is reported under its owner.
        assert!(
            !full
                .tables
                .iter()
                .any(|t| t.name.starts_with("vec_embeddings_")
                    || t.name.starts_with("fts_embeddings_")),
            "shadow tables leaked into the breakdown: {:?}",
            full.tables.iter().map(|t| &t.name).collect::<Vec<_>>()
        );

        store.forget_repo("repo:gh/acme/big").await.unwrap();
        let after = store.storage().await.unwrap();
        assert!(
            table_bytes(&after, "vec_embeddings") < vec_bytes,
            "the vector index did not shrink when its rows went"
        );
        assert!(
            table_bytes(&after, "fts_embeddings") < fts_bytes,
            "the full-text index did not shrink when its rows went"
        );
    }

    #[tokio::test]
    async fn storage_reports_dbstat_on_this_build() {
        // `dbstat` is a compile-time option. sqlx pulls libsqlite3-sys with `bundled`, which
        // passes -DSQLITE_ENABLE_DBSTAT_VTAB. If a dependency bump drops it, this test fails and
        // the fallback below is what ships.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/big", "acme/big", "A").await;
        let report = store.storage().await.unwrap();
        assert_eq!(report.method, StorageMethod::Dbstat);
        assert!(
            report.tables.iter().all(|t| !t.estimated),
            "a measured breakdown marked a table estimated"
        );
        assert!(
            report.groups.iter().all(|g| !g.estimated),
            "a measured breakdown marked a group estimated"
        );
        // `dbstat` sees every page not on the free list, so a measured breakdown accounts for
        // essentially the whole file. A wide gap would mean whole btrees were dropped.
        assert!(
            report.residual_bytes.abs() < report.page_size * 2,
            "measured attribution left {} bytes unexplained",
            report.residual_bytes
        );
    }

    #[tokio::test]
    async fn storage_estimate_fallback_still_answers() {
        // The fallback is for builds without `dbstat`; every build here has it, so call it
        // directly. It must produce the same shape, label every figure an estimate, and report a
        // search-index figure by scanning the virtual tables' shadow storage.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/big", "acme/big", "A").await;
        seed_bulk_chunks(&store, "repo:gh/acme/big", 200, 900).await;

        let measured = store.storage().await.unwrap();
        let est = store.storage_estimated().await.unwrap();
        assert_eq!(est.method, StorageMethod::PayloadEstimate);
        assert!(
            est.tables.iter().all(|t| t.estimated),
            "an estimated breakdown claimed a measured table"
        );
        assert!(
            est.groups.iter().all(|g| g.estimated),
            "an estimated breakdown claimed a measured group"
        );
        // The whole-file figures come from the same pragmas either way, so they must agree.
        assert_eq!(est.reserved_bytes, measured.reserved_bytes);
        assert_eq!(est.page_size, measured.page_size);
        assert!(!est.tables.is_empty(), "the fallback attributed nothing");
        assert!(
            table_bytes(&est, "vec_embeddings") > 0 && table_bytes(&est, "fts_embeddings") > 0,
            "the fallback missed the search index's shadow storage"
        );
        // Payload only: it cannot see indexes or page overhead, so it must come in under the
        // measured figure.
        assert!(
            est.attributed_bytes < measured.attributed_bytes,
            "the payload estimate ({}) exceeded the measured attribution ({})",
            est.attributed_bytes,
            measured.attributed_bytes
        );
    }

    #[tokio::test]
    async fn storage_breakdown_reconciles() {
        // The parts must reconstruct the whole under both methods: the residual is not clamped at
        // zero and comes only from the reported figures.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/big", "acme/big", "A").await;
        seed_bulk_chunks(&store, "repo:gh/acme/big", 50, 400).await;
        for report in [
            store.storage().await.unwrap(),
            store.storage_estimated().await.unwrap(),
        ] {
            assert_eq!(
                report.attributed_bytes + report.free_bytes + report.residual_bytes,
                report.reserved_bytes,
                "{:?} does not reconcile",
                report.method
            );
            assert_eq!(
                report.reserved_bytes,
                report.page_count * report.page_size,
                "reserved bytes are not page_count * page_size"
            );
            let grouped: i64 = report.groups.iter().map(|g| g.bytes).sum();
            assert_eq!(
                grouped, report.attributed_bytes,
                "the group roll-up lost bytes"
            );
        }
    }

    #[tokio::test]
    async fn every_table_is_classified() {
        // The group map is hand-written. A table added without classification lands in `Other` and
        // fails here, naming it.
        let store = Store::open_in_memory().await.unwrap();
        let mut names = store.table_names().await.unwrap();
        // The btrees `table_names` filters out but a report still has to classify.
        names.extend(["_sqlx_migrations", "sqlite_schema", "sqlite_sequence"].map(str::to_string));
        let unclassified: Vec<&String> = names
            .iter()
            .filter(|n| classify(n) == StorageGroup::Other)
            .collect();
        assert!(
            unclassified.is_empty(),
            "these tables are in no storage group: {unclassified:?}"
        );
    }

    /// `table_group` applied as a report applies it, through the shadow-table roll-up first, so
    /// `fts_embeddings_data` is classified as the search index.
    fn classify(name: &str) -> StorageGroup {
        table_group(crate::storage::owning_table(name))
    }

    #[tokio::test]
    async fn repo_storage_separates_repos() {
        // Per-repo attribution must distinguish repos: a global total or a wrong-column key would
        // give both repos the same number or take the neighbour down on a forget.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/big", "acme/big", "A").await;
        seed_repo(&store, "repo:gh/acme/small", "acme/small", "B").await;
        seed_bulk_chunks(&store, "repo:gh/acme/big", 100, 900).await;
        seed_bulk_chunks(&store, "repo:gh/acme/small", 5, 40).await;

        let report = store.storage().await.unwrap();
        let big = report
            .repos
            .iter()
            .find(|r| r.repo_id == "repo:gh/acme/big")
            .unwrap()
            .clone();
        let small = report
            .repos
            .iter()
            .find(|r| r.repo_id == "repo:gh/acme/small")
            .unwrap()
            .clone();
        assert!(
            big.estimated_bytes > small.estimated_bytes * 5,
            "the bigger repo is not reported bigger: {} vs {}",
            big.estimated_bytes,
            small.estimated_bytes
        );
        assert_eq!(
            report.repos.first().map(|r| r.repo_id.as_str()),
            Some("repo:gh/acme/big"),
            "the list is not ordered by what a repo costs"
        );
        assert_eq!((big.commits, big.pull_requests, big.issues), (1, 1, 1));
        assert_eq!(big.code_files, 101, "100 bulk files plus the seeded one");

        store.forget_repo("repo:gh/acme/big").await.unwrap();
        let after = store.storage().await.unwrap();
        assert_eq!(after.repos.len(), 1, "the forgotten repo is still listed");
        assert_eq!(
            after.repos[0], small,
            "forgetting one repo changed the other's figures"
        );
    }

    #[tokio::test]
    async fn vector_bytes_follow_the_chunk_count() {
        // The vector payload is the one exact per-repo figure, derived from the store's real chunk
        // count. A hard-coded figure, or the code-chunk count alone, misses the commit and issue
        // chunks.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/big", "acme/big", "A").await;
        seed_bulk_chunks(&store, "repo:gh/acme/big", 7, 100).await;

        let report = store.storage().await.unwrap();
        let repo = &report.repos[0];
        // 7 bulk code chunks + the seed's code, commit, and issue chunk.
        assert_eq!(repo.index_chunks, 10);
        assert_eq!(report.embed_dim, core_embed::EMBED_DIM as i64);
        assert_eq!(
            repo.vector_bytes,
            repo.index_chunks * core_embed::EMBED_DIM as i64 * 4
        );
        let (total,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM embeddings")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        assert_eq!(
            repo.index_chunks, total,
            "the repo's chunk count disagrees with the index"
        );
        assert_eq!(repo.estimated_bytes, repo.content_bytes + repo.vector_bytes);
    }

    // ---- the storage budget ------------------------------------------------------------

    #[tokio::test]
    async fn storage_usage_tracks_what_is_stored() {
        // The budget is enforced against this number, so it must follow the database. Cheap: three
        // pragmas and two stats, no dbstat walk, since it runs on the sync path.
        let store = Store::open_in_memory().await.unwrap();
        let empty = store.storage_usage().await.unwrap();

        seed_repo(&store, "repo:gh/acme/big", "acme/big", "A").await;
        seed_bulk_chunks(&store, "repo:gh/acme/big", 200, 900).await;
        let full = store.storage_usage().await.unwrap();

        assert!(
            full.used_bytes > empty.used_bytes + 500_000,
            "200 chunks of ~900 bytes plus their vectors should move usage well past half a \
             megabyte: {} -> {}",
            empty.used_bytes,
            full.used_bytes
        );
        // An in-memory store has no file to stat, so usage falls back to the reserved pages.
        assert_eq!(full.db_bytes, None);
        assert_eq!(full.used_bytes, full.reserved_bytes);
        assert!(full.free_bytes >= 0 && full.free_bytes < full.reserved_bytes);
    }

    #[test]
    fn pressure_steps_at_the_two_thresholds() {
        // Three states from one number: warn at 80%, stop at 100%, never stop a user with no
        // limit. Boundaries are inclusive; 79 and 99 catch an off-by-one.
        let budget = 1_000i64;
        let at = |pct: i64| StoragePressure::for_usage(pct * budget / 100, Some(budget));
        assert_eq!(at(0), StoragePressure::Normal);
        assert_eq!(at(79), StoragePressure::Normal);
        assert_eq!(at(80), StoragePressure::Warning);
        assert_eq!(at(99), StoragePressure::Warning);
        assert_eq!(at(100), StoragePressure::Full);
        assert_eq!(at(150), StoragePressure::Full);

        // No budget, or a nonsensical one, means no limit.
        assert_eq!(
            StoragePressure::for_usage(i64::MAX, None),
            StoragePressure::Normal
        );
        assert_eq!(
            StoragePressure::for_usage(i64::MAX, Some(0)),
            StoragePressure::Normal
        );
        // Only the ceiling stops the lanes; the warning is a warning.
        let state = |pressure| BudgetState {
            pressure,
            usage: StorageUsage {
                db_bytes: None,
                wal_bytes: None,
                free_bytes: 0,
                reserved_bytes: 0,
                used_bytes: 0,
            },
            budget_bytes: Some(budget),
        };
        assert!(state(StoragePressure::Normal).allows_sync());
        assert!(state(StoragePressure::Warning).allows_sync());
        assert!(!state(StoragePressure::Full).allows_sync());
    }

    #[tokio::test]
    async fn the_budget_round_trips_through_settings() {
        // A default on a fresh store, a value that persists, and an explicit no-limit that
        // persists as null instead of reverting to the default.
        let store = Store::open_in_memory().await.unwrap();
        let fresh = store.settings().await.unwrap();
        assert_eq!(fresh.storage_budget_mb, Some(5120));

        let write = |mb| {
            let store = &store;
            async move {
                let mut s = store.settings().await.unwrap();
                s.storage_budget_mb = mb;
                store.update_settings(&s).await.unwrap();
                store.settings().await.unwrap().storage_budget_mb
            }
        };
        assert_eq!(write(Some(250)).await, Some(250));
        assert_eq!(write(None).await, None);

        // And the state derived from it: no budget never stops, a tiny budget does.
        assert_eq!(
            store.budget_state().await.unwrap().pressure,
            StoragePressure::Normal
        );
        assert_eq!(write(Some(1)).await, Some(1));
        let state = store.budget_state().await.unwrap();
        assert_eq!(state.budget_bytes, Some(MB));
        assert_eq!(state.pressure, StoragePressure::Normal);
    }

    #[test]
    fn every_storage_group_states_whether_it_is_refetchable() {
        // The storage rule as code: a thing may be discarded only if the app can get it back from
        // a forge on its own. Metric history cannot be back-filled and configuration is the
        // user's, so neither is ever eligible, at any budget pressure.
        for g in [
            StorageGroup::SearchIndex,
            StorageGroup::ChunkText,
            StorageGroup::Code,
            StorageGroup::Activity,
        ] {
            assert!(g.refetchable(), "{} should be re-fetchable", g.as_str());
        }
        for g in [
            StorageGroup::History,
            StorageGroup::Config,
            StorageGroup::Internal,
            // An unclassified table is not a table this app may delete.
            StorageGroup::Other,
        ] {
            assert!(!g.refetchable(), "{} must not be discardable", g.as_str());
        }
        // The tables that carry the two irreplaceable things must land in the two ineligible
        // groups.
        for table in [
            "metric_snapshots",
            "digests",
            "work_item_status_history",
            "sprint_commitments",
        ] {
            assert!(
                !table_group(table).refetchable(),
                "{table} holds history that no re-sync brings back"
            );
        }
        for table in ["boards", "sources", "forges", "settings", "board_repos"] {
            assert!(
                !table_group(table).refetchable(),
                "{table} is the user's own configuration"
            );
        }
    }

    #[tokio::test]
    async fn dropping_a_repo_index_takes_the_index_and_nothing_else() {
        // Reclaim drops the chunks, vectors, full-text rows and per-file markers; the next sync
        // rebuilds them. The repo's activity and daily metric snapshots stay.
        // The two repo ids differ by one character (`_` against `-`): a `LIKE`-based prefix would
        // take the neighbour's index too.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/my_repo", "acme/my_repo", "A").await;
        seed_repo(&store, "repo:gh/acme/my-repo", "acme/my-repo", "B").await;
        seed_bulk_chunks(&store, "repo:gh/acme/my_repo", 30, 200).await;
        seed_bulk_chunks(&store, "repo:gh/acme/my-repo", 30, 200).await;
        store
            .upsert_metric_snapshot(&MetricSnapshot {
                scope_kind: "repo".into(),
                scope_id: "repo:gh/acme/my_repo".into(),
                captured_on: "2026-08-01".into(),
                wip: 3,
                stale_open_prs: 0,
                merged_without_review: 0,
                attention_count: 0,
                median_cycle_time_secs: None,
                median_pickup_secs: None,
                median_review_secs: None,
            })
            .await
            .unwrap();

        let count = |sql: &'static str| {
            let store = &store;
            async move {
                let (n,): (i64,) = sqlx::query_as(sql).fetch_one(&store.pool).await.unwrap();
                n
            }
        };
        let chunks_of = |repo: &'static str| {
            let store = &store;
            async move {
                let (n,): (i64,) = sqlx::query_as(
                    "SELECT COUNT(*) FROM embeddings \
                     WHERE ref_kind = 'code' AND substr(ref_id, 1, length(?)) = ?",
                )
                .bind(format!("{repo}#"))
                .bind(format!("{repo}#"))
                .fetch_one(&store.pool)
                .await
                .unwrap();
                n
            }
        };
        let before_neighbour = chunks_of("repo:gh/acme/my-repo").await;
        assert!(before_neighbour > 0);
        let vectors_before = count("SELECT COUNT(*) FROM vec_embeddings").await;
        // The long snapshot format writes one row per metric per day, so count rather than assume;
        // the number must not change.
        let snapshots_before =
            count("SELECT COUNT(*) FROM metric_snapshots WHERE scope_id = 'repo:gh/acme/my_repo'")
                .await;
        assert!(snapshots_before > 0);

        let dropped = store.drop_repo_index("repo:gh/acme/my_repo").await.unwrap();
        assert!(dropped.found);
        assert!(dropped.chunks > 30, "chunks removed: {}", dropped.chunks);
        // 30 from the bulk seed plus the one `src/lib.rs` that `seed_repo` indexes.
        assert_eq!(dropped.code_files, 31);

        // The index is gone for this repo...
        assert_eq!(chunks_of("repo:gh/acme/my_repo").await, 0);
        assert_eq!(
            count("SELECT COUNT(*) FROM code_files WHERE repo_id = 'repo:gh/acme/my_repo'").await,
            0
        );
        assert!(count("SELECT COUNT(*) FROM vec_embeddings").await < vectors_before);
        // ...its activity is not...
        assert_eq!(
            count("SELECT COUNT(*) FROM commits WHERE repo_id = 'repo:gh/acme/my_repo'").await,
            1
        );
        assert_eq!(
            count("SELECT COUNT(*) FROM repos WHERE id = 'repo:gh/acme/my_repo'").await,
            1
        );
        // ...and neither is the one thing no re-sync could rebuild.
        assert_eq!(
            count("SELECT COUNT(*) FROM metric_snapshots WHERE scope_id = 'repo:gh/acme/my_repo'")
                .await,
            snapshots_before,
            "a metric snapshot was discarded by an action that may only touch re-fetchable data"
        );
        // ...and the neighbour whose id differs by one character keeps its index.
        assert_eq!(chunks_of("repo:gh/acme/my-repo").await, before_neighbour);
    }

    #[tokio::test]
    async fn reclaiming_free_space_shrinks_the_file_and_keeps_the_rows() {
        // A delete does not shrink a SQLite file: pages go on the free list and the file stays the
        // same size. Without a compacting step a user could drop every index and still be over the
        // ceiling.
        // File-backed because the point is the size on disk. A no-op reclaim leaves the size
        // unchanged and `freed_bytes` at zero; a reclaim that dropped rows fails the survivor
        // check.
        let path = format!(
            "{}/orgonzola-reclaim-{}.db",
            std::env::temp_dir().to_string_lossy(),
            std::process::id()
        );
        for p in [path.clone(), format!("{path}-wal"), format!("{path}-shm")] {
            let _ = std::fs::remove_file(&p);
        }

        let store = Store::open(&path).await.unwrap();
        seed_repo(&store, "repo:gh/acme/gone", "acme/gone", "A").await;
        seed_repo(&store, "repo:gh/acme/stays", "acme/stays", "B").await;
        seed_bulk_chunks(&store, "repo:gh/acme/gone", 400, 900).await;
        seed_bulk_chunks(&store, "repo:gh/acme/stays", 20, 900).await;

        let grown = store.storage_usage().await.unwrap();
        store.drop_repo_index("repo:gh/acme/gone").await.unwrap();
        let deleted = store.storage_usage().await.unwrap();
        assert!(
            deleted.free_bytes > 500_000,
            "the dropped index did not land on the free list: {}",
            deleted.free_bytes
        );

        let report = store.reclaim_free_space().await.unwrap();
        let after = store.storage_usage().await.unwrap();
        assert!(
            report.freed_bytes > 500_000,
            "reclaim reported {} freed, from {} of free space",
            report.freed_bytes,
            deleted.free_bytes
        );
        assert!(
            after.used_bytes < grown.used_bytes,
            "the file did not shrink: {} -> {}",
            grown.used_bytes,
            after.used_bytes
        );
        assert!(after.free_bytes < deleted.free_bytes);
        assert_eq!(report.used_before - report.used_after, report.freed_bytes);

        // A VACUUM rewrites the file; it does not delete anything.
        let (survivors,): (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM embeddings \
             WHERE ref_kind = 'code' AND substr(ref_id, 1, length(?)) = ?",
        )
        .bind("repo:gh/acme/stays#")
        .bind("repo:gh/acme/stays#")
        .fetch_one(&store.pool)
        .await
        .unwrap();
        // 20 bulk chunks plus `seed_repo`'s own `src/lib.rs`, all untouched by the VACUUM.
        assert_eq!(survivors, 21);
        assert_eq!(store.count_repos().await.unwrap(), 2);

        drop(store);
        for p in [path.clone(), format!("{path}-wal"), format!("{path}-shm")] {
            let _ = std::fs::remove_file(&p);
        }
    }

    #[tokio::test]
    async fn the_budget_message_says_which_state_it_is_in() {
        // The three banner messages must differ, name the numbers they are about, and the stopped
        // one must say nothing was deleted.
        let store = Store::open_in_memory().await.unwrap();
        seed_repo(&store, "repo:gh/acme/big", "acme/big", "A").await;
        seed_bulk_chunks(&store, "repo:gh/acme/big", 200, 900).await;

        let with_budget = |mb: Option<i64>| {
            let store = &store;
            async move {
                let mut s = store.settings().await.unwrap();
                s.storage_budget_mb = mb;
                store.update_settings(&s).await.unwrap();
                store.budget_state().await.unwrap()
            }
        };

        let unlimited = with_budget(None).await;
        assert_eq!(unlimited.pressure, StoragePressure::Normal);
        assert!(unlimited.message().contains("No storage budget is set"));

        // The store is a couple of megabytes here, so these budgets put it in each band.
        let used = unlimited.usage.used_bytes;
        let normal = with_budget(Some(used / MB * 4 + 4)).await;
        let warning = with_budget(Some((used as f64 / 0.85) as i64 / MB)).await;
        let full = with_budget(Some(1)).await;
        assert_eq!(normal.pressure, StoragePressure::Normal);
        assert_eq!(warning.pressure, StoragePressure::Warning);
        assert_eq!(full.pressure, StoragePressure::Full);

        let (n, w, f) = (normal.message(), warning.message(), full.message());
        assert_ne!(n, w);
        assert_ne!(w, f);
        assert!(w.contains("Nothing has stopped yet"), "{w}");
        assert!(f.contains("has stopped syncing and indexing"), "{f}");
        assert!(f.contains("Nothing has been deleted"), "{f}");
        // Every message names what is being used and against what.
        for m in [&n, &w, &f] {
            assert!(m.starts_with("Using "), "{m}");
            assert!(m.contains("storage budget"), "{m}");
        }
    }
}
