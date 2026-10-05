//! Storage accounting: bytes per table, group and repo, each labelled as measured or estimated.
//! `dbstat` (`SQLITE_ENABLE_DBSTAT_VTAB`) is a compile-time option, so its presence is probed at
//! run time. It is on in this build (bundled `libsqlite3-sys`). [`StorageReport::method`] says
//! which path ran; the fallback marks every table figure an estimate.
//! `vec_embeddings` (vec0) and `fts_embeddings` (FTS5) cannot be sized by querying them. `dbstat`
//! reports their shadow tables and [`owning_table`] rolls those up into the owning virtual table.
//! Per-repo bytes are not available from SQLite (pages belong to tables, not rows), so the
//! per-repo figures are summed content lengths, always estimates. They order repos against each
//! other and do not reconcile against the file size.

use serde::{Deserialize, Serialize};
use sqlx::Row;

use crate::{Store, StoreError};

/// How a storage breakdown was obtained. Carried in the report so an estimate is not shown as a
/// measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageMethod {
    /// Real per-btree page counts from SQLite's `dbstat` virtual table. Measured.
    Dbstat,
    /// Summed byte length of the stored values, per table. An estimate: it sees payload only, so
    /// it misses every index, every page header, and all slack inside pages.
    PayloadEstimate,
}

/// What a table's bytes are for. [`StorageGroup::Other`] marks an unclassified table; a test fails
/// when a migration adds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageGroup {
    /// The vector and full-text indexes: `vec_embeddings` and `fts_embeddings` with their shadow
    /// tables.
    SearchIndex,
    /// The chunk text those indexes point back to, and the hash markers that avoid re-embedding it.
    ChunkText,
    /// Synced forge activity: commits, pull requests, issues, reviews, CI, work items.
    Activity,
    /// Source code and what is derived from it: the file list, file health, dependencies,
    /// ownership.
    Code,
    /// Daily metric snapshots and digests. The one group a re-sync cannot rebuild.
    History,
    /// What the user configured plus the catalogue it points at: boards, sources, forges, repos,
    /// people.
    Config,
    /// SQLite's own bookkeeping and the migration ledger.
    Internal,
    /// Unclassified. A table that reached here is a gap in the map, not a category.
    Other,
}

impl StorageGroup {
    /// Whether the app could get this group's bytes back from a forge on its own.
    /// Bounds what any action the app offers may discard. `History` (daily snapshots) cannot be
    /// rebuilt by a re-sync, `Config` is user data, `Internal` is SQLite and the migration ledger,
    /// and `Other` is false because an unclassified table must not be deleted.
    pub fn refetchable(self) -> bool {
        match self {
            // Derived from the forge's blobs and the store's own rows; the next sync rebuilds them.
            Self::SearchIndex | Self::ChunkText | Self::Code => true,
            // Commits, pull requests, issues, reviews, CI: a re-sync pulls what the forge still
            // has.
            Self::Activity => true,
            Self::History | Self::Config | Self::Internal | Self::Other => false,
        }
    }

    /// A stable wire/display name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SearchIndex => "search_index",
            Self::ChunkText => "chunk_text",
            Self::Activity => "activity",
            Self::Code => "code",
            Self::History => "history",
            Self::Config => "config",
            Self::Internal => "internal",
            Self::Other => "other",
        }
    }
}

/// One table's share of the database.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableStorage {
    /// The owning table. Shadow tables are reported under their virtual table, indexes under
    /// their table.
    pub name: String,
    pub group: StorageGroup,
    pub bytes: i64,
    /// True when `bytes` came from summed value lengths rather than from page counts.
    pub estimated: bool,
}

/// One group's rolled-up share of the database.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupStorage {
    pub group: StorageGroup,
    pub bytes: i64,
    pub estimated: bool,
}

/// What one repo costs, as an estimate. Content bytes, not comparable to the database page figures.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoStorage {
    pub repo_id: String,
    pub full_name: String,
    pub commits: i64,
    pub pull_requests: i64,
    pub issues: i64,
    pub code_files: i64,
    /// Chunks this repo has in the search index, across code, commit, and issue kinds.
    pub index_chunks: i64,
    /// Every row this repo owns in the tables below, across all of them.
    pub rows: i64,
    /// Summed byte length of the values stored in the repo's activity, code, and chunk-text rows.
    /// Payload only: no index, no page overhead, and not the vector data (which is not text).
    pub content_bytes: i64,
    /// `index_chunks * EMBED_DIM * 4`. Raw float payload only; excludes vec0 chunk headers, rowid
    /// map and metadata, so it is smaller than the `dbstat` size of `vec_embeddings`.
    pub vector_bytes: i64,
    /// `content_bytes + vector_bytes`. Sort key for the repo list. An estimate.
    pub estimated_bytes: i64,
}

/// The whole storage picture for the database. Assets outside the database are the host's to
/// report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageReport {
    /// Which path produced `tables` and `groups`.
    pub method: StorageMethod,
    /// The main database file's size on disk, or `None` for an in-memory database.
    pub file_bytes: Option<i64>,
    /// The write-ahead log beside it, if present. It can be a large share of the total.
    pub wal_bytes: Option<i64>,
    /// The shared-memory index beside it, if present.
    pub shm_bytes: Option<i64>,
    pub page_size: i64,
    pub page_count: i64,
    pub freelist_pages: i64,
    /// `page_count * page_size`: what SQLite has claimed inside the file. Measured under both
    /// methods.
    pub reserved_bytes: i64,
    /// `freelist_pages * page_size`: claimed but holding nothing. Only a `VACUUM` returns them to
    /// the file system.
    pub free_bytes: i64,
    /// The sum of `tables`.
    pub attributed_bytes: i64,
    /// `reserved - attributed - free`. Near zero under `dbstat`; large under the estimate, which
    /// cannot see indexes or overhead.
    pub residual_bytes: i64,
    /// Per owning table, largest first.
    pub tables: Vec<TableStorage>,
    /// The same bytes rolled up by group, largest first.
    pub groups: Vec<GroupStorage>,
    /// Per repo, largest estimate first.
    pub repos: Vec<RepoStorage>,
    /// The embedding width the vector figures are computed from.
    pub embed_dim: i64,
}

/// The table a btree belongs to, for reporting.
/// Indexes are resolved through `sqlite_master.tbl_name` before this is called. This handles
/// shadow tables, whose `tbl_name` is themselves. The prefix match requires the `_` separator, so
/// an unrelated table sharing the letters is not captured.
pub(crate) fn owning_table(name: &str) -> &str {
    for base in ["vec_embeddings", "fts_embeddings"] {
        if name == base
            || name
                .strip_prefix(base)
                .is_some_and(|rest| rest.starts_with('_'))
        {
            return base;
        }
    }
    name
}

/// What a table's bytes are for. An unlisted table returns [`StorageGroup::Other`], and
/// `every_table_is_classified` fails when a migration adds one.
pub fn table_group(name: &str) -> StorageGroup {
    match name {
        "vec_embeddings" | "fts_embeddings" => StorageGroup::SearchIndex,
        "embeddings" | "indexed_text" => StorageGroup::ChunkText,
        "commits" | "pull_requests" | "issues" | "reviews" | "releases" | "ci_runs"
        | "pr_files" | "pr_closing_issues" | "work_items" | "jira_issues" | "jira_sprints"
        | "sprints" | "sprint_work_items" | "links" => StorageGroup::Activity,
        "code_files" | "file_health" | "dependencies" | "ownership" => StorageGroup::Code,
        "metric_snapshots" | "digests" | "work_item_status_history" | "sprint_commitments" => {
            StorageGroup::History
        }
        "repos"
        | "sources"
        | "forges"
        | "settings"
        | "trackers"
        | "boards"
        | "board_repos"
        | "board_people"
        | "board_trackers"
        | "board_discovered_repos"
        | "board_disabled_signals"
        | "teams"
        | "team_members"
        | "identities"
        | "identity_accounts"
        | "repo_index_prefs" => StorageGroup::Config,
        "meta" | "sync_cursors" | "_sqlx_migrations" | "sqlite_schema" | "sqlite_sequence" => {
            StorageGroup::Internal
        }
        _ => StorageGroup::Other,
    }
}

/// The tables a repo owns rows in: `(table, join, key)`, with the table aliased `t`. `embeddings`
/// is handled separately because its rows are keyed three ways.
const REPO_TABLES: &[(&str, &str, &str)] = &[
    ("commits", "", "t.repo_id"),
    ("pull_requests", "", "t.repo_id"),
    ("issues", "", "t.repo_id"),
    ("releases", "", "t.repo_id"),
    ("ci_runs", "", "t.repo_id"),
    ("dependencies", "", "t.repo_id"),
    ("code_files", "", "t.repo_id"),
    ("file_health", "", "t.repo_id"),
    (
        "reviews",
        "JOIN pull_requests p ON p.id = t.pr_id",
        "p.repo_id",
    ),
    (
        "pr_files",
        "JOIN pull_requests p ON p.id = t.pr_id",
        "p.repo_id",
    ),
];

impl Store {
    /// Measure the database. Read-only.
    /// Uses `dbstat`, falling back to summed payload lengths when it is not compiled in. Costly:
    /// `dbstat` walks every page. Do not call on a refresh path.
    pub async fn storage(&self) -> Result<StorageReport, StoreError> {
        let (tables, method) = match self.table_bytes_from_dbstat().await {
            Ok(t) => (t, StorageMethod::Dbstat),
            // `dbstat` is a compile-time option, so its absence is expected. Any other SQL error
            // also lands here and gets the rougher fallback.
            Err(_) => (
                self.table_bytes_from_payload().await?,
                StorageMethod::PayloadEstimate,
            ),
        };
        self.assemble(tables, method).await
    }

    /// The fallback path alone, so the estimate can be tested where `dbstat` is present.
    pub async fn storage_estimated(&self) -> Result<StorageReport, StoreError> {
        let tables = self.table_bytes_from_payload().await?;
        self.assemble(tables, StorageMethod::PayloadEstimate).await
    }

    /// Turn a per-table byte map into the full report.
    async fn assemble(
        &self,
        tables: Vec<(String, i64)>,
        method: StorageMethod,
    ) -> Result<StorageReport, StoreError> {
        let estimated = method == StorageMethod::PayloadEstimate;
        let page_size = self.pragma_i64("page_size").await?;
        let page_count = self.pragma_i64("page_count").await?;
        let freelist_pages = self.pragma_i64("freelist_count").await?;
        let reserved_bytes = page_count * page_size;
        let free_bytes = freelist_pages * page_size;

        let mut tables: Vec<TableStorage> = tables
            .into_iter()
            .map(|(name, bytes)| TableStorage {
                group: table_group(&name),
                name,
                bytes,
                estimated,
            })
            .collect();
        tables.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));
        let attributed_bytes: i64 = tables.iter().map(|t| t.bytes).sum();

        let mut groups: Vec<GroupStorage> = Vec::new();
        for t in &tables {
            match groups.iter_mut().find(|g| g.group == t.group) {
                Some(g) => g.bytes += t.bytes,
                None => groups.push(GroupStorage {
                    group: t.group,
                    bytes: t.bytes,
                    estimated,
                }),
            }
        }
        groups.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.group.cmp(&b.group)));

        let (file_bytes, wal_bytes, shm_bytes) = self.file_sizes().await?;
        Ok(StorageReport {
            method,
            file_bytes,
            wal_bytes,
            shm_bytes,
            page_size,
            page_count,
            freelist_pages,
            reserved_bytes,
            free_bytes,
            attributed_bytes,
            // Keeps `attributed + free + residual == reserved`. Large under the estimate, which
            // cannot see indexes or overhead.
            residual_bytes: reserved_bytes - attributed_bytes - free_bytes,
            tables,
            groups,
            repos: self.repo_storage().await?,
            embed_dim: core_embed::EMBED_DIM as i64,
        })
    }

    /// Per-table bytes from `dbstat` page counts.
    /// Btrees are joined to `sqlite_master` to fold indexes into their table; btrees without a row
    /// there (`sqlite_schema`, `sqlite_sequence`) keep their own name. Shadow tables are rolled up
    /// in Rust. Errors when `dbstat` is not compiled in.
    async fn table_bytes_from_dbstat(&self) -> Result<Vec<(String, i64)>, StoreError> {
        let rows = sqlx::query(
            "SELECT COALESCE(m.tbl_name, d.name) AS owner, SUM(d.pgsize) AS bytes \
             FROM dbstat AS d LEFT JOIN sqlite_master AS m ON m.name = d.name \
             GROUP BY owner",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut out: Vec<(String, i64)> = Vec::new();
        for row in rows {
            let owner: String = row.try_get("owner")?;
            let bytes: i64 = row.try_get("bytes")?;
            add(&mut out, owning_table(&owner), bytes);
        }
        Ok(out)
    }

    /// Per-table bytes as summed value lengths, the fallback when `dbstat` is absent.
    /// Virtual tables are skipped and their shadow tables summed instead. Columns are cast to
    /// `BLOB` first because `length()` on TEXT counts characters, not bytes.
    async fn table_bytes_from_payload(&self) -> Result<Vec<(String, i64)>, StoreError> {
        let names: Vec<(String,)> = sqlx::query_as(
            "SELECT name FROM sqlite_master \
             WHERE type = 'table' AND COALESCE(sql, '') NOT LIKE 'CREATE VIRTUAL TABLE%' \
             ORDER BY name",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut out: Vec<(String, i64)> = Vec::new();
        for (name,) in names {
            let expr = self.payload_expr(&name, "").await?;
            // A table with no columns would produce `SUM()`.
            if expr.is_empty() {
                continue;
            }
            let sql = format!("SELECT COALESCE(SUM({expr}), 0) FROM \"{name}\"");
            let (bytes,): (i64,) = sqlx::query_as(&sql).fetch_one(&self.pool).await?;
            add(&mut out, owning_table(&name), bytes);
        }
        Ok(out)
    }

    /// SQL expression for the stored byte length of one row of `table`, through an optional alias.
    /// Built from `pragma_table_info`, so new columns are picked up. Identifiers come from the
    /// schema and are quoted.
    async fn payload_expr(&self, table: &str, alias: &str) -> Result<String, StoreError> {
        let cols: Vec<(String,)> = sqlx::query_as("SELECT name FROM pragma_table_info(?)")
            .bind(table)
            .fetch_all(&self.pool)
            .await?;
        let qualify = if alias.is_empty() {
            String::new()
        } else {
            format!("{alias}.")
        };
        Ok(cols
            .into_iter()
            .map(|(c,)| format!("COALESCE(length(CAST({qualify}\"{c}\" AS BLOB)), 0)"))
            .collect::<Vec<_>>()
            .join(" + "))
    }

    /// Per-repo content estimates plus row counts.
    /// One grouped query per table, not per repo. Chunks are counted separately because
    /// `embeddings` rows are keyed three ways: code by a `<repo_id>#<path>` prefix, commits by
    /// SHA, issues by issue id.
    pub(crate) async fn repo_storage(&self) -> Result<Vec<RepoStorage>, StoreError> {
        let repos: Vec<(String, String)> =
            sqlx::query_as("SELECT id, full_name FROM repos ORDER BY full_name")
                .fetch_all(&self.pool)
                .await?;
        let mut by_id: Vec<RepoStorage> = repos
            .into_iter()
            .map(|(repo_id, full_name)| RepoStorage {
                repo_id,
                full_name,
                ..Default::default()
            })
            .collect();
        let index: std::collections::HashMap<String, usize> = by_id
            .iter()
            .enumerate()
            .map(|(i, r)| (r.repo_id.clone(), i))
            .collect();

        for (table, join, key) in REPO_TABLES {
            let expr = self.payload_expr(table, "t").await?;
            let sql = format!(
                "SELECT {key} AS rid, COUNT(*) AS n, COALESCE(SUM({expr}), 0) AS bytes \
                 FROM \"{table}\" AS t {join} GROUP BY rid"
            );
            for row in sqlx::query(&sql).fetch_all(&self.pool).await? {
                // A NULL repo id (a PR-keyed child whose parent is gone) belongs to no known repo
                // and is left out of every repo's figure.
                let Ok(rid) = row.try_get::<String, _>("rid") else {
                    continue;
                };
                let Some(repo) = index.get(&rid).map(|i| &mut by_id[*i]) else {
                    continue;
                };
                let n: i64 = row.try_get("n")?;
                repo.rows += n;
                repo.content_bytes += row.try_get::<i64, _>("bytes")?;
                match *table {
                    "commits" => repo.commits = n,
                    "pull_requests" => repo.pull_requests = n,
                    "issues" => repo.issues = n,
                    "code_files" => repo.code_files = n,
                    _ => {}
                }
            }
        }

        // The three chunk kinds as one repo-keyed set. A repo id never contains `#`, so the first
        // `#` in a code chunk's ref_id is the separator.
        let chunks = sqlx::query(
            "SELECT rid, COUNT(*) AS n, COALESCE(SUM(bytes), 0) AS bytes FROM ( \
               SELECT substr(e.ref_id, 1, instr(e.ref_id, '#') - 1) AS rid, \
                      length(CAST(e.chunk AS BLOB)) AS bytes \
                 FROM embeddings e WHERE e.ref_kind = 'code' AND instr(e.ref_id, '#') > 0 \
               UNION ALL \
               SELECT c.repo_id, length(CAST(e.chunk AS BLOB)) \
                 FROM embeddings e JOIN commits c ON c.sha = e.ref_id WHERE e.ref_kind = 'commit' \
               UNION ALL \
               SELECT i.repo_id, length(CAST(e.chunk AS BLOB)) \
                 FROM embeddings e JOIN issues i ON i.id = e.ref_id WHERE e.ref_kind = 'issue' \
             ) GROUP BY rid",
        )
        .fetch_all(&self.pool)
        .await?;
        for row in chunks {
            let rid: String = row.try_get("rid")?;
            let Some(repo) = index.get(&rid).map(|i| &mut by_id[*i]) else {
                continue;
            };
            let n: i64 = row.try_get("n")?;
            repo.index_chunks = n;
            repo.rows += n;
            repo.content_bytes += row.try_get::<i64, _>("bytes")?;
        }

        let dim = core_embed::EMBED_DIM as i64;
        for r in &mut by_id {
            // One f32 per dimension per chunk. Excludes vec0 per-chunk overhead, so this is a
            // floor.
            r.vector_bytes = r.index_chunks * dim * 4;
            r.estimated_bytes = r.content_bytes + r.vector_bytes;
        }
        by_id.sort_by(|a, b| {
            b.estimated_bytes
                .cmp(&a.estimated_bytes)
                .then_with(|| a.full_name.cmp(&b.full_name))
        });
        Ok(by_id)
    }

    /// The database file and its sidecars. `None` for an in-memory database (empty file name).
    async fn file_sizes(&self) -> Result<(Option<i64>, Option<i64>, Option<i64>), StoreError> {
        let path: Option<String> =
            sqlx::query_scalar("SELECT file FROM pragma_database_list WHERE name = 'main'")
                .fetch_optional(&self.pool)
                .await?
                .filter(|p: &String| !p.is_empty());
        let Some(path) = path else {
            return Ok((None, None, None));
        };
        let size = |p: String| {
            std::fs::metadata(p)
                .ok()
                .map(|m| m.len().min(i64::MAX as u64) as i64)
        };
        Ok((
            size(path.clone()),
            size(format!("{path}-wal")),
            size(format!("{path}-shm")),
        ))
    }

    /// Read an integer pragma. `name` is a literal from this module; pragmas take no bind
    /// parameters.
    async fn pragma_i64(&self, name: &str) -> Result<i64, StoreError> {
        let (n,): (i64,) = sqlx::query_as(&format!("PRAGMA {name}"))
            .fetch_one(&self.pool)
            .await?;
        Ok(n)
    }
}

/// Accumulate `bytes` under `name`, folding shadow tables and indexes into one entry.
fn add(out: &mut Vec<(String, i64)>, name: &str, bytes: i64) {
    match out.iter_mut().find(|(n, _)| n == name) {
        Some((_, b)) => *b += bytes,
        None => out.push((name.to_string(), bytes)),
    }
}

// ---- storage budget ----

/// The share of the budget at which the app warns. Nothing stops here.
pub const WARN_FRACTION: f64 = 0.80;

/// One megabyte, the unit the budget is stored in.
pub const MB: i64 = 1024 * 1024;

/// How close the store is to the budget, and therefore whether the sync lanes may run.
/// Three states, no throttling band: over the ceiling everything stops at once. Nothing is ever
/// discarded automatically; the app reports that it has stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoragePressure {
    /// Under [`WARN_FRACTION`], or no budget set. Everything runs.
    Normal,
    /// At or over [`WARN_FRACTION`] and under the budget. Everything still runs; the app says the
    /// ceiling is coming.
    Warning,
    /// At or over the budget. Syncing and indexing stop. Nothing is deleted.
    Full,
}

impl StoragePressure {
    /// The stable wire/display name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Warning => "warning",
            Self::Full => "full",
        }
    }

    /// The state for a usage against a budget. A missing or non-positive budget means no limit.
    pub fn for_usage(used_bytes: i64, budget_bytes: Option<i64>) -> Self {
        let Some(budget) = budget_bytes.filter(|b| *b > 0) else {
            return Self::Normal;
        };
        let fraction = used_bytes as f64 / budget as f64;
        if fraction >= 1.0 {
            Self::Full
        } else if fraction >= WARN_FRACTION {
            Self::Warning
        } else {
            Self::Normal
        }
    }
}

/// What the database costs right now, cheap enough to read on the sync path. Unlike
/// [`StorageReport`], it does not walk pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageUsage {
    /// The main database file on disk. `None` for an in-memory database.
    pub db_bytes: Option<i64>,
    /// The write-ahead log beside it. New pages live here until a checkpoint.
    pub wal_bytes: Option<i64>,
    /// `freelist_count * page_size`: claimed pages holding nothing. What a `VACUUM` would give
    /// back.
    pub free_bytes: i64,
    /// `page_count * page_size`: what SQLite has claimed inside the file.
    pub reserved_bytes: i64,
    /// The figure the budget is enforced against: file plus log, or the reserved bytes for an
    /// in-memory database.
    pub used_bytes: i64,
}

/// The live usage, the budget, and what the app may still do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetState {
    pub pressure: StoragePressure,
    pub usage: StorageUsage,
    /// The configured budget in bytes, or `None` for no limit.
    pub budget_bytes: Option<i64>,
}

impl BudgetState {
    /// Whether the sync and index lanes may run. False only at the ceiling.
    pub fn allows_sync(&self) -> bool {
        self.pressure != StoragePressure::Full
    }

    /// The plain-language sentence the banner and the Settings card show. Every non-normal message
    /// names the usage, the budget, what has stopped, and that nothing was deleted.
    pub fn message(&self) -> String {
        let used = format_bytes(self.usage.used_bytes);
        let Some(budget) = self.budget_bytes.filter(|b| *b > 0) else {
            return format!("Using {used}. No storage budget is set, so nothing is capped.");
        };
        let budget = format_bytes(budget);
        match self.pressure {
            StoragePressure::Normal => {
                format!("Using {used} of the {budget} storage budget.")
            }
            StoragePressure::Warning => format!(
                "Using {used} of the {budget} storage budget. Nothing has stopped yet: at the budget \
                 orgonzola stops syncing and indexing until there is room. It never deletes your data \
                 to make room."
            ),
            StoragePressure::Full => {
                let mut msg = format!(
                    "Using {used} of the {budget} storage budget. orgonzola has stopped syncing and \
                     indexing, so boards will not update until there is room. Nothing has been \
                     deleted. Raise the budget, or free space on the storage page"
                );
                // Mention free space only when there is some; with an empty free list the reclaim
                // button would do nothing.
                if self.usage.free_bytes >= MB {
                    msg.push_str(&format!(
                        " ({} of it is already free space waiting to be reclaimed)",
                        format_bytes(self.usage.free_bytes)
                    ));
                }
                msg.push('.');
                msg
            }
        }
    }
}

/// What a [`Store::drop_repo_index`] removed. Counts only rebuildable data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DroppedIndex {
    pub repo_id: String,
    /// Whether the repo was in the store at all.
    pub found: bool,
    /// Chunks removed from `embeddings`, with their vector and full-text rows.
    pub chunks: i64,
    /// Per-file blob-SHA markers removed, so the next sync re-fetches and re-embeds each file.
    pub code_files: i64,
    /// Every row this removed, across all of it.
    pub total_rows: i64,
}

/// What a [`Store::reclaim_free_space`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReclaimReport {
    pub used_before: i64,
    pub used_after: i64,
    /// `used_before - used_after`, floored at zero. A `VACUUM` with an empty free list can leave
    /// the file a page larger.
    pub freed_bytes: i64,
}

/// A byte count for a person to read. Rust-side because the messages above are built here.
fn format_bytes(bytes: i64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut n = bytes.unsigned_abs() as f64;
    let mut unit = 0;
    while n >= 1024.0 && unit < UNITS.len() - 1 {
        n /= 1024.0;
        unit += 1;
    }
    let sign = if bytes < 0 { "-" } else { "" };
    if unit == 0 {
        format!("{sign}{} {}", n.round(), UNITS[unit])
    } else if n >= 100.0 {
        format!("{sign}{n:.0} {}", UNITS[unit])
    } else {
        format!("{sign}{n:.1} {}", UNITS[unit])
    }
}

impl Store {
    /// What the database costs right now: three pragmas and two `stat` calls.
    pub async fn storage_usage(&self) -> Result<StorageUsage, StoreError> {
        let page_size = self.pragma_i64("page_size").await?;
        let page_count = self.pragma_i64("page_count").await?;
        let freelist_pages = self.pragma_i64("freelist_count").await?;
        let (db_bytes, wal_bytes, _shm) = self.file_sizes().await?;
        let reserved_bytes = page_count * page_size;
        // File plus log. The shared-memory sidecar is rebuildable index state and is not counted.
        let used_bytes = match db_bytes {
            Some(db) => db + wal_bytes.unwrap_or(0),
            None => reserved_bytes,
        };
        Ok(StorageUsage {
            db_bytes,
            wal_bytes,
            free_bytes: freelist_pages * page_size,
            reserved_bytes,
            used_bytes,
        })
    }

    /// The live usage against the configured budget. Read fresh, since the budget can change while
    /// a sync is running.
    pub async fn budget_state(&self) -> Result<BudgetState, StoreError> {
        let usage = self.storage_usage().await?;
        let budget_bytes = self
            .settings()
            .await?
            .storage_budget_mb
            .filter(|mb| *mb > 0)
            .map(|mb| mb.saturating_mul(MB));
        Ok(BudgetState {
            pressure: StoragePressure::for_usage(usage.used_bytes, budget_bytes),
            usage,
            budget_bytes,
        })
    }

    /// Drop one repo's search index. Activity and metric snapshots are kept; the next sync
    /// rebuilds it.
    /// Removes the repo's chunks across all three kinds, their vector and full-text rows, the
    /// per-file blob-SHA markers, the derived file-health rows and the two index cursors.
    /// Idempotent, and safe on a pinned or watched repo.
    pub async fn drop_repo_index(&self, repo_id: &str) -> Result<DroppedIndex, StoreError> {
        // Anchored with `substr`, not `LIKE`: a repo id can contain `_`, which LIKE treats as a
        // wildcard, so `my_repo` would also match `my-repo#...`.
        let prefix = format!("{repo_id}#");
        // The repo's rows across the three chunk kinds. `commits` and `issues` are left in place.
        const INDEX_ROWS: &str = "(ref_kind = 'code' AND substr(ref_id, 1, length(?)) = ?) \
             OR (ref_kind = 'commit' AND ref_id IN (SELECT sha FROM commits WHERE repo_id = ?)) \
             OR (ref_kind = 'issue' AND ref_id IN (SELECT id FROM issues WHERE repo_id = ?))";

        let mut tx = self.pool.begin().await?;
        let (found,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM repos WHERE id = ?")
            .bind(repo_id)
            .fetch_one(&mut *tx)
            .await?;
        let (code_files,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM code_files WHERE repo_id = ?")
                .bind(repo_id)
                .fetch_one(&mut *tx)
                .await?;

        let mut total = 0i64;
        let mut chunks = 0i64;
        // Vector and full-text rows first, by rowid, while `embeddings` still holds them: neither
        // has a foreign key or trigger back to it, and a stranded vector keeps occupying KNN
        // candidate slots.
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
                .execute(&mut *tx)
                .await?
                .rows_affected() as i64;
            total += n;
            if sql.starts_with("DELETE FROM embeddings") {
                chunks = n;
            }
        }
        for sql in [
            // Without dropping the markers the next sync would find the blob SHAs unchanged and
            // never re-embed.
            "DELETE FROM code_files WHERE repo_id = ?",
            "DELETE FROM file_health WHERE repo_id = ?",
            // The code watermark and the activity checkpoint.
            "DELETE FROM sync_cursors WHERE repo_id = ? AND entity IN ('code', 'activity_index')",
        ] {
            total += sqlx::query(sql)
                .bind(repo_id)
                .execute(&mut *tx)
                .await?
                .rows_affected() as i64;
        }
        tx.commit().await?;
        Ok(DroppedIndex {
            repo_id: repo_id.to_string(),
            found: found > 0,
            chunks,
            code_files,
            total_rows: total,
        })
    }

    /// Give free pages back to the file system: `VACUUM`, then truncate the write-ahead log.
    /// Deletes nothing.
    /// Deleting rows does not shrink a SQLite file. Needs room for a second copy of the file and
    /// holds the write lock throughout, so it is user-triggered, never part of a sync.
    pub async fn reclaim_free_space(&self) -> Result<ReclaimReport, StoreError> {
        let before = self.storage_usage().await?;
        sqlx::query("VACUUM").execute(&self.pool).await?;
        // VACUUM writes through the log; checkpoint so the file on disk is not briefly old size
        // plus a large log.
        let _ = sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
            .fetch_all(&self.pool)
            .await;
        let after = self.storage_usage().await?;
        Ok(ReclaimReport {
            used_before: before.used_bytes,
            used_after: after.used_bytes,
            freed_bytes: (before.used_bytes - after.used_bytes).max(0),
        })
    }
}
