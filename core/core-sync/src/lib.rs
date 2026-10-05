//! Incremental sync. Pulls a forge's activity into the store through the [`core_forge::Forge`]
//! trait; an `unchanged` delta persists nothing. The engine owns orchestration, cursors, store
//! writes and indexing; the forge owns endpoints, wire shapes, pagination and cursor meaning.
//! Works with the LLM disabled.

use core_embed::{chunk_code, chunk_text, source_language, Embedder, HashEmbedder};
use core_forge::{Forge, ForgeError, RepoRef};
use core_graph::{parse_issue_references, parse_manifest};
use core_store::{
    backfill_entity, Board, BoardKind, DependencyRow, PullRequest, Repo, SourceRow, Store,
    StoreError, BACKFILL_DONE,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{interval, MissedTickBehavior};

/// The package manifests `sync_dependencies` looks for, in order. Each maps to a `core-graph`
/// parser.
const MANIFEST_FILES: [&str; 5] = [
    "Cargo.toml",
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "go.mod",
];

#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    #[error(transparent)]
    Forge(#[from] ForgeError),
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A repo or board named a forge the engine has not been given. Skipped per repo/board, not
    /// fatal to the pass.
    #[error("no forge configured for id {0:?}")]
    UnknownForge(String),
    /// The store is at the storage budget, so this sync did not run. Carries the budget state's
    /// sentence. A refusal, not a failure: nothing was attempted or written.
    #[error("{0}")]
    OverBudget(String),
}

impl SyncError {
    /// A plain-language message for a non-technical user. Forge failures delegate to
    /// [`ForgeError::user_message`]; store/config failures get a generic sentence.
    pub fn user_message(&self) -> String {
        match self {
            SyncError::Forge(e) => e.user_message(),
            SyncError::UnknownForge(_) => {
                "A board points at a connection that is not set up. Connect it under Settings > Forges.".into()
            }
            SyncError::Store(detail) => {
                format!("Could not save the synced data locally. Try again. ({detail})")
            }
            // Already a plain-language sentence; passed through unwrapped.
            SyncError::OverBudget(message) => message.clone(),
        }
    }
}

/// Forge id a single-forge engine registers under (`SyncEngine::new`); also used in test repo ids.
pub const DEFAULT_FORGE: &str = "default";

/// Build a namespaced repo id: `repo:<forge_id>/<owner>/<name>`. Forge ids contain no `/`, so the
/// first segment parses back unambiguously.
pub fn repo_id(forge_id: &str, full_name: &str) -> String {
    format!("repo:{forge_id}/{full_name}")
}

/// Extract the forge id from `repo:<forge>/<owner>/<name>`. `None` if the id has another shape.
pub fn forge_id_of(repo_id: &str) -> Option<&str> {
    repo_id
        .strip_prefix("repo:")?
        .split_once('/')
        .map(|(forge, _)| forge)
}

/// The git host of a forge, from its API base URL authority: `https://gitea.example.com/api/v1` ->
/// `gitea.example.com`. Special case: github.com's API is `api.github.com` but its repos live on
/// `github.com`. No general `api.` stripping, since elsewhere that prefix may be the real host.
/// `None` when `base_url` has no authority.
fn forge_host(base_url: &str) -> Option<String> {
    let authority = base_url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(base_url)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .trim();
    match authority {
        "" => None,
        "api.github.com" => Some("github.com".to_string()),
        host => Some(host.to_string()),
    }
}

/// Extract `owner/name` from a namespaced repo id. `None` if the id is not in that shape.
fn full_name_of(repo_id: &str) -> Option<&str> {
    repo_id
        .strip_prefix("repo:")?
        .split_once('/')
        .map(|(_, full)| full)
}

/// What one `sync_repo` run did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncReport {
    pub fetched: usize,
    pub persisted: usize,
    /// True when the conditional GET returned 304 and nothing was pulled.
    pub unchanged: bool,
}

/// Cap on repos discovered per board, so a busy person or large org cannot pull in an unbounded
/// set. A partial scan is reported.
const MAX_DISCOVERED_REPOS: usize = 50;

/// The largest blob the forge file APIs return. GitHub's git-blobs endpoint has a 100 MB ceiling;
/// a larger blob cannot be fetched and would fail every pass, holding the `code` cursor back. Such
/// a file is skipped and reported with its size.
const MAX_BLOB_BYTES: i64 = 100 * 1024 * 1024;

/// How many changed files one pass fetches before leaving the rest for the next pass. A bound on
/// work, not on what exists: indexed files are skipped by blob SHA and the `code` cursor holds
/// while work remains, so the repo reports `partial` until all arrive. Sized for memory (about 30
/// MB of 20 kB files in flight) and API budget (one call per file).
const CODE_FILES_PER_PASS: usize = 1_500;

/// How long the scheduler waits before rechecking while stopped at the storage budget. Short, so
/// freeing space resumes syncing within about a minute.
const BUDGET_RECHECK: Duration = Duration::from_secs(60);

/// Merged PRs to back-fill changed-files for per pass. Open PRs are always refreshed; merged-PR
/// diffs never change, so they trickle in to protect the rate budget on a big repo's first sync.
const MERGED_PR_FILES_PER_PASS: i64 = 50;

/// The `sync_cursors` entity holding a repo's activity-index checkpoint. `clear_index` drops it
/// with the `code` cursor on a model reset.
const ACTIVITY_INDEX_CURSOR: &str = "activity_index";

/// Drives sync against a [`Forge`] and the store. `new` defaults to the deterministic
/// `HashEmbedder`; `with_embedder` swaps in a real model.
pub struct SyncEngine<F> {
    /// Configured forges by id. Repo ids (`repo:<forge_id>/...`) and `board.forge_id` route each
    /// sync. A single-forge engine (`new`) holds one entry under [`DEFAULT_FORGE`].
    forges: std::collections::BTreeMap<String, F>,
    store: Store,
    embedder: Arc<dyn Embedder>,
    /// When set, the fetch lane hands indexing to this background queue so fetching never waits on
    /// embedding. `None` indexes inline.
    index_tx: Option<tokio::sync::mpsc::Sender<IndexJob>>,
    /// Current per-repo index state. Source for the Debug "Queues" panel; survives the UI
    /// unmounting.
    index_status: Arc<std::sync::Mutex<std::collections::BTreeMap<String, IndexProgress>>>,
    /// One generation number per index pass. The fetch lane and index worker write the status map
    /// from separate tasks, so a write must say which pass it came from; otherwise an old pass
    /// finishing removes a newer re-queue's entry. Counts from 1; 0 means no pass owns the entry.
    index_generation: std::sync::atomic::AtomicU64,
}

impl<F: Forge> SyncEngine<F> {
    /// A single-forge engine: the forge is registered under [`DEFAULT_FORGE`], so repo ids are
    /// `repo:default/<owner>/<name>`. Used by tests; the desktop host uses [`SyncEngine::empty`] +
    /// [`SyncEngine::with_forge`].
    pub fn new(forge: F, store: Store) -> Self {
        Self::empty(store).with_forge(DEFAULT_FORGE, forge)
    }

    /// An engine with no forges; add them with [`SyncEngine::with_forge`]. A repo whose forge is
    /// not registered is skipped with `SyncError::UnknownForge`.
    pub fn empty(store: Store) -> Self {
        Self {
            forges: std::collections::BTreeMap::new(),
            store,
            embedder: Arc::new(HashEmbedder::default()),
            index_tx: None,
            index_status: Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new())),
            index_generation: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Register a forge under its id (builder style). Re-registering the same id replaces it.
    pub fn with_forge(mut self, id: impl Into<String>, forge: F) -> Self {
        self.forges.insert(id.into(), forge);
        self
    }

    /// The forge that syncs a repo, from the forge id in its repo id. Errors if the id is
    /// malformed or the forge is not configured.
    fn forge_for_repo(&self, repo_id: &str) -> Result<&F, SyncError> {
        let forge_id =
            forge_id_of(repo_id).ok_or_else(|| SyncError::UnknownForge(repo_id.to_string()))?;
        self.forge_by_id(forge_id)
    }

    fn forge_by_id(&self, forge_id: &str) -> Result<&F, SyncError> {
        self.forges
            .get(forge_id)
            .ok_or_else(|| SyncError::UnknownForge(forge_id.to_string()))
    }

    /// Replace the indexing embedder (e.g. with the fastembed model).
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = embedder;
        self
    }

    /// Attach the background index queue: the fetch lane then enqueues each repo's indexing onto
    /// `tx` instead of indexing inline. Pair with `run_index_worker` over the matching receiver.
    pub fn with_index_queue(mut self, tx: tokio::sync::mpsc::Sender<IndexJob>) -> Self {
        self.index_tx = Some(tx);
        self
    }

    /// Jobs waiting in the background index queue, or `None` if none is attached. The job being
    /// indexed is not counted.
    pub fn index_queue_depth(&self) -> Option<usize> {
        self.index_tx
            .as_ref()
            .map(|tx| tx.max_capacity() - tx.capacity())
    }

    /// Generation for the next index pass. One counter for the engine: two repos never share a
    /// status-map entry, so one sequence orders every colliding write. Starts at 1 so an
    /// unsequenced record (generation 0) never wins.
    fn next_index_generation(&self) -> u64 {
        self.index_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    /// Record a repo's index state. A terminal `Indexed` state is dropped from the map; `Queued`,
    /// `Indexing` and `Error` are kept so `index_status()` reflects the live queue.
    fn record_index_status(&self, p: IndexProgress) {
        record_index_status_in(&self.index_status, p);
    }

    /// Repos currently in the index lane (queued, indexing or errored).
    pub fn index_status(&self) -> Vec<IndexProgress> {
        self.index_status
            .lock()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Sync a repo's commits incrementally: fetch commits `since` the stored cursor (a commit
    /// date), upsert each, then advance the cursor to the newest commit date. The cursor is
    /// persisted, unlike the in-memory ETag. First sync (no cursor) pulls from the start up to the
    /// page cap.
    pub async fn sync_repo(&self, repo: &Repo) -> Result<SyncReport, SyncError> {
        self.store.upsert_repo(repo).await?;
        let cursor = self.store.sync_cursor(&repo.id, "commits").await?;
        let delta = self
            .forge_for_repo(&repo.id)?
            .commits(&repo.into(), cursor)
            .await?;
        let fetched = delta.items.len();
        for c in &delta.items {
            self.store.upsert_commit(c).await?;
        }
        // Record the backfill walk before advancing the watermark past the un-fetched tail.
        self.record_backfill(&repo.id, "commits", delta.backfill.as_deref())
            .await?;
        if !delta.unchanged {
            if let Some(cur) = delta.cursor {
                self.store
                    .set_sync_cursor(&repo.id, "commits", &cur)
                    .await?;
            }
        }
        Ok(SyncReport {
            fetched,
            persisted: fetched,
            unchanged: delta.unchanged,
        })
    }

    /// Sync a repo's pull requests incrementally. The forge owns pagination; this upserts the PRs,
    /// advances the cursor, and returns the changed PRs (id + number) so only they get reviews
    /// fetched.
    pub async fn sync_pull_requests(
        &self,
        repo: &Repo,
    ) -> Result<(SyncReport, Vec<(String, i64)>), SyncError> {
        self.store.upsert_repo(repo).await?;
        let cursor = self.store.sync_cursor(&repo.id, "pulls").await?;
        let delta = self
            .forge_for_repo(&repo.id)?
            .pull_requests(&repo.into(), cursor)
            .await?;
        let persisted = delta.items.len();
        for p in &delta.items {
            self.store.upsert_pull_request(p).await?;
        }
        self.record_backfill(&repo.id, "pulls", delta.backfill.as_deref())
            .await?;
        if !delta.unchanged {
            if let Some(cur) = delta.cursor {
                self.store.set_sync_cursor(&repo.id, "pulls", &cur).await?;
            }
        }
        Ok((
            SyncReport {
                fetched: persisted,
                persisted,
                unchanged: delta.unchanged,
            },
            delta.changed,
        ))
    }

    /// Sync a repo's issues incrementally. The forge skips PRs returned by the issues endpoint and
    /// advances the cursor across everything, so a PR-only update still moves it.
    pub async fn sync_issues(&self, repo: &Repo) -> Result<SyncReport, SyncError> {
        self.store.upsert_repo(repo).await?;
        let cursor = self.store.sync_cursor(&repo.id, "issues").await?;
        let delta = self
            .forge_for_repo(&repo.id)?
            .issues(&repo.into(), cursor)
            .await?;
        let persisted = delta.items.len();
        for i in &delta.items {
            self.store.upsert_issue(i).await?;
        }
        if !delta.unchanged {
            if let Some(cur) = delta.cursor {
                self.store.set_sync_cursor(&repo.id, "issues", &cur).await?;
            }
        }
        Ok(SyncReport {
            fetched: persisted,
            persisted,
            unchanged: delta.unchanged,
        })
    }

    // ---- history backfill ----

    /// Put a forward pass's leftover history on the books, if any. Call before advancing that
    /// entity's forward watermark: the watermark may pass an un-fetched tail only once something
    /// owns it.
    /// `None` leaves any walk in progress alone. `Some` always overwrites it: that walk started
    /// below this pass's window and would never reach the new gap, so it restarts from the higher
    /// position. The overlap re-fetches rows we hold (writes are upserts by forge id), costing API
    /// calls but leaving no hole.
    async fn record_backfill(
        &self,
        repo_id: &str,
        entity: &str,
        token: Option<&str>,
    ) -> Result<(), SyncError> {
        let Some(token) = token else {
            return Ok(());
        };
        self.store
            .set_sync_cursor(repo_id, &backfill_entity(entity), token)
            .await?;
        Ok(())
    }

    /// The resume token for a repo entity's backfill walk, or `None` when none is active.
    async fn pending_backfill(
        &self,
        repo_id: &str,
        entity: &str,
    ) -> Result<Option<String>, SyncError> {
        let cursor = self
            .store
            .sync_cursor(repo_id, &backfill_entity(entity))
            .await?;
        Ok(cursor.filter(|c| c != BACKFILL_DONE))
    }

    /// Persist the next resume token, or the completion marker once the forge reports end of
    /// history.
    async fn advance_backfill(
        &self,
        repo_id: &str,
        entity: &str,
        next: Option<&str>,
    ) -> Result<(), SyncError> {
        self.store
            .set_sync_cursor(
                repo_id,
                &backfill_entity(entity),
                next.unwrap_or(BACKFILL_DONE),
            )
            .await?;
        Ok(())
    }

    /// Walk one bounded chunk of a repo's un-fetched history.
    /// When a cold window exceeds the forge's page cap, the forward lane keeps the newest window
    /// fresh and hands the older tail here as an opaque resume token; each pass walks one chunk
    /// down. It runs last in `sync_all_steps` and a chunk is a fraction of a forward page cap, so
    /// it does not starve the forward lane. The token lives in `sync_cursors`, so a restart costs
    /// one chunk. The forge skips the chunk when its tracked rate budget is low, leaving the token
    /// untouched.
    /// Covers commits and pull requests, whose listings are newest-first and do not drain forward.
    /// Backfilled PRs carry no reviews (one call per PR would dwarf the lane's budget).
    pub async fn backfill_repo(&self, repo: &Repo) -> Result<BackfillReport, SyncError> {
        let forge = self.forge_for_repo(&repo.id)?;
        let repo_ref: RepoRef = repo.into();
        let mut report = BackfillReport::default();

        if let Some(token) = self.pending_backfill(&repo.id, "commits").await? {
            let delta = forge.commits(&repo_ref, Some(token)).await?;
            for c in &delta.items {
                self.store.upsert_commit(c).await?;
            }
            report.commits = delta.items.len();
            report.pending |= delta.backfill.is_some();
            self.advance_backfill(&repo.id, "commits", delta.backfill.as_deref())
                .await?;
        }

        if let Some(token) = self.pending_backfill(&repo.id, "pulls").await? {
            let delta = forge.pull_requests(&repo_ref, Some(token)).await?;
            for p in &delta.items {
                self.store.upsert_pull_request(p).await?;
            }
            report.pull_requests = delta.items.len();
            report.pending |= delta.backfill.is_some();
            self.advance_backfill(&repo.id, "pulls", delta.backfill.as_deref())
                .await?;
        }

        Ok(report)
    }

    /// Sync a repo's releases (ETag-conditional in the forge): a 304 leaves stored releases and
    /// cursor untouched; otherwise upsert and advance the cursor.
    pub async fn sync_releases(&self, repo: &Repo) -> Result<SyncReport, SyncError> {
        self.store.upsert_repo(repo).await?;
        let cursor = self.store.sync_cursor(&repo.id, "releases").await?;
        let delta = self
            .forge_for_repo(&repo.id)?
            .releases(&repo.into(), cursor)
            .await?;
        if delta.unchanged {
            return Ok(SyncReport {
                fetched: 0,
                persisted: 0,
                unchanged: true,
            });
        }
        let fetched = delta.items.len();
        for r in &delta.items {
            self.store.upsert_release(r).await?;
        }
        if let Some(cur) = delta.cursor {
            self.store
                .set_sync_cursor(&repo.id, "releases", &cur)
                .await?;
        }
        Ok(SyncReport {
            fetched,
            persisted: fetched,
            unchanged: false,
        })
    }

    /// Sync reviews for the PRs that changed this pass. The reviews endpoint is per-PR, so only
    /// the `changed` (pr_id, number) pairs from `sync_pull_requests` are fetched. Also captures
    /// each changed PR's closing-issue references on forges that report them, which the linker
    /// prefers over body-parsed `#N`.
    pub async fn sync_reviews_for(
        &self,
        repo: &Repo,
        changed: &[(String, i64)],
    ) -> Result<SyncReport, SyncError> {
        let repo_ref: RepoRef = repo.into();
        let forge = self.forge_for_repo(&repo.id)?;
        let closing_refs = forge.capabilities().closing_issue_refs;
        let mut fetched = 0;
        let mut persisted = 0;
        for (pr_id, number) in changed {
            let reviews = forge.reviews(&repo_ref, *number, pr_id).await?;
            fetched += reviews.len();
            for rv in &reviews {
                self.store.upsert_review(rv).await?;
                persisted += 1;
            }
            if closing_refs {
                // Best-effort: a GraphQL failure must not fail the pass (body-parse fallback still
                // links). A successful empty result clears stale references for this PR.
                if let Ok(numbers) = forge.pr_closing_issues(&repo_ref, *number).await {
                    self.store
                        .replace_pr_closing_issues(pr_id, &numbers)
                        .await?;
                }
            }
        }
        Ok(SyncReport {
            fetched,
            persisted,
            unchanged: changed.is_empty(),
        })
    }

    /// Sync a repo's CI runs (ETag-conditional in the forge): a 304 leaves stored runs and cursor
    /// untouched; otherwise upsert and advance the cursor.
    pub async fn sync_ci_runs(&self, repo: &Repo) -> Result<SyncReport, SyncError> {
        self.store.upsert_repo(repo).await?;
        let cursor = self.store.sync_cursor(&repo.id, "ci_runs").await?;
        let delta = self
            .forge_for_repo(&repo.id)?
            .ci_runs(&repo.into(), cursor)
            .await?;
        if delta.unchanged {
            return Ok(SyncReport {
                fetched: 0,
                persisted: 0,
                unchanged: true,
            });
        }
        let fetched = delta.items.len();
        for run in &delta.items {
            self.store.upsert_ci_run(run).await?;
        }
        if let Some(cur) = delta.cursor {
            self.store
                .set_sync_cursor(&repo.id, "ci_runs", &cur)
                .await?;
        }
        Ok(SyncReport {
            fetched,
            persisted: fetched,
            unchanged: false,
        })
    }

    /// Sync everything for one repo: the six activity types in dependency order (reviews after
    /// PRs) plus the declared dependency graph. Returns the per-type persisted counts. Shared by
    /// the shell command and the scheduler.
    pub async fn sync_all(&self, repo: &Repo) -> Result<FullSyncReport, SyncError> {
        self.sync_all_steps(repo, true, |_, _, _| {}).await
    }

    /// Fetch a repo's activity without indexing it (no code fetch, embeddings or link pass). Used
    /// for org boards, which only need activity for metrics and never embed.
    pub async fn sync_metrics_only(&self, repo: &Repo) -> Result<FullSyncReport, SyncError> {
        self.sync_all_steps(repo, false, |_, _, _| {}).await
    }

    /// Human-readable sub-steps of a repo's fetch (the activity lane), in run order, for header
    /// progress. Code fetch and indexing run in the index lane. The index is the step's position.
    pub const SYNC_STEPS: [&'static str; 9] = [
        "commits",
        "pull requests",
        "issues",
        "releases",
        "reviews",
        "CI runs",
        "dependencies",
        "PR files",
        // Last, and a no-op unless history is still owed. A repo stuck on this step is backfilling.
        "history backfill",
    ];

    /// Fetch a repo fully (the activity lane) and hand its indexing to the background queue. Calls
    /// `on_step` before each sub-step for header progress. Enqueues an `IndexJob` if a queue is
    /// attached, else indexes inline. The report covers the fetch; `code_files` counts changed
    /// source files queued.
    pub async fn sync_all_steps<Cb>(
        &self,
        repo: &Repo,
        index: bool,
        mut on_step: Cb,
    ) -> Result<FullSyncReport, SyncError>
    where
        Cb: FnMut(&str, usize, usize),
    {
        // Gate 1 of 4: single-repo entry point, checked before any network call. Pass-level gates
        // mean the normal path never refuses here; a direct caller gets a stated reason, not a
        // silent no-op.
        let budget = self.store.budget_state().await?;
        if !budget.allows_sync() {
            return Err(SyncError::OverBudget(budget.message()));
        }

        let n = Self::SYNC_STEPS.len();
        let step = |i: usize, f: &mut Cb| f(Self::SYNC_STEPS[i], i, n);

        // Record whether this repo is a fork and its upstream, so a fork can be folded out of the
        // board's flagged set. Best-effort: a forge that cannot report it leaves the repo a
        // non-fork. The repo row must exist first (sync_repo upserts it).
        step(0, &mut on_step);
        let commits = self.sync_repo(repo).await?;
        if let Ok(meta) = self.forge_for_repo(&repo.id)?.repo_meta(&repo.into()).await {
            let parent_id = meta
                .parent_full_name
                .as_deref()
                .map(|p| repo_id(forge_id_of(&repo.id).unwrap_or_default(), p));
            self.store
                .set_repo_fork(&repo.id, meta.is_fork, parent_id.as_deref())
                .await?;
        }
        step(1, &mut on_step);
        let (pull_requests, changed_prs) = self.sync_pull_requests(repo).await?;
        step(2, &mut on_step);
        let issues = self.sync_issues(repo).await?;
        step(3, &mut on_step);
        let releases = self.sync_releases(repo).await?;
        step(4, &mut on_step);
        let reviews = self.sync_reviews_for(repo, &changed_prs).await?;
        step(5, &mut on_step);
        let ci_runs = self.sync_ci_runs(repo).await?;
        step(6, &mut on_step);
        let dependencies = self.sync_dependencies(repo).await?;
        // Best-effort: a files-endpoint failure does not fail the sync.
        step(7, &mut on_step);
        let _ = self.sync_pr_files(repo).await;
        // Last, so old history never delays fresh activity. Best-effort: a failure is logged and
        // the walk resumes from its stored token next pass.
        step(8, &mut on_step);
        let backfilled = match self.backfill_repo(repo).await {
            Ok(r) => r.commits + r.pull_requests,
            Err(e) => {
                eprintln!("sync: history backfill failed for {}: {e}", repo.full_name);
                0
            }
        };

        // Hand indexing (code fetch + embed, activity embed, links) to the background queue so
        // activity surfaces now and slow code work stays off the critical path. With no queue
        // (tests, standalone), index inline. Org boards pass `index = false` and skip the job.
        if !index {
            return Ok(FullSyncReport {
                commits: commits.persisted,
                pull_requests: pull_requests.persisted,
                issues: issues.persisted,
                releases: releases.persisted,
                ci_runs: ci_runs.persisted,
                reviews: reviews.persisted,
                dependencies: dependencies.persisted,
                code_files: 0,
                backfilled,
            });
        }
        // One generation per pass, allocated on both paths so every record is tagged with it.
        let job = IndexJob {
            repo: repo.clone(),
            generation: self.next_index_generation(),
        };
        match &self.index_tx {
            Some(tx) => {
                // Written before the send so the worker cannot report on this pass before the map
                // holds its generation; the terminal removal can then require an exact match.
                self.record_index_status(IndexProgress::state(
                    &repo.id,
                    &repo.full_name,
                    IndexState::Queued,
                    job.generation,
                ));
                let _ = tx.send(job).await; // a closed receiver just drops the job (best-effort)
            }
            None => {
                // No queue: index inline. Log a failure, since there is no Error badge.
                match self.process_index_job(&job).await {
                    Err(e) => eprintln!("index error for {}: {e}", job.repo.full_name),
                    // Log a partly-indexed repo too; this is the path tests and standalone engines
                    // take.
                    Ok(outcome) if outcome.partial() => {
                        eprintln!(
                            "index partial for {}: {}",
                            job.repo.full_name,
                            outcome.note()
                        )
                    }
                    Ok(_) => {}
                }
            }
        }

        Ok(FullSyncReport {
            commits: commits.persisted,
            pull_requests: pull_requests.persisted,
            issues: issues.persisted,
            releases: releases.persisted,
            ci_runs: ci_runs.persisted,
            reviews: reviews.persisted,
            dependencies: dependencies.persisted,
            // Code is fetched and indexed off the critical path; the count arrives via
            // IndexProgress.
            code_files: 0,
            backfilled,
        })
    }

    /// Do one repo's indexing off the critical path: fetch changed code, derive cross-reference
    /// links, embed activity text and code. Run by the index worker, or inline with no queue.
    /// Returns the number of code files (re-)embedded.
    pub async fn process_index_job(&self, job: &IndexJob) -> Result<IndexOutcome, SyncError> {
        self.process_index_job_progress(job, |_, _| {}).await
    }

    /// Like `process_index_job`, but reports `(done, total)` progress. `total` is commits, issues and
    /// changed code files; `done` advances per unit embedded or skipped. The activity count is
    /// queried up front to set the total.
    pub async fn process_index_job_progress<Cb>(
        &self,
        job: &IndexJob,
        mut on_progress: Cb,
    ) -> Result<IndexOutcome, SyncError>
    where
        Cb: FnMut(usize, usize) + Send,
    {
        // Fetch code only when the repo opts into code indexing. A real fetch failure (rate limit,
        // 5xx, bad payload, DB error) must surface as an index Error, not an empty fetch reported
        // as "Indexed". The tolerated cases (empty repo, no default branch: 404/409) are handled
        // in fetch_changed_code.
        let code = if self.store.repo_index_code(&job.repo.id).await? {
            Some(self.fetch_changed_code(&job.repo).await?)
        } else {
            None
        };

        // Activity-index checkpoint: embedding every commit message and issue title and
        // re-deriving links is O(repo history), and nothing changes unless a commit, PR or issue
        // did. The checkpoint is the activity fetch cursors plus the embedder identity; when it
        // matches the last indexed value, the activity scan and linking are skipped. Code is gated
        // separately on `pushed_at`.
        let checkpoint = self.activity_index_checkpoint(&job.repo.id).await?;
        let stored = self
            .store
            .sync_cursor(&job.repo.id, ACTIVITY_INDEX_CURSOR)
            .await?;
        let do_activity = stored.as_deref() != Some(checkpoint.as_str());

        // The progress total counts only work about to run, so a skipped scan shows no stalled bar.
        let (n_commits, n_issues) = if do_activity {
            (
                self.store.commits_for_repo(&job.repo.id).await?.len(),
                self.store.issues_for_repo(&job.repo.id).await?.len(),
            )
        } else {
            (0, 0)
        };
        let changed_files = code.as_ref().map(|c| c.changed.len()).unwrap_or(0);
        let total = n_commits + n_issues + changed_files;
        let mut done = 0usize;
        on_progress(done, total);
        let mut tick = || {
            done += 1;
            on_progress(done, total);
        };
        if do_activity {
            self.link_references(&job.repo).await?;
            self.index_repo_with(&job.repo, self.embedder.as_ref(), &mut tick)
                .await?;
            // Record the checkpoint only after activity work succeeds, so a failure retries next
            // time.
            self.store
                .set_sync_cursor(&job.repo.id, ACTIVITY_INDEX_CURSOR, &checkpoint)
                .await?;
        }
        match code {
            Some(code) => {
                let files = self.index_code_with(&job.repo, &code, &mut tick).await?;
                Ok(IndexOutcome {
                    files,
                    pending: code.pending,
                    skipped: code.skipped,
                })
            }
            None => Ok(IndexOutcome::default()),
        }
    }

    /// The embedder and chunking-scheme identity, `name:dim:scheme`. Any change means a different
    /// index, so it is part of the model reconcile and the activity-index checkpoint.
    fn embedder_identity(&self) -> String {
        format!(
            "{}:{}:{}",
            self.embedder.name(),
            self.embedder.dimensions(),
            core_embed::INDEX_SCHEME_VERSION
        )
    }

    /// The activity-index checkpoint for a repo: the `commits`, `pulls` and `issues` fetch cursors
    /// plus the embedder identity. The cursors advance only when something is added or updated, so
    /// an unchanged checkpoint means nothing new to embed or link.
    async fn activity_index_checkpoint(&self, repo_id: &str) -> Result<String, SyncError> {
        let cursor = |entity: &'static str| self.store.sync_cursor(repo_id, entity);
        let commits = cursor("commits").await?.unwrap_or_default();
        let pulls = cursor("pulls").await?.unwrap_or_default();
        let issues = cursor("issues").await?.unwrap_or_default();
        Ok(format!(
            "{commits}|{pulls}|{issues}|{}",
            self.embedder_identity()
        ))
    }

    /// Run the background index worker until the queue closes. Consumes `IndexJob`s one at a time
    /// and emits an `IndexProgress` as each repo starts and finishes. A job whose indexing errors
    /// emits `IndexState::Error` and the worker continues.
    pub async fn run_index_worker<Cb>(
        &self,
        mut rx: tokio::sync::mpsc::Receiver<IndexJob>,
        mut emit: Cb,
    ) where
        Cb: FnMut(IndexProgress) + Send,
    {
        while let Some(job) = rx.recv().await {
            let repo_id = job.repo.id.clone();
            let full_name = job.repo.full_name.clone();
            // Every record below belongs to this pass; a superseded pass must not touch the map.
            let generation = job.generation;
            // Gate 3 of 4: between jobs, so an index is either done or not started, never
            // abandoned mid-embed. The job is dropped, not held: the next pass re-queues it, and
            // holding it would grow a queue while the store is full.
            if let Ok(budget) = self.store.budget_state().await {
                if !budget.allows_sync() {
                    let paused = IndexProgress {
                        note: Some(budget.message()),
                        ..IndexProgress::state(&repo_id, &full_name, IndexState::Paused, generation)
                    };
                    self.record_index_status(paused.clone());
                    emit(paused);
                    continue;
                }
            }
            // count not known until the code is fetched + embedded
            let indexing =
                IndexProgress::state(&repo_id, &full_name, IndexState::Indexing, generation);
            self.record_index_status(indexing.clone());
            emit(indexing);
            // Emit `Indexing` only when the whole-percent figure changes (at most ~100 events per
            // repo). The callback captures the shared status map, not `self`, so it stays `Send`
            // for any engine.
            let mut last_pct = usize::MAX;
            let status = self.index_status.clone();
            let result = self
                .process_index_job_progress(&job, |done, total| {
                    let pct = (done * 100).checked_div(total).unwrap_or(0);
                    if pct == last_pct {
                        return;
                    }
                    last_pct = pct;
                    let p = IndexProgress {
                        done,
                        total,
                        ..IndexProgress::state(
                            &repo_id,
                            &full_name,
                            IndexState::Indexing,
                            generation,
                        )
                    };
                    record_index_status_in(&status, p.clone());
                    emit(p);
                })
                .await;
            let finished = match result {
                // A pass that left files owed or could not read some of them must say so, since an
                // empty code-search result over a partly-read repo looks like a miss.
                Ok(outcome) => {
                    let partial = outcome.partial();
                    IndexProgress {
                        files: outcome.files,
                        pending: outcome.pending,
                        skipped: outcome.skipped.len(),
                        note: partial.then(|| outcome.note()),
                        ..IndexProgress::state(
                            &repo_id,
                            &full_name,
                            if partial {
                                IndexState::Partial
                            } else {
                                IndexState::Indexed
                            },
                            generation,
                        )
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    // Surface the reason so a transient DB lock is distinguishable from a data
                    // problem. No tracing in this crate.
                    eprintln!("index error for {full_name}: {msg}");
                    IndexProgress {
                        error: Some(msg),
                        ..IndexProgress::state(&repo_id, &full_name, IndexState::Error, generation)
                    }
                }
            };
            self.record_index_status(finished.clone());
            emit(finished);
        }
    }

    /// Run the periodic poll loop until the task is dropped. Polls one source per sub-tick,
    /// cycling through `repos` so every source is synced once per `interval`, evenly spread. Each
    /// poll calls `emit` with a `SyncTick` carrying the source id and its `FullSyncReport` or
    /// error string. Returns at once if `repos` is empty.
    pub async fn run_scheduler<Cb>(&self, repos: &[Repo], interval_period: Duration, mut emit: Cb)
    where
        Cb: FnMut(SyncProgress),
    {
        if repos.is_empty() {
            return;
        }
        // Rebuild the index if the bundled model changed since last run, before any embedding.
        if let Err(e) = self.reconcile_index_model().await {
            eprintln!("reconcile index model: {e}");
        }
        // One source per sub-tick: the whole registry is covered once per `interval_period`.
        let total = repos.len();
        let step = (interval_period / total as u32).max(Duration::from_millis(1));
        let mut ticker = interval(step);
        // Under load, skip rather than burst-poll to catch up (shared rate budget).
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut idx = 0usize;
        loop {
            ticker.tick().await;
            let repo = &repos[idx % total];
            let report = self.sync_all(repo).await.map_err(|e| e.to_string());
            let tick = SyncTick {
                source_id: repo.id.clone(),
                report,
            };
            emit(SyncProgress::finished(
                &repo.full_name,
                (idx % total) + 1,
                total,
                tick,
            ));
            idx = idx.wrapping_add(1);
        }
    }

    /// Fetch and persist the changed files for a repo's PRs, replacing any prior set per PR. Open
    /// PRs are refreshed every sync (conditional GET). Merged PR diffs never change, so they are
    /// back-filled a bounded batch per pass. Returns the total files stored.
    pub async fn sync_pr_files(&self, repo: &Repo) -> Result<usize, SyncError> {
        let open: Vec<PullRequest> = self
            .store
            .pull_requests(&repo.id)
            .await?
            .into_iter()
            .filter(|p| p.state == "open")
            .collect();
        let backfill = self
            .store
            .merged_pull_requests_missing_files(&repo.id, MERGED_PR_FILES_PER_PASS)
            .await?;

        let mut total = 0;
        for pr in open.iter().chain(backfill.iter()) {
            total += self.sync_one_pr_files(repo, pr).await?;
        }
        Ok(total)
    }

    /// Fetch + persist one PR's changed files. Returns the count stored.
    async fn sync_one_pr_files(&self, repo: &Repo, pr: &PullRequest) -> Result<usize, SyncError> {
        let rows = self
            .forge_for_repo(&repo.id)?
            .pr_files(&repo.into(), pr.number, &pr.id)
            .await?;
        self.store.replace_pr_files(&pr.id, &rows).await?;
        // Mark the PR's files fetched even when `rows` is empty, so a merged PR with zero changed
        // files is not re-listed by `merged_pull_requests_missing_files` forever.
        self.store.mark_pr_files_synced(&pr.id).await?;
        Ok(rows.len())
    }

    /// Sync a repo's declared dependencies: for each known manifest, fetch its text (conditional
    /// on the stored cursor), parse it with `core-graph`, and replace the edges. A missing or
    /// malformed manifest contributes zero rows without erroring. `fetched`/`persisted` count
    /// parsed edges; `unchanged` is true only if every manifest reported no change.
    pub async fn sync_dependencies(&self, repo: &Repo) -> Result<SyncReport, SyncError> {
        self.store.upsert_repo(repo).await?;
        let repo_ref: RepoRef = repo.into();
        let forge = self.forge_for_repo(&repo.id)?;
        let mut fetched = 0;
        let mut persisted = 0;
        let mut all_unchanged = true;
        for manifest in MANIFEST_FILES {
            let entity = format!("deps:{manifest}");
            let cursor = self.store.sync_cursor(&repo.id, &entity).await?;
            let file = forge.file_text(&repo_ref, manifest, cursor).await?;
            // A repo without this manifest is normal, not an error; a 304 keeps the stored deps +
            // cursor.
            if file.missing || file.unchanged {
                continue;
            }
            all_unchanged = false;
            // Record the cursor now: this manifest version is handled whatever the parse outcome
            // (bad text counts as zero deps), so a restart skips instead of re-fetching and
            // re-failing.
            if let Some(cur) = &file.cursor {
                self.store.set_sync_cursor(&repo.id, &entity, cur).await?;
            }
            // Missing or malformed manifests give zero deps, but the manifest's stored rows are
            // still replaced so a removed dependency does not linger and drive a phantom
            // cross-repo link. `source = manifest` scopes the replace.
            let rows: Vec<DependencyRow> = file
                .content
                .as_deref()
                .and_then(|text| parse_manifest(manifest, text).ok())
                .map(|deps| {
                    deps.iter()
                        .map(|d| DependencyRow {
                            repo_id: repo.id.clone(),
                            ecosystem: d.ecosystem.clone(),
                            name: d.name.clone(),
                            version_req: d.version_req.clone(),
                            kind: d.kind.clone(),
                            source: manifest.to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let n = rows.len();
            self.store
                .replace_dependencies_from_source(&repo.id, manifest, &rows)
                .await?;
            fetched += n;
            persisted += n;
        }
        Ok(SyncReport {
            fetched,
            persisted,
            unchanged: all_unchanged,
        })
    }

    /// Index a repo's synced activity text for semantic search, incrementally: chunk and embed
    /// each commit message and issue title only when its text changed. A content hash per entity
    /// (`indexed_text`) lets unchanged text skip the model. Returns how many entities were
    /// (re-)embedded. No network.
    pub async fn index_repo<E: Embedder + ?Sized>(
        &self,
        repo: &Repo,
        embedder: &E,
    ) -> Result<usize, SyncError> {
        self.index_repo_with(repo, embedder, &mut || {}).await
    }

    /// Activity embedding (as `index_repo`), calling `tick` once per entity processed (embedded or
    /// skipped) for index progress.
    async fn index_repo_with<E: Embedder + ?Sized>(
        &self,
        repo: &Repo,
        embedder: &E,
        tick: &mut (dyn FnMut() + Send),
    ) -> Result<usize, SyncError> {
        const CHUNK_CHARS: usize = 256;
        let mut indexed = 0;

        let embed_chunks = |text: &str| -> Vec<(String, Vec<f32>)> {
            chunk_text(text, CHUNK_CHARS)
                .into_iter()
                .map(|c| {
                    let v = embedder.embed(&c.text);
                    (c.text, v)
                })
                .collect()
        };

        // Embed `text` for `(ref_kind, ref_id)` only if its content hash changed. Returns true
        // when it (re-)embedded.
        for commit in self.store.commits_for_repo(&repo.id).await? {
            if self
                .index_text_if_changed("commit", &commit.sha, &commit.message, &embed_chunks)
                .await?
            {
                indexed += 1;
            }
            tick();
        }
        for issue in self.store.issues_for_repo(&repo.id).await? {
            if self
                .index_text_if_changed("issue", &issue.id, &issue.title, &embed_chunks)
                .await?
            {
                indexed += 1;
            }
            tick();
        }
        Ok(indexed)
    }

    /// Embed `text` under `(ref_kind, ref_id)` only if its content hash differs from the last
    /// indexed one; records the new hash when it embeds. Returns whether it (re-)embedded.
    async fn index_text_if_changed(
        &self,
        ref_kind: &str,
        ref_id: &str,
        text: &str,
        embed_chunks: &impl Fn(&str) -> Vec<(String, Vec<f32>)>,
    ) -> Result<bool, SyncError> {
        let hash = format!("{:016x}", core_embed::content_hash(text));
        if self.store.indexed_text_hash(ref_kind, ref_id).await? == Some(hash.clone()) {
            return Ok(false); // text unchanged since last index: skip the model
        }
        let chunks = embed_chunks(text);
        if chunks.is_empty() {
            return Ok(false);
        }
        self.store
            .replace_embeddings(ref_kind, ref_id, &chunks)
            .await?;
        self.store
            .set_indexed_text_hash(ref_kind, ref_id, &hash)
            .await?;
        Ok(true)
    }

    /// Index a repo's source code for semantic code search. Fetches the file tree, then for each
    /// source file (by extension) whose blob SHA differs from the last indexed SHA, fetches the
    /// blob, chunks it by line window, embeds it, and stores chunks under `ref_kind = "code"`,
    /// `ref_id = "<repo_id>#<path>"`. Incremental: an unchanged tree (304) fetches no blobs, an
    /// unchanged file is not refetched, and a file gone from the tree has its embeddings and
    /// tracking row pruned. Returns the number of files (re-)embedded. Uses the engine's embedder.
    pub async fn sync_code(&self, repo: &Repo) -> Result<usize, SyncError> {
        let fetch = self.fetch_changed_code(repo).await?;
        self.index_code(repo, &fetch).await
    }

    /// Fetch a repo's changed source files (the API half of code indexing). Gates on `pushed_at`,
    /// diffs the tree against stored blob SHAs, and fetches the decoded text of each changed file.
    /// No embedding (that is `index_code`, run by the index worker). Best-effort per blob: an
    /// unreadable one is skipped. Stays in the serialized fetch lane for the rate budget.
    pub async fn fetch_changed_code(&self, repo: &Repo) -> Result<CodeFetch, SyncError> {
        self.store.upsert_repo(repo).await?;
        let repo_ref: RepoRef = repo.into();
        let forge = self.forge_for_repo(&repo.id)?;
        let pushed_at = forge
            .repo_meta(&repo_ref)
            .await
            .ok()
            .and_then(|m| m.pushed_at);
        let code_cursor = self.store.sync_cursor(&repo.id, "code").await?;
        if let Some(pushed) = &pushed_at {
            if code_cursor.as_deref() == Some(pushed.as_str()) {
                return Ok(CodeFetch {
                    changed: Vec::new(),
                    removed: Vec::new(),
                    pushed_at,
                    gated: true,
                    complete: true,
                    pending: 0,
                    skipped: Vec::new(),
                });
            }
        }

        let tree = match forge.tree(&repo_ref).await {
            Ok(t) => t,
            // No default branch / empty repo: nothing to index, not an error.
            Err(ForgeError::EmptyRepo) => {
                return Ok(CodeFetch {
                    changed: Vec::new(),
                    removed: Vec::new(),
                    pushed_at,
                    gated: false,
                    complete: true,
                    pending: 0,
                    skipped: Vec::new(),
                })
            }
            Err(e) => return Err(e.into()),
        };
        // Every source file in the tree is wanted; only the transport's blob ceiling excludes one,
        // and that is recorded.
        let mut wanted: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
        let mut skipped: Vec<SkippedFile> = Vec::new();
        for entry in tree {
            if entry.kind != "blob" || source_language(&entry.path).is_none() {
                continue;
            }
            let size = entry.size.unwrap_or(0);
            if size > MAX_BLOB_BYTES {
                skipped.push(SkippedFile {
                    path: entry.path,
                    bytes: size,
                    reason: "larger than the forge will return for a single file",
                });
                continue;
            }
            wanted.insert(entry.path, entry.sha);
        }

        let stored = self.store.code_file_shas(&repo.id).await?;
        let mut changed = Vec::new();
        // A transient blob-fetch failure leaves the pass incomplete: skip that file, and do not
        // advance the `code` cursor below, or the `pushed_at` gate would skip it until the next
        // push. A deterministic skip (undecodable blob) does not block the cursor; retrying would
        // not help.
        let mut complete = true;
        // Changed files not fetched this pass. They are not pruned: the next pass re-diffs the
        // tree, finds files whose stored SHA still differs, and takes the next batch. `pending`
        // reports them.
        let mut pending = 0usize;
        for (file_path, blob_sha) in &wanted {
            if stored.get(file_path) == Some(blob_sha) {
                continue; // content unchanged since last index: do not refetch the blob
            }
            if changed.len() >= CODE_FILES_PER_PASS {
                pending += 1;
                continue;
            }
            match forge.blob_text(&repo_ref, blob_sha).await {
                Ok(Some(text)) => changed.push((file_path.clone(), blob_sha.clone(), text)),
                // Present but not decodable text: a deterministic skip (retrying would not help).
                Ok(None) => continue,
                // A transient failure marks the pass incomplete so the cursor stays behind and the
                // next sync retries this blob.
                Err(_) => {
                    complete = false;
                    continue;
                }
            }
        }
        // Files indexed before but gone from the tree: deleted, or unfetchable at the transport's
        // ceiling (also in `skipped`, so reported). A file waiting for a later pass is still in
        // `wanted`, so it is not pruned.
        let removed: Vec<String> = stored
            .keys()
            .filter(|p| !wanted.contains_key(*p))
            .cloned()
            .collect();
        Ok(CodeFetch {
            changed,
            removed,
            pushed_at,
            gated: false,
            // Work still owed keeps the `code` cursor behind, so the next sync is not gated by
            // `pushed_at`.
            complete: complete && pending == 0,
            pending,
            skipped,
        })
    }

    /// Embed the changed code and prune removed files (the CPU/DB half of code indexing, run by
    /// the index worker). Chunks and embeds each changed file, replaces its `code` chunks, records
    /// its blob SHA, prunes removed files, and advances the `code` cursor to `pushed_at`. Returns
    /// the number of files (re-)embedded.
    pub async fn index_code(&self, repo: &Repo, fetch: &CodeFetch) -> Result<usize, SyncError> {
        self.index_code_with(repo, fetch, &mut || {}).await
    }

    /// Code embedding (as `index_code`), calling `tick` once per changed file embedded, for index
    /// progress.
    async fn index_code_with(
        &self,
        repo: &Repo,
        fetch: &CodeFetch,
        tick: &mut (dyn FnMut() + Send),
    ) -> Result<usize, SyncError> {
        let mut indexed = 0;
        for (file_path, blob_sha, text) in &fetch.changed {
            let lang = source_language(file_path).unwrap_or("");
            let chunks: Vec<(String, Vec<f32>)> = chunk_code(text, lang)
                .into_iter()
                .map(|c| {
                    let v = self.embedder.embed(&c.text);
                    (c.text, v)
                })
                .collect();
            let ref_id = format!("{}#{}", repo.id, file_path);
            self.store
                .replace_embeddings("code", &ref_id, &chunks)
                .await?;
            self.store
                .upsert_code_file(&repo.id, file_path, blob_sha)
                .await?;
            // Compute and store AST health metrics while the source text is in memory.
            // `health_score` is None for a language with no grammar.
            let metrics = core_codehealth::compute_metrics(text, lang);
            let score = core_codehealth::health_score(&metrics);
            self.store
                .upsert_file_health(
                    &repo.id,
                    file_path,
                    metrics.loc,
                    metrics.functions,
                    metrics.branches,
                    score,
                )
                .await?;
            indexed += 1;
            tick();
        }
        for old_path in &fetch.removed {
            let ref_id = format!("{}#{}", repo.id, old_path);
            self.store.replace_embeddings("code", &ref_id, &[]).await?;
            self.store.delete_code_file(&repo.id, old_path).await?;
            self.store
                .delete_file_health_path(&repo.id, old_path)
                .await?;
        }
        // Advance the `code` cursor only when every wanted blob was fetched. After a transient
        // failure (`complete == false`) the cursor stays behind so the next sync is not gated by
        // `pushed_at` and re-fetches the missing blob.
        if fetch.complete {
            if let Some(pushed) = &fetch.pushed_at {
                self.store.set_sync_cursor(&repo.id, "code", pushed).await?;
            }
        }
        Ok(indexed)
    }

    /// Resolve declared dependencies into `repo -> repo` `depends_on` links between watched repos.
    /// Two exact matches: (1) the dependency `name` equals another repo's short name (cargo crate
    /// `widget` -> `acme/widget`); (2) the dependency encodes `owner/repo` on the git host of a
    /// configured forge (a go module path `gitea.example.com/acme/widget`, or an npm/pip git URL;
    /// `core_graph::repo_on_host`), matched against that forge's repos. The host routes the second
    /// match, so two forges watching the same `owner/name` stay distinct. An unresolved dependency
    /// is not linked. Idempotent (`add_link` ignores duplicates); no self-links. Cross-repo, so it
    /// runs once over the whole store. Returns the links written.
    pub async fn link_cross_repo_dependencies(&self) -> Result<usize, SyncError> {
        let repos = self.store.repos().await?;
        // short name (e.g. "widget" from "acme/widget") -> repo id. Last wins on a name collision.
        let name_to_id: std::collections::HashMap<&str, &str> = repos
            .iter()
            .map(|r| (r.name.as_str(), r.id.as_str()))
            .collect();
        // (forge id, full name) -> repo id, for host-path resolution. Keyed by forge too, so
        // `github.com/acme/widget` cannot link to a same-named repo on another forge.
        let full_to_id: std::collections::HashMap<(&str, &str), &str> = repos
            .iter()
            .filter_map(|r| Some(((forge_id_of(&r.id)?, r.full_name.as_str()), r.id.as_str())))
            .collect();
        // Each forge's git host, from its stored API base URL. A forge whose base URL has no
        // authority has no host: its repos still link by bare name.
        let hosts: Vec<(String, String)> = self
            .store
            .forges()
            .await?
            .into_iter()
            .filter_map(|f| forge_host(&f.base_url).map(|host| (f.id, host)))
            .collect();

        // De-dup edges in-process so a dep matched by both name and host path writes once.
        let mut edges: std::collections::BTreeSet<(String, String)> =
            std::collections::BTreeSet::new();
        for repo in &repos {
            for dep in self.store.dependencies_for_repo(&repo.id).await? {
                if let Some(&dst_id) = name_to_id.get(dep.name.as_str()) {
                    if dst_id != repo.id {
                        edges.insert((repo.id.clone(), dst_id.to_string()));
                    }
                }
                for (forge_id, host) in &hosts {
                    let Some(full) = core_graph::repo_on_host(
                        &dep.ecosystem,
                        &dep.name,
                        dep.version_req.as_deref(),
                        host,
                    ) else {
                        continue;
                    };
                    if let Some(&dst_id) = full_to_id.get(&(forge_id.as_str(), full.as_str())) {
                        if dst_id != repo.id {
                            edges.insert((repo.id.clone(), dst_id.to_string()));
                        }
                    }
                }
            }
        }
        for (src, dst) in &edges {
            self.store
                .add_link("repo", src, "repo", dst, "depends_on")
                .await?;
        }
        Ok(edges.len())
    }

    /// Project this repo's GitHub issues into the generic `work_items` table and derive
    /// cross-reference links. Each issue becomes a `work_item` (`source = 'github'`, id reused,
    /// `status_category` from its state and whether an open PR links it); each `#N` in a PR/issue
    /// body that resolves to a stored issue becomes a link to that work item (`closes` |
    /// `mentions`). Order-independent (reads persisted bodies and the number -> id map).
    /// Idempotent: `add_link` and `upsert_github_work_item` ignore or merge duplicates.
    /// Self-references are skipped and unresolved numbers dropped. Returns the number of links
    /// written.
    pub async fn link_references(&self, repo: &Repo) -> Result<usize, SyncError> {
        use std::collections::{HashMap, HashSet};
        let issues = self.store.issues_for_repo(&repo.id).await?;
        let prs = self.store.pull_requests(&repo.id).await?;
        let number_to_id: HashMap<i64, String> =
            issues.iter().map(|i| (i.number, i.id.clone())).collect();

        // Resolve a body's references to edges (src -> issue id), dropping unresolved numbers and
        // an issue's self-reference. Pure; the async writes come after.
        let edges_from =
            |src_kind: &str, src_id: &str, body: &str| -> Vec<(String, String, String)> {
                parse_issue_references(body)
                    .into_iter()
                    .filter_map(|r| {
                        let dst_id = number_to_id.get(&r.number)?;
                        if src_kind == "issue" && dst_id == src_id {
                            return None; // an issue referencing itself
                        }
                        Some((src_id.to_string(), dst_id.clone(), r.relation))
                    })
                    .collect()
            };

        // If the forge reports authoritative closing-issue references, `closes` comes from them
        // and a body `#N` that is not one of them is a weaker `mention`.
        let supports_closing_refs = self
            .forge_for_repo(&repo.id)
            .map(|f| f.capabilities().closing_issue_refs)
            .unwrap_or(false);
        // Authoritative `pull_request -> work_item` closes pairs captured during sync.
        let mut authoritative: HashSet<(String, String)> = HashSet::new();
        for pr in &prs {
            for number in self.store.pr_closing_issues(&pr.id).await? {
                if let Some(issue_id) = number_to_id.get(&number) {
                    authoritative.insert((pr.id.clone(), issue_id.clone()));
                }
            }
        }

        let mut edges: Vec<(String, String, String, String)> = Vec::new();
        // Authoritative closes first; body-parsed references fill in mentions (or both, on a forge
        // with no closing-ref support, where the body keyword is the only signal).
        for (pr_id, issue_id) in &authoritative {
            edges.push((
                "pull_request".to_string(),
                pr_id.clone(),
                issue_id.clone(),
                "closes".to_string(),
            ));
        }
        for pr in &prs {
            if let Some(body) = &pr.body {
                for (src, dst, rel) in edges_from("pull_request", &pr.id, body) {
                    if authoritative.contains(&(pr.id.clone(), dst.clone())) {
                        continue; // already an authoritative closes edge
                    }
                    let rel = if supports_closing_refs {
                        "mentions".to_string()
                    } else {
                        rel
                    };
                    edges.push(("pull_request".to_string(), src, dst, rel));
                }
            }
        }
        for issue in &issues {
            if let Some(body) = &issue.body {
                for (src, dst, rel) in edges_from("issue", &issue.id, body) {
                    edges.push(("issue".to_string(), src, dst, rel));
                }
            }
        }

        // Issues with an open PR linking them ("work started"); drives the in-progress status.
        let open_pr: HashSet<&str> = prs
            .iter()
            .filter(|p| p.merged_at.is_none() && p.state == "open")
            .map(|p| p.id.as_str())
            .collect();
        let issue_has_open_pr: HashSet<&str> = edges
            .iter()
            .filter(|(src_kind, src_id, _, _)| {
                src_kind == "pull_request" && open_pr.contains(src_id.as_str())
            })
            .map(|(_, _, dst_id, _)| dst_id.as_str())
            .collect();

        // A PR's creation time, to date when an issue entered "in progress" (earliest linked PR's
        // open).
        let pr_created: HashMap<&str, &str> = prs
            .iter()
            .map(|p| (p.id.as_str(), p.created_at.as_str()))
            .collect();
        let earliest_linked_pr_at = |issue_id: &str| -> Option<&str> {
            edges
                .iter()
                .filter(|(src_kind, _, dst, _)| src_kind == "pull_request" && dst == issue_id)
                .filter_map(|(_, src_id, _, _)| pr_created.get(src_id.as_str()).copied())
                .min()
        };

        // Project every issue into work_items with its status_category: closed -> done; open with
        // a linked open PR -> indeterminate (in progress); open otherwise -> new. Status history
        // comes from real timestamps: new at creation, in-progress at the earliest linked PR, done
        // at close.
        for issue in &issues {
            let status = if issue.state == "closed" || issue.closed_at.is_some() {
                "done"
            } else if issue_has_open_pr.contains(issue.id.as_str()) {
                "indeterminate"
            } else {
                "new"
            };
            // A bug-labeled issue (any label containing "bug", case-insensitive, as
            // core_summary::is_bug) gets the `bug` kind, else `issue`.
            let kind = if issue
                .labels
                .split('\n')
                .any(|l| l.to_ascii_lowercase().contains("bug"))
            {
                "bug"
            } else {
                "issue"
            };
            self.store
                .upsert_github_work_item(&issue.id, &issue.title, kind, &issue.state, status)
                .await?;

            let mut history: Vec<(&str, String)> = vec![("new", issue.created_at.clone())];
            if let Some(at) = earliest_linked_pr_at(&issue.id) {
                history.push(("indeterminate", at.to_string()));
            }
            if issue.state == "closed" {
                if let Some(closed) = &issue.closed_at {
                    history.push(("done", closed.clone()));
                }
            }
            self.store
                .replace_work_item_status_history(&issue.id, &history)
                .await?;
        }

        for (src_kind, src_id, dst_id, relation) in &edges {
            self.store
                .add_link(src_kind, src_id, "work_item", dst_id, relation)
                .await?;
        }
        Ok(edges.len())
    }

    /// Like `run_scheduler`, but reads the source registry from the store at the start of every
    /// pass, so adding or removing a source takes effect on the next pass. The poll period is read
    /// from `settings.sync_period_secs` each pass too. Each pass polls every repo-kind source
    /// once, spread across the period. An empty registry sleeps one period and re-checks. Runs
    /// until the task is dropped.
    pub async fn run_scheduler_over_registry<Cb>(&self, mut emit: Cb)
    where
        Cb: FnMut(SyncProgress),
    {
        // Rebuild the index if the bundled model changed since last run, before any embedding.
        if let Err(e) = self.reconcile_index_model().await {
            eprintln!("reconcile index model: {e}");
        }
        loop {
            let period = self.scheduler_period().await;
            // Gate 2 of 4: before the pass plans anything, so nothing is fetched, discovered or
            // written while the store is at the ceiling. Nothing is recorded: the stop is
            // re-derived every pass, so resuming is free once usage falls under the budget. The
            // wait is shorter than the poll period so freeing space takes effect within about a
            // minute.
            match self.store.budget_state().await {
                Ok(budget) if !budget.allows_sync() => {
                    tokio::time::sleep(BUDGET_RECHECK.min(period)).await;
                    continue;
                }
                _ => {}
            }
            let repos: Vec<Repo> = match self.store.sources().await {
                Ok(sources) => sources.iter().filter_map(repo_from_source).collect(),
                Err(_) => Vec::new(),
            };
            if repos.is_empty() {
                // An empty registry is not an idle app. Only pinned repos become sources, so a
                // people-first board has none; skipping straight to the sleep would skip the
                // discovery call at the bottom of the loop. Discover first, then wait a full
                // period and re-read (which picks up new sources).
                let _ = self.discover_all_boards().await;
                tokio::time::sleep(period).await;
                continue;
            }
            let total = repos.len();
            let step = (period / total as u32).max(Duration::from_millis(1));
            let mut ticker = interval(step);
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            for (i, repo) in repos.iter().enumerate() {
                ticker.tick().await;
                let report = self.sync_all(repo).await.map_err(|e| e.to_string());
                let tick = SyncTick {
                    source_id: repo.id.clone(),
                    report,
                };
                emit(SyncProgress::finished(&repo.full_name, i + 1, total, tick));
            }
            // People-first boards: discover + sync the repos their people touch (best-effort).
            let _ = self.discover_all_boards().await;
            // Whole registry synced: resolve cross-repo dependency edges now that every watched
            // repo's deps are present. Best-effort: a failure does not stop the loop.
            let _ = self.link_cross_repo_dependencies().await;
        }
    }

    /// The scheduler poll period from `settings.sync_period_secs`. Falls back to 300s if settings
    /// cannot be read or the value is non-positive (zero would busy-loop).
    async fn scheduler_period(&self) -> Duration {
        let secs = self
            .store
            .settings()
            .await
            .map(|s| s.sync_period_secs)
            .unwrap_or(300);
        Duration::from_secs(secs.max(1) as u64)
    }

    /// Ensure the embedding index matches the active embedder and chunking scheme. The recorded
    /// identity is `name:dim:scheme` (`core_embed::INDEX_SCHEME_VERSION` is the chunking version).
    /// If it differs from the last sync, stored vectors and incremental markers are dropped
    /// (`clear_index`) so the next sync re-embeds everything. A fresh store just records the
    /// identity. Returns whether it reset the index.
    pub async fn reconcile_index_model(&self) -> Result<bool, SyncError> {
        const KEY: &str = "embedder_id";
        let current = self.embedder_identity();
        let stored = self.store.meta_get(KEY).await?;
        if stored.as_deref() == Some(current.as_str()) {
            return Ok(false);
        }
        // Only clear when a different model produced the existing index.
        let reset = stored.is_some();
        if reset {
            eprintln!(
                "embedder changed ({} -> {current}); clearing the index to re-embed",
                stored.as_deref().unwrap_or("?")
            );
            self.store.clear_index().await?;
        }
        self.store.meta_set(KEY, &current).await?;
        Ok(reset)
    }

    /// Sync the full target repo set once, now, with no stagger (the manual "sync now" trigger).
    /// It first plans: the deduped union of registry source repos, every board's pinned repos, and
    /// every people/org board's freshly enumerated discovered repos (the listing calls happen
    /// here). Then it syncs each repo, calling `emit` with a `SyncProgress` (`done`/`total`) as
    /// each completes. Finally it records each board's discovered set and resolves cross-repo
    /// dependency edges. Returns the number of repos synced; one bad repo does not stop the rest.
    pub async fn sync_all_sources<Cb>(&self, mut emit: Cb) -> Result<usize, SyncError>
    where
        Cb: FnMut(SyncProgress),
    {
        use std::collections::{BTreeMap, HashSet};

        // Gate 2 of 4, the manual half: refuse before planning, so "Sync now" over the ceiling
        // makes no network call and writes nothing. Returns zero, not an error: nothing failed.
        // The caller reads the budget state to say why, and the last-synced stamp stays put.
        if !self.store.budget_state().await?.allows_sync() {
            return Ok(0);
        }

        // A bundled-model change invalidates existing vectors/markers; drop them up front.
        self.reconcile_index_model().await?;

        // ---- plan ----
        let mut targets: BTreeMap<String, Repo> = BTreeMap::new();
        // Repo ids wanted for indexing (code + activity embed): those on at least one non-org
        // board. An org-only repo is fetched for its metrics but never embedded.
        let mut indexed: HashSet<String> = HashSet::new();
        for source in self.store.sources().await? {
            if let Some(repo) = repo_from_source(&source) {
                indexed.insert(repo.id.clone());
                targets.insert(repo.id.clone(), repo);
            }
        }
        let boards = self.store.boards().await?;
        let board_total = boards.len();
        // For each discovery board, the repo ids it watches this pass. Updated as each repo
        // finishes, so a repo surfaces on its board as soon as its fetch completes.
        let mut discovered: Vec<(String, HashSet<String>)> = Vec::new();
        for (bi, board) in boards.iter().enumerate() {
            emit(SyncProgress::planning(&board.name, bi + 1, board_total));
            if board.repos.is_empty() {
                // Best-effort: a failed discovery (private org, transient 5xx) must not abort the
                // sync. Fall back to the board's last-known discovered repos and leave its
                // discovered set untouched.
                let repos = match self.enumerate_board_repos(board).await {
                    Ok(repos) => repos,
                    Err(e) => {
                        eprintln!(
                            "discovery failed for board {}: {e}; using last-known repos",
                            board.name
                        );
                        let prev = self.store.board_discovered_repos(&board.id).await?;
                        let ids: HashSet<String> = prev.iter().cloned().collect();
                        if board.kind != BoardKind::Org {
                            indexed.extend(ids.iter().cloned());
                        }
                        for repo_id in &prev {
                            if let Some(repo) = self.store.get_repo(repo_id).await? {
                                targets.entry(repo.id.clone()).or_insert(repo);
                            }
                        }
                        discovered.push((board.id.clone(), ids));
                        continue;
                    }
                };
                let ids: HashSet<String> = repos.iter().map(|r| r.id.clone()).collect();
                if board.kind != BoardKind::Org {
                    indexed.extend(ids.iter().cloned());
                }
                // Prune the board to repos still discovered this pass, but keep still-valid ones
                // visible while the sync runs.
                let prev = self.store.board_discovered_repos(&board.id).await?;
                let kept: Vec<String> = prev.into_iter().filter(|id| ids.contains(id)).collect();
                self.store
                    .replace_board_discovered_repos(&board.id, &kept)
                    .await?;
                for repo in repos {
                    targets.entry(repo.id.clone()).or_insert(repo);
                }
                discovered.push((board.id.clone(), ids));
            } else {
                // Pinned repos are the board's explicit set; prefer the stored row, else construct
                // it.
                for repo_id in &board.repos {
                    let repo = match self.store.get_repo(repo_id).await? {
                        Some(r) => r,
                        None => match repo_from_id(repo_id) {
                            Some(r) => r,
                            None => continue,
                        },
                    };
                    if board.kind != BoardKind::Org {
                        indexed.insert(repo.id.clone());
                    }
                    targets.entry(repo.id.clone()).or_insert(repo);
                }
            }
        }

        // ---- sync ----
        let repos: Vec<Repo> = targets.into_values().collect();
        let total = repos.len();
        let mut synced = 0usize;
        for (i, repo) in repos.iter().enumerate() {
            // Gate 4 of 4: a pass that crosses the ceiling stops at a repo boundary. The repo just
            // written is finished, and the remaining ones are left alone, not reported as failures.
            if !self.store.budget_state().await?.allows_sync() {
                break;
            }
            let done = i + 1;
            let report = self
                .sync_all_steps(repo, indexed.contains(&repo.id), |step, sd, st| {
                    emit(SyncProgress::step(
                        &repo.full_name,
                        done,
                        total,
                        step,
                        sd,
                        st,
                    ));
                })
                .await
                // Plain-language per-source error, shown on the board.
                .map_err(|e| e.user_message());
            if report.is_ok() {
                // Surface this repo on every discovery board that watches it now, so the overview
                // picks it up on the completion event.
                for (board_id, ids) in &discovered {
                    if ids.contains(&repo.id) {
                        self.store
                            .add_board_discovered_repo(board_id, &repo.id)
                            .await?;
                    }
                }
            }
            emit(SyncProgress::finished(
                &repo.full_name,
                done,
                total,
                SyncTick {
                    source_id: repo.id.clone(),
                    report,
                },
            ));
            synced += 1;
        }

        let _ = self.link_cross_repo_dependencies().await;
        // Count what was synced, not what was planned: a pass stopped at the ceiling is not the
        // whole set.
        Ok(synced)
    }

    /// Repos a person works in on a forge: recently touched (per-user activity feed) plus owned.
    /// Returns distinct `owner/name`, recent activity first. Empty when the forge has no such
    /// discovery.
    pub async fn discover_person_repos(
        &self,
        forge_id: &str,
        login: &str,
    ) -> Result<Vec<String>, SyncError> {
        Ok(self.forge_by_id(forge_id)?.person_repos(login).await?)
    }

    /// All repos in an org on a forge. Returns distinct `owner/name`. Empty when the forge has no
    /// org concept.
    pub async fn enumerate_org_repos(
        &self,
        forge_id: &str,
        org: &str,
    ) -> Result<Vec<String>, SyncError> {
        Ok(self.forge_by_id(forge_id)?.org_repos(org).await?)
    }

    /// Apply a team board's archived-repo discovery policy, keeping forge relevance order.
    /// Activity events carry no archive state, so when archived repos are excluded the per-repo
    /// metadata endpoint is used for every candidate. Calls are sequential for rate budgets and
    /// stop once the board cap is filled. Opting in skips them. A transient metadata failure
    /// aborts the enumeration so callers keep the last-known discovered set; a repository that
    /// disappeared between listing and lookup is skipped.
    async fn filter_discovered_repos(
        &self,
        board: &Board,
        forge_id: &str,
        candidates: Vec<String>,
    ) -> Result<Vec<Repo>, SyncError> {
        let forge = self.forge_by_id(forge_id)?;
        let mut repos = Vec::new();
        for full in candidates {
            let Some(repo) = repo_from_full(forge_id, &full) else {
                continue;
            };
            if !board.include_archived {
                match forge.repo_meta(&RepoRef::from(&repo)).await {
                    Ok(meta) if meta.archived => continue,
                    Ok(_) => {}
                    // A stale event/listing entry is not a discovery-wide failure. Other failures
                    // are transient and must not yield a partial prune.
                    Err(ForgeError::NotFound | ForgeError::EmptyRepo) => continue,
                    Err(error) => return Err(error.into()),
                }
            }
            repos.push(repo);
            if repos.len() >= MAX_DISCOVERED_REPOS {
                break;
            }
        }
        Ok(repos)
    }

    /// What the named forge supports, so a caller can degrade (e.g. report contributor import as
    /// unsupported rather than implying a repo has none).
    pub fn forge_capabilities(
        &self,
        forge_id: &str,
    ) -> Result<core_forge::Capabilities, SyncError> {
        Ok(self.forge_by_id(forge_id)?.capabilities())
    }

    /// User logins matching `query` on a forge, for the source pickers. Empty on a forge
    /// without search (see [`core_forge::Capabilities::search`]).
    pub async fn search_users(
        &self,
        forge_id: &str,
        query: &str,
    ) -> Result<Vec<String>, SyncError> {
        Ok(self.forge_by_id(forge_id)?.search_users(query).await?)
    }

    /// Organization logins matching `query` on a forge. Empty when unsupported.
    pub async fn search_orgs(&self, forge_id: &str, query: &str) -> Result<Vec<String>, SyncError> {
        Ok(self.forge_by_id(forge_id)?.search_orgs(query).await?)
    }

    /// The "owner/name" of repos matching `query` on a forge. Empty when unsupported.
    pub async fn search_repos(
        &self,
        forge_id: &str,
        query: &str,
    ) -> Result<Vec<String>, SyncError> {
        Ok(self.forge_by_id(forge_id)?.search_repos(query).await?)
    }

    /// Enumerate the repos a board watches by discovery, without syncing them. A `Team` discovers
    /// the repos its people touch or own (filtered to `org` if set); an `Org` takes its org's
    /// repos, most-recently-active first; a `Repo` board (or any board with pinned repos) returns
    /// empty. Archived repos are excluded from a team's people-derived discovery unless
    /// `include_archived`; explicit pins and org enumeration bypass this. Capped at
    /// [`MAX_DISCOVERED_REPOS`]; both kinds keep the forge's order through the cap. A team's cap
    /// is shared round-robin across its people (everyone's top repo, then everyone's second), so
    /// one prolific member cannot spend it alone. Returning the set up front gives a caller the
    /// full sync target and progress total.
    pub async fn enumerate_board_repos(&self, board: &Board) -> Result<Vec<Repo>, SyncError> {
        if !board.repos.is_empty() {
            return Ok(Vec::new());
        }
        // The board must name a configured forge to discover against. Resolving it up front
        // validates it and gives the forge id the discovered repo ids are namespaced with.
        let forge_id = board
            .forge_id
            .as_deref()
            .ok_or_else(|| SyncError::UnknownForge(format!("board {}", board.id)))?;
        self.forge_by_id(forge_id)?; // fail early if the named forge is not configured

        match board.kind {
            BoardKind::Org => {
                // The whole org, most-recently-active first (the forge returns it pushed-desc).
                // Keep that order through dedupe and cap so the slice is the top-N active repos.
                let Some(org) = &board.org else {
                    return Ok(Vec::new());
                };
                let mut seen = std::collections::HashSet::new();
                let mut ordered = Vec::new();
                for full in self.enumerate_org_repos(forge_id, org).await? {
                    if seen.insert(full.clone()) {
                        ordered.push(full);
                    }
                }
                Ok(ordered
                    .into_iter()
                    .take(MAX_DISCOVERED_REPOS)
                    .filter_map(|f| repo_from_full(forge_id, &f))
                    .collect())
            }
            // Team is people-first; a repo board's repo is pinned (handled by the early return
            // above).
            BoardKind::Team | BoardKind::Repo => {
                // The forge returns each person's repos most-relevant-first. Keep that order so
                // the cap holds the active slice, as in the org branch.
                let mut per_person: Vec<Vec<String>> = Vec::new();
                for login in &board.people {
                    // Best-effort per person: a stale or renamed login (404) or transient error
                    // for one member must not drop the board's other repos.
                    let touched = match self.discover_person_repos(forge_id, login).await {
                        Ok(repos) => repos,
                        Err(e) => {
                            eprintln!("discovery failed for {login}: {e}");
                            continue;
                        }
                    };
                    let mut theirs = Vec::new();
                    for full in touched {
                        if let Some(org) = &board.org {
                            if full.split('/').next() != Some(org.as_str()) {
                                continue;
                            }
                        }
                        theirs.push(full);
                    }
                    per_person.push(theirs);
                }
                // Take the cap a round at a time across the people (everyone's top repo, then
                // everyone's second) so one prolific member cannot spend it all. A person with a
                // short list stops contributing in later rounds, handing their share to the others.
                let rounds = per_person.iter().map(Vec::len).max().unwrap_or(0);
                let mut seen = std::collections::HashSet::new();
                let mut ordered: Vec<String> = Vec::new();
                for round in 0..rounds {
                    for theirs in &per_person {
                        let Some(full) = theirs.get(round) else {
                            continue;
                        };
                        if seen.insert(full.clone()) {
                            ordered.push(full.clone());
                        }
                    }
                }
                self.filter_discovered_repos(board, forge_id, ordered).await
            }
        }
    }

    /// Discover and sync a board's repos, recording them as its discovered set. A no-op for a
    /// board with pinned repos. Used by the background scheduler; the manual sync inlines the
    /// enumeration so discovery counts in its progress. Returns the count synced.
    pub async fn discover_board_repos(&self, board: &Board) -> Result<usize, SyncError> {
        if !board.repos.is_empty() {
            return Ok(0);
        }
        let repos = self.enumerate_board_repos(board).await?;
        // Org boards fetch activity for the rollup but never embed; team boards index fully.
        let index = board.kind != BoardKind::Org;
        // A refresh failure is not a scope change: keep an already-visible, still-eligible repo on
        // the board and retry next pass. Only a fresh enumeration that omits the repo may remove
        // it.
        let previous: std::collections::HashSet<String> = self
            .store
            .board_discovered_repos(&board.id)
            .await?
            .into_iter()
            .collect();
        let mut repo_ids = Vec::new();
        let mut synced_count = 0usize;
        for repo in &repos {
            // Best-effort: a repo we cannot sync (private/404) does not stop the rest of the board.
            let synced = if index {
                self.sync_all(repo).await
            } else {
                self.sync_metrics_only(repo).await
            };
            if synced.is_ok() {
                synced_count += 1;
                repo_ids.push(repo.id.clone());
            } else if previous.contains(&repo.id) {
                repo_ids.push(repo.id.clone());
            }
        }
        self.store
            .replace_board_discovered_repos(&board.id, &repo_ids)
            .await?;
        Ok(synced_count)
    }

    /// Run repo discovery for every board (after each scheduler pass or manual sync). Best-effort
    /// per board.
    pub async fn discover_all_boards(&self) -> Result<(), SyncError> {
        for board in self.store.boards().await? {
            let _ = self.discover_board_repos(&board).await;
        }
        Ok(())
    }
}

/// The live index-lane map. Shared with the index worker's progress callback so the worker can
/// update it without borrowing the generic engine.
type IndexStatusMap = std::sync::Mutex<std::collections::BTreeMap<String, IndexProgress>>;

/// Record a repo's index state into the shared map. A terminal `Indexed` state is dropped; other
/// states are kept so `index_status()` reflects the live queue. A record from a superseded pass is
/// ignored.
fn record_index_status_in(map: &IndexStatusMap, p: IndexProgress) {
    if let Ok(mut map) = map.lock() {
        // The fetch lane and index worker write this map from separate tasks, so an old pass can
        // report after the repo was re-queued. Drop such a record rather than let it remove or
        // overwrite the newer entry. Equal is the ordinary case (one pass writing several times);
        // newer than the stored entry means that entry belongs to a finished pass.
        if map
            .get(&p.repo_id)
            .is_some_and(|current| p.generation < current.generation)
        {
            return;
        }
        // `Indexed` is the one terminal state dropped. `Partial` and `Paused` are terminal too but
        // are kept so the user can still see them after navigating away and back.
        if p.state == IndexState::Indexed {
            map.remove(&p.repo_id);
        } else {
            map.insert(p.repo_id.clone(), p);
        }
    }
}

/// Parse a forge's `owner/name` into an observed `Repo` with a forge-namespaced id
/// (`repo:<forge_id>/<owner>/<name>`). `None` if `full` is not `owner/name`.
fn repo_from_full(forge_id: &str, full: &str) -> Option<Repo> {
    let (owner, name) = full.split_once('/')?;
    Some(Repo {
        id: repo_id(forge_id, full),
        owner: owner.to_string(),
        name: name.to_string(),
        full_name: full.to_string(),
        ownership: "observed".to_string(),
    })
}

/// Parse a stored repo id (`repo:<forge_id>/<owner>/<name>`) into a `Repo`. Used for a pinned
/// board repo not yet synced (so not in the repos table); pinned repos are `owned`. `None` if
/// malformed.
fn repo_from_id(id: &str) -> Option<Repo> {
    let forge = forge_id_of(id)?;
    let full = full_name_of(id)?;
    repo_from_full(forge, full).map(|mut r| {
        r.ownership = "owned".to_string();
        r
    })
}

/// Map a repo-kind source row to the `Repo` the engine syncs, namespaced by the source's forge.
/// `None` for non-repo sources, a source with no forge assigned, or a malformed `owner/name`.
pub fn repo_from_source(source: &SourceRow) -> Option<Repo> {
    if source.kind != "repo" {
        return None;
    }
    let forge_id = source.forge_id.as_deref()?;
    repo_from_full(forge_id, &source.name).map(|mut r| {
        r.ownership = source.ownership.clone();
        r
    })
}

/// Per-type persisted counts from one full-repo sync.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullSyncReport {
    pub commits: usize,
    pub pull_requests: usize,
    pub issues: usize,
    pub releases: usize,
    pub reviews: usize,
    pub ci_runs: usize,
    pub dependencies: usize,
    /// Source files that changed and were queued for (or did) indexing this pass; 0 when
    /// nothing was pushed.
    pub code_files: usize,
    /// Older commits + pull requests the history backfill lane pulled in this pass; 0 when none is
    /// owed.
    #[serde(default)]
    pub backfilled: usize,
}

/// What one [`SyncEngine::backfill_repo`] chunk did: older items pulled in per entity walk, and
/// whether any walk still has history left.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackfillReport {
    pub commits: usize,
    pub pull_requests: usize,
    pub pending: bool,
}

/// The changed-code payload the fetch lane produces and the index lane consumes: changed source
/// files (path, blob sha, decoded text), paths gone from the tree, and the repo's `pushed_at`
/// watermark. `gated` is true when nothing was pushed since the last index. `complete` is false
/// when a wanted blob's fetch failed transiently, so the `code` cursor must not advance to
/// `pushed_at` (the next sync retries the blob).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeFetch {
    pub changed: Vec<(String, String, String)>,
    pub removed: Vec<String>,
    pub pushed_at: Option<String>,
    pub gated: bool,
    /// True when this pass fetched everything it wanted, so the `code` cursor may advance. False
    /// when a blob failed transiently or when files are still owed (`pending`).
    pub complete: bool,
    /// Changed files left for a later pass. Not pruned; the next pass finds them by stored blob
    /// SHA.
    pub pending: usize,
    /// Files that could not be fetched at all, with the reason.
    pub skipped: Vec<SkippedFile>,
}

/// A source file the index could not take, and why. Reported so truncation is visible to the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFile {
    pub path: String,
    pub bytes: i64,
    /// A short, stable phrase for the UI. Static: add new causes here, not formatted at the call
    /// site.
    pub reason: &'static str,
}

/// A unit of background indexing work: the repo to index off the critical path. The index worker
/// fetches the repo's changed code and embeds it, plus links and activity embedding, all reading
/// from the store. The fetch lane does only activity, so a repo appears in the UI before its code
/// fetch.
#[derive(Debug, Clone)]
pub struct IndexJob {
    pub repo: Repo,
    /// Which index pass this job is, allocated by the engine when the repo is queued. Every status
    /// record the job produces carries it, so a record from a superseded pass can be told apart
    /// from the current entry. A job built outside the engine (a test) can leave this 0.
    pub generation: u64,
}

/// What one repo's indexing pass did, including files still owed or unreadable, not just a count.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexOutcome {
    /// Source files (re-)embedded this pass.
    pub files: usize,
    /// Changed files left for the next pass (found by their stored blob SHA).
    pub pending: usize,
    /// Files this pass could not fetch at all, with the reason.
    pub skipped: Vec<SkippedFile>,
}

impl IndexOutcome {
    /// Whether the index is short of the tree (files owed or skipped).
    pub fn partial(&self) -> bool {
        self.pending > 0 || !self.skipped.is_empty()
    }

    /// One sentence saying what is missing and why, with counts, for the badge hover and Debug
    /// panel.
    pub fn note(&self) -> String {
        let mut parts = Vec::new();
        if self.pending > 0 {
            parts.push(format!(
                "{} file(s) still to index; the next sync continues where this one stopped",
                self.pending
            ));
        }
        if let Some(first) = self.skipped.first() {
            parts.push(format!(
                "{} file(s) skipped, {}: {}",
                self.skipped.len(),
                first.reason,
                first.path
            ));
        }
        parts.join(". ")
    }
}

/// Where a repo is in the background index queue, for the per-repo UI badge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexState {
    Queued,
    Indexing,
    /// Everything the repo has is indexed.
    Indexed,
    /// Indexed as far as this pass went, with files still owed (`pending`) or unfetchable
    /// (`skipped`). An empty code-search result would otherwise read as "nothing matches".
    Partial,
    /// Indexing did not run because the store is at the storage budget. Reported so it is not
    /// mistaken for up to date.
    Paused,
    Error,
}

/// A per-repo index-progress update from the queue/worker. Distinct from `SyncProgress` (the
/// header's fetch progress).
#[derive(Debug, Clone)]
pub struct IndexProgress {
    pub repo_id: String,
    pub full_name: String,
    pub state: IndexState,
    /// Files (re-)embedded (on `Indexed`) or queued (on `Queued`).
    pub files: usize,
    /// Units processed / total to process this pass (commits + issues + changed code files), so the
    /// UI can show an indexing percentage. Both 0 outside the `Indexing` state (no meaningful
    /// fraction).
    pub done: usize,
    pub total: usize,
    /// The failure reason on `Error`. `None` for every other state.
    pub error: Option<String>,
    /// Files this repo still owes the index. Non-zero only on `Partial`.
    pub pending: usize,
    /// Files this pass could not fetch at all. Non-zero only on `Partial`.
    pub skipped: usize,
    /// Why a `Partial` or `Paused` state is what it is, as a UI sentence. Separate from `error`,
    /// since neither is a failure.
    pub note: Option<String>,
    /// Which index pass produced this record, copied from the [`IndexJob`]. Internal ordering, not
    /// carried across the host boundary.
    pub generation: u64,
}

impl IndexProgress {
    /// A progress record with the optional fields at their defaults.
    fn state(repo_id: &str, full_name: &str, state: IndexState, generation: u64) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            full_name: full_name.to_string(),
            state,
            files: 0,
            done: 0,
            total: 0,
            error: None,
            pending: 0,
            skipped: 0,
            note: None,
            generation,
        }
    }
}

/// What the scheduler emits after polling one source: the source id and its outcome (a full
/// report, or the error string if that source's sync failed). One bad source does not stop the
/// loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncTick {
    pub source_id: String,
    pub report: Result<FullSyncReport, String>,
}

/// Which phase of a manual sync a progress update belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPhase {
    /// Enumerating the target set (listing each board's repos) before any syncing.
    Planning,
    /// Syncing the repos.
    Syncing,
}

/// A granular sync progress update for hierarchical progress. It names the phase, the current
/// `item` (a board being planned or a repo being synced) and its position (`item_done` of
/// `item_total`), and while syncing a repo the current sub-`step` and its position. `finished`
/// carries the repo's result on the event that completes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncProgress {
    pub phase: SyncPhase,
    pub item: String,
    pub item_done: usize,
    pub item_total: usize,
    pub step: Option<String>,
    pub step_done: usize,
    pub step_total: usize,
    pub finished: Option<SyncTick>,
}

impl SyncProgress {
    /// A planning update: enumerating board `item` (`done` of `total`).
    fn planning(item: &str, done: usize, total: usize) -> Self {
        Self {
            phase: SyncPhase::Planning,
            item: item.to_string(),
            item_done: done,
            item_total: total,
            step: None,
            step_done: 0,
            step_total: 0,
            finished: None,
        }
    }

    /// A step update: syncing repo `item` (`done` of `total`), currently on `step` (`sd` of `st`).
    fn step(item: &str, done: usize, total: usize, step: &str, sd: usize, st: usize) -> Self {
        Self {
            phase: SyncPhase::Syncing,
            item: item.to_string(),
            item_done: done,
            item_total: total,
            step: Some(step.to_string()),
            step_done: sd,
            step_total: st,
            finished: None,
        }
    }

    /// A repo-completion update: repo `item` (`done` of `total`) finished with `tick`.
    fn finished(item: &str, done: usize, total: usize, tick: SyncTick) -> Self {
        Self {
            phase: SyncPhase::Syncing,
            item: item.to_string(),
            item_done: done,
            item_total: total,
            step: None,
            step_done: 0,
            step_total: 0,
            finished: Some(tick),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_forge::fake::FakeForge;
    use core_forge::{Capabilities, TreeEntry};
    use core_store::{CiRun, Commit, DependencyRow, Issue, PrFile, Release, Review};

    // Repo ids are forge-namespaced; the single-forge helpers register under `DEFAULT_FORGE`, so
    // test repo ids carry that segment.
    const RID: &str = "repo:default/acme/widget";

    // ---- domain builders ----

    fn repo() -> Repo {
        Repo {
            id: RID.into(),
            owner: "acme".into(),
            name: "widget".into(),
            full_name: "acme/widget".into(),
            ownership: "owned".into(),
        }
    }

    /// An index job for the test repo, unsequenced. A test sending a job straight to the worker is
    /// the only pass in flight; ordering tests set generations explicitly.
    fn job() -> IndexJob {
        IndexJob {
            repo: repo(),
            generation: 0,
        }
    }

    fn commit(sha: &str, date: &str) -> Commit {
        Commit {
            sha: sha.into(),
            repo_id: RID.into(),
            author_login: Some("octocat".into()),
            message: format!("msg {sha}"),
            committed_at: date.into(),
        }
    }

    fn pr(id: &str, number: i64, state: &str, merged_at: Option<&str>, body: &str) -> PullRequest {
        PullRequest {
            id: id.into(),
            repo_id: RID.into(),
            number,
            title: format!("PR {number}"),
            state: state.into(),
            author_login: None,
            body: Some(body.into()),
            created_at: "2026-06-01T00:00:00Z".into(),
            merged_at: merged_at.map(str::to_string),
            html_url: None,
        }
    }

    fn issue(id: &str, number: i64, title: &str) -> Issue {
        Issue {
            id: id.into(),
            repo_id: RID.into(),
            number,
            title: title.into(),
            state: "open".into(),
            author_login: None,
            body: Some(String::new()),
            created_at: "2026-06-01T00:00:00Z".into(),
            closed_at: None,
            labels: String::new(),
            html_url: None,
        }
    }

    fn release(id: &str, tag: &str) -> Release {
        Release {
            id: id.into(),
            repo_id: RID.into(),
            tag: tag.into(),
            name: None,
            published_at: None,
        }
    }

    fn ci(id: &str, sha: &str, conclusion: Option<&str>) -> CiRun {
        CiRun {
            id: id.into(),
            repo_id: RID.into(),
            commit_sha: Some(sha.into()),
            status: "completed".into(),
            conclusion: conclusion.map(str::to_string),
            completed_at: None,
            html_url: None,
            run_attempt: None,
        }
    }

    fn review(id: &str, pr_id: &str, login: Option<&str>) -> Review {
        Review {
            id: id.into(),
            pr_id: pr_id.into(),
            reviewer_login: login.map(str::to_string),
            state: "APPROVED".into(),
            submitted_at: None,
        }
    }

    fn prfile(pr_id: &str, name: &str, add: i64, del: i64, patch: Option<&str>) -> PrFile {
        PrFile {
            pr_id: pr_id.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: add,
            deletions: del,
            patch: patch.map(str::to_string),
        }
    }

    fn te(path: &str, sha: &str, kind: &str, size: Option<i64>) -> TreeEntry {
        TreeEntry {
            path: path.into(),
            sha: sha.into(),
            kind: kind.into(),
            size,
        }
    }

    async fn store_mem() -> Store {
        Store::open_in_memory().await.unwrap()
    }

    async fn engine(forge: FakeForge) -> SyncEngine<FakeForge> {
        SyncEngine::new(forge, store_mem().await)
    }

    /// Upsert the `default` forge config row (the `forge_id` foreign-key target for
    /// boards/sources). The in-memory `FakeForge` the engine routes through is separate.
    async fn seed_default_forge(store: &Store) {
        seed_forge(store, DEFAULT_FORGE, "github", "https://api.example").await;
    }

    /// A forge with one of every activity type for `repo()`. Two PRs (5 and 6), each with one
    /// review.
    fn full_forge() -> FakeForge {
        FakeForge::new()
            .with_commits(
                RID,
                vec![
                    commit("aaa", "2026-06-01T00:00:00Z"),
                    commit("bbb", "2026-06-02T00:00:00Z"),
                ],
                Some("2026-06-02T00:00:00Z"),
            )
            .with_pulls(
                RID,
                vec![
                    pr("101", 5, "open", None, ""),
                    pr("102", 6, "closed", Some("2026-06-03T00:00:00Z"), ""),
                ],
                Some("2026-06-03T00:00:00Z"),
            )
            .with_issues(
                RID,
                vec![issue("201", 10, "bug")],
                Some("2026-06-01T00:00:00Z"),
            )
            .with_releases(
                RID,
                vec![release("900", "v1.0"), release("901", "v1.1")],
                Some("rel-v1"),
            )
            .with_ci_runs(RID, vec![ci("800", "aaa", Some("success"))], Some("ci-v1"))
            .with_reviews(RID, 5, vec![review("700", "101", Some("alice"))])
            .with_reviews(RID, 6, vec![review("701", "102", None)])
    }

    // ---- history backfill ----

    /// The token the fake forge hands back for the tail of a capped commits pass.
    const TAIL_1: &str = "bf:c:1:2019-05-01T00:00:00Z";
    const TAIL_2: &str = "bf:c:6:2019-05-01T00:00:00Z";

    /// A forge whose commits pass is capped: it returns the newest window plus a resume token for
    /// the older history it could not reach.
    fn capped_commits_forge() -> FakeForge {
        FakeForge::new()
            .with_commits(
                RID,
                vec![
                    commit("new1", "2026-06-01T00:00:00Z"),
                    commit("new2", "2026-06-02T00:00:00Z"),
                ],
                Some("2026-06-02T00:00:00Z"),
            )
            .with_commits_capped(RID, TAIL_1)
    }

    #[tokio::test]
    async fn capped_forward_pass_records_the_tail_then_advances_the_watermark() {
        // A capped pass records the tail as a backfill walk and still advances the watermark.
        let eng = engine(capped_commits_forge()).await;
        eng.sync_repo(&repo()).await.unwrap();

        assert_eq!(
            eng.store().sync_cursor(RID, "commits").await.unwrap(),
            Some("2026-06-02T00:00:00Z".into()),
        );
        assert_eq!(
            eng.store()
                .sync_cursor(RID, &backfill_entity("commits"))
                .await
                .unwrap(),
            Some(TAIL_1.into()),
        );
    }

    #[tokio::test]
    async fn an_uncapped_pass_does_not_disturb_a_walk_in_progress() {
        // A forward pass that fits under the cap leaves a running walk alone.
        let forge = FakeForge::new().with_commits(
            RID,
            vec![commit("new1", "2026-06-01T00:00:00Z")],
            Some("2026-06-01T00:00:00Z"),
        );
        let eng = engine(forge).await;
        eng.store()
            .set_sync_cursor(RID, &backfill_entity("commits"), TAIL_2)
            .await
            .unwrap();
        eng.sync_repo(&repo()).await.unwrap();
        assert_eq!(
            eng.store()
                .sync_cursor(RID, &backfill_entity("commits"))
                .await
                .unwrap(),
            Some(TAIL_2.into()),
        );
    }

    #[tokio::test]
    async fn backfill_fetches_the_tail_across_passes_and_then_stops() {
        // History older than the first pass's window lands in the store: two chunks, then the
        // forge reports the end of history.
        let forge = capped_commits_forge().with_commit_backfill(
            RID,
            TAIL_1,
            vec![commit("old1", "2019-04-02T00:00:00Z")],
            Some(TAIL_2),
        );
        let store = store_mem().await;
        // Pass 1: the forward fetch caps, opens the walk, and spends its first chunk.
        let eng = SyncEngine::new(forge, store.clone());
        eng.sync_all(&repo()).await.unwrap();

        // Pass 2, same store: the window now fits under the cap, so the forward pass leaves the
        // walk alone and the lane spends its second chunk, which the forge says is the last.
        let settled = FakeForge::new()
            .with_commits(
                RID,
                vec![commit("new3", "2026-06-03T00:00:00Z")],
                Some("2026-06-03T00:00:00Z"),
            )
            .with_commit_backfill(
                RID,
                TAIL_2,
                vec![commit("old2", "2019-03-01T00:00:00Z")],
                None,
            );
        let eng = SyncEngine::new(settled.clone(), store);
        eng.sync_all(&repo()).await.unwrap();

        let shas: Vec<String> = eng
            .store()
            .commits_for_repo(RID)
            .await
            .unwrap()
            .into_iter()
            .map(|c| c.sha)
            .collect();
        assert!(shas.contains(&"old1".to_string()), "got {shas:?}");
        assert!(shas.contains(&"old2".to_string()), "got {shas:?}");
        assert!(shas.contains(&"new1".to_string()));

        // The walk is recorded complete, and a third pass asks for no further chunk.
        let states = eng.store().backfill_states().await.unwrap();
        let commits = states.iter().find(|s| s.entity == "commits").unwrap();
        assert!(commits.complete);
        let before = settled.calls().len();
        let report = eng.backfill_repo(&repo()).await.unwrap();
        assert_eq!(report, BackfillReport::default());
        assert_eq!(
            settled.calls().len(),
            before,
            "a finished walk still called out"
        );
    }

    #[tokio::test]
    async fn backfill_runs_only_after_every_forward_fetch() {
        // The backfill chunk is the last thing a repo's pass does: fresh activity outranks 2019.
        let forge = full_forge()
            .with_commits_capped(RID, TAIL_1)
            .with_commit_backfill(
                RID,
                TAIL_1,
                vec![commit("old1", "2019-04-02T00:00:00Z")],
                None,
            );
        let eng = engine(forge.clone()).await;
        eng.sync_all(&repo()).await.unwrap();

        let calls = forge.calls();
        let backfill_at = calls
            .iter()
            .position(|(label, cursor)| label == "commits" && cursor.as_deref() == Some(TAIL_1))
            .expect("the backfill chunk ran");
        for entity in ["pulls", "issues", "releases", "ci_runs"] {
            let forward_at = calls
                .iter()
                .rposition(|(label, _)| label == entity)
                .unwrap_or_else(|| panic!("no {entity} fetch"));
            assert!(
                backfill_at > forward_at,
                "backfill ran at {backfill_at}, before the {entity} fetch at {forward_at}"
            );
        }
        // And it is the last commits call, not the first.
        assert_eq!(
            calls.iter().rposition(|(l, _)| l == "commits"),
            Some(backfill_at)
        );
    }

    #[tokio::test]
    async fn backfill_resumes_from_the_stored_token_after_a_restart() {
        // A big backfill spans sessions: the walk must resume where it stopped, not start over.
        let path =
            std::env::temp_dir().join(format!("orgonzola-backfill-{}.db", std::process::id()));
        let path = path.to_string_lossy().to_string();
        let cleanup = || {
            for p in [path.clone(), format!("{path}-wal"), format!("{path}-shm")] {
                let _ = std::fs::remove_file(&p);
            }
        };
        cleanup();

        // Session one: the forward pass opens the walk and spends the first chunk, which reports
        // more history left (TAIL_2).
        let first = capped_commits_forge().with_commit_backfill(
            RID,
            TAIL_1,
            vec![commit("old1", "2019-04-02T00:00:00Z")],
            Some(TAIL_2),
        );
        {
            let eng = SyncEngine::new(first, Store::open(&path).await.unwrap());
            eng.sync_all(&repo()).await.unwrap();
        }

        // Session two: a brand new store handle and a brand new forge with an empty call log.
        let second = capped_commits_forge().with_commit_backfill(
            RID,
            TAIL_2,
            vec![commit("old2", "2019-03-01T00:00:00Z")],
            None,
        );
        let store = Store::open(&path).await.unwrap();
        let eng = SyncEngine::new(second.clone(), store);
        let report = eng.backfill_repo(&repo()).await.unwrap();

        assert_eq!(report.commits, 1);
        let cursors: Vec<Option<String>> = second
            .calls()
            .into_iter()
            .filter(|(label, _)| label == "commits")
            .map(|(_, cursor)| cursor)
            .collect();
        // One chunk, asked for with the token the previous session left behind.
        assert_eq!(cursors, vec![Some(TAIL_2.to_string())]);
        let shas: Vec<String> = eng
            .store()
            .commits_for_repo(RID)
            .await
            .unwrap()
            .into_iter()
            .map(|c| c.sha)
            .collect();
        assert!(
            shas.contains(&"old1".to_string()),
            "session one's chunk was lost: {shas:?}"
        );
        assert!(
            shas.contains(&"old2".to_string()),
            "session two's chunk is missing: {shas:?}"
        );
        cleanup();
    }

    #[tokio::test]
    async fn pull_request_backfill_stores_older_prs_without_fetching_their_reviews() {
        // Reviews are one call per PR, so a backfilled PR arrives without them. The PR row must
        // still land.
        let forge = FakeForge::new()
            .with_pulls(
                RID,
                vec![pr("101", 5, "open", None, "")],
                Some("2026-06-03T00:00:00Z"),
            )
            .with_pulls_capped(RID, "bf:p:1")
            .with_pull_backfill(
                RID,
                "bf:p:1",
                vec![pr(
                    "9",
                    1,
                    "closed",
                    Some("2015-01-02T00:00:00Z"),
                    "the first one",
                )],
                None,
            );
        let eng = engine(forge.clone()).await;
        eng.sync_all(&repo()).await.unwrap();

        let numbers: Vec<i64> = eng
            .store()
            .pull_requests(RID)
            .await
            .unwrap()
            .into_iter()
            .map(|p| p.number)
            .collect();
        assert!(
            numbers.contains(&1),
            "the backfilled PR is missing: {numbers:?}"
        );
        // Only the forward pass's PR triggered a review fetch.
        let reviews: Vec<String> = forge
            .calls()
            .into_iter()
            .map(|(label, _)| label)
            .filter(|l| l.starts_with("reviews:"))
            .collect();
        assert_eq!(reviews, vec!["reviews:5".to_string()]);
    }

    // ---- per-entity orchestration ----

    #[tokio::test]
    async fn sync_repo_persists_commits_and_advances_cursor() {
        let forge = FakeForge::new().with_commits(
            RID,
            vec![
                commit("aaa", "2026-06-01T00:00:00Z"),
                commit("bbb", "2026-06-02T00:00:00Z"),
            ],
            Some("2026-06-02T00:00:00Z"),
        );
        let eng = engine(forge).await;
        let report = eng.sync_repo(&repo()).await.unwrap();
        assert_eq!(report.persisted, 2);
        assert!(!report.unchanged);
        assert_eq!(eng.store().count_commits().await.unwrap(), 2);
        assert_eq!(
            eng.store().sync_cursor(RID, "commits").await.unwrap(),
            Some("2026-06-02T00:00:00Z".to_string())
        );
    }

    #[tokio::test]
    async fn resync_feeds_the_stored_cursor_back_to_the_forge() {
        // First sync stores the cursor; a restart (fresh forge, same store) must hand it back, and
        // an empty delta persists nothing and no duplicates.
        let store = store_mem().await;
        let eng1 = SyncEngine::new(
            FakeForge::new().with_commits(
                RID,
                vec![commit("aaa", "2026-06-02T00:00:00Z")],
                Some("2026-06-02T00:00:00Z"),
            ),
            store.clone(),
        );
        eng1.sync_repo(&repo()).await.unwrap();

        // Nothing new since the cursor: an unchanged delta.
        let probe = FakeForge::new().with_commits(RID, vec![], None);
        let eng2 = SyncEngine::new(probe.clone(), store.clone());
        let second = eng2.sync_repo(&repo()).await.unwrap();
        assert!(second.unchanged);
        assert_eq!(store.count_commits().await.unwrap(), 1);
        assert_eq!(
            probe.calls(),
            vec![(
                "commits".to_string(),
                Some("2026-06-02T00:00:00Z".to_string())
            )]
        );
    }

    #[tokio::test]
    async fn sync_pull_requests_persists_and_returns_changed() {
        let forge = FakeForge::new().with_pulls(
            RID,
            vec![
                pr("102", 6, "closed", Some("2026-06-03T00:00:00Z"), ""),
                pr("101", 5, "open", None, ""),
            ],
            Some("2026-06-03T00:00:00Z"),
        );
        let eng = engine(forge).await;
        let (report, changed) = eng.sync_pull_requests(&repo()).await.unwrap();
        assert_eq!(report.persisted, 2);
        assert_eq!(changed.len(), 2);
        assert_eq!(eng.store().pull_requests(RID).await.unwrap().len(), 2);
        assert_eq!(
            eng.store().sync_cursor(RID, "pulls").await.unwrap(),
            Some("2026-06-03T00:00:00Z".to_string())
        );
    }

    #[tokio::test]
    async fn sync_issues_persists_and_advances_cursor() {
        let forge = FakeForge::new().with_issues(
            RID,
            vec![issue("201", 10, "bug")],
            Some("2026-06-02T00:00:00Z"),
        );
        let eng = engine(forge).await;
        let report = eng.sync_issues(&repo()).await.unwrap();
        assert_eq!(report.persisted, 1);
        assert_eq!(eng.store().issues_for_repo(RID).await.unwrap().len(), 1);
        assert_eq!(
            eng.store().sync_cursor(RID, "issues").await.unwrap(),
            Some("2026-06-02T00:00:00Z".to_string())
        );
    }

    #[tokio::test]
    async fn sync_releases_persists_and_unchanged_skips() {
        let store = store_mem().await;
        let eng1 = SyncEngine::new(
            FakeForge::new().with_releases(
                RID,
                vec![release("900", "v1.0"), release("901", "v1.1")],
                Some("v1"),
            ),
            store.clone(),
        );
        assert_eq!(eng1.sync_releases(&repo()).await.unwrap().persisted, 2);
        assert_eq!(
            store.sync_cursor(RID, "releases").await.unwrap(),
            Some("v1".to_string())
        );

        // An unchanged delta (a 304) leaves the stored releases and the cursor untouched.
        let eng2 = SyncEngine::new(
            FakeForge::new().with_releases(RID, vec![], None),
            store.clone(),
        );
        let second = eng2.sync_releases(&repo()).await.unwrap();
        assert!(second.unchanged);
        assert_eq!(store.releases_for_repo(RID).await.unwrap().len(), 2);
        assert_eq!(
            store.sync_cursor(RID, "releases").await.unwrap(),
            Some("v1".to_string())
        );
    }

    #[tokio::test]
    async fn sync_ci_runs_persists_and_unchanged_skips() {
        let store = store_mem().await;
        let eng1 = SyncEngine::new(
            FakeForge::new().with_ci_runs(
                RID,
                vec![ci("800", "aaa", Some("success")), ci("801", "bbb", None)],
                Some("v1"),
            ),
            store.clone(),
        );
        assert_eq!(eng1.sync_ci_runs(&repo()).await.unwrap().persisted, 2);
        assert_eq!(store.ci_runs_for_repo(RID).await.unwrap().len(), 2);

        let eng2 = SyncEngine::new(
            FakeForge::new().with_ci_runs(RID, vec![], None),
            store.clone(),
        );
        assert!(eng2.sync_ci_runs(&repo()).await.unwrap().unchanged);
        assert_eq!(store.ci_runs_for_repo(RID).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn sync_reviews_for_persists_per_changed_pr() {
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        store
            .upsert_pull_request(&pr("101", 5, "open", None, ""))
            .await
            .unwrap();
        let forge = FakeForge::new().with_reviews(
            RID,
            5,
            vec![
                review("700", "101", Some("alice")),
                review("701", "101", None),
            ],
        );
        let eng = SyncEngine::new(forge, store);
        let report = eng
            .sync_reviews_for(&repo(), &[("101".to_string(), 5)])
            .await
            .unwrap();
        assert_eq!(report.persisted, 2);
        assert_eq!(eng.store().reviews_for_pr("101").await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn sync_all_runs_every_activity_type() {
        let eng = engine(full_forge()).await;
        let report = eng.sync_all(&repo()).await.unwrap();
        assert_eq!(report.commits, 2);
        assert_eq!(report.pull_requests, 2);
        assert_eq!(report.issues, 1);
        assert_eq!(report.releases, 2);
        assert_eq!(report.reviews, 2);
        assert_eq!(report.ci_runs, 1);
        assert_eq!(report.dependencies, 0);
        assert_eq!(report.code_files, 0);
    }

    // ---- scheduler ----

    #[tokio::test]
    async fn scheduler_emits_a_tick_per_polled_source() {
        use tokio::sync::mpsc;
        let eng = engine(full_forge()).await;
        let repos = vec![repo()];
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::select! {
            _ = eng.run_scheduler(&repos, Duration::from_millis(5), move |p| { let _ = tx.send(p); }) => {
                panic!("scheduler returned early");
            }
            progress = rx.recv() => {
                let progress = progress.unwrap();
                let tick = progress.finished.expect("scheduler emits a completion event per repo");
                assert_eq!(tick.source_id, RID);
                assert_eq!(progress.item_done, 1);
                assert_eq!(progress.item_total, 1);
                let report = tick.report.expect("first source synced ok");
                assert_eq!(report.commits, 2);
                assert_eq!(report.ci_runs, 1);
            }
        }
    }

    #[tokio::test]
    async fn scheduler_over_registry_polls_persisted_sources() {
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .upsert_source(&SourceRow {
                id: RID.into(),
                kind: "repo".into(),
                name: "acme/widget".into(),
                ownership: "owned".into(),
                filters: vec![],
                stale_pr_days: 7,
                forge_id: Some(DEFAULT_FORGE.into()),
            })
            .await
            .unwrap();
        store
            .update_settings(&core_store::Settings {
                sync_period_secs: 1,
                stale_pr_days: 7,
                digest_webhook_url: None,
                digest_schedule_hours: None,
                llm_enabled: true,
                llm_model: None,
                storage_budget_mb: None,
            })
            .await
            .unwrap();
        let eng = SyncEngine::new(full_forge(), store);

        use tokio::sync::mpsc;
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::select! {
            _ = eng.run_scheduler_over_registry(move |p| { let _ = tx.send(p); }) => {
                panic!("scheduler returned early");
            }
            progress = rx.recv() => {
                let progress = progress.unwrap();
                let tick = progress.finished.expect("scheduler emits a completion event per repo");
                assert_eq!(tick.source_id, RID);
                assert_eq!(progress.item_total, 1);
                assert!(tick.report.is_ok());
            }
        }
    }

    // ---- PR files ----

    #[tokio::test]
    async fn sync_pr_files_stores_changed_files_for_open_prs() {
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        store
            .upsert_pull_request(&pr("101", 5, "open", None, ""))
            .await
            .unwrap();
        let forge = FakeForge::new().with_pr_files(
            RID,
            5,
            vec![
                prfile("101", "a.rs", 6, 1, Some("@@ a @@")),
                prfile("101", "b.rs", 4, 0, None),
            ],
        );
        let eng = SyncEngine::new(forge, store);
        assert_eq!(eng.sync_pr_files(&repo()).await.unwrap(), 2);
        let stored = eng.store().pr_files("101").await.unwrap();
        assert_eq!(stored.len(), 2);
    }

    #[tokio::test]
    async fn sync_pr_files_backfills_merged_pr_diffs_once() {
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        store
            .upsert_pull_request(&pr("201", 7, "closed", Some("2026-06-02T00:00:00Z"), ""))
            .await
            .unwrap();
        let forge = FakeForge::new().with_pr_files(
            RID,
            7,
            vec![prfile("201", "x.rs", 3, 2, Some("@@ x @@"))],
        );
        let eng = SyncEngine::new(forge, store);
        // First pass back-fills the merged PR's files; second pass: not open + has files ->
        // nothing.
        assert_eq!(eng.sync_pr_files(&repo()).await.unwrap(), 1);
        assert_eq!(eng.store().pr_files("201").await.unwrap().len(), 1);
        assert_eq!(eng.sync_pr_files(&repo()).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn sync_pr_files_does_not_refetch_a_zero_file_merged_pr() {
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        store
            .upsert_pull_request(&pr("202", 8, "closed", Some("2026-06-02T00:00:00Z"), ""))
            .await
            .unwrap();
        let forge = FakeForge::new().with_pr_files(RID, 8, vec![]); // genuinely zero files
        let eng = SyncEngine::new(forge, store);
        assert_eq!(eng.sync_pr_files(&repo()).await.unwrap(), 0);
        assert!(
            eng.store()
                .merged_pull_requests_missing_files(RID, 50)
                .await
                .unwrap()
                .is_empty(),
            "a zero-file merged PR is marked synced, not re-listed as missing files"
        );
        assert_eq!(eng.sync_pr_files(&repo()).await.unwrap(), 0);
    }

    // ---- discovery + manual sync ----

    #[tokio::test]
    async fn discover_person_repos_delegates_to_forge() {
        let eng = engine(
            FakeForge::new().with_person_repos("alice", vec!["acme/api".into(), "acme/web".into()]),
        )
        .await;
        assert_eq!(
            eng.discover_person_repos(DEFAULT_FORGE, "alice")
                .await
                .unwrap(),
            vec!["acme/api".to_string(), "acme/web".to_string()]
        );
    }

    #[tokio::test]
    async fn enumerate_org_repos_delegates_to_forge() {
        let eng = engine(
            FakeForge::new().with_org_repos("acme", vec!["acme/api".into(), "acme/web".into()]),
        )
        .await;
        assert_eq!(
            eng.enumerate_org_repos(DEFAULT_FORGE, "acme")
                .await
                .unwrap(),
            vec!["acme/api".to_string(), "acme/web".to_string()]
        );
    }

    #[tokio::test]
    async fn people_discovery_excludes_archived_candidates_by_default_and_can_include_them() {
        const ARCHIVED: &str = "repo:default/acme/retired";
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        let eng = SyncEngine::new(
            FakeForge::new()
                .with_person_repos("alice", vec!["acme/widget".into(), "acme/retired".into()])
                .with_archived(ARCHIVED),
            store,
        );

        let board = eng.store().board("board:1").await.unwrap().unwrap();
        assert!(
            !board.include_archived,
            "the persisted default is restrictive"
        );
        let repos = eng.enumerate_board_repos(&board).await.unwrap();
        assert_eq!(
            repos
                .iter()
                .map(|r| r.full_name.as_str())
                .collect::<Vec<_>>(),
            vec!["acme/widget"]
        );

        eng.store()
            .set_board_include_archived("board:1", true)
            .await
            .unwrap();
        let board = eng.store().board("board:1").await.unwrap().unwrap();
        let repos = eng.enumerate_board_repos(&board).await.unwrap();
        assert_eq!(
            repos
                .iter()
                .map(|r| r.full_name.as_str())
                .collect::<Vec<_>>(),
            vec!["acme/widget", "acme/retired"]
        );
    }

    #[tokio::test]
    async fn archived_recent_activity_candidate_is_excluded() {
        // `person_repos` is the forge-normalized union whose front is recent-event candidates. The
        // archive check happens after that boundary because event payloads carry only a repo name.
        const RECENT_ARCHIVED: &str = "repo:default/acme/recent-but-retired";
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        let eng = SyncEngine::new(
            FakeForge::new()
                .with_person_repos("alice", vec!["acme/recent-but-retired".into()])
                .with_archived(RECENT_ARCHIVED),
            store,
        );

        let board = eng.store().board("board:1").await.unwrap().unwrap();
        assert!(eng.enumerate_board_repos(&board).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_pinned_archived_repo_remains_a_manual_sync_target() {
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.upsert_repo(&repo()).await.unwrap();
        store.add_board_repo("board:1", RID).await.unwrap();
        let eng = SyncEngine::new(full_forge().with_archived(RID), store);

        assert_eq!(eng.sync_all_sources(|_| {}).await.unwrap(), 1);
        assert_eq!(
            eng.store()
                .board_effective_repo_ids("board:1")
                .await
                .unwrap(),
            vec![RID.to_string()]
        );
    }

    #[tokio::test]
    async fn successful_manual_discovery_prunes_a_previously_discovered_archived_repo() {
        const ARCHIVED: &str = "repo:default/acme/retired";
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        store
            .set_board_include_archived("board:1", true)
            .await
            .unwrap();
        let cached = repo_from_full(DEFAULT_FORGE, "acme/retired").unwrap();
        store.upsert_repo(&cached).await.unwrap();
        store
            .replace_board_discovered_repos("board:1", &[ARCHIVED.to_string()])
            .await
            .unwrap();
        // Model a board that discovered this repo while the option was enabled, then turned it off.
        store
            .set_board_include_archived("board:1", false)
            .await
            .unwrap();
        let eng = SyncEngine::new(
            FakeForge::new()
                .with_person_repos("alice", vec!["acme/retired".into()])
                .with_archived(ARCHIVED),
            store,
        );

        assert_eq!(eng.sync_all_sources(|_| {}).await.unwrap(), 0);
        assert!(
            eng.store()
                .board_discovered_repos("board:1")
                .await
                .unwrap()
                .is_empty(),
            "a fresh successful plan reconciles the board association instead of appending"
        );
        assert!(
            eng.store().get_repo(ARCHIVED).await.unwrap().is_some(),
            "cached repo data remains available for explicit storage cleanup"
        );
    }

    #[tokio::test]
    async fn background_discovery_replaces_a_previously_discovered_archived_repo() {
        const ARCHIVED: &str = "repo:default/acme/retired";
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        let cached = repo_from_full(DEFAULT_FORGE, "acme/retired").unwrap();
        store.upsert_repo(&cached).await.unwrap();
        store
            .replace_board_discovered_repos("board:1", &[ARCHIVED.to_string()])
            .await
            .unwrap();
        let eng = SyncEngine::new(
            FakeForge::new()
                .with_person_repos("alice", vec!["acme/retired".into()])
                .with_archived(ARCHIVED),
            store,
        );
        let board = eng.store().board("board:1").await.unwrap().unwrap();

        assert_eq!(eng.discover_board_repos(&board).await.unwrap(), 0);
        assert!(eng
            .store()
            .board_discovered_repos("board:1")
            .await
            .unwrap()
            .is_empty());
        assert!(eng.store().get_repo(ARCHIVED).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn background_refresh_failure_keeps_a_still_eligible_repo_visible() {
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store.upsert_repo(&repo()).await.unwrap();
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        store
            .replace_board_discovered_repos("board:1", &[RID.to_string()])
            .await
            .unwrap();
        // Force sync_all to fail before refreshing activity without changing discovery eligibility.
        fill_store(&store, 2 * core_store::MB).await;
        assert!(!set_budget(&store, Some(1)).await.allows_sync());
        let eng = SyncEngine::new(
            full_forge().with_person_repos("alice", vec!["acme/widget".into()]),
            store,
        );
        let board = eng.store().board("board:1").await.unwrap().unwrap();

        assert_eq!(eng.discover_board_repos(&board).await.unwrap(), 0);
        assert_eq!(
            eng.store().board_discovered_repos("board:1").await.unwrap(),
            vec![RID.to_string()],
            "a transient refresh failure is not a discovery-scope removal"
        );
    }

    #[tokio::test]
    async fn sync_all_sources_polls_registry_once_immediately() {
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .upsert_source(&SourceRow {
                id: RID.into(),
                kind: "repo".into(),
                name: "acme/widget".into(),
                ownership: "owned".into(),
                filters: vec![],
                stale_pr_days: 7,
                forge_id: Some(DEFAULT_FORGE.into()),
            })
            .await
            .unwrap();
        let eng = SyncEngine::new(full_forge(), store);
        let mut progress = Vec::new();
        let total = eng.sync_all_sources(|p| progress.push(p)).await.unwrap();
        assert_eq!(total, 1);
        assert!(progress
            .iter()
            .any(|p| p.phase == SyncPhase::Syncing && p.step.as_deref() == Some("commits")));
        let finished: Vec<&SyncProgress> =
            progress.iter().filter(|p| p.finished.is_some()).collect();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].finished.as_ref().unwrap().source_id, RID);
    }

    #[tokio::test]
    async fn sync_all_sources_counts_board_repos() {
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store.add_board_repo("board:1", RID).await.unwrap();
        let eng = SyncEngine::new(full_forge(), store);
        let mut progress = Vec::new();
        let total = eng.sync_all_sources(|p| progress.push(p)).await.unwrap();
        assert_eq!(total, 1);
        assert!(progress.iter().any(|p| p.phase == SyncPhase::Planning));
    }

    #[tokio::test]
    async fn sync_all_sources_enumerates_discovery_into_the_total() {
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        let eng = SyncEngine::new(
            full_forge().with_person_repos("alice", vec!["acme/api".into()]),
            store,
        );
        let mut progress = Vec::new();
        let total = eng.sync_all_sources(|p| progress.push(p)).await.unwrap();
        assert_eq!(total, 1);
        let finished: Vec<&SyncProgress> =
            progress.iter().filter(|p| p.finished.is_some()).collect();
        assert_eq!(finished.len(), 1);
        assert_eq!(
            finished[0].finished.as_ref().unwrap().source_id,
            "repo:default/acme/api"
        );
        assert_eq!(
            eng.store().board_discovered_repos("board:1").await.unwrap(),
            vec!["repo:default/acme/api".to_string()]
        );
    }

    #[tokio::test]
    async fn discovery_tolerates_one_failing_login_and_does_not_abort_the_sync() {
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        store.add_board_person("board:1", "bob").await.unwrap();
        let forge = full_forge()
            .with_person_repos_error("alice")
            .with_person_repos("bob", vec!["acme/api".into()]);
        let eng = SyncEngine::new(forge, store);
        let total = eng.sync_all_sources(|_| {}).await.unwrap();
        assert_eq!(total, 1, "bob's repo still syncs despite alice's failure");
        assert_eq!(
            eng.store().board_discovered_repos("board:1").await.unwrap(),
            vec!["repo:default/acme/api".to_string()]
        );
    }

    #[tokio::test]
    async fn sync_all_sources_prunes_stale_discovered_repos() {
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        store
            .replace_board_discovered_repos(
                "board:1",
                &[
                    "repo:default/acme/api".to_string(),
                    "repo:default/acme/old".to_string(),
                ],
            )
            .await
            .unwrap();
        let eng = SyncEngine::new(
            full_forge().with_person_repos("alice", vec!["acme/api".into()]),
            store,
        );
        eng.sync_all_sources(|_| {}).await.unwrap();
        assert_eq!(
            eng.store().board_discovered_repos("board:1").await.unwrap(),
            vec!["repo:default/acme/api".to_string()]
        );
    }

    // ---- multi-forge routing ----

    #[tokio::test]
    async fn two_forges_route_same_owner_name_to_distinct_repos() {
        // The same `owner/name` on two forges are distinct repos: each carries its forge in its
        // id, and the engine routes each sync by it.
        let mk_commit = |repo_id: &str, sha: &str, date: &str| Commit {
            sha: sha.into(),
            repo_id: repo_id.into(),
            author_login: None,
            message: format!("msg {sha}"),
            committed_at: date.into(),
        };
        let gh_id = "repo:gh/tinygrad/tinygrad";
        let gitea_id = "repo:gitea/tinygrad/tinygrad";
        let gh = FakeForge::new().with_commits(
            gh_id,
            vec![mk_commit(gh_id, "gh-sha", "2026-01-01T00:00:00Z")],
            Some("2026-01-01T00:00:00Z"),
        );
        let gitea = FakeForge::new().with_commits(
            gitea_id,
            vec![mk_commit(gitea_id, "gitea-sha", "2026-01-02T00:00:00Z")],
            Some("2026-01-02T00:00:00Z"),
        );
        let eng = SyncEngine::empty(store_mem().await)
            .with_forge("gh", gh)
            .with_forge("gitea", gitea);

        let gh_repo = repo_from_full("gh", "tinygrad/tinygrad").unwrap();
        let gitea_repo = repo_from_full("gitea", "tinygrad/tinygrad").unwrap();
        eng.sync_repo(&gh_repo).await.unwrap();
        eng.sync_repo(&gitea_repo).await.unwrap();

        // Two distinct repo rows, each with only its own forge's commit.
        assert_eq!(eng.store().count_repos().await.unwrap(), 2);
        let gh_commits = eng.store().commits_for_repo(gh_id).await.unwrap();
        assert_eq!(gh_commits.len(), 1);
        assert_eq!(gh_commits[0].sha, "gh-sha");
        let gitea_commits = eng.store().commits_for_repo(gitea_id).await.unwrap();
        assert_eq!(gitea_commits.len(), 1);
        assert_eq!(gitea_commits[0].sha, "gitea-sha");

        // A repo naming a forge the engine was not given is skipped with UnknownForge, not a panic.
        let orphan = repo_from_full("nope", "x/y").unwrap();
        assert!(matches!(
            eng.sync_repo(&orphan).await,
            Err(SyncError::UnknownForge(_))
        ));
    }

    // ---- dependencies ----

    #[tokio::test]
    async fn sync_dependencies_parses_and_treats_missing_as_empty() {
        let cargo = "[dependencies]\nserde = \"1\"\ntokio = { version = \"1\" }\n\n[dev-dependencies]\nwiremock = \"0.6\"\n";
        let requirements = "requests==2.31\nflask\n";
        let forge = FakeForge::new()
            .with_file_text(RID, "Cargo.toml", cargo, Some("v1"))
            .with_file_text(RID, "requirements.txt", requirements, Some("v1"));
        let eng = engine(forge).await;
        let report = eng.sync_dependencies(&repo()).await.unwrap();
        assert_eq!(report.persisted, 5); // serde, tokio, wiremock (cargo) + requests, flask (pip)
        let deps = eng.store().dependencies_for_repo(RID).await.unwrap();
        assert_eq!(deps.len(), 5);
        assert!(deps
            .iter()
            .any(|d| d.name == "serde" && d.ecosystem == "cargo"));
        assert!(deps.iter().any(|d| d.name == "wiremock" && d.kind == "dev"));
        assert!(deps
            .iter()
            .any(|d| d.name == "requests" && d.ecosystem == "pip"));
    }

    #[tokio::test]
    async fn sync_dependencies_prunes_deps_removed_from_a_manifest() {
        let store = store_mem().await;
        let eng1 = SyncEngine::new(
            FakeForge::new().with_file_text(
                RID,
                "Cargo.toml",
                "[dependencies]\nserde = \"1\"\ntokio = \"1\"\n",
                Some("v1"),
            ),
            store.clone(),
        );
        eng1.sync_dependencies(&repo()).await.unwrap();
        assert_eq!(store.dependencies_for_repo(RID).await.unwrap().len(), 2);

        // Cargo.toml declares only serde; tokio must be pruned.
        let eng2 = SyncEngine::new(
            FakeForge::new().with_file_text(
                RID,
                "Cargo.toml",
                "[dependencies]\nserde = \"1\"\n",
                Some("v2"),
            ),
            store.clone(),
        );
        eng2.sync_dependencies(&repo()).await.unwrap();
        let deps = store.dependencies_for_repo(RID).await.unwrap();
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].name, "serde");
        assert_eq!(deps[0].source, "Cargo.toml");
    }

    #[tokio::test]
    async fn sync_dependencies_unchanged_manifest_keeps_deps_and_cursor() {
        let store = store_mem().await;
        let eng1 = SyncEngine::new(
            FakeForge::new().with_file_text(
                RID,
                "Cargo.toml",
                "[dependencies]\nserde = \"1\"\n",
                Some("v1"),
            ),
            store.clone(),
        );
        eng1.sync_dependencies(&repo()).await.unwrap();
        assert_eq!(
            store.sync_cursor(RID, "deps:Cargo.toml").await.unwrap(),
            Some("v1".to_string())
        );

        let eng2 = SyncEngine::new(
            FakeForge::new().with_file_unchanged(RID, "Cargo.toml"),
            store.clone(),
        );
        let report = eng2.sync_dependencies(&repo()).await.unwrap();
        assert!(report.unchanged);
        assert_eq!(store.dependencies_for_repo(RID).await.unwrap().len(), 1);
    }

    // ---- indexing + code ----

    /// An embedder that counts its `embed` calls, so a test can tell whether the activity scan
    /// ran. Delegates vectors to a `HashEmbedder`.
    struct CountingEmbedder {
        inner: core_embed::HashEmbedder,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl core_embed::Embedder for CountingEmbedder {
        fn dimensions(&self) -> usize {
            core_embed::Embedder::dimensions(&self.inner)
        }
        fn embed(&self, text: &str) -> Vec<f32> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            core_embed::Embedder::embed(&self.inner, text)
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    #[tokio::test]
    async fn index_job_skips_activity_scan_when_nothing_changed() {
        use std::sync::atomic::Ordering::SeqCst;
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        // Isolate the test to the activity path: no code fetch/embed.
        store.set_repo_index_code(RID, false).await.unwrap();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let eng = SyncEngine::new(
            FakeForge::new().with_commits(
                RID,
                vec![
                    commit("aaa", "2026-06-01T00:00:00Z"),
                    commit("bbb", "2026-06-02T00:00:00Z"),
                ],
                Some("2026-06-02T00:00:00Z"),
            ),
            store.clone(),
        )
        .with_embedder(std::sync::Arc::new(CountingEmbedder {
            inner: core_embed::HashEmbedder::default(),
            calls: calls.clone(),
        }));
        eng.sync_repo(&repo()).await.unwrap();

        // First index: both commit messages embed, and the checkpoint is recorded.
        eng.process_index_job(&job()).await.unwrap();
        let after_first = calls.load(SeqCst);
        assert!(after_first >= 2, "first pass embeds both commits");
        assert!(
            store
                .sync_cursor(RID, "activity_index")
                .await
                .unwrap()
                .is_some(),
            "the checkpoint is recorded after a successful index"
        );

        // Second index, nothing changed: the activity scan + linking are skipped (no new embed
        // calls).
        eng.process_index_job(&job()).await.unwrap();
        assert_eq!(
            calls.load(SeqCst),
            after_first,
            "an unchanged repo re-index embeds nothing - the scan is skipped"
        );

        // A new commit advances the commits cursor, so the checkpoint changes and the scan runs
        // again; the per-item content hash limits the work to the new commit.
        store
            .upsert_commit(&commit("ccc", "2026-06-03T00:00:00Z"))
            .await
            .unwrap();
        store
            .set_sync_cursor(RID, "commits", "2026-06-03T00:00:00Z")
            .await
            .unwrap();
        eng.process_index_job(&job()).await.unwrap();
        assert_eq!(
            calls.load(SeqCst),
            after_first + 1,
            "only the new commit is embedded on the next change"
        );
    }

    #[tokio::test]
    async fn clear_index_drops_the_activity_checkpoint() {
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        store
            .set_sync_cursor(RID, "activity_index", "a|b|c|hash:768:2")
            .await
            .unwrap();
        assert!(store
            .sync_cursor(RID, "activity_index")
            .await
            .unwrap()
            .is_some());
        store.clear_index().await.unwrap();
        assert!(
            store
                .sync_cursor(RID, "activity_index")
                .await
                .unwrap()
                .is_none(),
            "clearing the index drops the activity checkpoint so the next sync re-indexes"
        );
    }

    #[tokio::test]
    async fn index_repo_embeds_commit_text_for_search() {
        use core_embed::{Embedder, HashEmbedder};
        let eng = engine(FakeForge::new().with_commits(
            RID,
            vec![
                commit("aaa", "2026-06-01T00:00:00Z"),
                commit("bbb", "2026-06-02T00:00:00Z"),
            ],
            Some("c"),
        ))
        .await;
        eng.sync_repo(&repo()).await.unwrap();
        let embedder = HashEmbedder::default();
        let n = eng.index_repo(&repo(), &embedder).await.unwrap();
        assert_eq!(n, 2);
        let hits = eng
            .store()
            .nearest_embeddings(&embedder.embed("msg aaa"), 2)
            .await
            .unwrap();
        assert!(hits.iter().any(|h| h.ref_kind == "commit"));
        // A second pass over unchanged commits re-embeds nothing.
        assert_eq!(eng.index_repo(&repo(), &embedder).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn metrics_only_sync_fetches_activity_without_embedding() {
        // The org-board path: sync_metrics_only fetches activity (so the rollup has data) but
        // enqueues no indexing. sync_all on the same forge does embed, so the `index` flag is the
        // only gate.
        let make = || {
            FakeForge::new()
                .with_commits(RID, vec![commit("aaa", "2026-06-01T00:00:00Z")], Some("c"))
                .with_pushed_at(RID, "2026-06-10T00:00:00Z")
                .with_tree(RID, vec![te("src/x.rs", "blobA", "blob", Some(20))])
                .with_blob(RID, "blobA", "fn x() {}")
        };

        let metrics = engine(make()).await;
        metrics.sync_metrics_only(&repo()).await.unwrap();
        let c = metrics.store().counts().await.unwrap();
        assert!(c.commits >= 1, "activity is fetched for the metrics rollup");
        assert_eq!(c.embeddings_total, 0, "a metrics-only sync embeds nothing");

        let full = engine(make()).await;
        full.sync_all(&repo()).await.unwrap();
        assert!(
            full.store().counts().await.unwrap().embeddings_total > 0,
            "a full sync embeds - the index flag is the only difference"
        );
    }

    #[tokio::test]
    async fn sync_records_a_fork_with_its_namespaced_upstream() {
        // A synced repo's fork relationship is captured from forge metadata (here a programmed
        // fork of upstream/widget), so an observed fork can be folded out of the board later.
        let forge = FakeForge::new()
            .with_commits(RID, vec![commit("aaa", "2026-06-01T00:00:00Z")], Some("c"))
            .with_fork(RID, "upstream/widget");
        let eng = engine(forge).await;
        eng.sync_metrics_only(&repo()).await.unwrap();
        let forks = eng.store().repo_forks().await.unwrap();
        assert_eq!(forks.len(), 1);
        assert_eq!(forks[0].repo_id, RID);
        assert!(forks[0].is_fork);
        assert_eq!(
            forks[0].parent_repo_id.as_deref(),
            Some("repo:default/upstream/widget"),
            "the upstream is stored as a namespaced repo id, matchable against the board's repos"
        );
    }

    #[tokio::test]
    async fn sync_leaves_a_non_fork_unflagged() {
        // No fork programmed -> the repo is not reported as a fork, so it is analyzed normally.
        let forge = FakeForge::new().with_commits(
            RID,
            vec![commit("aaa", "2026-06-01T00:00:00Z")],
            Some("c"),
        );
        let eng = engine(forge).await;
        eng.sync_metrics_only(&repo()).await.unwrap();
        assert!(
            eng.store().repo_forks().await.unwrap().is_empty(),
            "a non-fork repo is never in the fork set"
        );
    }

    #[tokio::test]
    async fn enumerate_org_board_takes_org_repos_in_order() {
        // An org board enumerates its org's repos (most-recently-active first, capped); no people
        // needed.
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:o", "Acme", BoardKind::Org, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:o", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.set_board_org("board:o", Some("acme")).await.unwrap();
        let eng = SyncEngine::new(
            full_forge().with_org_repos("acme", vec!["acme/api".into(), "acme/web".into()]),
            store,
        );
        let board = eng.store().board("board:o").await.unwrap().unwrap();
        let repos = eng.enumerate_board_repos(&board).await.unwrap();
        assert_eq!(
            repos
                .iter()
                .map(|r| r.full_name.as_str())
                .collect::<Vec<_>>(),
            vec!["acme/api", "acme/web"],
            "the org's repos, forge order preserved"
        );
    }

    #[tokio::test]
    async fn enumerate_team_board_caps_in_discovery_order() {
        // The forge returns a person's repos most-relevant-first, so the cap must keep that head,
        // not the alphabetical first N. Discovery order here runs r59 down to r00, so the kept
        // slice starts at r59.
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:t", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:t", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:t", "alice").await.unwrap();
        let n = MAX_DISCOVERED_REPOS + 10;
        let found: Vec<String> = (0..n).map(|i| format!("acme/r{:02}", n - 1 - i)).collect();
        let eng = SyncEngine::new(
            full_forge().with_person_repos("alice", found.clone()),
            store,
        );
        let board = eng.store().board("board:t").await.unwrap().unwrap();
        let repos = eng.enumerate_board_repos(&board).await.unwrap();
        assert_eq!(
            repos
                .iter()
                .map(|r| r.full_name.clone())
                .collect::<Vec<_>>(),
            found[..MAX_DISCOVERED_REPOS].to_vec(),
            "the cap keeps the first N in discovery order"
        );
    }

    #[tokio::test]
    async fn enumerate_team_board_shares_the_cap_across_its_people() {
        // The cap belongs to the board, not to whoever is listed first. Alice alone overflows it,
        // so a fill that walks people in order would never query bob. Round-robin gives everyone a
        // turn per round, so bob's repo lands second, right behind alice's most relevant one.
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:t", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:t", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:t", "alice").await.unwrap();
        store.add_board_person("board:t", "bob").await.unwrap();
        let hers: Vec<String> = (0..MAX_DISCOVERED_REPOS + 10)
            .map(|i| format!("acme/a{i:02}"))
            .collect();
        let eng = SyncEngine::new(
            full_forge()
                .with_person_repos("alice", hers.clone())
                .with_person_repos("bob", vec!["acme/bobs-repo".into()]),
            store,
        );
        let board = eng.store().board("board:t").await.unwrap().unwrap();
        let repos = eng.enumerate_board_repos(&board).await.unwrap();
        let names: Vec<&str> = repos.iter().map(|r| r.full_name.as_str()).collect();
        assert_eq!(names.len(), MAX_DISCOVERED_REPOS, "still capped");
        assert_eq!(
            names[..2],
            ["acme/a00", "acme/bobs-repo"],
            "one round is one repo per person, each person's most relevant first"
        );
        // Bob's single repo does not cost him a share of the rest: his unused turns go to alice.
        assert_eq!(names[2], "acme/a01");
    }

    #[tokio::test]
    async fn scheduler_discovers_boards_with_an_empty_source_registry() {
        // Only pinned repos become sources, so a people-first board leaves the registry empty. The
        // empty-registry branch must still discover, or such a board would sync only on a manual
        // "Sync now".
        let store = store_mem().await;
        seed_default_forge(&store).await;
        store
            .create_board("board:1", "Team", BoardKind::Team, "2026-06-01T00:00:00Z")
            .await
            .unwrap();
        store
            .set_board_forge("board:1", Some(DEFAULT_FORGE))
            .await
            .unwrap();
        store.add_board_person("board:1", "alice").await.unwrap();
        store
            .update_settings(&core_store::Settings {
                sync_period_secs: 1,
                stale_pr_days: 7,
                digest_webhook_url: None,
                digest_schedule_hours: None,
                llm_enabled: true,
                llm_model: None,
                storage_budget_mb: None,
            })
            .await
            .unwrap();
        assert!(
            store.sources().await.unwrap().is_empty(),
            "the board pins nothing, so there is no source to poll"
        );
        let eng = SyncEngine::new(
            full_forge().with_person_repos("alice", vec!["acme/widget".into()]),
            store,
        );

        tokio::select! {
            _ = eng.run_scheduler_over_registry(|_| {}) => {
                panic!("scheduler returned early");
            }
            discovered = async {
                // Discovery runs before the period sleep, so it lands well inside this window; a
                // scheduler that skips it just leaves the set empty until the poll budget runs out.
                for _ in 0..100 {
                    let found = eng.store().board_discovered_repos("board:1").await.unwrap();
                    if !found.is_empty() {
                        return found;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Vec::new()
            } => {
                assert_eq!(discovered, vec![RID.to_string()]);
            }
        }
    }

    #[tokio::test]
    async fn sync_code_indexes_source_files_and_skips_non_source() {
        use core_embed::{Embedder, HashEmbedder};
        let forge = FakeForge::new()
            .with_pushed_at(RID, "2026-06-10T00:00:00Z")
            .with_tree(
                RID,
                vec![
                    te("src/graph.rs", "blobA", "blob", Some(120)),
                    te("README.md", "blobR", "blob", Some(50)),
                    te("bundle.js", "blobBig", "blob", Some(999_999)),
                    te("src", "t1", "tree", None),
                ],
            )
            .with_blob(RID, "blobA", "fn parse_manifest() {}");
        let eng = engine(forge).await;
        assert_eq!(eng.sync_code(&repo()).await.unwrap(), 1); // README + oversized bundle skipped

        let embedder = HashEmbedder::default();
        let hits = eng
            .store()
            .nearest_embeddings_of_kind(&embedder.embed("parse manifest"), 5, "code")
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].ref_id, "repo:default/acme/widget#src/graph.rs");
        let shas = eng.store().code_file_shas(RID).await.unwrap();
        assert_eq!(shas.get("src/graph.rs"), Some(&"blobA".to_string()));
    }

    #[tokio::test]
    async fn sync_code_is_incremental_skips_unchanged_and_prunes_removed() {
        let store = store_mem().await;
        let eng1 = SyncEngine::new(
            FakeForge::new()
                .with_pushed_at(RID, "2026-06-10T00:00:00Z")
                .with_tree(
                    RID,
                    vec![
                        te("a.rs", "shaA", "blob", Some(30)),
                        te("b.rs", "shaB", "blob", Some(30)),
                    ],
                )
                .with_blob(RID, "shaA", "fn a() {}")
                .with_blob(RID, "shaB", "fn b() {}"),
            store.clone(),
        );
        assert_eq!(eng1.sync_code(&repo()).await.unwrap(), 2);

        // New push, tree now has only a.rs (same sha): a.rs is unchanged (no blob fetch), b.rs is
        // pruned.
        let eng2 = SyncEngine::new(
            FakeForge::new()
                .with_pushed_at(RID, "2026-06-11T00:00:00Z")
                .with_tree(RID, vec![te("a.rs", "shaA", "blob", Some(30))]),
            store.clone(),
        );
        assert_eq!(eng2.sync_code(&repo()).await.unwrap(), 0);
        let shas = store.code_file_shas(RID).await.unwrap();
        assert_eq!(shas.len(), 1);
        assert!(shas.contains_key("a.rs"));
    }

    #[tokio::test]
    async fn sync_code_stores_file_health_prunes_removed_and_skips_unchanged_blobs() {
        let store = store_mem().await;
        let eng1 = SyncEngine::new(
            FakeForge::new()
                .with_pushed_at(RID, "2026-06-10T00:00:00Z")
                .with_tree(
                    RID,
                    vec![
                        te("a.rs", "shaA", "blob", Some(60)),
                        te("b.rs", "shaB", "blob", Some(60)),
                        te("App.kt", "shaK", "blob", Some(60)),
                    ],
                )
                .with_blob(
                    RID,
                    "shaA",
                    "fn a(x: i32) -> i32 { if x > 0 { x } else { -x } }",
                )
                .with_blob(RID, "shaB", "fn b() {}")
                .with_blob(RID, "shaK", "fun main() { if (true) { println(1) } }"),
            store.clone(),
        );
        assert_eq!(eng1.sync_code(&repo()).await.unwrap(), 3);

        // Only the two parsed Rust files are scored. Kotlin has no grammar, so it is stored as
        // not-analyzed.
        let health = store.repo_file_health(RID).await.unwrap();
        assert_eq!(
            health.iter().map(|h| h.0.as_str()).collect::<Vec<_>>(),
            vec!["a.rs", "b.rs"],
            "App.kt has no score, so it is not reported"
        );
        let (_, _, functions, branches, score) = health[0].clone();
        assert_eq!((functions, branches), (1, 1), "a.rs: one fn, one if");
        assert_eq!(score, 10);

        // Tamper with a stored row, then re-sync a tree where a.rs is unchanged (same sha) and
        // b.rs is gone. The unchanged blob is not refetched, so its row survives; the removed
        // file's row is deleted.
        store
            .upsert_file_health(RID, "a.rs", 999, 9, 99, Some(2))
            .await
            .unwrap();
        let eng2 = SyncEngine::new(
            FakeForge::new()
                .with_pushed_at(RID, "2026-06-11T00:00:00Z")
                .with_tree(RID, vec![te("a.rs", "shaA", "blob", Some(60))]),
            store.clone(),
        );
        assert_eq!(eng2.sync_code(&repo()).await.unwrap(), 0);
        assert_eq!(
            store.repo_file_health(RID).await.unwrap(),
            vec![("a.rs".to_string(), 999, 9, 99, 2)],
            "unchanged blob: not recomputed; b.rs + App.kt: pruned"
        );
    }

    #[tokio::test]
    async fn transient_blob_failure_keeps_code_cursor_so_next_sync_retries() {
        let store = store_mem().await;
        let pushed = "2026-06-10T00:00:00Z";
        let eng1 = SyncEngine::new(
            FakeForge::new()
                .with_pushed_at(RID, pushed)
                .with_tree(RID, vec![te("a.rs", "shaA", "blob", Some(30))])
                .with_blob_transient(RID, "shaA"),
            store.clone(),
        );
        assert_eq!(eng1.sync_code(&repo()).await.unwrap(), 0);
        assert_eq!(
            store.sync_cursor(RID, "code").await.unwrap(),
            None,
            "an incomplete pass must not advance the code cursor"
        );

        let eng2 = SyncEngine::new(
            FakeForge::new()
                .with_pushed_at(RID, pushed)
                .with_tree(RID, vec![te("a.rs", "shaA", "blob", Some(30))])
                .with_blob(RID, "shaA", "fn a() {}"),
            store.clone(),
        );
        assert_eq!(eng2.sync_code(&repo()).await.unwrap(), 1);
        assert_eq!(
            store.sync_cursor(RID, "code").await.unwrap(),
            Some(pushed.to_string()),
            "a complete pass advances the cursor"
        );
    }

    #[tokio::test]
    async fn index_job_skips_code_fetch_when_index_code_is_disabled() {
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        store.set_repo_index_code(RID, false).await.unwrap();
        let probe = FakeForge::new();
        let eng = SyncEngine::new(probe.clone(), store);
        let n = eng.process_index_job(&job()).await.unwrap();
        assert_eq!(
            n,
            IndexOutcome::default(),
            "no code is fetched or embedded when the toggle is off"
        );
        assert!(
            !probe
                .calls()
                .iter()
                .any(|(l, _)| l == "tree" || l.starts_with("blob:")),
            "the disabled toggle must skip the tree/blob fetch entirely"
        );
    }

    #[tokio::test]
    async fn index_queue_depth_reports_pending_jobs() {
        use tokio::sync::mpsc;
        let (tx, _rx) = mpsc::channel(8);
        let probe = tx.clone();
        let eng = SyncEngine::new(FakeForge::new(), store_mem().await).with_index_queue(tx);
        assert_eq!(eng.index_queue_depth(), Some(0));
        probe.send(job()).await.unwrap();
        probe.send(job()).await.unwrap();
        assert_eq!(eng.index_queue_depth(), Some(2));
        assert_eq!(engine(FakeForge::new()).await.index_queue_depth(), None);
    }

    #[tokio::test]
    async fn index_worker_fetches_and_indexes_a_repo_and_reports() {
        use core_embed::{Embedder, HashEmbedder};
        use tokio::sync::mpsc;
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        let forge = FakeForge::new()
            .with_pushed_at(RID, "2026-06-10T00:00:00Z")
            .with_tree(RID, vec![te("a.rs", "shaA", "blob", Some(30))])
            .with_blob(RID, "shaA", "fn authenticate() {}");
        let eng = SyncEngine::new(forge, store);

        let (tx, rx) = mpsc::channel(4);
        tx.send(job()).await.unwrap();
        drop(tx);

        let mut events = Vec::new();
        eng.run_index_worker(rx, |p| events.push((p.state, p.files, p.done, p.total)))
            .await;
        assert!(events.len() >= 2);
        assert_eq!(events.first().unwrap().0, IndexState::Indexing);
        let last = events.last().unwrap();
        assert_eq!(last.0, IndexState::Indexed);
        assert_eq!(last.1, 1);
        let hits = eng
            .store()
            .nearest_embeddings_of_kind(&HashEmbedder::default().embed("authenticate"), 5, "code")
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].ref_id, "repo:default/acme/widget#a.rs");
        assert_eq!(
            eng.store().sync_cursor(RID, "code").await.unwrap(),
            Some("2026-06-10T00:00:00Z".to_string())
        );
        assert!(eng.index_status().is_empty());
    }

    /// The fetch lane and index worker write the status map from separate tasks, so an old pass
    /// can finish after the repo was re-queued. Its `Indexed` must not remove the fresh entry, or
    /// the Debug "Queues" panel reports the repo idle while it waits to be indexed.
    #[tokio::test]
    async fn a_stale_indexed_does_not_remove_a_requeued_repo() {
        let eng = engine(FakeForge::new()).await;
        // Pass 1 is running; the fetch lane finishes another fetch and re-queues the repo as pass
        // 2.
        eng.record_index_status(IndexProgress::state(
            RID,
            "acme/widget",
            IndexState::Indexing,
            1,
        ));
        eng.record_index_status(IndexProgress::state(
            RID,
            "acme/widget",
            IndexState::Queued,
            2,
        ));
        // Pass 1 now finishes.
        eng.record_index_status(IndexProgress::state(
            RID,
            "acme/widget",
            IndexState::Indexed,
            1,
        ));

        let live = eng.index_status();
        assert_eq!(live.len(), 1, "the re-queued pass must still be listed");
        assert_eq!(live[0].state, IndexState::Queued);
        assert_eq!(live[0].generation, 2);
    }

    /// The ordinary case: a repo with nothing else outstanding still leaves the map when its pass
    /// reports `Indexed`.
    #[tokio::test]
    async fn a_current_indexed_removes_the_entry() {
        let eng = engine(FakeForge::new()).await;
        eng.record_index_status(IndexProgress::state(
            RID,
            "acme/widget",
            IndexState::Queued,
            7,
        ));
        eng.record_index_status(IndexProgress::state(
            RID,
            "acme/widget",
            IndexState::Indexing,
            7,
        ));
        assert_eq!(eng.index_status().len(), 1);

        eng.record_index_status(IndexProgress::state(
            RID,
            "acme/widget",
            IndexState::Indexed,
            7,
        ));
        assert!(
            eng.index_status().is_empty(),
            "a finished pass that still owns the entry must drop it"
        );
    }

    /// Same race, non-terminal shape: an old pass's percentage must not overwrite a newer pass's
    /// entry, or the old pass's `Indexed` would then match it and remove the newer entry.
    #[tokio::test]
    async fn a_stale_progress_update_does_not_overwrite_a_newer_generation() {
        let eng = engine(FakeForge::new()).await;
        eng.record_index_status(IndexProgress::state(
            RID,
            "acme/widget",
            IndexState::Queued,
            2,
        ));
        eng.record_index_status(IndexProgress {
            done: 40,
            total: 100,
            ..IndexProgress::state(RID, "acme/widget", IndexState::Indexing, 1)
        });

        let live = eng.index_status();
        assert_eq!(live.len(), 1);
        assert_eq!(
            live[0].state,
            IndexState::Queued,
            "the newer pass owns the entry"
        );
        assert_eq!(live[0].generation, 2);
        assert_eq!(
            (live[0].done, live[0].total),
            (0, 0),
            "an older pass must not write its progress onto a newer entry"
        );
    }

    /// Through the real enqueue path: two syncs of the same repo must hand the worker two
    /// distinguishable passes, and the live entry must belong to the second.
    #[tokio::test]
    async fn each_enqueue_allocates_a_fresh_generation() {
        use tokio::sync::mpsc;
        let (tx, mut rx) = mpsc::channel(8);
        let eng = SyncEngine::new(FakeForge::new(), store_mem().await).with_index_queue(tx);

        eng.sync_all(&repo()).await.unwrap();
        eng.sync_all(&repo()).await.unwrap();

        let first = rx.recv().await.unwrap();
        let second = rx.recv().await.unwrap();
        assert_eq!(
            (first.generation, second.generation),
            (1, 2),
            "each enqueue gets its own generation, in order"
        );
        let live = eng.index_status();
        assert_eq!(live.len(), 1);
        assert_eq!(
            live[0].generation, 2,
            "the entry belongs to the pass that is actually outstanding"
        );
    }

    #[tokio::test]
    async fn reconcile_index_model_resets_only_on_model_change() {
        use core_embed::{Embedder, HashEmbedder};
        let eng = engine(FakeForge::new()).await;
        let embedder = HashEmbedder::default();
        let id = format!(
            "{}:{}:{}",
            embedder.name(),
            embedder.dimensions(),
            core_embed::INDEX_SCHEME_VERSION
        );
        eng.store()
            .replace_embeddings("commit", "c1", &[("x".to_string(), embedder.embed("x"))])
            .await
            .unwrap();

        assert!(!eng.reconcile_index_model().await.unwrap());
        assert_eq!(
            eng.store().meta_get("embedder_id").await.unwrap(),
            Some(id.clone())
        );
        assert_eq!(eng.store().counts().await.unwrap().embeddings_total, 1);

        assert!(!eng.reconcile_index_model().await.unwrap());
        assert_eq!(eng.store().counts().await.unwrap().embeddings_total, 1);

        eng.store()
            .meta_set("embedder_id", "different-model:512")
            .await
            .unwrap();
        assert!(eng.reconcile_index_model().await.unwrap());
        assert_eq!(eng.store().counts().await.unwrap().embeddings_total, 0);
        assert_eq!(eng.store().meta_get("embedder_id").await.unwrap(), Some(id));
    }

    #[tokio::test]
    async fn sync_code_unchanged_push_skips_tree_and_blobs() {
        let probe = FakeForge::new()
            .with_pushed_at(RID, "2026-06-10T00:00:00Z")
            .with_tree(RID, vec![te("a.rs", "shaA", "blob", Some(30))])
            .with_blob(RID, "shaA", "fn a() {}");
        let eng = SyncEngine::new(probe.clone(), store_mem().await);
        assert_eq!(eng.sync_code(&repo()).await.unwrap(), 1);
        assert_eq!(eng.sync_code(&repo()).await.unwrap(), 0); // same pushed_at -> gate short-circuits
        assert_eq!(
            probe.calls().iter().filter(|(l, _)| l == "tree").count(),
            1,
            "the unchanged-push gate must skip the tree fetch on the second pass"
        );
    }

    // ---- linking ----

    #[tokio::test]
    async fn link_references_resolves_pr_body_to_stored_issue() {
        // A forge without authoritative closing-issue references (Gitea-style): the body keyword is
        // the only signal, so "Fixes #10" is a `closes` edge.
        let forge = FakeForge::new()
            .with_capabilities(Capabilities {
                closing_issue_refs: false,
                ..Capabilities::ALL
            })
            .with_pulls(
                RID,
                vec![pr("101", 5, "open", None, "Fixes #10, see #11")],
                Some("p"),
            )
            .with_issues(RID, vec![issue("10", 10, "a bug")], Some("i"));
        let eng = engine(forge).await;
        eng.sync_pull_requests(&repo()).await.unwrap();
        eng.sync_issues(&repo()).await.unwrap();

        let n = eng.link_references(&repo()).await.unwrap();
        assert_eq!(n, 1); // #10 resolves; #11 has no stored issue, dropped
        let links = eng.store().links_from("pull_request", "101").await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].dst_id, "10");
        assert_eq!(links[0].relation, "closes");

        eng.link_references(&repo()).await.unwrap();
        assert_eq!(
            eng.store()
                .links_from("pull_request", "101")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn link_references_prefers_authoritative_closing_refs() {
        // GitHub-style forge: the body only *mentions* #10, but the forge reports PR #5 closes
        // #10. The edge must be `closes`, not `mentions`.
        let forge = FakeForge::new()
            .with_pulls(RID, vec![pr("101", 5, "open", None, "see #10")], Some("p"))
            .with_issues(RID, vec![issue("10", 10, "a bug")], Some("i"))
            .with_pr_closing_issues(RID, 5, vec![10]);
        let eng = engine(forge).await;
        let (_report, changed) = eng.sync_pull_requests(&repo()).await.unwrap();
        eng.sync_reviews_for(&repo(), &changed).await.unwrap(); // captures closing refs
        eng.sync_issues(&repo()).await.unwrap();
        eng.link_references(&repo()).await.unwrap();

        let links = eng.store().links_from("pull_request", "101").await.unwrap();
        assert_eq!(
            links.len(),
            1,
            "the mention and the close are the same pair"
        );
        assert_eq!(links[0].dst_kind, "work_item");
        assert_eq!(links[0].dst_id, "10");
        assert_eq!(
            links[0].relation, "closes",
            "authoritative wins over the body mention"
        );
    }

    #[tokio::test]
    async fn link_references_derives_bug_kind_from_labels() {
        let mut bug = issue("10", 10, "broken");
        bug.labels = "Type: Bug".into();
        let forge =
            FakeForge::new().with_issues(RID, vec![bug, issue("11", 11, "a feature")], Some("i"));
        let eng = engine(forge).await;
        eng.sync_issues(&repo()).await.unwrap();
        eng.link_references(&repo()).await.unwrap();
        assert_eq!(
            eng.store().work_item("10").await.unwrap().unwrap().kind,
            "bug"
        );
        assert_eq!(
            eng.store().work_item("11").await.unwrap().unwrap().kind,
            "issue"
        );
    }

    #[tokio::test]
    async fn link_references_projects_issues_into_work_items_with_status() {
        // i10: open, closed-by an OPEN PR -> indeterminate. i11: open, no PR -> new. i12: closed ->
        // done.
        let mut closed = issue("12", 12, "shipped");
        closed.state = "closed".into();
        closed.closed_at = Some("2026-06-05T00:00:00Z".into());
        let forge = FakeForge::new()
            .with_pulls(
                RID,
                vec![pr("101", 5, "open", None, "Fixes #10")],
                Some("p"),
            )
            .with_issues(
                RID,
                vec![issue("10", 10, "a bug"), issue("11", 11, "later"), closed],
                Some("i"),
            );
        let eng = engine(forge).await;
        eng.sync_pull_requests(&repo()).await.unwrap();
        eng.sync_issues(&repo()).await.unwrap();
        eng.link_references(&repo()).await.unwrap();

        let store = eng.store();
        let wi10 = store.work_item("10").await.unwrap().unwrap();
        assert_eq!(wi10.source, "github");
        assert_eq!(
            wi10.status_category.as_deref(),
            Some("indeterminate"),
            "open issue with an open linked PR is in progress"
        );
        let wi11 = store.work_item("11").await.unwrap().unwrap();
        assert_eq!(wi11.status_category.as_deref(), Some("new"));
        let wi12 = store.work_item("12").await.unwrap().unwrap();
        assert_eq!(wi12.status_category.as_deref(), Some("done"));

        // The planning edge points at the work_item, not a bare issue kind.
        let links = eng.store().links_from("pull_request", "101").await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].dst_kind, "work_item");
        assert_eq!(links[0].dst_id, "10");

        // Status history from real timestamps: #10 new + in-progress (its linked PR), #11 new only
        // (no PR), #12 new + done (its close time).
        let kinds = |h: Vec<(String, String)>| h.into_iter().map(|(k, _)| k).collect::<Vec<_>>();
        let h10 = kinds(store.work_item_status_history("10").await.unwrap());
        assert!(h10.contains(&"new".to_string()) && h10.contains(&"indeterminate".to_string()));
        assert_eq!(
            kinds(store.work_item_status_history("11").await.unwrap()),
            vec!["new".to_string()]
        );
        let h12 = store.work_item_status_history("12").await.unwrap();
        assert!(h12
            .iter()
            .any(|(k, at)| k == "done" && at == "2026-06-05T00:00:00Z"));
        assert!(h12.iter().any(|(k, _)| k == "new"));
    }

    #[tokio::test]
    async fn cross_repo_dependencies_link_by_name_match() {
        let eng = engine(FakeForge::new()).await;
        let store = eng.store();
        for (id, name) in [("repo:acme/app", "app"), ("repo:acme/widget", "widget")] {
            store
                .upsert_repo(&Repo {
                    id: id.into(),
                    owner: "acme".into(),
                    name: name.into(),
                    full_name: format!("acme/{name}"),
                    ownership: "owned".into(),
                })
                .await
                .unwrap();
        }
        for dep in ["widget", "serde"] {
            store
                .upsert_dependency(&DependencyRow {
                    repo_id: "repo:acme/app".into(),
                    ecosystem: "cargo".into(),
                    name: dep.into(),
                    version_req: Some("1".into()),
                    kind: "normal".into(),
                    source: "Cargo.toml".into(),
                })
                .await
                .unwrap();
        }
        let n = eng.link_cross_repo_dependencies().await.unwrap();
        assert_eq!(n, 1); // only "widget" matches a watched repo
        let links = store.links_from("repo", "repo:acme/app").await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].dst_id, "repo:acme/widget");
        assert_eq!(links[0].relation, "depends_on");
    }

    /// Upsert a forge config row so the linker can derive its git host.
    async fn seed_forge(store: &Store, id: &str, kind: &str, base_url: &str) {
        store
            .upsert_forge(&core_store::ForgeRow {
                id: id.into(),
                name: id.into(),
                kind: kind.into(),
                base_url: base_url.into(),
                oauth_client_id: None,
            })
            .await
            .unwrap();
    }

    /// Watch `owner/name` on `forge_id`, returning the namespaced repo id.
    async fn watch(store: &Store, forge_id: &str, owner: &str, name: &str) -> String {
        let id = repo_id(forge_id, &format!("{owner}/{name}"));
        store
            .upsert_repo(&Repo {
                id: id.clone(),
                owner: owner.into(),
                name: name.into(),
                full_name: format!("{owner}/{name}"),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        id
    }

    async fn dep(store: &Store, repo: &str, ecosystem: &str, name: &str, req: &str) {
        store
            .upsert_dependency(&DependencyRow {
                repo_id: repo.into(),
                ecosystem: ecosystem.into(),
                name: name.into(),
                version_req: Some(req.into()),
                kind: "normal".into(),
                source: "manifest".into(),
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn cross_repo_dependencies_link_by_github_path() {
        let eng = engine(FakeForge::new()).await;
        let store = eng.store();
        seed_forge(store, "gh", "github", "https://api.github.com").await;
        let app = watch(store, "gh", "acme", "app").await;
        let widget = watch(store, "gh", "acme", "widget").await;
        // A go module path and an npm git URL, both naming acme/widget on github.com. The npm
        // package is not named "widget", so only the path route can link it.
        dep(store, &app, "go", "github.com/acme/widget/v2", "v2.1.0").await;
        dep(
            store,
            &app,
            "npm",
            "widget-js",
            "git+https://github.com/acme/widget.git#v1",
        )
        .await;

        let n = eng.link_cross_repo_dependencies().await.unwrap();
        assert_eq!(n, 1); // both resolve to acme/widget -> one de-duped edge
        let links = store.links_from("repo", &app).await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].dst_id, widget);
    }

    #[tokio::test]
    async fn cross_repo_dependencies_link_by_gitea_path() {
        let eng = engine(FakeForge::new()).await;
        let store = eng.store();
        seed_forge(store, "gitea", "gitea", "https://gitea.example.com/api/v1").await;
        let app = watch(store, "gitea", "acme", "app").await;
        let widget = watch(store, "gitea", "acme", "widget").await;
        dep(store, &app, "go", "gitea.example.com/acme/widget", "v1.0.0").await;
        dep(
            store,
            &app,
            "npm",
            "widget-js",
            "git+https://gitea.example.com/acme/widget.git",
        )
        .await;

        let n = eng.link_cross_repo_dependencies().await.unwrap();
        assert_eq!(n, 1);
        let links = store.links_from("repo", &app).await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].dst_id, widget);
    }

    #[tokio::test]
    async fn cross_repo_dependencies_link_to_the_forge_hosting_the_dependency() {
        let eng = engine(FakeForge::new()).await;
        let store = eng.store();
        seed_forge(store, "gh", "github", "https://api.github.com").await;
        seed_forge(store, "gitea", "gitea", "https://gitea.example.com/api/v1").await;
        // The same owner/name is watched on both connections; only the GitHub one is the
        // dependency.
        let app = watch(store, "gitea", "acme", "app").await;
        let gh_widget = watch(store, "gh", "acme", "widget").await;
        let gitea_widget = watch(store, "gitea", "acme", "widget").await;
        dep(store, &app, "go", "github.com/acme/widget", "v1.0.0").await;

        let n = eng.link_cross_repo_dependencies().await.unwrap();
        assert_eq!(n, 1);
        let links = store.links_from("repo", &app).await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].dst_id, gh_widget);
        assert_ne!(links[0].dst_id, gitea_widget);
    }

    #[test]
    fn forge_host_takes_the_authority_and_maps_the_github_api_root() {
        // GitHub's API root is api.github.com; its repos (what a dependency names) are on
        // github.com.
        assert_eq!(
            forge_host("https://api.github.com").as_deref(),
            Some("github.com")
        );
        // A self-hosted Gitea / Forgejo API root already carries the right host, path and all.
        assert_eq!(
            forge_host("https://gitea.example.com/api/v1").as_deref(),
            Some("gitea.example.com")
        );
        // GitHub Enterprise: no `api.` subdomain to undo.
        assert_eq!(
            forge_host("https://ghe.example.com/api/v3").as_deref(),
            Some("ghe.example.com")
        );
        // A port belongs to the host and is kept, so a URL dependency on it still matches.
        assert_eq!(
            forge_host("http://gitea.example.com:3000/api/v1").as_deref(),
            Some("gitea.example.com:3000")
        );
        // Nothing to take.
        assert_eq!(forge_host(""), None);
        assert_eq!(forge_host("https://"), None);
    }

    #[test]
    fn repo_from_source_maps_repo_kind_and_skips_others() {
        let repo = repo_from_source(&SourceRow {
            id: RID.into(),
            kind: "repo".into(),
            name: "acme/widget".into(),
            ownership: "owned".into(),
            filters: vec![],
            stale_pr_days: 7,
            forge_id: Some(DEFAULT_FORGE.into()),
        })
        .unwrap();
        assert_eq!(repo.owner, "acme");
        assert_eq!(repo.name, "widget");
        assert_eq!(repo.full_name, "acme/widget");
        // The id is namespaced by the source's forge.
        assert_eq!(repo.id, RID);
        assert_eq!(forge_id_of(&repo.id), Some(DEFAULT_FORGE));
        // A non-repo source is skipped.
        assert!(repo_from_source(&SourceRow {
            id: "org:acme".into(),
            kind: "org".into(),
            name: "acme".into(),
            ownership: "observed".into(),
            filters: vec![],
            stale_pr_days: 7,
            forge_id: Some(DEFAULT_FORGE.into()),
        })
        .is_none());
        // A repo source with no forge assigned is unrouted, so it is skipped.
        assert!(repo_from_source(&SourceRow {
            id: RID.into(),
            kind: "repo".into(),
            name: "acme/widget".into(),
            ownership: "owned".into(),
            filters: vec![],
            stale_pr_days: 7,
            forge_id: None,
        })
        .is_none());
    }

    // ---- storage budget ----

    /// Grow the store past `target` bytes so a budget in whole megabytes can put it in a chosen
    /// state. Commit messages, because they are the cheapest way to move real pages.
    async fn fill_store(store: &Store, target: i64) {
        let mut n = 0;
        loop {
            for _ in 0..40 {
                n += 1;
                store
                    .upsert_commit(&Commit {
                        sha: format!("fill{n:06}"),
                        repo_id: RID.into(),
                        author_login: Some("dev".into()),
                        message: "x".repeat(4096),
                        committed_at: "2026-01-01T00:00:00Z".into(),
                    })
                    .await
                    .unwrap();
            }
            if store.storage_usage().await.unwrap().used_bytes >= target {
                return;
            }
        }
    }

    /// Set the budget to a whole number of megabytes and return the resulting state.
    async fn set_budget(store: &Store, mb: Option<i64>) -> core_store::BudgetState {
        let mut s = store.settings().await.unwrap();
        s.storage_budget_mb = mb;
        store.update_settings(&s).await.unwrap();
        store.budget_state().await.unwrap()
    }

    #[tokio::test]
    async fn the_budget_stops_the_sync_and_a_raised_budget_resumes_it() {
        // Over the ceiling nothing is fetched, nothing is deleted to make room, and raising the
        // budget resumes syncing (the stop is a state, not a latch).
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        fill_store(&store, 3 * core_store::MB).await;

        let forge = FakeForge::new().with_commits(
            RID,
            vec![commit("newsha", "2026-07-01T00:00:00Z")],
            Some("2026-07-01T00:00:00Z"),
        );
        let eng = SyncEngine::new(forge, store.clone());

        // A budget under what is already stored: stopped.
        let used = store.storage_usage().await.unwrap().used_bytes;
        let state = set_budget(&store, Some(used / core_store::MB)).await;
        assert_eq!(state.pressure, core_store::StoragePressure::Full);

        let rows_before = store.total_row_count().await.unwrap();
        let err = eng.sync_all(&repo()).await.unwrap_err();
        assert!(
            matches!(err, SyncError::OverBudget(_)),
            "a stop must be a stated refusal, not {err:?}"
        );
        // The refusal carries the budget state's own sentence, so `user_message` tells the user
        // why.
        assert!(err
            .user_message()
            .contains("has stopped syncing and indexing"));
        assert!(
            !store
                .commits_for_repo(RID)
                .await
                .unwrap()
                .iter()
                .any(|c| c.sha == "newsha"),
            "the forge's new commit was fetched despite the stop"
        );
        assert_eq!(
            store.total_row_count().await.unwrap(),
            rows_before,
            "a stopped sync must not delete anything to make room"
        );

        // The manual "sync now" path refuses before planning: no forge call, no error, nothing
        // synced.
        assert_eq!(eng.sync_all_sources(|_| {}).await.unwrap(), 0);

        // Raise the ceiling and nothing else; the same engine syncs.
        set_budget(&store, Some(used / core_store::MB * 8 + 8)).await;
        eng.sync_all(&repo()).await.unwrap();
        assert!(store
            .commits_for_repo(RID)
            .await
            .unwrap()
            .iter()
            .any(|c| c.sha == "newsha"));
    }

    #[tokio::test]
    async fn a_full_store_pauses_the_index_worker_instead_of_failing_it() {
        // A repo whose indexing was refused must say `paused`: not `error` (nothing failed) and
        // not nothing (indistinguishable from up to date).
        use tokio::sync::mpsc;
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        fill_store(&store, 3 * core_store::MB).await;
        let used = store.storage_usage().await.unwrap().used_bytes;
        set_budget(&store, Some(used / core_store::MB)).await;

        let forge = FakeForge::new()
            .with_pushed_at(RID, "2026-06-10T00:00:00Z")
            .with_tree(RID, vec![te("a.rs", "shaA", "blob", Some(30))])
            .with_blob(RID, "shaA", "fn authenticate() {}");
        let eng = SyncEngine::new(forge, store.clone());

        let (tx, rx) = mpsc::channel(4);
        tx.send(job()).await.unwrap();
        drop(tx);
        let mut events = Vec::new();
        eng.run_index_worker(rx, |p| events.push(p)).await;

        assert_eq!(events.len(), 1, "a paused job must not report indexing too");
        assert_eq!(events[0].state, IndexState::Paused);
        assert!(events[0].error.is_none(), "a pause is not a failure");
        assert!(events[0]
            .note
            .as_deref()
            .is_some_and(|n| n.contains("storage budget")));
        // A terminal `Indexed` is dropped from the status map; a `Paused` is not.
        assert_eq!(eng.index_status().len(), 1);
        assert_eq!(
            store.code_file_shas(RID).await.unwrap().len(),
            0,
            "nothing was indexed"
        );
    }

    /// A tree of `n` small source files plus three special cases: a file over 100 kB, a file that
    /// sorts after every other path, and one blob too large for the forge to return at all.
    fn wide_tree(n: usize) -> (Vec<TreeEntry>, Vec<(String, String)>) {
        let mut tree = Vec::new();
        let mut blobs = Vec::new();
        for i in 0..n {
            let (path, sha) = (format!("src/f{i:05}.rs"), format!("sha{i:05}"));
            tree.push(te(&path, &sha, "blob", Some(30)));
            blobs.push((sha, format!("fn f{i}() {{}}")));
        }
        // A 200 kB file: indexed as a few more chunks.
        tree.push(te("src/big.rs", "shabig", "blob", Some(200_000)));
        blobs.push(("shabig".into(), "fn big() {}".to_string()));
        // Sorts after every `src/...` path.
        tree.push(te("zzz_last.rs", "shalast", "blob", Some(30)));
        blobs.push(("shalast".into(), "fn last() {}".to_string()));
        // Past what the forge's file API will return: skipped, but reported.
        tree.push(te(
            "src/huge.bin.rs",
            "shahuge",
            "blob",
            Some(300 * 1024 * 1024),
        ));
        (tree, blobs)
    }

    #[tokio::test]
    async fn every_source_file_is_indexed_across_passes_whatever_its_name_or_size() {
        // Work is bounded per pass, not what exists: across passes the union is every file
        // (including the one that sorts last and the large one), and the unfetchable file is
        // reported, not omitted.
        let n = CODE_FILES_PER_PASS + 1;
        let (tree, blobs) = wide_tree(n);
        let mut forge = FakeForge::new().with_tree(RID, tree);
        for (sha, text) in &blobs {
            forge = forge.with_blob(RID, sha, text);
        }
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        let eng = SyncEngine::new(forge, store.clone());

        // Pass one: a full batch, with the rest owed rather than dropped.
        let first = eng.fetch_changed_code(&repo()).await.unwrap();
        assert_eq!(first.changed.len(), CODE_FILES_PER_PASS);
        assert_eq!(first.pending, n + 2 - CODE_FILES_PER_PASS);
        assert!(
            !first.complete,
            "work is still owed, so the cursor must wait"
        );
        assert_eq!(first.skipped.len(), 1);
        assert_eq!(first.skipped[0].path, "src/huge.bin.rs");
        assert_eq!(first.skipped[0].bytes, 300 * 1024 * 1024);
        // A file waiting for a later pass is not pruned.
        assert!(first.removed.is_empty());
        eng.index_code(&repo(), &first).await.unwrap();

        // Pass two: the remainder, and the result is complete.
        let second = eng.fetch_changed_code(&repo()).await.unwrap();
        assert_eq!(second.pending, 0);
        assert!(second.complete);
        eng.index_code(&repo(), &second).await.unwrap();

        let indexed = store.code_file_shas(RID).await.unwrap();
        assert_eq!(
            indexed.len(),
            n + 2,
            "every source file the forge could return should be indexed after two passes"
        );
        assert!(
            indexed.contains_key("zzz_last.rs"),
            "the file that sorts last was never indexed - the alphabetical truncation is back"
        );
        assert!(
            indexed.contains_key("src/big.rs"),
            "a 200 kB source file was excluded for its size"
        );
        assert!(
            !indexed.contains_key("src/huge.bin.rs"),
            "a blob the forge cannot return must not be recorded as indexed"
        );
    }

    #[tokio::test]
    async fn a_repo_with_files_still_owed_reports_partial() {
        // Work left over must be visible: a repo only half read reports how much it still owes and
        // why, not `Indexed`.
        let n = CODE_FILES_PER_PASS + 1;
        let (tree, blobs) = wide_tree(n);
        let mut forge = FakeForge::new().with_tree(RID, tree);
        for (sha, text) in &blobs {
            forge = forge.with_blob(RID, sha, text);
        }
        let store = store_mem().await;
        store.upsert_repo(&repo()).await.unwrap();
        let eng = SyncEngine::new(forge, store.clone());

        let first = eng.process_index_job(&job()).await.unwrap();
        assert!(first.partial());
        assert_eq!(first.files, CODE_FILES_PER_PASS);
        assert_eq!(first.pending, n + 2 - CODE_FILES_PER_PASS);
        let note = first.note();
        assert!(note.contains("still to index"), "{note}");
        assert!(note.contains("skipped"), "{note}");

        // The next pass finishes the tree. One unfetchable file remains, so it is still partial
        // and says so.
        let second = eng.process_index_job(&job()).await.unwrap();
        assert_eq!(second.pending, 0);
        assert_eq!(second.skipped.len(), 1);
        assert!(second.partial());
    }
}
