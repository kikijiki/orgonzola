//! Gitea / Forgejo implementation of the [`core_forge::Forge`] trait.
//! Gitea's REST API is close to GitHub's and accepts `Authorization: Bearer <token>`, so
//! `GiteaForge` reuses `core-github`'s `GithubClient` + `HttpTransport` for HTTP, auth and
//! conditional GET, and swaps in Gitea's endpoint paths and query params. The base URL is the API
//! root (e.g. `http://host/api/v1`). Forgejo shares the API, so this covers both.
//! Differences from the GitHub forge that this encodes:
//! - pagination is `limit` (max 50) + `page`, not `per_page`;
//! - issues come from `?type=issues` (Gitea separates issues and PRs, so there is no PR to skip);
//! - pulls sort by `recentupdate` (desc) for the stop-at-cursor incremental;
//! - the tree uses `recursive=true` (Gitea truncates very large trees);
//! - the code-gate stamp is the repo's `updated_at`;
//! - CI runs and per-user activity discovery are not exposed the same way, so those capabilities
//!   are off and return empty (boards on a Gitea forge pin repos / use org discovery).

use base64::{engine::general_purpose::STANDARD, Engine};
use core_forge::{
    Capabilities, Delta, FileText, Forge, ForgeError, PullDelta, RepoMeta, RepoRef, TreeEntry,
};
use core_github::{GithubClient, GithubError, TokenProvider, Transport};
use core_store::{CiRun, Commit, Issue, PrFile, PullRequest, Release, Review};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::future::Future;

/// Items per page (Gitea caps `limit` at 50) and a safety cap on pages per entity per fetch.
const PER_PAGE: usize = 50;
const MAX_PAGES: usize = 20;

/// Pages per **backfill** chunk, a fraction of `MAX_PAGES` so walking old history costs less than
/// a forward fetch. Gitea's pages are half GitHub's, so the lane trickles.
const MAX_BACKFILL_PAGES: usize = 5;

/// Rate-limit budget below which the backfill lane does nothing, read from what `core-github`
/// tracks. Gitea usually sends no `x-ratelimit-remaining` header; then the budget is unknown and
/// the lane runs.
const BACKFILL_RATE_FLOOR: u32 = 200;

/// Prefixes of the opaque backfill tokens this forge mints; only this module reads them.
/// Commits page down a fixed `until` window; pull requests page up the `sort=oldest`
/// (created-ascending) listing, whose prefix is stable because new PRs only append to its end.
const COMMIT_BACKFILL: &str = "bf:c:";
const PULL_BACKFILL: &str = "bf:p:";

// --- Gitea JSON payloads (shapes match GitHub's where they overlap). ---

#[derive(Deserialize)]
struct GtContents {
    content: String,
    encoding: String,
}

#[derive(Deserialize)]
struct GtCommit {
    sha: String,
    commit: GtCommitDetail,
    author: Option<GtUser>,
}

#[derive(Deserialize)]
struct GtCommitDetail {
    message: String,
    author: GtCommitAuthor,
}

#[derive(Deserialize)]
struct GtCommitAuthor {
    date: String,
}

#[derive(Deserialize)]
struct GtUser {
    login: String,
}

#[derive(Deserialize)]
struct GtPull {
    id: u64,
    number: i64,
    title: String,
    state: String,
    user: Option<GtUser>,
    created_at: String,
    merged_at: Option<String>,
    #[serde(default)]
    updated_at: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    html_url: Option<String>,
}

#[derive(Deserialize)]
struct GtRepoMeta {
    #[serde(default)]
    updated_at: Option<String>,
    #[serde(default)]
    fork: bool,
    #[serde(default)]
    parent: Option<GtParent>,
    #[serde(default)]
    archived: bool,
}

// The `parent` object on a fork's repo metadata; we need its full name.
#[derive(Deserialize)]
struct GtParent {
    full_name: String,
}

#[derive(Deserialize)]
struct GtIssue {
    id: u64,
    number: i64,
    title: String,
    state: String,
    user: Option<GtUser>,
    created_at: String,
    closed_at: Option<String>,
    #[serde(default)]
    updated_at: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    labels: Vec<GtLabel>,
    #[serde(default)]
    html_url: Option<String>,
}

#[derive(Deserialize)]
struct GtLabel {
    name: String,
}

#[derive(Deserialize)]
struct GtRelease {
    id: u64,
    tag_name: String,
    name: Option<String>,
    published_at: Option<String>,
}

#[derive(Deserialize)]
struct GtReview {
    id: u64,
    user: Option<GtUser>,
    state: String,
    submitted_at: Option<String>,
}

#[derive(Deserialize)]
struct GtRepoItem {
    full_name: String,
}

#[derive(Deserialize)]
struct GtPrFile {
    filename: String,
    status: String,
    additions: i64,
    deletions: i64,
    #[serde(default)]
    patch: Option<String>,
}

#[derive(Deserialize)]
struct GtTree {
    #[serde(default)]
    tree: Vec<GtTreeEntry>,
}

#[derive(Deserialize)]
struct GtTreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
    sha: String,
    #[serde(default)]
    size: Option<i64>,
}

/// Mint a commits backfill token: page `page` of the "commits at or before `until`" window.
/// Page first, so the colon-laden `until` timestamp is the rest of the string.
fn commit_token(page: usize, until: &str) -> String {
    format!("{COMMIT_BACKFILL}{page}:{until}")
}

/// Read a commits backfill token back as `(page, until)`; `None` for a forward watermark.
fn parse_commit_token(cursor: &str) -> Option<(usize, &str)> {
    let (page, until) = cursor.strip_prefix(COMMIT_BACKFILL)?.split_once(':')?;
    Some((page.parse().ok()?, until))
}

/// Read a pull-request backfill token back as the page to resume at.
fn parse_pull_token(cursor: &str) -> Option<usize> {
    cursor.strip_prefix(PULL_BACKFILL)?.parse().ok()
}

fn backend(e: GithubError) -> ForgeError {
    match e {
        // Map as github does, else a 401 / rate-limit reads as a generic "could not reach".
        GithubError::Token(_) | GithubError::Unauthorized => ForgeError::Auth,
        GithubError::RateLimited => ForgeError::RateLimited,
        other => ForgeError::Backend(other.to_string()),
    }
}

fn parse_err(e: serde_json::Error) -> ForgeError {
    ForgeError::Backend(format!("parse: {e}"))
}

/// Decode a Gitea contents/blob payload to text. `None` when not base64 or undecodable.
fn decode_contents(contents: &GtContents) -> Option<String> {
    if contents.encoding != "base64" {
        return None;
    }
    let cleaned: String = contents.content.split_whitespace().collect();
    STANDARD
        .decode(cleaned.as_bytes())
        .ok()
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

/// The Gitea/Forgejo [`Forge`]: wraps a conditional-GET [`GithubClient`] at a Gitea API root.
pub struct GiteaForge<T, P> {
    client: GithubClient<T, P>,
}

impl<T, P> GiteaForge<T, P> {
    pub fn new(client: GithubClient<T, P>) -> Self {
        Self { client }
    }

    pub fn client(&self) -> &GithubClient<T, P> {
        &self.client
    }
}

impl<T: Transport + Sync, P: TokenProvider + Sync> GiteaForge<T, P> {
    /// Fetch all pages of a list endpoint (Gitea `limit`/`page`) up to the safety cap. Returns the
    /// items plus whether the cap was hit, so a non-draining caller can decline to advance its
    /// cursor past the un-fetched tail.
    async fn get_paginated<R: DeserializeOwned>(
        &self,
        base: &str,
    ) -> Result<(Vec<R>, bool), ForgeError> {
        let sep = if base.contains('?') { '&' } else { '?' };
        let mut all = Vec::new();
        let mut capped = true;
        for page in 1..=MAX_PAGES {
            let path = format!("{base}{sep}limit={PER_PAGE}&page={page}");
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let items: Vec<R> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
            let n = items.len();
            all.extend(items);
            if n < PER_PAGE {
                capped = false;
                break;
            }
        }
        if capped {
            eprintln!(
                "forge(gitea): {base} hit the {}-item page cap; items beyond it were not fetched this pass",
                MAX_PAGES * PER_PAGE
            );
        }
        Ok((all, capped))
    }

    /// True when the tracked rate-limit budget is known and too low to spend on history.
    fn backfill_budget_low(&self) -> bool {
        matches!(self.client.rate_limit_remaining(), Some(r) if r < BACKFILL_RATE_FLOOR)
    }

    /// One chunk of a backfill walk: up to [`MAX_BACKFILL_PAGES`] pages of `base` from `page`.
    /// Returns the items, the page to resume at, and whether a short page ended the listing. Stops
    /// early on a low rate budget, resuming at the first page it did not fetch.
    async fn backfill_pages<R: DeserializeOwned>(
        &self,
        base: &str,
        page: usize,
    ) -> Result<(Vec<R>, usize, bool), ForgeError> {
        let sep = if base.contains('?') { '&' } else { '?' };
        let mut all = Vec::new();
        let mut next = page;
        for p in page..page + MAX_BACKFILL_PAGES {
            if self.backfill_budget_low() {
                break;
            }
            let path = format!("{base}{sep}limit={PER_PAGE}&page={p}");
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let items: Vec<R> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
            let n = items.len();
            all.extend(items);
            next = p + 1;
            if n < PER_PAGE {
                return Ok((all, next, true));
            }
        }
        Ok((all, next, false))
    }

    /// One chunk of the commits backfill: page down a fixed `until` window.
    async fn commits_backfill(
        &self,
        repo_id: &str,
        full: &str,
        page: usize,
        until: &str,
    ) -> Result<Delta<Commit>, ForgeError> {
        if self.backfill_budget_low() {
            eprintln!("forge(gitea): backfill {full} commits: skipped this pass, rate budget is low; resuming later at page {page}");
            return Ok(Delta {
                items: Vec::new(),
                cursor: None,
                unchanged: true,
                backfill: Some(commit_token(page, until)),
            });
        }
        let base = format!("/repos/{full}/commits?until={until}");
        let (raw, next, exhausted) = self.backfill_pages::<GtCommit>(&base, page).await?;
        let items: Vec<Commit> = raw
            .into_iter()
            .map(|c| Commit {
                sha: c.sha,
                repo_id: repo_id.to_string(),
                author_login: c.author.map(|u| u.login),
                message: c.commit.message,
                committed_at: c.commit.author.date,
            })
            .collect();
        if exhausted {
            eprintln!(
                "forge(gitea): backfill {full} commits: +{} older commits, history complete",
                items.len()
            );
        } else {
            eprintln!(
                "forge(gitea): backfill {full} commits: +{} older commits, resuming at page {next} of the pre-{until} window",
                items.len()
            );
        }
        Ok(Delta {
            unchanged: items.is_empty(),
            items,
            cursor: None,
            backfill: (!exhausted).then(|| commit_token(next, until)),
        })
    }

    /// One chunk of the pull-request backfill: page up the `sort=oldest` listing. `changed`
    /// stays empty - reviews are not fetched for backfilled PRs.
    async fn pulls_backfill(
        &self,
        repo_id: &str,
        full: &str,
        page: usize,
    ) -> Result<PullDelta, ForgeError> {
        if self.backfill_budget_low() {
            eprintln!("forge(gitea): backfill {full} pulls: skipped this pass, rate budget is low; resuming later at page {page}");
            return Ok(PullDelta {
                items: Vec::new(),
                changed: Vec::new(),
                cursor: None,
                unchanged: true,
                backfill: Some(format!("{PULL_BACKFILL}{page}")),
            });
        }
        let base = format!("/repos/{full}/pulls?state=all&sort=oldest");
        let (raw, next, exhausted) = self.backfill_pages::<GtPull>(&base, page).await?;
        let items: Vec<PullRequest> = raw
            .into_iter()
            .map(|p| PullRequest {
                id: p.id.to_string(),
                repo_id: repo_id.to_string(),
                number: p.number,
                title: p.title,
                state: p.state,
                author_login: p.user.map(|u| u.login),
                body: p.body,
                created_at: p.created_at,
                merged_at: p.merged_at,
                html_url: p.html_url,
            })
            .collect();
        if exhausted {
            eprintln!(
                "forge(gitea): backfill {full} pulls: +{} older pull requests, history complete",
                items.len()
            );
        } else {
            eprintln!(
                "forge(gitea): backfill {full} pulls: +{} older pull requests, resuming at page {next}",
                items.len()
            );
        }
        Ok(PullDelta {
            unchanged: items.is_empty(),
            items,
            changed: Vec::new(),
            cursor: None,
            backfill: (!exhausted).then(|| format!("{PULL_BACKFILL}{next}")),
        })
    }
}

impl<T: Transport + Sync, P: TokenProvider + Sync> Forge for GiteaForge<T, P> {
    fn capabilities(&self) -> Capabilities {
        // No per-user events feed, and migrated repos carry no Actions runs, so those are off.
        // Search is off (endpoints differ across Gitea/Forgejo versions); source pickers fall back
        // to manual entry.
        Capabilities {
            person_discovery: false,
            org_discovery: true,
            search: false,
            ci: false,
            code: true,
            // No closing-issue references; the linker uses body-parsed `#N` for Gitea.
            closing_issue_refs: false,
        }
    }

    fn repo_meta(
        &self,
        repo: &RepoRef,
    ) -> impl Future<Output = Result<RepoMeta, ForgeError>> + Send {
        let full = repo.full_name.clone();
        async move {
            let path = format!("/repos/{full}");
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let meta: GtRepoMeta = serde_json::from_str(&outcome.body).map_err(parse_err)?;
            Ok(RepoMeta {
                pushed_at: meta.updated_at,
                is_fork: meta.fork,
                parent_full_name: meta.parent.map(|p| p.full_name),
                archived: meta.archived,
            })
        }
    }

    fn commits(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Commit>, ForgeError>> + Send {
        let (repo_id, full) = (repo.id.clone(), repo.full_name.clone());
        async move {
            // A token this forge minted routes to the backfill lane; anything else is a forward
            // watermark.
            if let Some((page, until)) = cursor.as_deref().and_then(parse_commit_token) {
                return self.commits_backfill(&repo_id, &full, page, until).await;
            }
            let mut base = format!("/repos/{full}/commits");
            if let Some(c) = &cursor {
                base.push_str(&format!("?since={c}"));
            }
            let (commits, capped): (Vec<GtCommit>, bool) = self.get_paginated(&base).await?;
            let mut newest = cursor.clone();
            let mut oldest: Option<&str> = None;
            let mut items = Vec::with_capacity(commits.len());
            for c in &commits {
                let date = c.commit.author.date.as_str();
                if newest.as_deref() < Some(date) {
                    newest = Some(c.commit.author.date.clone());
                }
                if oldest.is_none_or(|o| date < o) {
                    oldest = Some(date);
                }
                items.push(Commit {
                    sha: c.sha.clone(),
                    repo_id: repo_id.clone(),
                    author_login: c.author.as_ref().map(|u| u.login.clone()),
                    message: c.commit.message.clone(),
                    committed_at: c.commit.author.date.clone(),
                });
            }
            // Commits come back newest-first; a capped pass leaves an older un-fetched tail. Hand
            // it to the backfill lane and advance the watermark to the newest.
            let backfill = capped.then(|| oldest.map(|o| commit_token(1, o))).flatten();
            if let Some(token) = &backfill {
                eprintln!(
                    "forge(gitea): {full} commits: {} newest fetched, older history handed to the backfill lane ({token})",
                    items.len()
                );
            }
            Ok(Delta {
                unchanged: items.is_empty(),
                items,
                cursor: newest,
                backfill,
            })
        }
    }

    fn pull_requests(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<PullDelta, ForgeError>> + Send {
        let (repo_id, full) = (repo.id.clone(), repo.full_name.clone());
        async move {
            if let Some(page) = cursor.as_deref().and_then(parse_pull_token) {
                return self.pulls_backfill(&repo_id, &full, page).await;
            }
            let base = format!("/repos/{full}/pulls?state=all&sort=recentupdate");
            let mut items = Vec::new();
            let mut changed = Vec::new();
            let mut newest = cursor.clone();
            let mut stop = false;
            let mut capped = true;
            for page in 1..=MAX_PAGES {
                let path = format!("{base}&limit={PER_PAGE}&page={page}");
                let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
                let pulls: Vec<GtPull> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
                let n = pulls.len();
                for p in pulls {
                    if let Some(c) = &cursor {
                        if p.updated_at.as_str() <= c.as_str() {
                            stop = true;
                            break;
                        }
                    }
                    if newest.as_deref() < Some(p.updated_at.as_str()) {
                        newest = Some(p.updated_at.clone());
                    }
                    changed.push((p.id.to_string(), p.number));
                    items.push(PullRequest {
                        id: p.id.to_string(),
                        repo_id: repo_id.clone(),
                        number: p.number,
                        title: p.title,
                        state: p.state,
                        author_login: p.user.map(|u| u.login),
                        body: p.body,
                        created_at: p.created_at,
                        merged_at: p.merged_at,
                        html_url: p.html_url,
                    });
                }
                if stop || n < PER_PAGE {
                    capped = false;
                    break;
                }
            }
            // PRs come back newest-first (sort=recentupdate). A capped pass leaves an older tail;
            // the backfill lane takes it by walking `sort=oldest` from the start, so the watermark
            // may advance to the newest.
            let backfill = capped.then(|| format!("{PULL_BACKFILL}1"));
            if capped {
                eprintln!(
                    "forge(gitea): {full} pulls hit the {}-item page cap; older pull requests handed to the backfill lane",
                    MAX_PAGES * PER_PAGE
                );
            }
            Ok(PullDelta {
                unchanged: items.is_empty(),
                items,
                changed,
                cursor: newest,
                backfill,
            })
        }
    }

    fn reviews(
        &self,
        repo: &RepoRef,
        pr_number: i64,
        pr_id: &str,
    ) -> impl Future<Output = Result<Vec<Review>, ForgeError>> + Send {
        let (full, pr_id) = (repo.full_name.clone(), pr_id.to_string());
        async move {
            let path = format!("/repos/{full}/pulls/{pr_number}/reviews");
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let reviews: Vec<GtReview> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
            Ok(reviews
                .into_iter()
                .map(|rv| Review {
                    id: rv.id.to_string(),
                    pr_id: pr_id.clone(),
                    reviewer_login: rv.user.map(|u| u.login),
                    state: rv.state,
                    submitted_at: rv.submitted_at,
                })
                .collect())
        }
    }

    fn issues(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Issue>, ForgeError>> + Send {
        let (repo_id, full) = (repo.id.clone(), repo.full_name.clone());
        async move {
            // `type=issues` returns only issues (PRs are separate), so there is no PR to skip.
            // `since` bounds the incremental pull.
            let mut base = format!("/repos/{full}/issues?state=all&type=issues");
            if let Some(c) = &cursor {
                base.push_str(&format!("&since={c}"));
            }
            let (issues, capped): (Vec<GtIssue>, bool) = self.get_paginated(&base).await?;
            let unchanged = issues.is_empty();
            let mut newest = cursor.clone();
            let mut items = Vec::with_capacity(issues.len());
            for i in &issues {
                if newest.as_deref() < Some(i.updated_at.as_str()) {
                    newest = Some(i.updated_at.clone());
                }
                items.push(Issue {
                    id: i.id.to_string(),
                    repo_id: repo_id.clone(),
                    number: i.number,
                    title: i.title.clone(),
                    state: i.state.clone(),
                    author_login: i.user.as_ref().map(|u| u.login.clone()),
                    body: i.body.clone(),
                    created_at: i.created_at.clone(),
                    closed_at: i.closed_at.clone(),
                    labels: i
                        .labels
                        .iter()
                        .map(|l| l.name.as_str())
                        .collect::<Vec<_>>()
                        .join("\n"),
                    html_url: i.html_url.clone(),
                });
            }
            // Gitea's issues endpoint does not guarantee ascending order, so a capped pass cannot
            // be assumed to drain forward like GitHub's. Keep the input cursor when capped rather
            // than risk advancing past an un-fetched tail.
            let next = if capped { cursor } else { newest };
            Ok(Delta {
                items,
                cursor: next,
                unchanged,
                // No backfill lane for Gitea issues: with no guaranteed order there is no stable
                // window to walk, so a capped pass does not move the watermark.
                backfill: None,
            })
        }
    }

    fn releases(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Release>, ForgeError>> + Send {
        let (repo_id, full) = (repo.id.clone(), repo.full_name.clone());
        async move {
            let path = format!("/repos/{full}/releases?limit={PER_PAGE}");
            let fetch = self
                .client
                .get_with_etag(&path, cursor)
                .await
                .map_err(backend)?;
            let Some(body) = fetch.body else {
                return Ok(Delta {
                    items: Vec::new(),
                    cursor: None,
                    unchanged: true,
                    backfill: None,
                });
            };
            let releases: Vec<GtRelease> = serde_json::from_str(&body).map_err(parse_err)?;
            let items = releases
                .into_iter()
                .map(|r| Release {
                    id: r.id.to_string(),
                    repo_id: repo_id.clone(),
                    tag: r.tag_name,
                    name: r.name,
                    published_at: r.published_at,
                })
                .collect();
            Ok(Delta {
                items,
                cursor: fetch.etag,
                unchanged: false,
                backfill: None,
            })
        }
    }

    // Uniform `-> impl Future + Send` style across the trait impl; this one has no await.
    #[allow(clippy::manual_async_fn)]
    fn ci_runs(
        &self,
        _repo: &RepoRef,
        _cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<CiRun>, ForgeError>> + Send {
        // Gitea Actions are not exposed the same way and migrated repos carry no runs.
        async move {
            Ok(Delta {
                items: Vec::new(),
                cursor: None,
                unchanged: true,
                backfill: None,
            })
        }
    }

    fn pr_files(
        &self,
        repo: &RepoRef,
        pr_number: i64,
        pr_id: &str,
    ) -> impl Future<Output = Result<Vec<PrFile>, ForgeError>> + Send {
        let (full, pr_id) = (repo.full_name.clone(), pr_id.to_string());
        async move {
            let path = format!("/repos/{full}/pulls/{pr_number}/files?limit={PER_PAGE}");
            let outcome = match self.client.get_conditional(&path).await {
                Ok(o) => o,
                // Some Gitea versions 404 the files endpoint for certain PRs; treat as no files.
                Err(GithubError::Status(404)) => return Ok(Vec::new()),
                Err(e) => return Err(backend(e)),
            };
            let files: Vec<GtPrFile> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
            Ok(files
                .into_iter()
                .map(|f| PrFile {
                    pr_id: pr_id.clone(),
                    filename: f.filename,
                    status: f.status,
                    additions: f.additions,
                    deletions: f.deletions,
                    patch: f.patch,
                })
                .collect())
        }
    }

    fn file_text(
        &self,
        repo: &RepoRef,
        path: &str,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<FileText, ForgeError>> + Send {
        let (full, file_path) = (repo.full_name.clone(), path.to_string());
        async move {
            let path = format!("/repos/{full}/contents/{file_path}");
            let fetch = match self.client.get_with_etag(&path, cursor).await {
                Ok(f) => f,
                Err(GithubError::Status(404)) => {
                    return Ok(FileText {
                        content: None,
                        cursor: None,
                        unchanged: false,
                        missing: true,
                    })
                }
                Err(e) => return Err(backend(e)),
            };
            let Some(body) = fetch.body else {
                return Ok(FileText {
                    content: None,
                    cursor: None,
                    unchanged: true,
                    missing: false,
                });
            };
            let contents: GtContents = serde_json::from_str(&body).map_err(parse_err)?;
            Ok(FileText {
                content: decode_contents(&contents),
                cursor: fetch.etag,
                unchanged: false,
                missing: false,
            })
        }
    }

    fn tree(
        &self,
        repo: &RepoRef,
    ) -> impl Future<Output = Result<Vec<TreeEntry>, ForgeError>> + Send {
        let full = repo.full_name.clone();
        async move {
            // Gitea resolves HEAD and truncates very large trees.
            let path = format!("/repos/{full}/git/trees/HEAD?recursive=true&limit={PER_PAGE}");
            let outcome = match self.client.get_conditional(&path).await {
                Ok(o) => o,
                Err(GithubError::Status(404)) | Err(GithubError::Status(409)) => {
                    return Err(ForgeError::EmptyRepo)
                }
                Err(e) => return Err(backend(e)),
            };
            let tree: GtTree = serde_json::from_str(&outcome.body).map_err(parse_err)?;
            Ok(tree
                .tree
                .into_iter()
                .map(|e| TreeEntry {
                    path: e.path,
                    sha: e.sha,
                    kind: e.kind,
                    size: e.size,
                })
                .collect())
        }
    }

    fn blob_text(
        &self,
        repo: &RepoRef,
        sha: &str,
    ) -> impl Future<Output = Result<Option<String>, ForgeError>> + Send {
        let (full, sha) = (repo.full_name.clone(), sha.to_string());
        async move {
            let path = format!("/repos/{full}/git/blobs/{sha}");
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let Ok(contents) = serde_json::from_str::<GtContents>(&outcome.body) else {
                return Ok(None);
            };
            Ok(decode_contents(&contents))
        }
    }

    // No per-user events feed; person discovery is unsupported (capability off). Uniform
    // `-> impl Future + Send` style; no await here.
    #[allow(clippy::manual_async_fn)]
    fn person_repos(
        &self,
        _login: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        async move { Ok(Vec::new()) }
    }

    fn org_repos(&self, org: &str) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        let org = org.to_string();
        async move {
            let (repos, _capped): (Vec<GtRepoItem>, bool) =
                self.get_paginated(&format!("/orgs/{org}/repos")).await?;
            let mut seen = BTreeSet::new();
            for r in repos {
                seen.insert(r.full_name);
            }
            Ok(seen.into_iter().collect())
        }
    }

    // Search is off for Gitea (capability false); pickers fall back to manual entry. Empty so the
    // picker shows no suggestions rather than erroring.
    #[allow(clippy::manual_async_fn)]
    fn search_users(
        &self,
        _query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        async move { Ok(Vec::new()) }
    }

    #[allow(clippy::manual_async_fn)]
    fn search_orgs(
        &self,
        _query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        async move { Ok(Vec::new()) }
    }

    #[allow(clippy::manual_async_fn)]
    fn search_repos(
        &self,
        _query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        async move { Ok(Vec::new()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_github::{GhRequest, GhResponse, StaticTokenProvider, TransportError};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[test]
    fn backend_maps_auth_and_rate_limit_like_github() {
        // A 401 / rate-limit must surface as the actionable ForgeError, not a generic Backend.
        assert_eq!(backend(GithubError::Unauthorized), ForgeError::Auth);
        assert_eq!(backend(GithubError::RateLimited), ForgeError::RateLimited);
        assert_eq!(
            backend(GithubError::Status(500)),
            ForgeError::Backend("unexpected status 500".into())
        );
    }

    struct FakeTransport {
        responses: Mutex<VecDeque<GhResponse>>,
        /// Every requested path, in call order, so tests can assert what was requested.
        paths: Arc<Mutex<Vec<String>>>,
    }
    impl FakeTransport {
        fn new(responses: Vec<GhResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                paths: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }
    impl Transport for FakeTransport {
        fn send(
            &self,
            request: GhRequest,
        ) -> impl Future<Output = Result<GhResponse, TransportError>> + Send {
            self.paths.lock().unwrap().push(request.path);
            let next = self.responses.lock().unwrap().pop_front();
            async move { next.ok_or_else(|| TransportError::Failed("no response".into())) }
        }
    }

    fn ok(body: &str) -> GhResponse {
        GhResponse {
            status: 200,
            etag: None,
            body: body.to_string(),
            rate_limit_remaining: None,
        }
    }
    fn status(code: u16) -> GhResponse {
        GhResponse {
            status: code,
            etag: None,
            body: String::new(),
            rate_limit_remaining: None,
        }
    }
    fn forge(responses: Vec<GhResponse>) -> GiteaForge<FakeTransport, StaticTokenProvider> {
        GiteaForge::new(GithubClient::new(
            FakeTransport::new(responses),
            StaticTokenProvider("t".into()),
        ))
    }

    /// Like [`forge`], plus the log of requested paths (shared with the transport).
    #[allow(clippy::type_complexity)]
    fn forge_logging_paths(
        responses: Vec<GhResponse>,
    ) -> (
        GiteaForge<FakeTransport, StaticTokenProvider>,
        Arc<Mutex<Vec<String>>>,
    ) {
        let transport = FakeTransport::new(responses);
        let paths = Arc::clone(&transport.paths);
        (
            GiteaForge::new(GithubClient::new(
                transport,
                StaticTokenProvider("t".into()),
            )),
            paths,
        )
    }

    /// A response carrying an explicit rate-limit budget, for the backfill floor. Gitea does not
    /// normally send the header; this proves the lane honours it when present.
    fn ok_rl(body: &str, remaining: u32) -> GhResponse {
        GhResponse {
            status: 200,
            etag: None,
            body: body.to_string(),
            rate_limit_remaining: Some(remaining),
        }
    }
    fn repo() -> RepoRef {
        RepoRef {
            id: "repo:tinygrad/tinygrad".into(),
            full_name: "tinygrad/tinygrad".into(),
        }
    }

    #[tokio::test]
    async fn commits_map_and_advance_cursor() {
        let body = r#"[{"sha":"a83","commit":{"message":"support","author":{"date":"2026-06-11T17:10:47+08:00"}},"author":null}]"#;
        let delta = forge(vec![ok(body)]).commits(&repo(), None).await.unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].sha, "a83");
        assert_eq!(delta.items[0].author_login, None);
        assert_eq!(delta.cursor.as_deref(), Some("2026-06-11T17:10:47+08:00"));
    }

    fn commit_page() -> String {
        let items: Vec<String> = (0..PER_PAGE)
            .map(|i| {
                format!(
                    r#"{{"sha":"s{i}","commit":{{"message":"m","author":{{"date":"2026-02-{:02}T00:00:00Z"}}}},"author":null}}"#,
                    (i % 28) + 1
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    #[tokio::test]
    async fn commits_capped_advances_cursor_and_hands_the_tail_to_backfill() {
        // A full page in every slot means the cap was hit and an older tail went un-fetched. The
        // watermark advances to the newest commit because the tail leaves as a backfill token.
        let resps: Vec<GhResponse> = (0..MAX_PAGES).map(|_| ok(&commit_page())).collect();
        let delta = forge(resps)
            .commits(&repo(), Some("2026-01-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), MAX_PAGES * PER_PAGE);
        assert_eq!(delta.cursor.as_deref(), Some("2026-02-28T00:00:00Z"));
        assert_eq!(
            delta.backfill.as_deref(),
            Some("bf:c:1:2026-02-01T00:00:00Z")
        );
    }

    #[tokio::test]
    async fn commits_backfill_walks_the_until_window_and_resumes_deeper() {
        let resps: Vec<GhResponse> = (0..MAX_BACKFILL_PAGES)
            .map(|_| ok(&commit_page()))
            .collect();
        let (f, paths) = forge_logging_paths(resps);
        let delta = f
            .commits(&repo(), Some("bf:c:1:2026-02-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), MAX_BACKFILL_PAGES * PER_PAGE);
        // A chunk is cheaper than a forward fetch, so the lane cannot crowd it out.
        const { assert!(MAX_BACKFILL_PAGES < MAX_PAGES) };
        assert_eq!(
            delta.backfill.as_deref(),
            Some("bf:c:6:2026-02-01T00:00:00Z")
        );
        assert_eq!(delta.cursor, None); // a backfill never moves the forward watermark
        let paths = paths.lock().unwrap().clone();
        assert_eq!(paths.len(), MAX_BACKFILL_PAGES);
        assert!(paths[0].contains("until=2026-02-01T00:00:00Z"));
        assert!(paths[0].contains("limit=50"));
        assert!(paths[0].ends_with("page=1"));
        assert!(paths[4].ends_with("page=5"));
    }

    #[tokio::test]
    async fn commits_backfill_short_page_completes_the_walk() {
        let body = r#"[{"sha":"old","commit":{"message":"first","author":{"date":"2019-01-01T00:00:00Z"}},"author":null}]"#;
        let delta = forge(vec![ok(body)])
            .commits(&repo(), Some("bf:c:2:2020-01-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.backfill, None);
    }

    #[tokio::test]
    async fn commits_backfill_skips_the_pass_when_the_rate_budget_is_low() {
        // One call teaches the client the budget is nearly gone; the next chunk spends nothing.
        // Only one response is queued, so any request it made would fail with "no response".
        let f = forge(vec![ok_rl("[]", 10)]);
        let _ = f.commits(&repo(), None).await.unwrap();
        let delta = f
            .commits(&repo(), Some("bf:c:4:2020-01-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert!(delta.items.is_empty());
        assert_eq!(
            delta.backfill.as_deref(),
            Some("bf:c:4:2020-01-01T00:00:00Z")
        );
    }

    #[tokio::test]
    async fn pulls_capped_advances_cursor_and_hands_the_tail_to_backfill() {
        fn page() -> String {
            let items: Vec<String> = (0..PER_PAGE)
                .map(|i| {
                    format!(
                        r#"{{"id":{i},"number":{i},"title":"t","state":"open","user":null,"created_at":"2026-03-01T00:00:00Z","merged_at":null,"updated_at":"2026-03-{:02}T00:00:00Z"}}"#,
                        (i % 28) + 1
                    )
                })
                .collect();
            format!("[{}]", items.join(","))
        }
        let resps: Vec<GhResponse> = (0..MAX_PAGES).map(|_| ok(&page())).collect();
        let delta = forge(resps)
            .pull_requests(&repo(), Some("2026-01-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), MAX_PAGES * PER_PAGE);
        assert_eq!(delta.cursor.as_deref(), Some("2026-03-28T00:00:00Z"));
        assert_eq!(delta.backfill.as_deref(), Some("bf:p:1"));
    }

    #[tokio::test]
    async fn pulls_backfill_pages_the_oldest_first_listing() {
        // Gitea's `sort=oldest` is created-ascending, the same stable prefix GitHub's
        // `sort=created&direction=asc` gives, so a page number resumes cleanly.
        let body = r#"[{"id":1,"number":1,"title":"first ever","state":"closed","user":null,"created_at":"2015-01-01T00:00:00Z","merged_at":null,"updated_at":"2015-01-02T00:00:00Z"}]"#;
        let (f, paths) = forge_logging_paths(vec![ok(body)]);
        let delta = f
            .pull_requests(&repo(), Some("bf:p:2".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].title, "first ever");
        assert!(delta.changed.is_empty()); // no reviews implied for backfilled PRs
        assert_eq!(delta.backfill, None); // short page: history complete
        let paths = paths.lock().unwrap().clone();
        assert!(paths[0].contains("sort=oldest"));
        assert!(paths[0].ends_with("page=2"));
    }

    #[tokio::test]
    async fn issues_have_no_pr_to_skip() {
        let body = r#"[{"id":236,"number":1013,"title":"bug","state":"closed","user":{"login":"octo"},"created_at":"2023-06-21T03:45:53+09:00","updated_at":"2023-06-23T06:11:45+09:00","closed_at":"2023-06-23T06:11:45+09:00"}]"#;
        let delta = forge(vec![ok(body)]).issues(&repo(), None).await.unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].number, 1013);
        assert_eq!(delta.cursor.as_deref(), Some("2023-06-23T06:11:45+09:00"));
    }

    #[tokio::test]
    async fn releases_map() {
        let body = r#"[{"id":1,"tag_name":"v0.13.0","name":"tinygrad 0.13.0","published_at":"2026-05-23T05:12:45+09:00"}]"#;
        let delta = forge(vec![ok(body)]).releases(&repo(), None).await.unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].tag, "v0.13.0");
    }

    #[tokio::test]
    async fn repo_meta_reports_archived_state() {
        let f = forge(vec![ok(
            r#"{"updated_at":"2026-01-02T00:00:00Z","fork":false,"archived":true}"#,
        )]);
        let meta = f.repo_meta(&repo()).await.unwrap();
        assert!(meta.archived);
    }

    #[tokio::test]
    async fn tree_maps_and_blob_decodes() {
        let tree = r#"{"tree":[{"path":".coveragerc","type":"blob","sha":"4bb","size":38}],"truncated":true}"#;
        let f = forge(vec![ok(tree)]);
        let entries = f.tree(&repo()).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, ".coveragerc");
        // aGVsbG8= is base64 "hello"
        let blob = r#"{"content":"aGVsbG8=","encoding":"base64"}"#;
        let f2 = forge(vec![ok(blob)]);
        assert_eq!(
            f2.blob_text(&repo(), "4bb").await.unwrap(),
            Some("hello".to_string())
        );
    }

    #[tokio::test]
    async fn empty_repo_maps_to_forge_error() {
        assert_eq!(
            forge(vec![status(404)]).tree(&repo()).await,
            Err(ForgeError::EmptyRepo)
        );
    }

    #[tokio::test]
    async fn ci_is_unchanged_and_capabilities_reflect_gitea() {
        let f = forge(vec![]);
        assert!(f.ci_runs(&repo(), None).await.unwrap().unchanged);
        let caps = f.capabilities();
        assert!(!caps.ci);
        assert!(!caps.person_discovery);
        assert!(caps.org_discovery && caps.code);
        // person discovery is a no-op (no GitHub-style events feed).
        assert!(f.person_repos("x").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn missing_manifest_is_not_an_error() {
        let f = forge(vec![status(404)]);
        assert!(
            f.file_text(&repo(), "Cargo.toml", None)
                .await
                .unwrap()
                .missing
        );
    }
}
