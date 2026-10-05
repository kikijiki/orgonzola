//! GitHub implementation of the [`core_forge::Forge`] trait.
//! `GithubForge` wraps a [`core_github::GithubClient`] and owns everything GitHub-shaped: REST
//! paths, the `Gh*` serde structs, pagination, the pull-request stop-at-cursor loop, base64
//! decoding, and what each cursor means (a `since` timestamp, an ETag, a `pushed_at` stamp).
//! Methods return the normalized [`core_store`] types, so `core-sync` never sees a GitHub-specific
//! field.
//! Cursor contract (see [`core_forge::Delta`]): a `since`/watermark entity returns the newest
//! timestamp as the cursor and `unchanged` when nothing came back; an ETag entity returns the new
//! ETag and `unchanged = true` on a 304 (the consumer keeps the old cursor and writes nothing).

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

/// Items per page on list fetches (GitHub's max) and a cap on pages per entity per fetch, so a
/// huge first-time backlog stays finite.
const PER_PAGE: usize = 100;
const MAX_PAGES: usize = 10;

/// Pages per backfill chunk. A fraction of `MAX_PAGES` so a chunk costs less than a forward fetch;
/// a repo drains over several passes.
const MAX_BACKFILL_PAGES: usize = 5;

/// The rate-limit budget (from `core-github`'s `x-ratelimit-remaining` tracking) below which the
/// backfill lane does nothing. The forward lane is never gated by it.
const BACKFILL_RATE_FLOOR: u32 = 200;

/// Prefixes of the opaque backfill tokens this forge mints; `core-sync` replays them verbatim.
/// Commits walk down a fixed `until` window by page number; pull requests walk the
/// created-ascending listing up by page number (a stable prefix: new PRs only append).
const COMMIT_BACKFILL: &str = "bf:c:";
const PULL_BACKFILL: &str = "bf:p:";

// --- GitHub JSON payloads, reduced to what we map. ---

// GET /repos/{full}/contents/{path}: the file body is base64-encoded in `content` (with newlines).
#[derive(Deserialize)]
struct GhContents {
    content: String,
    encoding: String,
}

// GET /repos/{full}/commits.
#[derive(Deserialize)]
struct GhCommit {
    sha: String,
    commit: GhCommitDetail,
    author: Option<GhUser>,
}

#[derive(Deserialize)]
struct GhCommitDetail {
    message: String,
    author: GhCommitAuthor,
}

#[derive(Deserialize)]
struct GhCommitAuthor {
    date: String,
}

#[derive(Deserialize)]
struct GhUser {
    login: String,
}

// GET /repos/{full}/pulls (sorted by `updated` desc; `updated_at` drives the incremental cursor).
#[derive(Deserialize)]
struct GhPull {
    id: u64,
    number: i64,
    title: String,
    state: String,
    user: Option<GhUser>,
    created_at: String,
    merged_at: Option<String>,
    #[serde(default)]
    updated_at: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    html_url: Option<String>,
}

// GET /repos/{full}: the code-change gate, fork relationship, and archived state.
#[derive(Deserialize)]
struct GhRepoMeta {
    #[serde(default)]
    pushed_at: Option<String>,
    #[serde(default)]
    fork: bool,
    #[serde(default)]
    parent: Option<GhParent>,
    #[serde(default)]
    archived: bool,
}

// The `parent` object on a fork's repo metadata - we only need its full name.
#[derive(Deserialize)]
struct GhParent {
    full_name: String,
}

// GET /repos/{full}/issues. The issues endpoint also returns pull requests (they carry a
// `pull_request` object); those are skipped so issues and PRs do not collide.
#[derive(Deserialize)]
struct GhIssue {
    id: u64,
    number: i64,
    title: String,
    state: String,
    user: Option<GhUser>,
    created_at: String,
    closed_at: Option<String>,
    #[serde(default)]
    updated_at: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    pull_request: Option<serde_json::Value>,
    #[serde(default)]
    labels: Vec<GhLabel>,
    #[serde(default)]
    html_url: Option<String>,
}

#[derive(Deserialize)]
struct GhLabel {
    name: String,
}

// The GraphQL `closingIssuesReferences` response shape: the issue numbers a PR closes.
#[derive(Debug, Deserialize)]
struct GqlClosingResp {
    #[serde(default)]
    data: Option<GqlClosingData>,
}
#[derive(Debug, Deserialize)]
struct GqlClosingData {
    repository: Option<GqlClosingRepo>,
}
#[derive(Debug, Deserialize)]
struct GqlClosingRepo {
    #[serde(rename = "pullRequest")]
    pull_request: Option<GqlClosingPr>,
}
#[derive(Debug, Deserialize)]
struct GqlClosingPr {
    #[serde(rename = "closingIssuesReferences")]
    closing_issues_references: GqlClosingNodes,
}
#[derive(Debug, Deserialize)]
struct GqlClosingNodes {
    nodes: Vec<GqlIssueNumber>,
}
#[derive(Debug, Deserialize)]
struct GqlIssueNumber {
    number: i64,
}

// GET /repos/{full}/releases.
#[derive(Deserialize)]
struct GhRelease {
    id: u64,
    tag_name: String,
    name: Option<String>,
    published_at: Option<String>,
}

// GET /repos/{full}/pulls/{number}/reviews.
#[derive(Deserialize)]
struct GhReview {
    id: u64,
    user: Option<GhUser>,
    state: String,
    submitted_at: Option<String>,
}

// GET /users/{login}/events: the repo each event touched ("owner/name").
#[derive(Deserialize)]
struct GhEvent {
    repo: GhEventRepo,
}

#[derive(Deserialize)]
struct GhEventRepo {
    name: String,
}

// GET /orgs/{org}/repos and the items of GET /search/repositories: each repo's "owner/name".
#[derive(Deserialize)]
struct GhRepoItem {
    full_name: String,
}

// GET /search/users and /search/repositories wrap their hits in an `items` array (plus
// total_count, which we ignore). `T` is the per-item shape (an account login or a repo).
#[derive(Deserialize)]
struct GhSearch<T> {
    #[serde(default = "Vec::new")]
    items: Vec<T>,
}

// A /search/users item, and GET /user: the login of a matched (or the authenticated) user or org.
#[derive(Deserialize)]
struct GhAccount {
    login: String,
}

// GET /repos/{full}/pulls/{number}/files: one entry per changed file.
#[derive(Deserialize)]
struct GhPrFile {
    filename: String,
    status: String,
    additions: i64,
    deletions: i64,
    #[serde(default)]
    patch: Option<String>,
}

// GET /repos/{full}/git/trees/{ref}?recursive=1: the flattened file tree.
#[derive(Deserialize)]
struct GhTree {
    #[serde(default)]
    tree: Vec<GhTreeEntry>,
}

#[derive(Deserialize)]
struct GhTreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
    sha: String,
    #[serde(default)]
    size: Option<i64>,
}

// GET /repos/{full}/actions/runs: an object wrapping the runs array.
#[derive(Deserialize)]
struct GhWorkflowRuns {
    workflow_runs: Vec<GhWorkflowRun>,
}

#[derive(Deserialize)]
struct GhWorkflowRun {
    id: u64,
    head_sha: String,
    status: String,
    conclusion: Option<String>,
    updated_at: Option<String>,
    #[serde(default)]
    html_url: Option<String>,
    // Latest attempt number: 1 on the first try, incremented on each re-run. Absent on very old
    // runs or unexpected payloads (None, never counted flaky).
    #[serde(default)]
    run_attempt: Option<i64>,
}

/// Mint a commits backfill token: page `page` of the "commits at or before `until`" window.
/// The page comes first so the `until` timestamp, which contains colons, is the rest.
fn commit_token(page: usize, until: &str) -> String {
    format!("{COMMIT_BACKFILL}{page}:{until}")
}

/// Read a commits backfill token back as `(page, until)`. `None` for anything this forge did not
/// mint, such as a forward watermark; this is how one method serves both lanes.
fn parse_commit_token(cursor: &str) -> Option<(usize, &str)> {
    let (page, until) = cursor.strip_prefix(COMMIT_BACKFILL)?.split_once(':')?;
    Some((page.parse().ok()?, until))
}

/// Read a pull-request backfill token back, as the page to resume at.
fn parse_pull_token(cursor: &str) -> Option<usize> {
    cursor.strip_prefix(PULL_BACKFILL)?.parse().ok()
}

/// Map a `GithubError` onto `ForgeError`. Handles auth and rate limits; anything else is `Backend`.
fn backend(e: GithubError) -> ForgeError {
    match e {
        // No token, or the forge rejected it (401): both are "reconnect".
        GithubError::Token(_) | GithubError::Unauthorized => ForgeError::Auth,
        GithubError::RateLimited => ForgeError::RateLimited,
        other => ForgeError::Backend(other.to_string()),
    }
}

/// A JSON parse failure as a `Backend` error.
fn parse_err(e: serde_json::Error) -> ForgeError {
    ForgeError::Backend(format!("parse: {e}"))
}

/// Percent-encode a user-typed search term for a URL query value (RFC 3986 unreserved chars pass
/// through). The fixed qualifiers are appended after encoding, so a term like `acme/wid` cannot
/// add search qualifiers.
fn encode_query(q: &str) -> String {
    let mut out = String::with_capacity(q.len());
    for b in q.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Decode a GitHub contents/blob payload to text. `None` when the encoding is not base64 or the
/// bytes do not decode; a deterministic skip, unlike a transport failure.
fn decode_contents(contents: &GhContents) -> Option<String> {
    if contents.encoding != "base64" {
        return None;
    }
    // The API wraps base64 at column 60 with newlines; strip whitespace before decoding.
    let cleaned: String = contents.content.split_whitespace().collect();
    STANDARD
        .decode(cleaned.as_bytes())
        .ok()
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

/// The GitHub [`Forge`]: wraps a conditional-GET [`GithubClient`] and maps GitHub's REST API.
pub struct GithubForge<T, P> {
    client: GithubClient<T, P>,
}

impl<T, P> GithubForge<T, P> {
    pub fn new(client: GithubClient<T, P>) -> Self {
        Self { client }
    }

    /// The wrapped client (e.g. for the host to read the rate-limit budget).
    pub fn client(&self) -> &GithubClient<T, P> {
        &self.client
    }
}

impl<T: Transport + Sync, P: TokenProvider + Sync> GithubForge<T, P> {
    /// Fetch all pages of a list endpoint, appending `per_page` and `page=N` until a short page or
    /// the `MAX_PAGES` cap. `base` may already carry a query. Returns the items and whether the
    /// cap was hit, so a newest-first caller can decline to advance its cursor past the tail.
    async fn get_paginated<R: DeserializeOwned>(
        &self,
        base: &str,
    ) -> Result<(Vec<R>, bool), ForgeError> {
        let sep = if base.contains('?') { '&' } else { '?' };
        let mut all = Vec::new();
        let mut capped = true;
        for page in 1..=MAX_PAGES {
            let path = format!("{base}{sep}per_page={PER_PAGE}&page={page}");
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
            // The window held more than the cap; the remainder was not fetched this pass. Surface
            // it rather than truncating silently.
            eprintln!(
                "forge(github): {base} hit the {}-item page cap; items beyond it were not fetched this pass",
                MAX_PAGES * PER_PAGE
            );
        }
        Ok((all, capped))
    }

    /// True when the tracked `x-ratelimit-remaining` is known and too low to spend on history.
    /// Unknown means "go ahead": the forward lane runs first, so the number is real by then.
    fn backfill_budget_low(&self) -> bool {
        matches!(self.client.rate_limit_remaining(), Some(r) if r < BACKFILL_RATE_FLOOR)
    }

    /// One chunk of a backfill walk: up to [`MAX_BACKFILL_PAGES`] pages of `base` from `page`.
    /// Returns the items, the page to resume at, and whether the listing ended (short page). Stops
    /// early, resuming at the first page not fetched, when the rate budget drops below the floor.
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
            let path = format!("{base}{sep}per_page={PER_PAGE}&page={p}");
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let items: Vec<R> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
            let n = items.len();
            all.extend(items);
            next = p + 1;
            if n < PER_PAGE {
                // Listing ran out: the walk reached the end of history.
                return Ok((all, next, true));
            }
        }
        Ok((all, next, false))
    }

    /// One chunk of the commits backfill: page down a fixed `until` window. The window is fixed,
    /// so the page sequence is stable and each chunk resumes deeper than the last.
    async fn commits_backfill(
        &self,
        repo_id: &str,
        full: &str,
        page: usize,
        until: &str,
    ) -> Result<Delta<Commit>, ForgeError> {
        if self.backfill_budget_low() {
            eprintln!(
                "forge(github): backfill {full} commits: skipped this pass, rate budget is low ({} left); resuming later at page {page}",
                self.client.rate_limit_remaining().unwrap_or(0)
            );
            return Ok(Delta {
                items: Vec::new(),
                cursor: None,
                unchanged: true,
                backfill: Some(commit_token(page, until)),
            });
        }
        let base = format!("/repos/{full}/commits?until={until}");
        let (raw, next, exhausted) = self.backfill_pages::<GhCommit>(&base, page).await?;
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
                "forge(github): backfill {full} commits: +{} older commits, history complete",
                items.len()
            );
        } else {
            eprintln!(
                "forge(github): backfill {full} commits: +{} older commits, resuming at page {next} of the pre-{until} window",
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

    /// One chunk of the pull-request backfill: page up the created-ascending listing. That order
    /// is a stable prefix (new PRs only append), so a page number is a valid resume point.
    /// `changed` stays empty: reviews cost one call per PR, too much for backfill.
    async fn pulls_backfill(
        &self,
        repo_id: &str,
        full: &str,
        page: usize,
    ) -> Result<PullDelta, ForgeError> {
        if self.backfill_budget_low() {
            eprintln!(
                "forge(github): backfill {full} pulls: skipped this pass, rate budget is low ({} left); resuming later at page {page}",
                self.client.rate_limit_remaining().unwrap_or(0)
            );
            return Ok(PullDelta {
                items: Vec::new(),
                changed: Vec::new(),
                cursor: None,
                unchanged: true,
                backfill: Some(format!("{PULL_BACKFILL}{page}")),
            });
        }
        let base = format!("/repos/{full}/pulls?state=all&sort=created&direction=asc");
        let (raw, next, exhausted) = self.backfill_pages::<GhPull>(&base, page).await?;
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
                "forge(github): backfill {full} pulls: +{} older pull requests, history complete",
                items.len()
            );
        } else {
            eprintln!(
                "forge(github): backfill {full} pulls: +{} older pull requests, resuming at page {next}",
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

    /// Fetch one page of a `/search/*` endpoint and return its `items`. Not conditional, so
    /// distinct type-ahead queries do not fill the conditional-fetch cache.
    async fn search_hits<R: DeserializeOwned>(&self, path: &str) -> Result<Vec<R>, ForgeError> {
        let fetch = self
            .client
            .get_with_etag(path, None)
            .await
            .map_err(backend)?;
        // A first request with no ETag is never a 304; treat an absent body as no matches.
        let Some(body) = fetch.body else {
            return Ok(Vec::new());
        };
        let page: GhSearch<R> = serde_json::from_str(&body).map_err(parse_err)?;
        Ok(page.items)
    }

    /// The repos a person recently touched, from their activity feed. One page, newest event
    /// first; duplicates are left in so the caller can dedupe preserving that order.
    async fn person_event_repos(&self, login: &str) -> Result<Vec<String>, ForgeError> {
        let path = format!("/users/{login}/events?per_page={PER_PAGE}");
        let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
        let events: Vec<GhEvent> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
        Ok(events.into_iter().map(|e| e.repo.name).collect())
    }

    /// The repos a person owns, most-recently-pushed first. For the token's own account this is
    /// `/user/repos`, which includes private repos; for anyone else the public
    /// `/users/{login}/repos`. Owner-only; see the affiliation note below.
    async fn person_owned_repos(&self, login: &str) -> Result<Vec<String>, ForgeError> {
        let is_self = self
            .viewer_login()
            .await
            // GitHub logins are case-insensitive, so a board storing "Alice" is still the token
            // owner.
            .is_some_and(|viewer| viewer.eq_ignore_ascii_case(login));
        // Owner-only on both branches; do not add affiliations back. `organization_member` swept
        // in every repo of every org the person belongs to (e.g. EpicGames membership grants
        // Unreal Engine source access), burying boards under repos nobody added. `collaborator` is
        // the same problem. The events feed covers repos the person worked in but does not own.
        // `type=owner` is the public default, spelled out so both branches state the same rule.
        let path = if is_self {
            format!("/user/repos?per_page={PER_PAGE}&sort=pushed&direction=desc&affiliation=owner")
        } else {
            format!(
                "/users/{login}/repos?per_page={PER_PAGE}&sort=pushed&direction=desc&type=owner"
            )
        };
        let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
        let repos: Vec<GhRepoItem> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
        Ok(repos.into_iter().map(|r| r.full_name).collect())
    }

    /// The login the token authenticates as, or `None` when `/user` cannot be read. It only picks
    /// which owned-repos listing to ask for, so a miss falls back to the public one.
    async fn viewer_login(&self) -> Option<String> {
        let outcome = self.client.get_conditional("/user").await.ok()?;
        serde_json::from_str::<GhAccount>(&outcome.body)
            .ok()
            .map(|a| a.login)
    }
}

impl<T: Transport + Sync, P: TokenProvider + Sync> Forge for GithubForge<T, P> {
    fn capabilities(&self) -> Capabilities {
        Capabilities::ALL
    }

    fn repo_meta(
        &self,
        repo: &RepoRef,
    ) -> impl Future<Output = Result<RepoMeta, ForgeError>> + Send {
        let full = repo.full_name.clone();
        async move {
            let path = format!("/repos/{full}");
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let meta: GhRepoMeta = serde_json::from_str(&outcome.body).map_err(parse_err)?;
            Ok(RepoMeta {
                pushed_at: meta.pushed_at,
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
            // watermark. Both lanes share the method so the `Forge` trait does not change.
            if let Some((page, until)) = cursor.as_deref().and_then(parse_commit_token) {
                return self.commits_backfill(&repo_id, &full, page, until).await;
            }
            let mut base = format!("/repos/{full}/commits");
            if let Some(c) = &cursor {
                base.push_str(&format!("?since={c}"));
            }
            let (commits, capped): (Vec<GhCommit>, bool) = self.get_paginated(&base).await?;
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
            // Commits come back newest-first, so a capped pass leaves an older un-fetched tail.
            // Hand it to the backfill lane, which walks down from the oldest commit seen; the
            // watermark can then advance to the newest.
            let backfill = capped.then(|| oldest.map(|o| commit_token(1, o))).flatten();
            if let Some(token) = &backfill {
                eprintln!(
                    "forge(github): {full} commits: {} newest fetched, older history handed to the backfill lane ({token})",
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
            let base = format!("/repos/{full}/pulls?state=all&sort=updated&direction=desc");
            let mut items = Vec::new();
            let mut changed = Vec::new();
            let mut newest = cursor.clone();
            let mut stop = false;
            let mut capped = true;
            for page in 1..=MAX_PAGES {
                let path = format!("{base}&per_page={PER_PAGE}&page={page}");
                let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
                let pulls: Vec<GhPull> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
                let n = pulls.len();
                for p in pulls {
                    if let Some(c) = &cursor {
                        if p.updated_at.as_str() <= c.as_str() {
                            stop = true; // sorted desc: this and the rest are not newer than the cursor
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
            // PRs come back newest-first (sort=updated&direction=desc). A capped pass leaves an
            // older tail for the backfill lane (created-ascending listing), so the watermark may
            // advance to the newest.
            let backfill = capped.then(|| format!("{PULL_BACKFILL}1"));
            if capped {
                eprintln!(
                    "forge(github): {full} pulls hit the {}-item page cap; older pull requests handed to the backfill lane",
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
            let reviews: Vec<GhReview> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
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
            let mut base = format!("/repos/{full}/issues?state=all&sort=updated&direction=asc");
            if let Some(c) = &cursor {
                base.push_str(&format!("&since={c}"));
            }
            // Issues are fetched ascending by `updated` + `since`, so the cap drops the newest
            // items; the next pass re-fetches them (still `> cursor`). Advancing the cursor on a
            // capped pass is therefore safe here, unlike commits/PRs.
            let (issues, _capped): (Vec<GhIssue>, bool) = self.get_paginated(&base).await?;
            let unchanged = issues.is_empty();
            let mut newest = cursor.clone();
            let mut items = Vec::new();
            for i in &issues {
                // Advance the cursor across every returned item, including the PRs we skip, so a
                // PR-only update still moves it.
                if newest.as_deref() < Some(i.updated_at.as_str()) {
                    newest = Some(i.updated_at.clone());
                }
                if i.pull_request.is_some() {
                    continue; // a PR masquerading as an issue; handled by pull_requests
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
            Ok(Delta {
                items,
                cursor: newest,
                unchanged,
                // No backfill lane: the listing is ascending by `updated` with `since`, so a
                // capped pass drops the newest items and the next pass re-fetches them.
                backfill: None,
            })
        }
    }

    fn pr_closing_issues(
        &self,
        repo: &RepoRef,
        pr_number: i64,
    ) -> impl Future<Output = Result<Vec<i64>, ForgeError>> + Send {
        let full = repo.full_name.clone();
        async move {
            let (owner, name) = full.split_once('/').unwrap_or((full.as_str(), ""));
            let body = serde_json::json!({
                "query": "query($owner:String!,$repo:String!,$pr:Int!){repository(owner:$owner,name:$repo){pullRequest(number:$pr){closingIssuesReferences(first:50){nodes{number}}}}}",
                "variables": { "owner": owner, "repo": name, "pr": pr_number },
            })
            .to_string();
            let resp = self.client.graphql(body).await.map_err(backend)?;
            let parsed: GqlClosingResp = serde_json::from_str(&resp).map_err(parse_err)?;
            let numbers = parsed
                .data
                .and_then(|d| d.repository)
                .and_then(|r| r.pull_request)
                .map(|p| {
                    p.closing_issues_references
                        .nodes
                        .into_iter()
                        .map(|n| n.number)
                        .collect()
                })
                .unwrap_or_default();
            Ok(numbers)
        }
    }

    fn releases(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Release>, ForgeError>> + Send {
        let (repo_id, full) = (repo.id.clone(), repo.full_name.clone());
        async move {
            let path = format!("/repos/{full}/releases");
            let fetch = self
                .client
                .get_with_etag(&path, cursor)
                .await
                .map_err(backend)?;
            let Some(body) = fetch.body else {
                // 304: the stored releases are still current. No items; keep the old cursor.
                return Ok(Delta {
                    items: Vec::new(),
                    cursor: None,
                    unchanged: true,
                    backfill: None,
                });
            };
            let releases: Vec<GhRelease> = serde_json::from_str(&body).map_err(parse_err)?;
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
                // A single ETag-conditional page; no deeper history to walk.
                backfill: None,
            })
        }
    }

    fn ci_runs(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<CiRun>, ForgeError>> + Send {
        let (repo_id, full) = (repo.id.clone(), repo.full_name.clone());
        async move {
            let path = format!("/repos/{full}/actions/runs");
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
            let parsed: GhWorkflowRuns = serde_json::from_str(&body).map_err(parse_err)?;
            let items = parsed
                .workflow_runs
                .into_iter()
                .map(|run| CiRun {
                    id: run.id.to_string(),
                    repo_id: repo_id.clone(),
                    commit_sha: Some(run.head_sha),
                    status: run.status,
                    conclusion: run.conclusion,
                    completed_at: run.updated_at,
                    html_url: run.html_url,
                    run_attempt: run.run_attempt,
                })
                .collect();
            Ok(Delta {
                items,
                cursor: fetch.etag,
                unchanged: false,
                // A single ETag-conditional page; no deeper history to walk.
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
            let path = format!("/repos/{full}/pulls/{pr_number}/files?per_page=100");
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let files: Vec<GhPrFile> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
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
                // A repo without this file is normal, not an error.
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
                // 304: the stored content is still current.
                return Ok(FileText {
                    content: None,
                    cursor: None,
                    unchanged: true,
                    missing: false,
                });
            };
            let contents: GhContents = serde_json::from_str(&body).map_err(parse_err)?;
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
            let path = format!("/repos/{full}/git/trees/HEAD?recursive=1");
            let outcome = match self.client.get_conditional(&path).await {
                Ok(o) => o,
                // No default branch / empty repo: there is no tree to walk (not a failure).
                Err(GithubError::Status(404)) | Err(GithubError::Status(409)) => {
                    return Err(ForgeError::EmptyRepo)
                }
                Err(e) => return Err(backend(e)),
            };
            let tree: GhTree = serde_json::from_str(&outcome.body).map_err(parse_err)?;
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
            // A fetch failure is transient (Err, the caller retries). A present-but-undecodable
            // blob is a deterministic skip (Ok(None)).
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let Ok(contents) = serde_json::from_str::<GhContents>(&outcome.body) else {
                return Ok(None);
            };
            Ok(decode_contents(&contents))
        }
    }

    fn person_repos(
        &self,
        login: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        let login = login.to_string();
        async move {
            // The activity feed alone is narrow: capped at 300 events and roughly 90 days, and
            // public events only unless the token is theirs. Union it with the repos they own so a
            // repo they have not pushed to this quarter is still discoverable. The ownership half
            // stays owner-only. Best-effort per half: one failing still returns the other.
            let touched = self.person_event_repos(&login).await;
            let owned = self.person_owned_repos(&login).await;
            // Both failed: report the failure so the caller keeps the board's last-known repos
            // instead of pruning them.
            if let (Err(e), Err(_)) = (&touched, &owned) {
                return Err(e.clone());
            }
            // Recent activity first, then ownership, deduped in that order so a board that hits
            // its discovery cap keeps the active repos rather than an alphabetical slice.
            let mut seen = BTreeSet::new();
            let mut ordered = Vec::new();
            for full in touched
                .into_iter()
                .flatten()
                .chain(owned.into_iter().flatten())
            {
                if seen.insert(full.clone()) {
                    ordered.push(full);
                }
            }
            Ok(ordered)
        }
    }

    fn org_repos(&self, org: &str) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        let org = org.to_string();
        async move {
            // Most-recently-pushed first, so a capped org board keeps the active repos. Dedupe
            // preserving that order rather than re-sorting.
            let path = format!("/orgs/{org}/repos?per_page=100&sort=pushed&direction=desc");
            let outcome = self.client.get_conditional(&path).await.map_err(backend)?;
            let repos: Vec<GhRepoItem> = serde_json::from_str(&outcome.body).map_err(parse_err)?;
            let mut seen = BTreeSet::new();
            let mut ordered = Vec::new();
            for r in repos {
                if seen.insert(r.full_name.clone()) {
                    ordered.push(r.full_name);
                }
            }
            Ok(ordered)
        }
    }

    fn search_users(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        let path = format!(
            "/search/users?q={}+type:user&per_page=20",
            encode_query(query)
        );
        async move {
            let hits: Vec<GhAccount> = self.search_hits(&path).await?;
            Ok(hits.into_iter().map(|a| a.login).collect())
        }
    }

    fn search_orgs(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        let path = format!(
            "/search/users?q={}+type:org&per_page=20",
            encode_query(query)
        );
        async move {
            let hits: Vec<GhAccount> = self.search_hits(&path).await?;
            Ok(hits.into_iter().map(|a| a.login).collect())
        }
    }

    fn search_repos(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        let path = format!("/search/repositories?q={}&per_page=20", encode_query(query));
        async move {
            let hits: Vec<GhRepoItem> = self.search_hits(&path).await?;
            Ok(hits.into_iter().map(|r| r.full_name).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_github::{GhRequest, GhResponse, StaticTokenProvider, TransportError};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    /// A transport that returns programmed responses in call order, regardless of path - enough to
    /// drive the known per-method request sequence.
    struct FakeTransport {
        responses: Mutex<VecDeque<GhResponse>>,
        /// Every requested path, in call order - so a test can assert which endpoint was chosen.
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
            async move { next.ok_or_else(|| TransportError::Failed("no more responses".into())) }
        }

        // GraphQL (closing-issue references) pulls from the same programmed queue.
        fn post_json(
            &self,
            _path: String,
            _body: String,
        ) -> impl Future<Output = Result<GhResponse, TransportError>> + Send {
            let next = self.responses.lock().unwrap().pop_front();
            async move { next.ok_or_else(|| TransportError::Failed("no more responses".into())) }
        }
    }

    fn ok(body: &str) -> GhResponse {
        GhResponse {
            status: 200,
            etag: None,
            body: body.to_string(),
            rate_limit_remaining: Some(4999),
        }
    }

    /// Like [`ok`], but with an explicit `x-ratelimit-remaining` so a test can drive the budget
    /// the backfill lane reads.
    fn ok_rl(body: &str, remaining: u32) -> GhResponse {
        GhResponse {
            status: 200,
            etag: None,
            body: body.to_string(),
            rate_limit_remaining: Some(remaining),
        }
    }

    fn ok_etag(body: &str, etag: &str) -> GhResponse {
        GhResponse {
            status: 200,
            etag: Some(etag.to_string()),
            body: body.to_string(),
            rate_limit_remaining: Some(4999),
        }
    }

    fn status(code: u16) -> GhResponse {
        GhResponse {
            status: code,
            etag: None,
            body: String::new(),
            rate_limit_remaining: Some(4999),
        }
    }

    fn forge(responses: Vec<GhResponse>) -> GithubForge<FakeTransport, StaticTokenProvider> {
        GithubForge::new(GithubClient::new(
            FakeTransport::new(responses),
            StaticTokenProvider("t".into()),
        ))
    }

    /// Like [`forge`], plus the log of requested paths (shared with the transport).
    #[allow(clippy::type_complexity)]
    fn forge_logging_paths(
        responses: Vec<GhResponse>,
    ) -> (
        GithubForge<FakeTransport, StaticTokenProvider>,
        Arc<Mutex<Vec<String>>>,
    ) {
        let transport = FakeTransport::new(responses);
        let paths = Arc::clone(&transport.paths);
        (
            GithubForge::new(GithubClient::new(
                transport,
                StaticTokenProvider("t".into()),
            )),
            paths,
        )
    }

    fn repo() -> RepoRef {
        RepoRef {
            id: "repo:acme/widget".into(),
            full_name: "acme/widget".into(),
        }
    }

    // base64 of "hello" (no padding stripping needed; the API wraps with newlines, decode
    // tolerates it).
    const HELLO_B64: &str = "aGVsbG8=";

    #[tokio::test]
    async fn repo_meta_reports_archived_state() {
        let f = forge(vec![ok(
            r#"{"pushed_at":"2026-01-02T00:00:00Z","fork":false,"archived":true}"#,
        )]);
        let meta = f.repo_meta(&repo()).await.unwrap();
        assert!(meta.archived);
    }

    #[tokio::test]
    async fn commits_map_fields_and_advance_cursor() {
        let body = r#"[{"sha":"abc","commit":{"message":"init","author":{"date":"2026-01-02T00:00:00Z"}},"author":{"login":"octocat"}}]"#;
        let f = forge(vec![ok(body)]);
        let delta = f.commits(&repo(), None).await.unwrap();
        assert_eq!(delta.items.len(), 1);
        let c = &delta.items[0];
        assert_eq!(c.sha, "abc");
        assert_eq!(c.repo_id, "repo:acme/widget");
        assert_eq!(c.author_login.as_deref(), Some("octocat"));
        assert_eq!(c.message, "init");
        assert_eq!(delta.cursor.as_deref(), Some("2026-01-02T00:00:00Z"));
        assert!(!delta.unchanged);
    }

    #[tokio::test]
    async fn commits_empty_keeps_cursor_and_is_unchanged() {
        let f = forge(vec![ok("[]")]);
        let delta = f
            .commits(&repo(), Some("2026-01-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert!(delta.items.is_empty());
        assert!(delta.unchanged);
        assert_eq!(delta.cursor.as_deref(), Some("2026-01-01T00:00:00Z"));
    }

    /// A full page of commits.
    fn commit_page() -> String {
        let items: Vec<String> = (0..PER_PAGE)
            .map(|i| {
                format!(
                    r#"{{"sha":"s{i}","commit":{{"message":"m","author":{{"date":"2026-02-{:02}T00:00:00Z"}}}},"author":{{"login":"o"}}}}"#,
                    (i % 28) + 1
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    #[tokio::test]
    async fn commits_capped_advances_cursor_and_hands_the_tail_to_backfill() {
        let resps: Vec<GhResponse> = (0..MAX_PAGES).map(|_| ok(&commit_page())).collect();
        let f = forge(resps);
        let delta = f
            .commits(&repo(), Some("2026-01-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), MAX_PAGES * PER_PAGE);
        assert_eq!(delta.cursor.as_deref(), Some("2026-02-28T00:00:00Z"));
        // The token names the oldest commit this pass saw: page 1 of everything at or before it.
        assert_eq!(
            delta.backfill.as_deref(),
            Some("bf:c:1:2026-02-01T00:00:00Z")
        );
    }

    #[tokio::test]
    async fn commits_uncapped_leaves_no_backfill_token() {
        // A short page means the whole window fitted: nothing left behind, and a walk in progress
        // elsewhere must not be disturbed by a `Some` here.
        let body = r#"[{"sha":"abc","commit":{"message":"init","author":{"date":"2026-01-02T00:00:00Z"}},"author":null}]"#;
        let delta = forge(vec![ok(body)]).commits(&repo(), None).await.unwrap();
        assert_eq!(delta.backfill, None);
    }

    #[tokio::test]
    async fn commits_backfill_walks_the_until_window_and_resumes_deeper() {
        // Every page full: the chunk's budget ran out before history did, so the walk resumes at
        // the page after the last one fetched, in the same `until` window.
        let resps: Vec<GhResponse> = (0..MAX_BACKFILL_PAGES)
            .map(|_| ok(&commit_page()))
            .collect();
        let (f, paths) = forge_logging_paths(resps);
        let delta = f
            .commits(&repo(), Some("bf:c:1:2026-02-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), MAX_BACKFILL_PAGES * PER_PAGE);
        // A chunk is cheaper than a forward fetch, so the backfill lane cannot crowd it out.
        const { assert!(MAX_BACKFILL_PAGES < MAX_PAGES) };
        assert_eq!(
            delta.backfill.as_deref(),
            Some("bf:c:6:2026-02-01T00:00:00Z")
        );
        // It never touches the forward watermark.
        assert_eq!(delta.cursor, None);
        let paths = paths.lock().unwrap().clone();
        assert_eq!(paths.len(), MAX_BACKFILL_PAGES);
        assert!(paths[0].contains("until=2026-02-01T00:00:00Z"));
        assert!(paths[0].ends_with("page=1"));
        assert!(paths[4].ends_with("page=5"));
        // No `since=` on a backfill request: it walks down, it does not look forward.
        assert!(paths.iter().all(|p| !p.contains("since=")));
    }

    #[tokio::test]
    async fn commits_backfill_resumes_from_the_token_it_was_given() {
        // The token, not a fresh walk: a chunk that starts at page 6 requests page 6.
        let (f, paths) = forge_logging_paths(vec![ok("[]")]);
        let delta = f
            .commits(&repo(), Some("bf:c:6:2026-02-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert!(paths.lock().unwrap()[0].ends_with("page=6"));
        assert_eq!(delta.backfill, None); // a short page ends the walk
    }

    #[tokio::test]
    async fn commits_backfill_short_page_completes_the_walk() {
        let body = r#"[{"sha":"old","commit":{"message":"first","author":{"date":"2019-01-01T00:00:00Z"}},"author":null}]"#;
        let delta = forge(vec![ok(body)])
            .commits(&repo(), Some("bf:c:1:2020-01-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].sha, "old");
        // `None` means nothing older; the consumer records the walk as complete.
        assert_eq!(delta.backfill, None);
    }

    #[tokio::test]
    async fn commits_backfill_skips_the_pass_when_the_rate_budget_is_low() {
        // One forward call teaches the client the budget is nearly gone; the backfill chunk must
        // spend nothing and return the same token. Only one response is queued, so any request
        // would fail with "no more responses".
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
        // Every PR newer than the cursor and every page full: a capped pass with an older tail.
        fn page() -> String {
            let items: Vec<String> = (0..PER_PAGE)
                .map(|i| {
                    format!(
                        r#"{{"id":{i},"number":{i},"title":"t","state":"open","user":{{"login":"a"}},"created_at":"2026-03-01T00:00:00Z","merged_at":null,"updated_at":"2026-03-{:02}T00:00:00Z"}}"#,
                        (i % 28) + 1
                    )
                })
                .collect();
            format!("[{}]", items.join(","))
        }
        let resps: Vec<GhResponse> = (0..MAX_PAGES).map(|_| ok(&page())).collect();
        let f = forge(resps);
        let delta = f
            .pull_requests(&repo(), Some("2026-01-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), MAX_PAGES * PER_PAGE);
        assert_eq!(delta.cursor.as_deref(), Some("2026-03-28T00:00:00Z"));
        assert_eq!(delta.backfill.as_deref(), Some("bf:p:1"));
    }

    #[tokio::test]
    async fn pulls_backfill_pages_the_created_ascending_listing() {
        // The created-ascending listing is a stable prefix, so a page number is a valid resume
        // point. `changed` is empty: reviews are not fetched for backfill.
        fn page(base: usize) -> String {
            let items: Vec<String> = (0..PER_PAGE)
                .map(|i| {
                    let n = base + i;
                    format!(
                        r#"{{"id":{n},"number":{n},"title":"t","state":"closed","user":null,"created_at":"2019-01-01T00:00:00Z","merged_at":null,"updated_at":"2019-01-02T00:00:00Z"}}"#
                    )
                })
                .collect();
            format!("[{}]", items.join(","))
        }
        let resps: Vec<GhResponse> = (0..MAX_BACKFILL_PAGES)
            .map(|p| ok(&page(p * PER_PAGE)))
            .collect();
        let (f, paths) = forge_logging_paths(resps);
        let delta = f
            .pull_requests(&repo(), Some("bf:p:1".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), MAX_BACKFILL_PAGES * PER_PAGE);
        assert_eq!(delta.backfill.as_deref(), Some("bf:p:6"));
        assert_eq!(delta.cursor, None);
        assert!(delta.changed.is_empty());
        let paths = paths.lock().unwrap().clone();
        assert!(paths[0].contains("sort=created&direction=asc"));
        assert!(paths[0].ends_with("page=1"));
        assert!(paths[4].ends_with("page=5"));
    }

    #[tokio::test]
    async fn pulls_backfill_short_page_completes_the_walk() {
        let body = r#"[{"id":1,"number":1,"title":"first ever","state":"closed","user":null,"created_at":"2015-01-01T00:00:00Z","merged_at":null,"updated_at":"2015-01-02T00:00:00Z"}]"#;
        let delta = forge(vec![ok(body)])
            .pull_requests(&repo(), Some("bf:p:3".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].title, "first ever");
        assert_eq!(delta.backfill, None);
    }

    #[tokio::test]
    async fn pulls_stop_at_cursor_and_report_changed() {
        // Sorted desc: the first PR is newer than the cursor (keep), the second is at the cursor
        // (stop).
        let body = r#"[
            {"id":11,"number":7,"title":"new","state":"open","user":{"login":"a"},"created_at":"2026-01-03T00:00:00Z","merged_at":null,"updated_at":"2026-01-03T00:00:00Z"},
            {"id":10,"number":6,"title":"old","state":"closed","user":{"login":"b"},"created_at":"2026-01-01T00:00:00Z","merged_at":null,"updated_at":"2026-01-01T00:00:00Z"}
        ]"#;
        let f = forge(vec![ok(body)]);
        let delta = f
            .pull_requests(&repo(), Some("2026-01-01T00:00:00Z".into()))
            .await
            .unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].id, "11");
        assert_eq!(delta.items[0].number, 7);
        assert_eq!(delta.changed, vec![("11".to_string(), 7)]);
        assert_eq!(delta.cursor.as_deref(), Some("2026-01-03T00:00:00Z"));
        assert!(!delta.unchanged);
    }

    #[tokio::test]
    async fn pr_closing_issues_parses_graphql_nodes() {
        let body = r#"{"data":{"repository":{"pullRequest":{"closingIssuesReferences":{"nodes":[{"number":7},{"number":9}]}}}}}"#;
        let f = forge(vec![ok(body)]);
        let nums = f.pr_closing_issues(&repo(), 42).await.unwrap();
        assert_eq!(nums, vec![7, 9]);
    }

    #[tokio::test]
    async fn pr_closing_issues_empty_when_none() {
        let body =
            r#"{"data":{"repository":{"pullRequest":{"closingIssuesReferences":{"nodes":[]}}}}}"#;
        let f = forge(vec![ok(body)]);
        assert!(f.pr_closing_issues(&repo(), 42).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn issues_skip_prs_but_advance_cursor_across_them() {
        let body = r#"[
            {"id":1,"number":1,"title":"a bug","state":"open","user":{"login":"a"},"created_at":"2026-01-02T00:00:00Z","closed_at":null,"updated_at":"2026-01-02T00:00:00Z","labels":[{"name":"bug"},{"name":"P1"}],"html_url":"https://github.com/acme/widget/issues/1"},
            {"id":2,"number":2,"title":"a pr","state":"open","user":{"login":"b"},"created_at":"2026-01-03T00:00:00Z","closed_at":null,"updated_at":"2026-01-03T00:00:00Z","pull_request":{"url":"x"}}
        ]"#;
        let f = forge(vec![ok(body)]);
        let delta = f.issues(&repo(), None).await.unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].id, "1");
        // Labels are newline-joined; the issue carries its web URL.
        assert_eq!(delta.items[0].labels, "bug\nP1");
        assert_eq!(
            delta.items[0].html_url.as_deref(),
            Some("https://github.com/acme/widget/issues/1")
        );
        // The cursor advances to the PR's updated_at even though the PR itself is skipped.
        assert_eq!(delta.cursor.as_deref(), Some("2026-01-03T00:00:00Z"));
    }

    #[tokio::test]
    async fn releases_304_is_unchanged() {
        let f = forge(vec![status(304)]);
        let delta = f.releases(&repo(), Some("v1".into())).await.unwrap();
        assert!(delta.unchanged);
        assert!(delta.items.is_empty());
        assert!(delta.cursor.is_none());
    }

    #[tokio::test]
    async fn releases_200_maps_and_returns_new_etag() {
        let body =
            r#"[{"id":5,"tag_name":"v1.0","name":"One","published_at":"2026-01-01T00:00:00Z"}]"#;
        let f = forge(vec![ok_etag(body, "v2")]);
        let delta = f.releases(&repo(), None).await.unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].tag, "v1.0");
        assert_eq!(delta.cursor.as_deref(), Some("v2"));
        assert!(!delta.unchanged);
    }

    #[tokio::test]
    async fn ci_runs_map_head_sha_to_commit() {
        let body = r#"{"workflow_runs":[{"id":9,"head_sha":"deadbeef","status":"completed","conclusion":"failure","updated_at":"2026-01-04T00:00:00Z"}]}"#;
        let f = forge(vec![ok_etag(body, "c1")]);
        let delta = f.ci_runs(&repo(), None).await.unwrap();
        assert_eq!(delta.items.len(), 1);
        assert_eq!(delta.items[0].commit_sha.as_deref(), Some("deadbeef"));
        assert_eq!(delta.items[0].conclusion.as_deref(), Some("failure"));
        // No run_attempt in this payload -> None (never counted flaky).
        assert_eq!(delta.items[0].run_attempt, None);
    }

    #[tokio::test]
    async fn ci_runs_carry_run_attempt() {
        // A re-run-then-green run: attempt 2, success (flaky signal).
        let body = r#"{"workflow_runs":[{"id":9,"head_sha":"deadbeef","status":"completed","conclusion":"success","updated_at":"2026-01-04T00:00:00Z","run_attempt":2}]}"#;
        let f = forge(vec![ok_etag(body, "c1")]);
        let delta = f.ci_runs(&repo(), None).await.unwrap();
        assert_eq!(delta.items[0].run_attempt, Some(2));
    }

    #[tokio::test]
    async fn reviews_map_and_stamp_pr_id() {
        let body = r#"[{"id":3,"user":{"login":"rev"},"state":"APPROVED","submitted_at":"2026-01-02T00:00:00Z"}]"#;
        let f = forge(vec![ok(body)]);
        let reviews = f.reviews(&repo(), 7, "11").await.unwrap();
        assert_eq!(reviews.len(), 1);
        assert_eq!(reviews[0].pr_id, "11");
        assert_eq!(reviews[0].reviewer_login.as_deref(), Some("rev"));
    }

    #[tokio::test]
    async fn pr_files_map_and_stamp_pr_id() {
        let body = r#"[{"filename":"src/lib.rs","status":"modified","additions":3,"deletions":1,"patch":"@@"}]"#;
        let f = forge(vec![ok(body)]);
        let files = f.pr_files(&repo(), 7, "11").await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].pr_id, "11");
        assert_eq!(files[0].filename, "src/lib.rs");
        assert_eq!(files[0].additions, 3);
    }

    #[tokio::test]
    async fn file_text_missing_on_404() {
        let f = forge(vec![status(404)]);
        let ft = f.file_text(&repo(), "Cargo.toml", None).await.unwrap();
        assert!(ft.missing);
        assert!(ft.content.is_none());
        assert!(!ft.unchanged);
    }

    #[tokio::test]
    async fn file_text_decodes_base64_and_returns_etag() {
        let body = format!(r#"{{"content":"{HELLO_B64}","encoding":"base64"}}"#);
        let f = forge(vec![ok_etag(&body, "m1")]);
        let ft = f.file_text(&repo(), "Cargo.toml", None).await.unwrap();
        assert_eq!(ft.content.as_deref(), Some("hello"));
        assert_eq!(ft.cursor.as_deref(), Some("m1"));
        assert!(!ft.missing && !ft.unchanged);
    }

    #[tokio::test]
    async fn file_text_304_is_unchanged() {
        let f = forge(vec![status(304)]);
        let ft = f
            .file_text(&repo(), "Cargo.toml", Some("m1".into()))
            .await
            .unwrap();
        assert!(ft.unchanged);
        assert!(!ft.missing);
    }

    #[tokio::test]
    async fn tree_maps_entries() {
        let body = r#"{"tree":[{"path":"src/lib.rs","type":"blob","sha":"s1","size":42},{"path":"src","type":"tree","sha":"s2"}]}"#;
        let f = forge(vec![ok(body)]);
        let tree = f.tree(&repo()).await.unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree[0].path, "src/lib.rs");
        assert_eq!(tree[0].kind, "blob");
        assert_eq!(tree[0].size, Some(42));
    }

    #[tokio::test]
    async fn tree_empty_repo_maps_404_to_empty_repo_error() {
        let f = forge(vec![status(404)]);
        assert_eq!(f.tree(&repo()).await, Err(ForgeError::EmptyRepo));
    }

    #[tokio::test]
    async fn blob_text_decodes_present_blob() {
        let body = format!(r#"{{"content":"{HELLO_B64}","encoding":"base64"}}"#);
        let f = forge(vec![ok(&body)]);
        assert_eq!(
            f.blob_text(&repo(), "s1").await.unwrap(),
            Some("hello".to_string())
        );
    }

    #[tokio::test]
    async fn blob_text_none_for_non_base64() {
        let body = r#"{"content":"hi","encoding":"utf-8"}"#;
        let f = forge(vec![ok(body)]);
        assert_eq!(f.blob_text(&repo(), "s1").await.unwrap(), None);
    }

    #[tokio::test]
    async fn blob_text_err_on_transient_fetch_failure() {
        let f = forge(vec![status(500)]);
        assert!(f.blob_text(&repo(), "s1").await.is_err());
    }

    // Discovery calls, in order: the activity feed, GET /user (picks the listing), owned repos.
    #[tokio::test]
    async fn person_repos_unions_activity_and_owned_activity_first() {
        // The feed is newest-first, so deduping in order puts active repos ahead of merely owned
        // ones; a repo in both halves is listed once, at its activity position.
        let events = r#"[{"repo":{"name":"b/y"}},{"repo":{"name":"a/x"}},{"repo":{"name":"b/y"}}]"#;
        let owned = r#"[{"full_name":"a/x"},{"full_name":"c/z"}]"#;
        let f = forge(vec![
            ok(events),
            ok(r#"{"login":"someone-else"}"#),
            ok(owned),
        ]);
        assert_eq!(
            f.person_repos("octocat").await.unwrap(),
            vec!["b/y".to_string(), "a/x".to_string(), "c/z".to_string()]
        );
    }

    #[tokio::test]
    async fn person_repos_asks_the_authenticated_listing_for_the_token_owner() {
        // The token's own account gets /user/repos, which includes private repos.
        let (f, paths) = forge_logging_paths(vec![
            ok("[]"),
            ok(r#"{"login":"OctoCat"}"#),
            ok(r#"[{"full_name":"octocat/secret"}]"#),
        ]);
        assert_eq!(
            f.person_repos("octocat").await.unwrap(),
            vec!["octocat/secret".to_string()]
        );
        let paths = paths.lock().unwrap().clone();
        assert!(
            paths.iter().any(|p| p.starts_with("/user/repos?")),
            "expected the authenticated listing, got {paths:?}"
        );
    }

    #[tokio::test]
    async fn person_repos_asks_the_token_owner_for_owned_repos_only() {
        // Must not ask for `affiliation=owner,collaborator,organization_member`, which swept every
        // repo of every org the person belongs to onto their board.
        let (f, paths) = forge_logging_paths(vec![
            ok("[]"),
            ok(r#"{"login":"octocat"}"#),
            ok(r#"[{"full_name":"octocat/secret"}]"#),
        ]);
        f.person_repos("octocat").await.unwrap();
        let paths = paths.lock().unwrap().clone();
        let owned = paths
            .iter()
            .find(|p| p.starts_with("/user/repos?"))
            .unwrap_or_else(|| panic!("expected the authenticated listing, got {paths:?}"));
        assert!(
            owned.contains("affiliation=owner"),
            "the owned half must ask for ownership, got {owned}"
        );
        assert!(
            !owned.contains("organization_member"),
            "org membership is a blanket grant nobody chose per repo, got {owned}"
        );
        assert!(
            !owned.contains("collaborator"),
            "a repo someone else owns is not owned; the events feed covers work in it, got {owned}"
        );
    }

    #[tokio::test]
    async fn person_repos_asks_the_public_listing_for_anyone_else() {
        let (f, paths) = forge_logging_paths(vec![
            ok("[]"),
            ok(r#"{"login":"octocat"}"#),
            ok(r#"[{"full_name":"acme/api"}]"#),
        ]);
        assert_eq!(
            f.person_repos("alice").await.unwrap(),
            vec!["acme/api".to_string()]
        );
        let paths = paths.lock().unwrap().clone();
        assert!(
            paths.iter().any(|p| p.starts_with("/users/alice/repos?")),
            "expected the public listing, got {paths:?}"
        );
        assert!(
            !paths.iter().any(|p| p.starts_with("/user/repos?")),
            "the authenticated listing is only for the token owner, got {paths:?}"
        );
        // Owner-only on this branch too (GitHub's default, asked for explicitly).
        let public = paths
            .iter()
            .find(|p| p.starts_with("/users/alice/repos?"))
            .expect("checked above");
        assert!(
            public.contains("type=owner"),
            "the public listing must ask for owned repos, got {public}"
        );
    }

    #[tokio::test]
    async fn person_repos_keeps_activity_when_the_owned_listing_fails() {
        // Best-effort per half: a 500 on the owned listing still yields the touched repos, in feed
        // order.
        let f = forge(vec![
            ok(r#"[{"repo":{"name":"b/y"}},{"repo":{"name":"a/x"}}]"#),
            ok(r#"{"login":"someone-else"}"#),
            status(500),
        ]);
        assert_eq!(
            f.person_repos("alice").await.unwrap(),
            vec!["b/y".to_string(), "a/x".to_string()]
        );
    }

    #[tokio::test]
    async fn person_repos_keeps_owned_when_the_activity_feed_fails() {
        let f = forge(vec![
            status(500),
            ok(r#"{"login":"someone-else"}"#),
            ok(r#"[{"full_name":"acme/api"}]"#),
        ]);
        assert_eq!(
            f.person_repos("alice").await.unwrap(),
            vec!["acme/api".to_string()]
        );
    }

    #[tokio::test]
    async fn org_repos_preserve_forge_order_and_dedupe() {
        // The endpoint is requested pushed-desc; preserve that order and dedupe, no sorting.
        let body = r#"[{"full_name":"acme/b"},{"full_name":"acme/a"},{"full_name":"acme/b"}]"#;
        let f = forge(vec![ok(body)]);
        assert_eq!(
            f.org_repos("acme").await.unwrap(),
            vec!["acme/b".to_string(), "acme/a".to_string()]
        );
    }

    #[tokio::test]
    async fn search_users_maps_items_to_logins() {
        let body = r#"{"total_count":2,"items":[{"login":"octocat"},{"login":"octodog"}]}"#;
        let f = forge(vec![ok(body)]);
        assert_eq!(
            f.search_users("oct").await.unwrap(),
            vec!["octocat".to_string(), "octodog".to_string()]
        );
    }

    #[tokio::test]
    async fn search_orgs_maps_items_to_logins() {
        let body = r#"{"total_count":1,"items":[{"login":"acme-corp"}]}"#;
        let f = forge(vec![ok(body)]);
        assert_eq!(
            f.search_orgs("acme").await.unwrap(),
            vec!["acme-corp".to_string()]
        );
    }

    #[tokio::test]
    async fn search_repos_maps_items_to_full_names() {
        let body = r#"{"total_count":2,"items":[{"full_name":"acme/widget"},{"full_name":"acme/gadget"}]}"#;
        let f = forge(vec![ok(body)]);
        assert_eq!(
            f.search_repos("acme/").await.unwrap(),
            vec!["acme/widget".to_string(), "acme/gadget".to_string()]
        );
    }

    #[tokio::test]
    async fn search_empty_items_is_empty() {
        let f = forge(vec![ok(r#"{"total_count":0,"items":[]}"#)]);
        assert!(f.search_users("nobody-xyz").await.unwrap().is_empty());
    }

    #[test]
    fn encode_query_escapes_non_unreserved() {
        assert_eq!(encode_query("acme/wid get"), "acme%2Fwid%20get");
        assert_eq!(encode_query("oct-o.cat_1~"), "oct-o.cat_1~");
    }

    #[tokio::test]
    async fn capabilities_are_all() {
        assert_eq!(forge(vec![]).capabilities(), Capabilities::ALL);
    }
}
