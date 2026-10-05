//! Host-agnostic forge abstraction.
//! The [`Forge`] trait returns normalized [`core_store`] domain types for a repo and an opaque
//! cursor. An implementation (e.g. `core-forge-github`) owns endpoints, JSON, pagination and what
//! the cursor means; `core-sync` owns orchestration, cursor persistence, store writes and indexing.
//! The trait is async via `-> impl Future + Send` (like `core_github::Transport`), so it is
//! generic, not `dyn`. `core-sync` persists whatever cursor comes back and hands it back next
//! pass, skipping the write and advance when [`Delta::unchanged`]. [`Delta::backfill`] names older
//! history a capped pass left behind, so it can be walked down later without forge state.

use core_store::{CiRun, Commit, Issue, PrFile, PullRequest, Release, Repo, Review};
use std::future::Future;

pub mod fake;

/// The repo a forge call targets. `id` is the store key ("repo:owner/name"); `full_name` is
/// "owner/name".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoRef {
    pub id: String,
    pub full_name: String,
}

impl From<&Repo> for RepoRef {
    fn from(repo: &Repo) -> Self {
        Self {
            id: repo.id.clone(),
            full_name: repo.full_name.clone(),
        }
    }
}

/// A repo's metadata from the forge (one `GET /repos/{full}`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RepoMeta {
    /// Last push time (RFC-3339), or `None` if the forge cannot report it.
    pub pushed_at: Option<String>,
    /// Whether the forge reports this repo as a fork.
    pub is_fork: bool,
    /// The upstream parent's `owner/name`, when this is a fork.
    pub parent_full_name: Option<String>,
    /// Whether the forge marks the repository read-only/archived.
    pub archived: bool,
}

/// A page-collapsed delta for one entity: the new/changed `items` since the supplied cursor, the
/// `cursor` to persist, and `unchanged` (e.g. HTTP 304: skip the write and the advance).
/// `backfill` carries older history the pass left behind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delta<T> {
    pub items: Vec<T>,
    pub cursor: Option<String>,
    pub unchanged: bool,
    /// Opaque resume token for older history this pass did not fetch.
    /// The consumer persists it verbatim, separately from `cursor`, and passes it later as the
    /// `cursor` of the same fetch method to continue the walk.
    /// On a forward fetch, `Some` means the page cap was hit and an older tail went un-fetched;
    /// the consumer must store the token before advancing `cursor` past that tail. `None` says
    /// nothing about a walk already in progress (do not clear one).
    /// On a backfill fetch (token passed as the cursor), `Some` is the next resume point and
    /// `None` means the end of history. A backfill fetch returns `cursor: None` and never moves
    /// the forward watermark.
    pub backfill: Option<String>,
}

impl<T> Default for Delta<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            cursor: None,
            unchanged: false,
            backfill: None,
        }
    }
}

/// Like a [`Delta`] of pull requests, plus the `(pr_id, number)` of the PRs that changed, so the
/// consumer can fetch reviews only for those.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PullDelta {
    pub items: Vec<PullRequest>,
    /// `(pr_id, number)` for the PRs in `items` that are new or updated this pass.
    pub changed: Vec<(String, i64)>,
    pub cursor: Option<String>,
    pub unchanged: bool,
    /// Older history this pass left behind, with the same contract as [`Delta::backfill`].
    /// A backfill pass returns an empty `changed`: reviews are not fetched for backfilled PRs.
    pub backfill: Option<String>,
}

/// The text of a single repo file (a dependency manifest), with the same cursor and `unchanged`
/// contract as [`Delta`]. `missing` distinguishes an absent file from a present-but-empty one.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FileText {
    pub content: Option<String>,
    pub cursor: Option<String>,
    pub unchanged: bool,
    pub missing: bool,
}

/// One entry in a repo's source tree: path, content hash (`sha`, the forge's blob id), `kind`
/// ("blob" / "tree" / ...), and size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeEntry {
    pub path: String,
    pub sha: String,
    pub kind: String,
    pub size: Option<i64>,
}

/// What a forge supports, so the consumer degrades on a backend that lacks a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// Discover the repos a person recently touched (GitHub's events feed).
    pub person_discovery: bool,
    /// Enumerate an organization's repos.
    pub org_discovery: bool,
    /// Search users / orgs / repos by query string, for the source pickers.
    pub search: bool,
    /// Continuous-integration runs.
    pub ci: bool,
    /// Source-tree access (for the code index).
    pub code: bool,
    /// Authoritative PR -> closing-issue references (GitHub's GraphQL `closingIssuesReferences`).
    /// When false, the linker relies on body-parsed `#N` references only.
    pub closing_issue_refs: bool,
}

impl Capabilities {
    /// Every capability on, as a full-featured forge (GitHub) reports.
    pub const ALL: Self = Self {
        person_discovery: true,
        org_discovery: true,
        search: true,
        ci: true,
        code: true,
        closing_issue_refs: true,
    };
}

/// A forge error, reduced to what `core-sync` branches on. `NotFound` and `EmptyRepo` are
/// tolerated cases (nothing to sync); `Auth` and `Backend` are real errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ForgeError {
    /// The resource does not exist (e.g. a manifest the repo does not have). Often non-fatal.
    #[error("not found")]
    NotFound,
    /// The repo is empty or has no default branch.
    #[error("empty repository")]
    EmptyRepo,
    /// Authentication failed or no token was available.
    #[error("authentication failed")]
    Auth,
    /// The forge rate limit is exhausted; try again later.
    #[error("rate limited")]
    RateLimited,
    /// Any other failure (transport, unexpected status, parse error), with a message for the log.
    #[error("forge backend error: {0}")]
    Backend(String),
}

impl ForgeError {
    /// A plain-language, actionable message for a non-technical user, unlike the terse `Display`
    /// used in logs.
    pub fn user_message(&self) -> String {
        match self {
            ForgeError::Auth => {
                // The UI keys its "Open Settings" shortcut on the word "Reconnect" (locked by the
                // user_message test); keep it if you reword this.
                "Your GitHub connection has expired or was revoked. Reconnect it under Settings > Connections."
                    .into()
            }
            ForgeError::RateLimited => {
                "GitHub's rate limit was reached. orgonzola will catch up shortly - try again in a few minutes."
                    .into()
            }
            ForgeError::NotFound => {
                "Something was not found on the forge - a repository or organization may be private, renamed, or removed."
                    .into()
            }
            ForgeError::EmptyRepo => "The repository is empty - there is nothing to sync yet.".into(),
            // Network errors and unexpected statuses; the detail is kept for the log.
            ForgeError::Backend(detail) => {
                format!("Could not reach the forge. Check your network connection and try again. ({detail})")
            }
        }
    }
}

/// A code-hosting service the sync engine reads from. Each method returns normalized
/// [`core_store`] values; the implementation owns endpoints, JSON, pagination and the cursor.
/// Methods are `-> impl Future + Send` (not `async fn`) to keep the `Send` bound for tokio tasks,
/// so the trait is generic, not `dyn`.
pub trait Forge {
    /// What this forge supports (cheap, synchronous).
    fn capabilities(&self) -> Capabilities;

    /// The repo's metadata: last-push timestamp (the code-index gate: skip the tree/blob fetch
    /// when unchanged since the stored `code` cursor), fork relationship and archived state.
    fn repo_meta(
        &self,
        repo: &RepoRef,
    ) -> impl Future<Output = Result<RepoMeta, ForgeError>> + Send;

    /// Commits new since `cursor`, or, when `cursor` is a [`Delta::backfill`] token, the next
    /// chunk of older history.
    fn commits(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Commit>, ForgeError>> + Send;

    /// Pull requests changed since `cursor`, plus which ones changed (for review fetching). A
    /// [`Delta::backfill`] token as `cursor` fetches the next chunk of older pull requests.
    fn pull_requests(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<PullDelta, ForgeError>> + Send;

    /// Reviews on one pull request. `pr_id` stamps the returned [`Review::pr_id`]; `pr_number`
    /// identifies the PR to the forge.
    fn reviews(
        &self,
        repo: &RepoRef,
        pr_number: i64,
        pr_id: &str,
    ) -> impl Future<Output = Result<Vec<Review>, ForgeError>> + Send;

    /// Issues changed since `cursor` (excluding pull requests, which some forges return here).
    fn issues(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Issue>, ForgeError>> + Send;

    /// The issue numbers a pull request closes, per the forge's own link data (GitHub's
    /// `closingIssuesReferences`). Default: empty, so the linker uses body-parsed `#N` (see
    /// `Capabilities::closing_issue_refs`).
    fn pr_closing_issues(
        &self,
        repo: &RepoRef,
        pr_number: i64,
    ) -> impl Future<Output = Result<Vec<i64>, ForgeError>> + Send {
        let _ = (repo, pr_number);
        async { Ok(Vec::new()) }
    }

    /// Releases, conditional on `cursor`.
    fn releases(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Release>, ForgeError>> + Send;

    /// CI runs, conditional on `cursor`.
    fn ci_runs(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<CiRun>, ForgeError>> + Send;

    /// The changed files (diff) of one pull request. `pr_id` stamps the returned
    /// [`PrFile::pr_id`]; `pr_number` identifies the PR to the forge.
    fn pr_files(
        &self,
        repo: &RepoRef,
        pr_number: i64,
        pr_id: &str,
    ) -> impl Future<Output = Result<Vec<PrFile>, ForgeError>> + Send;

    /// One repo file's text (a dependency manifest), conditional on `cursor`. A missing file is
    /// reported via [`FileText::missing`], not an error.
    fn file_text(
        &self,
        repo: &RepoRef,
        path: &str,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<FileText, ForgeError>> + Send;

    /// The repo's full source tree (for the code index).
    fn tree(
        &self,
        repo: &RepoRef,
    ) -> impl Future<Output = Result<Vec<TreeEntry>, ForgeError>> + Send;

    /// One blob's text, by its tree `sha`. `Ok(None)` means the blob is present but not text or
    /// undecodable, so the consumer should skip it; `Err` is a transient failure worth retrying.
    fn blob_text(
        &self,
        repo: &RepoRef,
        sha: &str,
    ) -> impl Future<Output = Result<Option<String>, ForgeError>> + Send;

    /// The "owner/name" of repos a person recently touched. Empty when the forge has no such
    /// feed (see [`Capabilities::person_discovery`]).
    fn person_repos(
        &self,
        login: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send;

    /// The "owner/name" of an organization's repos. Empty when unsupported.
    fn org_repos(&self, org: &str) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send;

    /// User logins matching `query`, for the source pickers. Empty when the forge has no search
    /// (see [`Capabilities::search`]).
    fn search_users(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send;

    /// Organization logins matching `query`. Empty when unsupported.
    fn search_orgs(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send;

    /// The "owner/name" of repos matching `query`. Empty when unsupported.
    fn search_repos(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_message_is_plain_and_actionable() {
        // Auth points at reconnecting; rate-limit at waiting; neither leaks a status code.
        assert!(ForgeError::Auth.user_message().contains("Reconnect"));
        assert!(ForgeError::RateLimited
            .user_message()
            .contains("rate limit"));
        assert!(!ForgeError::Auth.user_message().contains("401"));
        // A backend error keeps the detail but leads with plain guidance.
        let m = ForgeError::Backend("transport timeout".into()).user_message();
        assert!(m.starts_with("Could not reach the forge"));
        assert!(m.contains("transport timeout"));
    }
}
