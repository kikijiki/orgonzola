//! A programmable in-memory [`Forge`] for tests. Program what each entity returns (items, an
//! opaque cursor, the `unchanged` flag), then assert on what the consumer does with it. Lookups
//! are keyed by `RepoRef::id` (plus pr number / path / sha). Unprogrammed entities return an
//! empty delta (empty tree, transient-miss blob, missing file). Every call is recorded (entity
//! label and cursor passed in). Build with `FakeForge::new()` and the `with_*` setters; the
//! convenience setters mark a delta `unchanged` when it has no items and no cursor.

use crate::{
    Capabilities, CiRun, Commit, Delta, FileText, Forge, ForgeError, Issue, PrFile, PullDelta,
    PullRequest, Release, RepoMeta, RepoRef, Review, TreeEntry,
};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

/// A programmed blob outcome: decoded text, present-but-undecodable (a deterministic skip), or a
/// transient failure. Mirrors [`Forge::blob_text`].
#[derive(Debug, Clone)]
enum BlobResult {
    Text(String),
    Undecodable,
    Transient,
}

/// A recorded forge call: the entity label plus the cursor it was called with.
pub type RecordedCall = (String, Option<String>);

#[derive(Debug, Clone)]
pub struct FakeForge {
    capabilities: Capabilities,
    commits: HashMap<String, Delta<Commit>>,
    pulls: HashMap<String, PullDelta>,
    /// (repo_id, backfill token) -> what a backfill pass with that token returns. Looked up
    /// before the forward delta.
    commit_backfills: HashMap<(String, String), Delta<Commit>>,
    pull_backfills: HashMap<(String, String), PullDelta>,
    issues: HashMap<String, Delta<Issue>>,
    releases: HashMap<String, Delta<Release>>,
    ci_runs: HashMap<String, Delta<CiRun>>,
    reviews: HashMap<(String, i64), Vec<Review>>,
    /// (repo_id, pr_number) -> the issue numbers the PR authoritatively closes.
    closing_issues: HashMap<(String, i64), Vec<i64>>,
    pr_files: HashMap<(String, i64), Vec<PrFile>>,
    files: HashMap<(String, String), FileText>,
    /// `Ok(entries)` or `Err(())` for an empty repo (-> [`ForgeError::EmptyRepo`]).
    trees: HashMap<String, Result<Vec<TreeEntry>, ()>>,
    blobs: HashMap<(String, String), BlobResult>,
    pushed_at: HashMap<String, Option<String>>,
    /// repo_id -> upstream `owner/name`: a programmed fork relationship.
    forks: HashMap<String, String>,
    archived: std::collections::HashSet<String>,
    /// `Ok(repos)` or `Err(())` to simulate a failed person discovery (-> a backend error).
    person_repos: HashMap<String, Result<Vec<String>, ()>>,
    org_repos: HashMap<String, Vec<String>>,
    calls: Arc<Mutex<Vec<RecordedCall>>>,
}

impl Default for FakeForge {
    fn default() -> Self {
        Self {
            capabilities: Capabilities::ALL,
            commits: HashMap::new(),
            pulls: HashMap::new(),
            commit_backfills: HashMap::new(),
            pull_backfills: HashMap::new(),
            issues: HashMap::new(),
            releases: HashMap::new(),
            ci_runs: HashMap::new(),
            reviews: HashMap::new(),
            closing_issues: HashMap::new(),
            pr_files: HashMap::new(),
            files: HashMap::new(),
            trees: HashMap::new(),
            blobs: HashMap::new(),
            pushed_at: HashMap::new(),
            forks: HashMap::new(),
            archived: std::collections::HashSet::new(),
            person_repos: HashMap::new(),
            org_repos: HashMap::new(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

fn cur(cursor: Option<&str>) -> Option<String> {
    cursor.map(str::to_string)
}

impl FakeForge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every call made (entity label + cursor passed in), in order. Shared across clones.
    pub fn calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().unwrap().clone()
    }

    fn record(&self, label: impl Into<String>, cursor: Option<&str>) {
        self.calls.lock().unwrap().push((label.into(), cur(cursor)));
    }

    pub fn with_capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_commits(mut self, repo_id: &str, items: Vec<Commit>, cursor: Option<&str>) -> Self {
        let unchanged = items.is_empty() && cursor.is_none();
        self.commits.insert(
            repo_id.to_string(),
            Delta {
                items,
                cursor: cur(cursor),
                unchanged,
                backfill: None,
            },
        );
        self
    }

    /// Program pull requests; `changed` defaults to the (id, number) of every supplied PR.
    pub fn with_pulls(
        mut self,
        repo_id: &str,
        items: Vec<PullRequest>,
        cursor: Option<&str>,
    ) -> Self {
        let unchanged = items.is_empty() && cursor.is_none();
        let changed = items.iter().map(|p| (p.id.clone(), p.number)).collect();
        self.pulls.insert(
            repo_id.to_string(),
            PullDelta {
                items,
                changed,
                cursor: cur(cursor),
                unchanged,
                backfill: None,
            },
        );
        self
    }

    /// Program a **capped** forward commits pass: the delta from `with_commits` also reports
    /// `backfill = Some(token)`. Pair with [`FakeForge::with_commit_backfill`].
    pub fn with_commits_capped(mut self, repo_id: &str, token: &str) -> Self {
        if let Some(d) = self.commits.get_mut(repo_id) {
            d.backfill = Some(token.to_string());
        }
        self
    }

    /// Program one chunk of the commits backfill walk: `commits` with `token` as cursor returns
    /// `items` and `next` as the resume token (`None` = exhausted).
    pub fn with_commit_backfill(
        mut self,
        repo_id: &str,
        token: &str,
        items: Vec<Commit>,
        next: Option<&str>,
    ) -> Self {
        self.commit_backfills.insert(
            (repo_id.to_string(), token.to_string()),
            Delta {
                items,
                cursor: None,
                unchanged: false,
                backfill: cur(next),
            },
        );
        self
    }

    /// The pull-request twin of [`FakeForge::with_commits_capped`].
    pub fn with_pulls_capped(mut self, repo_id: &str, token: &str) -> Self {
        if let Some(d) = self.pulls.get_mut(repo_id) {
            d.backfill = Some(token.to_string());
        }
        self
    }

    /// The pull-request twin of [`FakeForge::with_commit_backfill`]. `changed` is empty, as on
    /// the real forges.
    pub fn with_pull_backfill(
        mut self,
        repo_id: &str,
        token: &str,
        items: Vec<PullRequest>,
        next: Option<&str>,
    ) -> Self {
        self.pull_backfills.insert(
            (repo_id.to_string(), token.to_string()),
            PullDelta {
                items,
                changed: Vec::new(),
                cursor: None,
                unchanged: false,
                backfill: cur(next),
            },
        );
        self
    }

    /// Program the issue numbers a PR authoritatively closes (the `closingIssuesReferences`
    /// stand-in).
    pub fn with_pr_closing_issues(
        mut self,
        repo_id: &str,
        pr_number: i64,
        numbers: Vec<i64>,
    ) -> Self {
        self.closing_issues
            .insert((repo_id.to_string(), pr_number), numbers);
        self
    }

    pub fn with_issues(mut self, repo_id: &str, items: Vec<Issue>, cursor: Option<&str>) -> Self {
        let unchanged = items.is_empty() && cursor.is_none();
        self.issues.insert(
            repo_id.to_string(),
            Delta {
                items,
                cursor: cur(cursor),
                unchanged,
                backfill: None,
            },
        );
        self
    }

    pub fn with_releases(
        mut self,
        repo_id: &str,
        items: Vec<Release>,
        cursor: Option<&str>,
    ) -> Self {
        let unchanged = items.is_empty() && cursor.is_none();
        self.releases.insert(
            repo_id.to_string(),
            Delta {
                items,
                cursor: cur(cursor),
                unchanged,
                backfill: None,
            },
        );
        self
    }

    pub fn with_ci_runs(mut self, repo_id: &str, items: Vec<CiRun>, cursor: Option<&str>) -> Self {
        let unchanged = items.is_empty() && cursor.is_none();
        self.ci_runs.insert(
            repo_id.to_string(),
            Delta {
                items,
                cursor: cur(cursor),
                unchanged,
                backfill: None,
            },
        );
        self
    }

    pub fn with_reviews(mut self, repo_id: &str, pr_number: i64, reviews: Vec<Review>) -> Self {
        self.reviews
            .insert((repo_id.to_string(), pr_number), reviews);
        self
    }

    pub fn with_pr_files(mut self, repo_id: &str, pr_number: i64, files: Vec<PrFile>) -> Self {
        self.pr_files
            .insert((repo_id.to_string(), pr_number), files);
        self
    }

    /// Program a present file with decoded `content` and a cursor.
    pub fn with_file_text(
        mut self,
        repo_id: &str,
        path: &str,
        content: &str,
        cursor: Option<&str>,
    ) -> Self {
        self.files.insert(
            (repo_id.to_string(), path.to_string()),
            FileText {
                content: Some(content.to_string()),
                cursor: cur(cursor),
                unchanged: false,
                missing: false,
            },
        );
        self
    }

    /// Program a file that reports no change (a 304-equivalent).
    pub fn with_file_unchanged(mut self, repo_id: &str, path: &str) -> Self {
        self.files.insert(
            (repo_id.to_string(), path.to_string()),
            FileText {
                content: None,
                cursor: None,
                unchanged: true,
                missing: false,
            },
        );
        self
    }

    pub fn with_tree(mut self, repo_id: &str, tree: Vec<TreeEntry>) -> Self {
        self.trees.insert(repo_id.to_string(), Ok(tree));
        self
    }

    /// Mark a repo as empty (no default branch) so [`Forge::tree`] returns
    /// [`ForgeError::EmptyRepo`].
    pub fn with_empty_repo(mut self, repo_id: &str) -> Self {
        self.trees.insert(repo_id.to_string(), Err(()));
        self
    }

    pub fn with_blob(mut self, repo_id: &str, sha: &str, text: &str) -> Self {
        self.blobs.insert(
            (repo_id.to_string(), sha.to_string()),
            BlobResult::Text(text.to_string()),
        );
        self
    }

    pub fn with_blob_undecodable(mut self, repo_id: &str, sha: &str) -> Self {
        self.blobs.insert(
            (repo_id.to_string(), sha.to_string()),
            BlobResult::Undecodable,
        );
        self
    }

    pub fn with_blob_transient(mut self, repo_id: &str, sha: &str) -> Self {
        self.blobs.insert(
            (repo_id.to_string(), sha.to_string()),
            BlobResult::Transient,
        );
        self
    }

    pub fn with_pushed_at(mut self, repo_id: &str, pushed_at: &str) -> Self {
        self.pushed_at
            .insert(repo_id.to_string(), Some(pushed_at.to_string()));
        self
    }

    /// Program `repo_id` as a fork of `parent_full_name`, so `repo_meta` reports it.
    pub fn with_fork(mut self, repo_id: &str, parent_full_name: &str) -> Self {
        self.forks
            .insert(repo_id.to_string(), parent_full_name.to_string());
        self
    }

    /// Program `repo_id` as archived, so discovery-policy tests can exercise the metadata gate.
    pub fn with_archived(mut self, repo_id: &str) -> Self {
        self.archived.insert(repo_id.to_string());
        self
    }

    pub fn with_person_repos(mut self, login: &str, repos: Vec<String>) -> Self {
        self.person_repos.insert(login.to_string(), Ok(repos));
        self
    }

    /// Simulate a failed person discovery (a stale login or transient error).
    pub fn with_person_repos_error(mut self, login: &str) -> Self {
        self.person_repos.insert(login.to_string(), Err(()));
        self
    }

    pub fn with_org_repos(mut self, org: &str, repos: Vec<String>) -> Self {
        self.org_repos.insert(org.to_string(), repos);
        self
    }
}

impl Forge for FakeForge {
    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    fn repo_meta(
        &self,
        repo: &RepoRef,
    ) -> impl Future<Output = Result<RepoMeta, ForgeError>> + Send {
        self.record("repo_meta", None);
        let parent = self.forks.get(&repo.id).cloned();
        let meta = RepoMeta {
            pushed_at: self.pushed_at.get(&repo.id).cloned().unwrap_or(None),
            is_fork: parent.is_some(),
            parent_full_name: parent,
            archived: self.archived.contains(&repo.id),
        };
        async move { Ok(meta) }
    }

    fn commits(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Commit>, ForgeError>> + Send {
        self.record("commits", cursor.as_deref());
        // A programmed backfill token wins over the forward delta.
        let delta = cursor
            .as_deref()
            .and_then(|c| self.commit_backfills.get(&(repo.id.clone(), c.to_string())))
            .or_else(|| self.commits.get(&repo.id))
            .cloned()
            .unwrap_or_default();
        async move { Ok(delta) }
    }

    fn pull_requests(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<PullDelta, ForgeError>> + Send {
        self.record("pulls", cursor.as_deref());
        let delta = cursor
            .as_deref()
            .and_then(|c| self.pull_backfills.get(&(repo.id.clone(), c.to_string())))
            .or_else(|| self.pulls.get(&repo.id))
            .cloned()
            .unwrap_or_default();
        async move { Ok(delta) }
    }

    fn reviews(
        &self,
        repo: &RepoRef,
        pr_number: i64,
        _pr_id: &str,
    ) -> impl Future<Output = Result<Vec<Review>, ForgeError>> + Send {
        self.record(format!("reviews:{pr_number}"), None);
        let items = self
            .reviews
            .get(&(repo.id.clone(), pr_number))
            .cloned()
            .unwrap_or_default();
        async move { Ok(items) }
    }

    fn issues(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Issue>, ForgeError>> + Send {
        self.record("issues", cursor.as_deref());
        let delta = self.issues.get(&repo.id).cloned().unwrap_or_default();
        async move { Ok(delta) }
    }

    fn pr_closing_issues(
        &self,
        repo: &RepoRef,
        pr_number: i64,
    ) -> impl Future<Output = Result<Vec<i64>, ForgeError>> + Send {
        self.record("pr_closing_issues", None);
        let numbers = self
            .closing_issues
            .get(&(repo.id.clone(), pr_number))
            .cloned()
            .unwrap_or_default();
        async move { Ok(numbers) }
    }

    fn releases(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Release>, ForgeError>> + Send {
        self.record("releases", cursor.as_deref());
        let delta = self.releases.get(&repo.id).cloned().unwrap_or_default();
        async move { Ok(delta) }
    }

    fn ci_runs(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<CiRun>, ForgeError>> + Send {
        self.record("ci_runs", cursor.as_deref());
        let delta = self.ci_runs.get(&repo.id).cloned().unwrap_or_default();
        async move { Ok(delta) }
    }

    fn pr_files(
        &self,
        repo: &RepoRef,
        pr_number: i64,
        _pr_id: &str,
    ) -> impl Future<Output = Result<Vec<PrFile>, ForgeError>> + Send {
        self.record(format!("pr_files:{pr_number}"), None);
        let items = self
            .pr_files
            .get(&(repo.id.clone(), pr_number))
            .cloned()
            .unwrap_or_default();
        async move { Ok(items) }
    }

    fn file_text(
        &self,
        repo: &RepoRef,
        path: &str,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<FileText, ForgeError>> + Send {
        self.record(format!("file:{path}"), cursor.as_deref());
        // Unprogrammed file: "missing", the normal outcome for an absent manifest.
        let file = self
            .files
            .get(&(repo.id.clone(), path.to_string()))
            .cloned()
            .unwrap_or(FileText {
                content: None,
                cursor: None,
                unchanged: false,
                missing: true,
            });
        async move { Ok(file) }
    }

    fn tree(
        &self,
        repo: &RepoRef,
    ) -> impl Future<Output = Result<Vec<TreeEntry>, ForgeError>> + Send {
        self.record("tree", None);
        let result = self
            .trees
            .get(&repo.id)
            .cloned()
            .unwrap_or_else(|| Ok(Vec::new()));
        async move { result.map_err(|()| ForgeError::EmptyRepo) }
    }

    fn blob_text(
        &self,
        repo: &RepoRef,
        sha: &str,
    ) -> impl Future<Output = Result<Option<String>, ForgeError>> + Send {
        self.record(format!("blob:{sha}"), None);
        let outcome = self
            .blobs
            .get(&(repo.id.clone(), sha.to_string()))
            .cloned()
            .unwrap_or(BlobResult::Transient);
        async move {
            match outcome {
                BlobResult::Text(t) => Ok(Some(t)),
                BlobResult::Undecodable => Ok(None),
                BlobResult::Transient => Err(ForgeError::Backend("transient blob failure".into())),
            }
        }
    }

    fn person_repos(
        &self,
        login: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        self.record(format!("person:{login}"), None);
        let result = self
            .person_repos
            .get(login)
            .cloned()
            .unwrap_or(Ok(Vec::new()));
        async move { result.map_err(|()| ForgeError::Backend(format!("discovery failed: {login}"))) }
    }

    fn org_repos(&self, org: &str) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        self.record(format!("org:{org}"), None);
        let items = self.org_repos.get(org).cloned().unwrap_or_default();
        async move { Ok(items) }
    }

    fn search_users(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        self.record(format!("search_users:{query}"), None);
        async move { Ok(Vec::new()) }
    }

    fn search_orgs(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        self.record(format!("search_orgs:{query}"), None);
        async move { Ok(Vec::new()) }
    }

    fn search_repos(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        self.record(format!("search_repos:{query}"), None);
        async move { Ok(Vec::new()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> RepoRef {
        RepoRef {
            id: "repo:acme/widget".into(),
            full_name: "acme/widget".into(),
        }
    }

    #[tokio::test]
    async fn programmed_data_round_trips_and_records_the_cursor() {
        let commit = Commit {
            sha: "abc".into(),
            repo_id: "repo:acme/widget".into(),
            author_login: Some("octocat".into()),
            message: "init".into(),
            committed_at: "2026-01-01T00:00:00Z".into(),
        };
        let forge =
            FakeForge::new().with_commits("repo:acme/widget", vec![commit.clone()], Some("c2"));
        let delta = forge.commits(&repo(), Some("c1".into())).await.unwrap();
        assert_eq!(delta.items, vec![commit]);
        assert_eq!(delta.cursor.as_deref(), Some("c2"));
        assert!(!delta.unchanged);
        assert_eq!(
            forge.calls(),
            vec![("commits".to_string(), Some("c1".to_string()))]
        );
    }

    #[tokio::test]
    async fn unprogrammed_entities_are_empty_or_missing() {
        let forge = FakeForge::new();
        assert!(forge.commits(&repo(), None).await.unwrap().items.is_empty());
        assert!(forge.tree(&repo()).await.unwrap().is_empty());
        assert!(forge.person_repos("x").await.unwrap().is_empty());
        assert!(
            forge
                .file_text(&repo(), "Cargo.toml", None)
                .await
                .unwrap()
                .missing
        );
        assert!(forge.blob_text(&repo(), "s").await.is_err());
    }

    #[tokio::test]
    async fn empty_repo_and_person_error_map_to_forge_errors() {
        let forge = FakeForge::new()
            .with_empty_repo("repo:acme/widget")
            .with_person_repos_error("alice");
        assert_eq!(forge.tree(&repo()).await, Err(ForgeError::EmptyRepo));
        assert!(forge.person_repos("alice").await.is_err());
    }

    #[tokio::test]
    async fn capabilities_default_to_all() {
        assert_eq!(FakeForge::new().capabilities(), Capabilities::ALL);
    }
}
