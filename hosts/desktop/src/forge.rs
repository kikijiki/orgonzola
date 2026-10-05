//! `AnyForge`: the desktop host's forge type. `SyncEngine<F>` is generic over one forge type, so
//! this enum is the single `F` that delegates every [`core_forge::Forge`] call to a GitHub or
//! Gitea variant. A static enum (no `dyn`) keeps the engine monomorphized and the futures `Send`.
//! It lives in the host because it names both `GithubForge` and `GiteaForge`, which avoids the
//! core crates depending on each other.

// Methods keep the trait's `-> impl Future + Send` shape (a single delegating `async move` block)
// to match the `Forge` impls in the forge crates.
#![allow(clippy::manual_async_fn)]

use std::future::Future;

use core_forge::{
    Capabilities, Delta, FileText, Forge, ForgeError, PullDelta, RepoMeta, RepoRef, TreeEntry,
};
use core_forge_gitea::GiteaForge;
use core_forge_github::GithubForge;
use core_github::{GithubClient, HttpTransport, StaticTokenProvider};
use core_store::{CiRun, Commit, Issue, PrFile, Release, Review};

/// The transport and auth a host forge is built over: reqwest with a stored token.
type Client = GithubClient<HttpTransport, StaticTokenProvider>;

/// One configured forge: a GitHub or a Gitea/Forgejo backend over the shared HTTP client.
pub enum AnyForge {
    Github(GithubForge<HttpTransport, StaticTokenProvider>),
    Gitea(GiteaForge<HttpTransport, StaticTokenProvider>),
}

impl AnyForge {
    /// Build a forge from a stored config row's kind/base_url/token. `None` for an unknown kind or
    /// a transport that fails to construct.
    pub fn from_parts(kind: &str, base_url: &str, token: &str) -> Option<Self> {
        let transport = HttpTransport::new(base_url, token.to_string()).ok()?;
        let client: Client = GithubClient::new(transport, StaticTokenProvider(token.to_string()));
        match kind {
            "github" => Some(AnyForge::Github(GithubForge::new(client))),
            "gitea" => Some(AnyForge::Gitea(GiteaForge::new(client))),
            _ => None,
        }
    }
}

// Delegate every `Forge` method to the active variant. Owned args move into the one arm that
// runs; `repo` and `&str` args are borrowed for the returned future's lifetime.
impl Forge for AnyForge {
    fn capabilities(&self) -> Capabilities {
        match self {
            AnyForge::Github(f) => f.capabilities(),
            AnyForge::Gitea(f) => f.capabilities(),
        }
    }

    fn repo_meta(
        &self,
        repo: &RepoRef,
    ) -> impl Future<Output = Result<RepoMeta, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.repo_meta(repo).await,
                AnyForge::Gitea(f) => f.repo_meta(repo).await,
            }
        }
    }

    fn commits(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Commit>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.commits(repo, cursor).await,
                AnyForge::Gitea(f) => f.commits(repo, cursor).await,
            }
        }
    }

    fn pull_requests(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<PullDelta, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.pull_requests(repo, cursor).await,
                AnyForge::Gitea(f) => f.pull_requests(repo, cursor).await,
            }
        }
    }

    fn reviews(
        &self,
        repo: &RepoRef,
        pr_number: i64,
        pr_id: &str,
    ) -> impl Future<Output = Result<Vec<Review>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.reviews(repo, pr_number, pr_id).await,
                AnyForge::Gitea(f) => f.reviews(repo, pr_number, pr_id).await,
            }
        }
    }

    fn issues(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Issue>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.issues(repo, cursor).await,
                AnyForge::Gitea(f) => f.issues(repo, cursor).await,
            }
        }
    }

    fn releases(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<Release>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.releases(repo, cursor).await,
                AnyForge::Gitea(f) => f.releases(repo, cursor).await,
            }
        }
    }

    fn ci_runs(
        &self,
        repo: &RepoRef,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<Delta<CiRun>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.ci_runs(repo, cursor).await,
                AnyForge::Gitea(f) => f.ci_runs(repo, cursor).await,
            }
        }
    }

    fn pr_files(
        &self,
        repo: &RepoRef,
        pr_number: i64,
        pr_id: &str,
    ) -> impl Future<Output = Result<Vec<PrFile>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.pr_files(repo, pr_number, pr_id).await,
                AnyForge::Gitea(f) => f.pr_files(repo, pr_number, pr_id).await,
            }
        }
    }

    fn file_text(
        &self,
        repo: &RepoRef,
        path: &str,
        cursor: Option<String>,
    ) -> impl Future<Output = Result<FileText, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.file_text(repo, path, cursor).await,
                AnyForge::Gitea(f) => f.file_text(repo, path, cursor).await,
            }
        }
    }

    fn tree(
        &self,
        repo: &RepoRef,
    ) -> impl Future<Output = Result<Vec<TreeEntry>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.tree(repo).await,
                AnyForge::Gitea(f) => f.tree(repo).await,
            }
        }
    }

    fn blob_text(
        &self,
        repo: &RepoRef,
        sha: &str,
    ) -> impl Future<Output = Result<Option<String>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.blob_text(repo, sha).await,
                AnyForge::Gitea(f) => f.blob_text(repo, sha).await,
            }
        }
    }

    fn person_repos(
        &self,
        login: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.person_repos(login).await,
                AnyForge::Gitea(f) => f.person_repos(login).await,
            }
        }
    }

    fn org_repos(&self, org: &str) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.org_repos(org).await,
                AnyForge::Gitea(f) => f.org_repos(org).await,
            }
        }
    }

    fn search_users(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.search_users(query).await,
                AnyForge::Gitea(f) => f.search_users(query).await,
            }
        }
    }

    fn search_orgs(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.search_orgs(query).await,
                AnyForge::Gitea(f) => f.search_orgs(query).await,
            }
        }
    }

    fn search_repos(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<Vec<String>, ForgeError>> + Send {
        async move {
            match self {
                AnyForge::Github(f) => f.search_repos(query).await,
                AnyForge::Gitea(f) => f.search_repos(query).await,
            }
        }
    }
}
