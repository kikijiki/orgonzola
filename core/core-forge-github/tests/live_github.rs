//! Live test against the real GitHub API, syncing the private fixture repo
//! `kikijiki/orgonzola-linkcheck`. Ignored by default; needs a token with access to it:
//!   ORGONZOLA_GITHUB_TOKEN=<token> \
//!   cargo test -p core-forge-github --features http -- --ignored --nocapture
//! Asserts a `SyncEngine<GithubForge>` sync pulls the fixture's known commits and pull requests.
#![cfg(feature = "http")]

use core_forge_github::GithubForge;
use core_github::{GithubClient, HttpTransport, StaticTokenProvider};
use core_store::{Repo, Store};
use core_sync::{repo_id, SyncEngine, DEFAULT_FORGE};

const FIXTURE_REPO: &str = "kikijiki/orgonzola-linkcheck";

#[tokio::test]
#[ignore = "needs a real GitHub token with access to the private kikijiki/orgonzola-linkcheck fixture (ORGONZOLA_GITHUB_TOKEN)"]
async fn syncs_the_fixture_repo_and_pulls_its_commits_and_pull_requests() {
    let token = std::env::var("ORGONZOLA_GITHUB_TOKEN").expect("set ORGONZOLA_GITHUB_TOKEN");

    let transport =
        HttpTransport::new("https://api.github.com", token.clone()).expect("build transport");
    // The token goes to both the transport (Bearer header) and the token provider; an empty
    // provider fails auth with a bare "authentication failed".
    let forge = GithubForge::new(GithubClient::new(
        transport,
        StaticTokenProvider(token.clone()),
    ));
    let store = Store::open_in_memory().await.expect("open in-memory store");
    let engine = SyncEngine::new(forge, store);

    let rid = repo_id(DEFAULT_FORGE, FIXTURE_REPO);
    let (owner, name) = FIXTURE_REPO.split_once('/').expect("owner/repo");
    let repo = Repo {
        id: rid.clone(),
        owner: owner.to_string(),
        name: name.to_string(),
        full_name: FIXTURE_REPO.to_string(),
        ownership: "observed".into(),
    };

    let report = engine.sync_all(&repo).await.expect("sync_all");
    eprintln!(
        "live github sync: commits={} issues={} pulls={} releases={} reviews={} deps={}",
        report.commits,
        report.issues,
        report.pull_requests,
        report.releases,
        report.reviews,
        report.dependencies
    );

    // The default branch (master) carries the "SCRUM-2" and "SCRUM-4" commits plus the initial
    // commit(s). Only the default branch is fetched, so the PR branches' own commits are absent.
    assert!(
        report.commits >= 3,
        "expected at least 3 commits on the fixture's default branch, got {}",
        report.commits
    );
    assert_eq!(
        report.pull_requests, 2,
        "fixture has exactly 2 open PRs (#1 SCRUM-5, #2 SCRUM-8)"
    );

    let prs = engine.store().pull_requests(&rid).await.unwrap();
    assert_eq!(prs.len(), 2);
    assert!(
        prs.iter().any(|p| p.title.contains("SCRUM-5")),
        "expected a PR title carrying SCRUM-5: {prs:?}"
    );
    assert!(
        prs.iter().any(|p| p.title.contains("SCRUM-8")),
        "expected a PR title carrying SCRUM-8: {prs:?}"
    );
}
