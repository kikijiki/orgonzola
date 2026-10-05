//! Live integration test against a running Gitea (the local fixture). Ignored by default. Bring
//! one up with `just gitea-up && just gitea-mirror`, then:
//!   ORGONZOLA_FORGE_BASE_URL=http://localhost:3000/api/v1 \
//!   ORGONZOLA_FORGE_TOKEN=$(cat tools/gitea-fixture/.gitea-token) \
//!   cargo test -p core-forge-gitea --features http -- --ignored --nocapture
//! Drives the real `SyncEngine<GiteaForge>` against the mirrored tinygrad org and asserts commits
//! and issues reach the store.
#![cfg(feature = "http")]

use core_forge_gitea::GiteaForge;
use core_github::{GithubClient, HttpTransport, StaticTokenProvider};
use core_store::{Repo, Store};
use core_sync::{repo_id, SyncEngine, DEFAULT_FORGE};

#[tokio::test]
#[ignore = "needs a running Gitea fixture (ORGONZOLA_FORGE_BASE_URL + ORGONZOLA_FORGE_TOKEN)"]
async fn syncs_mirrored_tinygrad_from_a_live_gitea() {
    let base = std::env::var("ORGONZOLA_FORGE_BASE_URL").expect("set ORGONZOLA_FORGE_BASE_URL");
    let token = std::env::var("ORGONZOLA_FORGE_TOKEN").expect("set ORGONZOLA_FORGE_TOKEN");

    let transport = HttpTransport::new(base, token.clone()).expect("transport");
    let forge = GiteaForge::new(GithubClient::new(transport, StaticTokenProvider(token)));
    let store = Store::open_in_memory().await.expect("store");
    // `SyncEngine::new` registers the forge under `DEFAULT_FORGE`, so the repo id must carry that
    // forge segment (`repo:<forge>/<owner>/<name>`).
    let engine = SyncEngine::new(forge, store);

    let rid = repo_id(DEFAULT_FORGE, "tinygrad/tinygrad");
    let repo = Repo {
        id: rid.clone(),
        owner: "tinygrad".into(),
        name: "tinygrad".into(),
        full_name: "tinygrad/tinygrad".into(),
        ownership: "observed".into(),
    };

    let report = engine.sync_all(&repo).await.expect("sync_all");
    eprintln!(
        "live gitea sync: commits={} issues={} pulls={} releases={} reviews={} deps={}",
        report.commits,
        report.issues,
        report.pull_requests,
        report.releases,
        report.reviews,
        report.dependencies
    );

    // The mirrored tinygrad has real history and issues.
    assert!(
        report.commits > 0,
        "expected commits from the mirrored tinygrad"
    );
    assert!(
        report.issues > 0,
        "expected issues from the mirrored tinygrad"
    );
    assert_eq!(
        engine.store().count_commits().await.unwrap() as usize,
        report.commits,
        "persisted commit count should match the report"
    );
    let issues = engine.store().issues_for_repo(&rid).await.unwrap();
    assert_eq!(issues.len(), report.issues);
}
