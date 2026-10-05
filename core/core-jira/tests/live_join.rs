//! The end-to-end Jira-to-repo join, live: syncs the GitHub fixture repo and the Jira fixture
//! project, then runs `core_jira::link_keys` and checks the result. Jira is a read-only tracker,
//! not a forge.
//! Lives in core-jira because `link_keys` is owned here; core-jira takes `core-forge-github`,
//! `core-sync` and `core-github` as dev-dependencies for this test only. That is acyclic even
//! though core-sync dev-depends on core-forge-github: dev-deps never enter the library build graph
//! of a dependent crate.
//! Ignored by default - needs both fixtures' credentials:
//!   ORGONZOLA_GITHUB_TOKEN=<token> \
//!   ORGONZOLA_JIRA_BASE_URL=https://your-site.atlassian.net \
//!   ORGONZOLA_JIRA_EMAIL=you@example.com \
//!   ORGONZOLA_JIRA_TOKEN=<api token> \
//!   cargo test -p core-jira --features http -- --ignored --nocapture live_join
#![cfg(feature = "http")]

use core_forge_github::GithubForge;
use core_github::{GithubClient, HttpTransport, StaticTokenProvider};
use core_jira::{JiraClient, JiraError, JiraHttp};
use core_store::{Repo, Store};
use core_sync::{repo_id, SyncEngine, DEFAULT_FORGE};
use std::future::Future;

const FIXTURE_REPO: &str = "kikijiki/orgonzola-linkcheck";
const PROJECT: &str = "SCRUM";
const TRACKER_ID: &str = "live-test-tracker";

/// Duplicated from live_jira.rs rather than shared: each live test file stays self-contained, like
/// live_gitea.rs.
struct LiveJiraHttp {
    client: reqwest::Client,
    base_url: String,
    email: String,
    token: String,
}

impl LiveJiraHttp {
    fn new(base_url: &str, email: String, token: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            email,
            token,
        }
    }
}

impl JiraHttp for LiveJiraHttp {
    fn get(&self, path: &str) -> impl Future<Output = Result<String, JiraError>> + Send {
        let url = format!("{}{}", self.base_url, path);
        let client = self.client.clone();
        let email = self.email.clone();
        let token = self.token.clone();
        async move {
            let resp = client
                .get(&url)
                .header("Accept", "application/json")
                .basic_auth(&email, Some(&token))
                .send()
                .await
                .map_err(|e| JiraError::Http(e.to_string()))?;
            let status = resp.status();
            if !status.is_success() {
                return Err(JiraError::Http(format!("{} for {}", status.as_u16(), url)));
            }
            resp.text()
                .await
                .map_err(|e| JiraError::Http(e.to_string()))
        }
    }
}

#[tokio::test]
#[ignore = "needs a real GitHub token (ORGONZOLA_GITHUB_TOKEN) and a real Jira Cloud site (ORGONZOLA_JIRA_BASE_URL, ORGONZOLA_JIRA_EMAIL, ORGONZOLA_JIRA_TOKEN)"]
async fn github_and_jira_join_on_the_scrum_fixture_keys() {
    let github_token = std::env::var("ORGONZOLA_GITHUB_TOKEN").expect("set ORGONZOLA_GITHUB_TOKEN");
    let jira_base = std::env::var("ORGONZOLA_JIRA_BASE_URL").expect(
        "set ORGONZOLA_JIRA_BASE_URL (the site ROOT, e.g. https://your-site.atlassian.net - never the \
         /jira/ UI path)",
    );
    let jira_email = std::env::var("ORGONZOLA_JIRA_EMAIL").expect("set ORGONZOLA_JIRA_EMAIL");
    let jira_token = std::env::var("ORGONZOLA_JIRA_TOKEN").expect("set ORGONZOLA_JIRA_TOKEN");

    let store = Store::open_in_memory().await.expect("open in-memory store");

    // ---- sync GitHub: commits + pull requests for the fixture repo ----
    let transport = HttpTransport::new("https://api.github.com", github_token.clone())
        .expect("build github transport");
    // The token goes to both the transport and the token provider; an empty provider fails auth
    // with a bare "authentication failed" (see hosts/desktop/src/forge.rs's AnyForge::from_parts).
    let forge = GithubForge::new(GithubClient::new(
        transport,
        StaticTokenProvider(github_token),
    ));
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
    let gh_report = engine.sync_all(&repo).await.expect("github sync_all");
    assert!(
        gh_report.commits >= 3 && gh_report.pull_requests == 2,
        "expected the usual fixture shape (>=3 commits, 2 PRs), got {gh_report:?}"
    );

    // ---- sync Jira: the SCRUM project's issues ----
    let jira_client = JiraClient::new(LiveJiraHttp::new(&jira_base, jira_email, jira_token));
    let jira_report = core_jira::sync_project_issues(
        &jira_client,
        engine.store(),
        TRACKER_ID,
        &jira_base,
        PROJECT,
        None,
    )
    .await
    .expect("sync_project_issues");
    assert_eq!(jira_report.issues, 10, "fixture project has 10 issues");

    // ---- the join itself ----
    let written = core_jira::link_keys(engine.store(), TRACKER_ID, std::slice::from_ref(&rid))
        .await
        .expect("link_keys");
    assert_eq!(
        written, 4,
        "expected 4 links total: SCRUM-2/SCRUM-4 from commit messages, SCRUM-5/SCRUM-8 from PR titles"
    );

    // Assert both halves separately: the join matches keys from two different text sources
    // (commit messages vs. PR titles).
    let mut commit_links = 0usize;
    for c in engine.store().commits_for_repo(&rid).await.unwrap() {
        let links = engine.store().links_to("commit", &c.sha).await.unwrap();
        for l in &links {
            assert_eq!(l.relation, "implements");
            assert_eq!(l.src_kind, "work_item");
        }
        commit_links += links.len();
    }
    assert_eq!(
        commit_links, 2,
        "SCRUM-2 and SCRUM-4 should link from commit messages"
    );

    let mut pr_links = 0usize;
    for pr in engine.store().pull_requests(&rid).await.unwrap() {
        let links = engine
            .store()
            .links_to("pull_request", &pr.id)
            .await
            .unwrap();
        for l in &links {
            assert_eq!(l.relation, "implements");
            assert_eq!(l.src_kind, "work_item");
        }
        pr_links += links.len();
    }
    assert_eq!(
        pr_links, 2,
        "SCRUM-5 and SCRUM-8 should link from PR titles"
    );

    assert_eq!(commit_links + pr_links, written);
}
