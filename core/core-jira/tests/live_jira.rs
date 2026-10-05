//! Live tests against a real Jira Cloud site (the private `SCRUM` project fixture). Ignored by
//! default; each test needs real credentials:
//!   ORGONZOLA_JIRA_BASE_URL=https://your-site.atlassian.net \
//!   ORGONZOLA_JIRA_EMAIL=you@example.com \
//!   ORGONZOLA_JIRA_TOKEN=<api token> \
//!   cargo test -p core-jira --features http -- --ignored --nocapture
//! `ORGONZOLA_JIRA_BASE_URL` must be the site root (no `/jira/` suffix): core-jira builds every
//! API path relative to it, so the wrong root 404s everything.
//! These drive `core-jira`'s `JiraClient` (and, for the say-do / epic-progress checks, the
//! `core-summary` reads) against a live tenant. Each test uses its own in-memory store.
#![cfg(feature = "http")]

use core_jira::{JiraClient, JiraError, JiraHttp};
use core_store::{BoardKind, Store, Tracker};
use std::future::Future;

const PROJECT: &str = "SCRUM";
const TRACKER_ID: &str = "live-test-tracker";
const BOARD_ID: &str = "live-test-board";

/// Minimal reqwest-backed [`JiraHttp`], Basic `email:token` auth only. Mirrors
/// `hosts/desktop/src/jira.rs::JiraReqwest`; not depended on because `hosts/desktop` depends on
/// `core-jira`.
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
            // Same "<status> for <url>" shape core-jira's private `endpoint_absent` keys off, so a
            // real 404/410 exercises the production fallback path.
            if !status.is_success() {
                return Err(JiraError::Http(format!("{} for {}", status.as_u16(), url)));
            }
            resp.text()
                .await
                .map_err(|e| JiraError::Http(e.to_string()))
        }
    }
}

fn creds() -> (String, String, String) {
    let base = std::env::var("ORGONZOLA_JIRA_BASE_URL").expect(
        "set ORGONZOLA_JIRA_BASE_URL (the site ROOT, e.g. https://your-site.atlassian.net - never the \
         /jira/ UI path)",
    );
    let email = std::env::var("ORGONZOLA_JIRA_EMAIL").expect("set ORGONZOLA_JIRA_EMAIL");
    let token = std::env::var("ORGONZOLA_JIRA_TOKEN").expect("set ORGONZOLA_JIRA_TOKEN");
    (base, email, token)
}

/// Minimal percent-encoding for a JQL query value. Duplicates core-jira's private
/// `percent_encode` so this test only uses the crate's public surface.
fn enc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Sync the fixture project's issues into a fresh in-memory store, linked to a throwaway board so
/// board-scoped reads (`jira_issues_for_board`, `board_epics`, `jira_sprints_for_board`) have data.
async fn synced_store(base: &str, email: &str, token: &str) -> (Store, JiraClient<LiveJiraHttp>) {
    let store = Store::open_in_memory().await.expect("open in-memory store");
    store
        .create_board(
            BOARD_ID,
            "live test board",
            BoardKind::Team,
            "2026-09-17T00:00:00Z",
        )
        .await
        .expect("create_board");
    store
        .upsert_tracker(&Tracker {
            id: TRACKER_ID.into(),
            name: "live jira".into(),
            kind: "jira".into(),
            base_url: base.into(),
            email: Some(email.into()),
        })
        .await
        .expect("upsert_tracker");
    store
        .link_board_tracker(BOARD_ID, TRACKER_ID, PROJECT)
        .await
        .expect("link_board_tracker");

    let client = JiraClient::new(LiveJiraHttp::new(
        base,
        email.to_string(),
        token.to_string(),
    ));
    core_jira::sync_project_issues(&client, &store, TRACKER_ID, base, PROJECT, None)
        .await
        .expect("sync_project_issues");
    (store, client)
}

#[tokio::test]
#[ignore = "needs a real Jira Cloud site (ORGONZOLA_JIRA_BASE_URL, ORGONZOLA_JIRA_EMAIL, ORGONZOLA_JIRA_TOKEN)"]
async fn auth_and_enhanced_search_returns_all_ten_issues() {
    let (base, email, token) = creds();
    let http = LiveJiraHttp::new(&base, email, token);

    let jql = enc("project = SCRUM ORDER BY updated DESC");
    let path = format!("/rest/api/3/search/jql?jql={jql}&maxResults=50&fields=summary");
    let body = http.get(&path).await.expect("enhanced search request");
    let json: serde_json::Value = serde_json::from_str(&body).expect("parse enhanced search body");

    let issues = json["issues"].as_array().expect("issues array in response");
    assert_eq!(
        issues.len(),
        10,
        "fixture project SCRUM has 10 issues as of 2026-09-17: {body}"
    );
    assert_eq!(
        json.get("isLast").and_then(|v| v.as_bool()),
        Some(true),
        "10 issues fit in one page, isLast should be true: {body}"
    );
}

/// Confirms the legacy search endpoint is gone on this Cloud tenant and the public
/// `search_issues` still works. Observable proxy for the private `endpoint_absent` classifying the
/// 410 correctly.
#[tokio::test]
#[ignore = "needs a real Jira Cloud site (ORGONZOLA_JIRA_BASE_URL, ORGONZOLA_JIRA_EMAIL, ORGONZOLA_JIRA_TOKEN)"]
async fn legacy_search_endpoint_is_gone_and_the_cloud_path_still_works() {
    let (base, email, token) = creds();
    let http = LiveJiraHttp::new(&base, email.clone(), token.clone());

    let jql = enc("project = SCRUM");
    let legacy_path = format!("/rest/api/2/search?jql={jql}&maxResults=50");
    let err = http
        .get(&legacy_path)
        .await
        .expect_err("legacy /rest/api/2/search should be gone (410) on a Cloud tenant");
    let JiraError::Http(msg) = err else {
        panic!("expected JiraError::Http, got a different variant");
    };
    assert!(
        msg.starts_with("410 "),
        "expected a 410 Gone from the removed endpoint, got: {msg}"
    );

    let client = JiraClient::new(LiveJiraHttp::new(&base, email, token));
    let tickets = client
        .search_issues(PROJECT, None)
        .await
        .expect("search_issues should still succeed via the Cloud enhanced-search path");
    assert_eq!(tickets.len(), 10);
}

#[tokio::test]
#[ignore = "needs a real Jira Cloud site (ORGONZOLA_JIRA_BASE_URL, ORGONZOLA_JIRA_EMAIL, ORGONZOLA_JIRA_TOKEN)"]
async fn agile_endpoints_report_one_board_one_active_sprint_eight_members() {
    let (base, email, token) = creds();
    let client = JiraClient::new(LiveJiraHttp::new(&base, email, token));

    let boards = client.boards(PROJECT).await.expect("boards");
    assert_eq!(
        boards.len(),
        1,
        "fixture project has exactly one board: {boards:?}"
    );

    let sprints = client.sprints(boards[0].id).await.expect("sprints");
    let active: Vec<_> = sprints
        .iter()
        .filter(|s| s.state.as_deref() == Some("active"))
        .collect();
    assert_eq!(
        active.len(),
        1,
        "fixture has exactly one active sprint: {sprints:?}"
    );

    let members = client
        .sprint_issues(active[0].id)
        .await
        .expect("sprint_issues");
    assert_eq!(
        members.len(),
        8,
        "fixture's active sprint has 8 members: {members:?}"
    );
}

#[tokio::test]
#[ignore = "needs a real Jira Cloud site (ORGONZOLA_JIRA_BASE_URL, ORGONZOLA_JIRA_EMAIL, ORGONZOLA_JIRA_TOKEN)"]
async fn epic_parent_keys_persist_and_epic_progress_matches_the_fixture() {
    let (base, email, token) = creds();
    let (store, _client) = synced_store(&base, &email, &token).await;

    let issues = store
        .jira_issues_for_board(BOARD_ID)
        .await
        .expect("jira_issues_for_board");
    assert_eq!(
        issues.len(),
        10,
        "fixture project has 10 issues: {issues:?}"
    );
    let with_parent = issues.iter().filter(|i| i.parent_key.is_some()).count();
    assert_eq!(
        with_parent, 6,
        "6 of the 10 issues are epic children, 3 under each of SCRUM-9/SCRUM-10: {issues:?}"
    );

    let epics = core_summary::board_epics(&store, BOARD_ID)
        .await
        .expect("board_epics");
    assert_eq!(epics.len(), 2, "two epics: {epics:?}");
    let scrum9 = epics
        .iter()
        .find(|e| e.key == "SCRUM-9")
        .expect("SCRUM-9 present");
    let scrum10 = epics
        .iter()
        .find(|e| e.key == "SCRUM-10")
        .expect("SCRUM-10 present");
    // Tracks the fixture's state as of 2026-09-17.
    assert_eq!(
        (scrum9.total, scrum9.done, scrum9.in_progress),
        (3, 1, 2),
        "SCRUM-9 (Board experience): {scrum9:?}"
    );
    assert_eq!(
        (scrum10.total, scrum10.done, scrum10.in_progress),
        (3, 2, 0),
        "SCRUM-10 (Sync reliability): {scrum10:?}"
    );
}

#[tokio::test]
#[ignore = "needs a real Jira Cloud site (ORGONZOLA_JIRA_BASE_URL, ORGONZOLA_JIRA_EMAIL, ORGONZOLA_JIRA_TOKEN)"]
async fn active_sprint_commitment_freezes_and_say_do_reports_the_fixture_ratio() {
    let (base, email, token) = creds();
    let (store, client) = synced_store(&base, &email, &token).await;

    let boards = client.boards(PROJECT).await.expect("boards");
    let board = boards.first().expect("fixture has one board");
    let sprints = client.sprints(board.id).await.expect("sprints");
    core_jira::sync_sprints(&store, TRACKER_ID, PROJECT, &sprints)
        .await
        .expect("sync_sprints");
    // First-seen-active stamp: a fresh in-memory store has not observed this sprint, so the
    // commitment freezes here. The exact date does not affect say-do's counts.
    let captured_on = "2026-09-17";
    for sprint in &sprints {
        core_jira::sync_sprint_issues(&client, &store, TRACKER_ID, sprint, captured_on)
            .await
            .expect("sync_sprint_issues");
    }

    let rows = store
        .jira_sprints_for_board(BOARD_ID)
        .await
        .expect("jira_sprints_for_board");
    let active = rows
        .iter()
        .find(|r| r.state.as_deref() == Some("active"))
        .expect("one active sprint row");
    let say_do = core_summary::sprint_say_do(&store, active)
        .await
        .expect("sprint_say_do");

    assert!(
        say_do.committed_on.is_some(),
        "commitment should have frozen on first sight of the active sprint: {say_do:?}"
    );
    // Tracks the fixture's state as of 2026-09-17: 8 committed, 4 of them done.
    assert_eq!(say_do.committed, 8, "{say_do:?}");
    assert_eq!(say_do.delivered, 4, "{say_do:?}");
    assert_eq!(say_do.carryover, 4, "{say_do:?}");
    assert_eq!(say_do.ratio, Some(0.5), "{say_do:?}");
}
