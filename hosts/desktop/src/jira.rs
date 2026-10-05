//! Host wiring for the Jira tracker: the reqwest transport and the read-only sync pass that runs
//! alongside the forge sync. Mapping and linking live in `core-jira`.

use crate::CredentialStore;
use core_jira::{JiraClient, JiraError, JiraHttp};
use core_store::Store;
use std::future::Future;

/// A reqwest-backed [`JiraHttp`]. Auth is optional: Jira Cloud uses Basic `email:token`,
/// Server/DC a bearer PAT, and a public instance needs none. Anonymous requests still sync
/// issues; the agile/sprint endpoints 401, which the sync treats as "no sprints".
pub struct JiraReqwest {
    client: reqwest::Client,
    base_url: String,
    email: Option<String>,
    token: Option<String>,
}

impl JiraReqwest {
    pub fn new(base_url: &str, email: Option<String>, token: Option<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            email,
            token,
        }
    }
}

impl JiraHttp for JiraReqwest {
    fn get(&self, path: &str) -> impl Future<Output = Result<String, JiraError>> + Send {
        let url = format!("{}{}", self.base_url, path);
        let client = self.client.clone();
        let email = self.email.clone();
        let token = self.token.clone();
        async move {
            let mut req = client.get(&url).header("Accept", "application/json");
            req = match (&email, &token) {
                (Some(email), Some(token)) => req.basic_auth(email, Some(token)),
                (None, Some(token)) => req.bearer_auth(token),
                _ => req, // anonymous
            };
            let resp = req
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

fn cursor_key(tracker_id: &str, project: &str) -> String {
    format!("jira:{tracker_id}:{project}")
}

/// Run the read-only Jira sync for every board-linked project: incremental issue sync into
/// `work_items`/`jira_issues`, best-effort sprint sync, then `PROJ-123` linking over each
/// board's repos. A failing project or tracker is logged and skipped.
pub async fn sync_jira(store: &Store, credentials: &dyn CredentialStore) {
    let Ok(links) = store.all_board_trackers().await else {
        return;
    };
    // Sync each unique (tracker, project) once.
    let mut seen = std::collections::HashSet::new();
    for bt in &links {
        if !seen.insert((bt.tracker_id.clone(), bt.project_key.clone())) {
            continue;
        }
        if let Err(e) = sync_one(store, credentials, &bt.tracker_id, &bt.project_key).await {
            eprintln!(
                "jira: sync {}:{} failed: {e}",
                bt.tracker_id, bt.project_key
            );
        }
    }
    // Link keys per board over its effective repos.
    for bt in &links {
        let Ok(repo_ids) = store.board_effective_repo_ids(&bt.board_id).await else {
            continue;
        };
        if let Err(e) = core_jira::link_keys(store, &bt.tracker_id, &repo_ids).await {
            eprintln!("jira: link {}:{} failed: {e}", bt.tracker_id, bt.board_id);
        }
    }
}

async fn sync_one(
    store: &Store,
    credentials: &dyn CredentialStore,
    tracker_id: &str,
    project: &str,
) -> Result<(), JiraError> {
    let Some(tracker) = store.tracker(tracker_id).await? else {
        return Ok(());
    };
    let token = credentials.get(tracker_id);
    let client = JiraClient::new(JiraReqwest::new(
        &tracker.base_url,
        tracker.email.clone(),
        token,
    ));

    let key = cursor_key(tracker_id, project);
    let since = store.sync_cursor(&key, "issues").await?;
    let report = core_jira::sync_project_issues(
        &client,
        store,
        tracker_id,
        &tracker.base_url,
        project,
        since.as_deref(),
    )
    .await?;
    if let Some(cursor) = report.cursor {
        store.set_sync_cursor(&key, "issues", &cursor).await?;
    }

    // Sprints (best-effort: the agile API is often auth-gated). Also sync each sprint's issue
    // membership and freeze a start-of-sprint commitment the first time it is seen active.
    let today = &crate::now_rfc3339()[..10];
    if let Ok(boards) = client.boards(project).await {
        for board in boards {
            if let Ok(sprints) = client.sprints(board.id).await {
                let _ = core_jira::sync_sprints(store, tracker_id, project, &sprints).await;
                for sprint in &sprints {
                    let _ =
                        core_jira::sync_sprint_issues(&client, store, tracker_id, sprint, today)
                            .await;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod live_tests {
    use super::*;

    /// End-to-end against a real Jira Cloud instance. Ignored by default: needs network and live
    /// credentials. Run with the env set:
    ///   set -a; . ./.env.local; set +a
    ///   cargo test -p orgonzola-desktop --lib live_say_do -- --ignored --nocapture
    /// Drives the real client and sync functions into an in-memory store and prints say-do.
    #[tokio::test]
    #[ignore = "needs live Jira credentials + network"]
    async fn live_say_do() {
        let (Ok(base), Ok(email), Ok(token)) = (
            std::env::var("JIRA_BASE_URL"),
            std::env::var("JIRA_EMAIL"),
            std::env::var("JIRA_TOKEN"),
        ) else {
            eprintln!("skipping: JIRA_BASE_URL / JIRA_EMAIL / JIRA_TOKEN not set");
            return;
        };
        let project = std::env::var("JIRA_PROJECT").unwrap_or_else(|_| "SCRUM".into());
        let tracker_id = "trk:live";

        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_tracker(&core_store::Tracker {
                id: tracker_id.into(),
                name: "Live".into(),
                kind: "jira".into(),
                base_url: base.clone(),
                email: Some(email.clone()),
            })
            .await
            .unwrap();
        store
            .create_board(
                "board:live",
                "Live",
                core_store::BoardKind::Team,
                &crate::now_rfc3339(),
            )
            .await
            .unwrap();
        store
            .link_board_tracker("board:live", tracker_id, &project)
            .await
            .unwrap();

        let client = JiraClient::new(JiraReqwest::new(&base, Some(email), Some(token)));
        // Issues first (say-do reads their status), then sprints, membership and the commitment
        // snapshot.
        core_jira::sync_project_issues(&client, &store, tracker_id, &base, &project, None)
            .await
            .expect("issue sync");
        let today = &crate::now_rfc3339()[..10];
        let boards = client.boards(&project).await.expect("boards");
        assert!(!boards.is_empty(), "the project should have an agile board");
        for board in &boards {
            let sprints = client.sprints(board.id).await.expect("sprints");
            core_jira::sync_sprints(&store, tracker_id, &project, &sprints)
                .await
                .unwrap();
            for sprint in &sprints {
                let n = core_jira::sync_sprint_issues(&client, &store, tracker_id, sprint, today)
                    .await
                    .expect("sprint membership");
                eprintln!("sprint {:?} ({:?}): {n} members", sprint.name, sprint.state);
            }
        }

        let say_do = core_summary::board_say_do(&store, "board:live")
            .await
            .expect("say-do");
        for sd in &say_do {
            eprintln!(
                "say-do {}: committed={} delivered={} carryover={} added={} removed={} ratio={:?} on={:?}",
                sd.sprint_name,
                sd.committed,
                sd.delivered,
                sd.carryover,
                sd.added,
                sd.removed,
                sd.ratio,
                sd.committed_on,
            );
        }
        // An active sprint observed during this run must have a frozen commitment.
        let committed_any = say_do.iter().any(|s| s.committed_on.is_some());
        assert!(
            committed_any,
            "expected at least one sprint with a frozen commitment snapshot"
        );
    }
}
