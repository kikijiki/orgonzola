//! Read-only Jira tracker integration.
//! Syncs a Jira project's issues (and, where allowed, sprints) into the generic `work_items` /
//! `sprints` model (`source = "jira"`), and links `PROJ-123` keys found in PRs, branches and
//! commits to work items via the `links` table. Never writes to Jira.
//! Transport is injected (`JiraHttp`), so the host wires HTTP and auth and tests use recorded
//! JSON.

use core_store::{JiraIssueRow, JiraSprintRow, Store, StoreError};
use serde::Deserialize;
use std::future::Future;

#[derive(Debug, thiserror::Error)]
pub enum JiraError {
    #[error("jira http error: {0}")]
    Http(String),
    #[error("jira parse error: {0}")]
    Parse(String),
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// Carries a GET to a Jira site. `path` is the API path (e.g. `/rest/api/2/search?...`), relative
/// to the site root. The host adds auth; tests use a recorded-response fake.
pub trait JiraHttp {
    fn get(&self, path: &str) -> impl Future<Output = Result<String, JiraError>> + Send;
}

// ---- normalized output ----

/// One Jira issue, normalized from the REST shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JiraTicket {
    pub key: String,
    pub summary: String,
    pub issue_type: String,
    pub status: String,
    pub status_category: Option<String>,
    pub assignee: Option<String>,
    pub created: Option<String>,
    pub updated: Option<String>,
    pub resolved: Option<String>,
    /// The parent/epic issue key (`fields.parent.key`). `None` if top-level.
    pub parent_key: Option<String>,
}

/// A Jira sprint, normalized from the Agile REST shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JiraSprintInfo {
    pub id: u64,
    pub name: String,
    pub state: Option<String>,
    pub start: Option<String>,
    pub end: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JiraBoardInfo {
    pub id: u64,
    pub name: String,
}

// ---- raw REST JSON ----

#[derive(Deserialize)]
struct RawSearch {
    #[serde(default)]
    issues: Vec<RawIssue>,
    #[serde(default)]
    total: u32,
}

#[derive(Deserialize)]
struct RawIssue {
    key: String,
    fields: RawFields,
}

/// The enhanced-search (`/rest/api/3/search/jql`) page shape. Jira Cloud removed the legacy offset
/// `/search` endpoint, so this pages by `nextPageToken` and has no total; `isLast` or an absent
/// token marks the end.
#[derive(Deserialize)]
struct RawJqlSearch {
    #[serde(default)]
    issues: Vec<RawIssue>,
    #[serde(default, rename = "nextPageToken")]
    next_page_token: Option<String>,
    #[serde(default, rename = "isLast")]
    is_last: Option<bool>,
}

#[derive(Deserialize)]
struct RawFields {
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    issuetype: Option<RawNamed>,
    #[serde(default)]
    status: Option<RawStatus>,
    #[serde(default)]
    assignee: Option<RawUser>,
    #[serde(default)]
    created: Option<String>,
    #[serde(default)]
    updated: Option<String>,
    #[serde(default)]
    resolutiondate: Option<String>,
    #[serde(default)]
    parent: Option<RawParent>,
}

#[derive(Deserialize)]
struct RawNamed {
    #[serde(default)]
    name: Option<String>,
}

/// The `fields.parent` object (epic in a team-managed project, or a sub-task's parent). Only the
/// key is kept.
#[derive(Deserialize)]
struct RawParent {
    #[serde(default)]
    key: Option<String>,
}

#[derive(Deserialize)]
struct RawStatus {
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "statusCategory")]
    status_category: Option<RawCategory>,
}

#[derive(Deserialize)]
struct RawCategory {
    #[serde(default)]
    key: Option<String>,
}

#[derive(Deserialize)]
struct RawUser {
    #[serde(default, rename = "displayName")]
    display_name: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
struct RawSprintList {
    #[serde(default)]
    values: Vec<RawSprint>,
}

#[derive(Deserialize)]
struct RawSprint {
    id: u64,
    name: String,
    #[serde(default)]
    state: Option<String>,
    #[serde(default, rename = "startDate")]
    start_date: Option<String>,
    #[serde(default, rename = "endDate")]
    end_date: Option<String>,
}

#[derive(Deserialize)]
struct RawBoardList {
    #[serde(default)]
    values: Vec<RawBoard>,
}

#[derive(Deserialize)]
struct RawBoard {
    id: u64,
    name: String,
}

impl RawIssue {
    fn into_ticket(self) -> JiraTicket {
        let f = self.fields;
        JiraTicket {
            key: self.key,
            summary: f.summary.unwrap_or_default(),
            issue_type: f
                .issuetype
                .and_then(|t| t.name)
                .unwrap_or_else(|| "Issue".into()),
            status: f
                .status
                .as_ref()
                .and_then(|s| s.name.clone())
                .unwrap_or_else(|| "Unknown".into()),
            status_category: f.status.and_then(|s| s.status_category).and_then(|c| c.key),
            assignee: f.assignee.and_then(|u| u.display_name.or(u.name)),
            created: f.created,
            updated: f.updated,
            resolved: f.resolutiondate,
            parent_key: f.parent.and_then(|p| p.key),
        }
    }
}

// ---- client ----

/// Issues per search page, and the cap on pages one sync walks. The incremental cursor catches up
/// over later runs on very large projects.
const PAGE: usize = 50;
const MAX_PAGES: usize = 40;

/// The issue fields requested.
const FIELDS: &str = "summary,issuetype,status,assignee,created,updated,resolutiondate,parent";

pub struct JiraClient<H: JiraHttp> {
    http: H,
}

impl<H: JiraHttp> JiraClient<H> {
    pub fn new(http: H) -> Self {
        Self { http }
    }

    /// A project's issues, newest-updated first, stopping once an issue is not newer than `since`
    /// (the stored cursor on `updated`). `None` walks back to the page cap. Read-only JQL search.
    /// Uses Cloud enhanced search (`/rest/api/3/search/jql`, token-paginated), since Cloud removed
    /// the legacy `/rest/api/2/search` (410). Server/DC lacks the new endpoint, so a 404/410 falls
    /// back to the legacy offset search.
    pub async fn search_issues(
        &self,
        project: &str,
        since: Option<&str>,
    ) -> Result<Vec<JiraTicket>, JiraError> {
        match self.search_issues_cloud(project, since).await {
            Err(JiraError::Http(msg)) if endpoint_absent(&msg) => {
                self.search_issues_legacy(project, since).await
            }
            other => other,
        }
    }

    /// Cloud enhanced search: page by `nextPageToken`, stop at `isLast` or an absent token.
    async fn search_issues_cloud(
        &self,
        project: &str,
        since: Option<&str>,
    ) -> Result<Vec<JiraTicket>, JiraError> {
        let encoded = percent_encode(&format!("project = \"{project}\" ORDER BY updated DESC"));
        let mut out = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..MAX_PAGES {
            let mut path =
                format!("/rest/api/3/search/jql?jql={encoded}&maxResults={PAGE}&fields={FIELDS}");
            if let Some(t) = &token {
                path.push_str("&nextPageToken=");
                path.push_str(&percent_encode(t));
            }
            let body = self.http.get(&path).await?;
            let parsed: RawJqlSearch =
                serde_json::from_str(&body).map_err(|e| JiraError::Parse(e.to_string()))?;
            let n = parsed.issues.len();
            let stop = collect_until_cursor(parsed.issues, since, &mut out);
            token = parsed.next_page_token;
            if stop || token.is_none() || parsed.is_last == Some(true) || n == 0 {
                break;
            }
        }
        Ok(out)
    }

    /// Legacy offset search (`/rest/api/2/search`) for Server/DC instances.
    async fn search_issues_legacy(
        &self,
        project: &str,
        since: Option<&str>,
    ) -> Result<Vec<JiraTicket>, JiraError> {
        let encoded = percent_encode(&format!("project = \"{project}\" ORDER BY updated DESC"));
        let mut out = Vec::new();
        for page in 0..MAX_PAGES {
            let start = page * PAGE;
            let path = format!(
                "/rest/api/2/search?jql={encoded}&startAt={start}&maxResults={PAGE}&fields={FIELDS}"
            );
            let body = self.http.get(&path).await?;
            let parsed: RawSearch =
                serde_json::from_str(&body).map_err(|e| JiraError::Parse(e.to_string()))?;
            let n = parsed.issues.len();
            let total = parsed.total as usize;
            let stop = collect_until_cursor(parsed.issues, since, &mut out);
            if stop || n < PAGE || start + n >= total {
                break;
            }
        }
        Ok(out)
    }

    /// Scrum/Kanban boards for a project (Agile API, requires auth on most instances).
    pub async fn boards(&self, project: &str) -> Result<Vec<JiraBoardInfo>, JiraError> {
        let path = format!(
            "/rest/agile/1.0/board?projectKeyOrId={}&maxResults=50",
            percent_encode(project)
        );
        let body = self.http.get(&path).await?;
        let parsed: RawBoardList =
            serde_json::from_str(&body).map_err(|e| JiraError::Parse(e.to_string()))?;
        Ok(parsed
            .values
            .into_iter()
            .map(|b| JiraBoardInfo {
                id: b.id,
                name: b.name,
            })
            .collect())
    }

    /// A board's sprints (Agile API, requires auth on most instances).
    pub async fn sprints(&self, board_id: u64) -> Result<Vec<JiraSprintInfo>, JiraError> {
        let path = format!("/rest/agile/1.0/board/{board_id}/sprint?maxResults=50");
        let body = self.http.get(&path).await?;
        let parsed: RawSprintList =
            serde_json::from_str(&body).map_err(|e| JiraError::Parse(e.to_string()))?;
        Ok(parsed
            .values
            .into_iter()
            .map(|s| JiraSprintInfo {
                id: s.id,
                name: s.name,
                state: s.state,
                start: s.start_date,
                end: s.end_date,
            })
            .collect())
    }

    /// The issue keys in a sprint (Agile API, requires auth). Membership only; issue facts come
    /// from the project issue sync. Walks up to `MAX_PAGES`.
    pub async fn sprint_issues(&self, sprint_id: u64) -> Result<Vec<String>, JiraError> {
        let mut keys = Vec::new();
        let mut start = 0usize;
        for _ in 0..MAX_PAGES {
            let path = format!(
                "/rest/agile/1.0/sprint/{sprint_id}/issue?fields=summary&maxResults={PAGE}&startAt={start}"
            );
            let body = self.http.get(&path).await?;
            let parsed: RawSearch =
                serde_json::from_str(&body).map_err(|e| JiraError::Parse(e.to_string()))?;
            let n = parsed.issues.len();
            keys.extend(parsed.issues.into_iter().map(|i| i.key));
            start += n;
            if n < PAGE || start as u32 >= parsed.total {
                break;
            }
        }
        Ok(keys)
    }
}

/// The result of a project sync.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct JiraSyncReport {
    pub issues: usize,
    pub sprints: usize,
    /// The newest `updated` seen, to persist as the next incremental cursor.
    pub cursor: Option<String>,
}

/// The namespaced work-item id for a Jira issue: `jira:<tracker>:<KEY>`.
pub fn work_item_id(tracker_id: &str, key: &str) -> String {
    format!("jira:{tracker_id}:{key}")
}

/// Sync a project's issues into `work_items` + `jira_issues`. `base_url` builds the browse URL
/// (`<base>/browse/KEY`). Sprints are synced separately.
pub async fn sync_project_issues<H: JiraHttp>(
    client: &JiraClient<H>,
    store: &Store,
    tracker_id: &str,
    base_url: &str,
    project: &str,
    since: Option<&str>,
) -> Result<JiraSyncReport, JiraError> {
    let tickets = client.search_issues(project, since).await?;
    let base = base_url.trim_end_matches('/');
    let mut cursor: Option<String> = None;
    for t in &tickets {
        if cursor.as_deref() < t.updated.as_deref() {
            cursor = t.updated.clone();
        }
        store
            .upsert_jira_issue(&JiraIssueRow {
                work_item_id: work_item_id(tracker_id, &t.key),
                tracker_id: tracker_id.to_string(),
                project: project.to_string(),
                issue_key: t.key.clone(),
                title: t.summary.clone(),
                issue_type: t.issue_type.clone(),
                status: t.status.clone(),
                status_category: t.status_category.clone(),
                assignee: t.assignee.clone(),
                url: Some(format!("{base}/browse/{}", t.key)),
                created_at: t.created.clone(),
                updated_at: t.updated.clone(),
                resolved_at: t.resolved.clone(),
                parent_key: t.parent_key.clone(),
            })
            .await?;
    }
    Ok(JiraSyncReport {
        issues: tickets.len(),
        sprints: 0,
        cursor,
    })
}

/// Persist a board's sprints into `sprints` + `jira_sprints`.
pub async fn sync_sprints(
    store: &Store,
    tracker_id: &str,
    project: &str,
    sprints: &[JiraSprintInfo],
) -> Result<usize, JiraError> {
    for s in sprints {
        store
            .upsert_jira_sprint(&JiraSprintRow {
                sprint_id: format!("jira:{tracker_id}:sprint:{}", s.id),
                tracker_id: tracker_id.to_string(),
                project: project.to_string(),
                name: s.name.clone(),
                state: s.state.clone(),
                starts_on: s.start.clone(),
                ends_on: s.end.clone(),
                committed_at: None,
            })
            .await?;
    }
    Ok(sprints.len())
}

/// Sync a sprint's issue membership: store the sprint's issue keys as the live member set
/// (wholesale replace), and freeze the start-of-sprint commitment the first time the sprint is
/// observed `active`. `captured_on` is the host's observation date. Returns the member count.
pub async fn sync_sprint_issues<H: JiraHttp>(
    client: &JiraClient<H>,
    store: &Store,
    tracker_id: &str,
    sprint: &JiraSprintInfo,
    captured_on: &str,
) -> Result<usize, JiraError> {
    let sprint_id = format!("jira:{tracker_id}:sprint:{}", sprint.id);
    let keys = client.sprint_issues(sprint.id).await?;
    let ids: Vec<String> = keys.iter().map(|k| work_item_id(tracker_id, k)).collect();
    store.replace_sprint_work_items(&sprint_id, &ids).await?;
    if sprint.state.as_deref() == Some("active")
        && !store.sprint_commitment_taken(&sprint_id).await?
    {
        store
            .record_sprint_commitment(&sprint_id, &ids, captured_on)
            .await?;
    }
    Ok(ids.len())
}

/// Link `PROJ-123` keys found in a board's PRs (title + body) and commits (message) to the
/// matching synced Jira work item via the `links` table. Keys with no synced issue are ignored.
/// Returns links written.
pub async fn link_keys(
    store: &Store,
    tracker_id: &str,
    repo_ids: &[String],
) -> Result<usize, JiraError> {
    let mut written = 0usize;
    for repo_id in repo_ids {
        for pr in store.pull_requests(repo_id).await? {
            let text = format!("{} {}", pr.title, pr.body.unwrap_or_default());
            for key in core_graph::parse_issue_keys(&text) {
                if let Some(issue) = store.jira_issue_by_key(tracker_id, &key).await? {
                    store
                        .add_link(
                            "work_item",
                            &issue.work_item_id,
                            "pull_request",
                            &pr.id,
                            "implements",
                        )
                        .await?;
                    written += 1;
                }
            }
        }
        for c in store.commits_for_repo(repo_id).await? {
            for key in core_graph::parse_issue_keys(&c.message) {
                if let Some(issue) = store.jira_issue_by_key(tracker_id, &key).await? {
                    store
                        .add_link(
                            "work_item",
                            &issue.work_item_id,
                            "commit",
                            &c.sha,
                            "implements",
                        )
                        .await?;
                    written += 1;
                }
            }
        }
    }
    Ok(written)
}

/// Push a page of raw issues onto `out`, newest-first, stopping at the first issue not newer than
/// `since`. Returns whether the walk should stop. Shared by the Cloud and legacy search paths.
fn collect_until_cursor(
    issues: Vec<RawIssue>,
    since: Option<&str>,
    out: &mut Vec<JiraTicket>,
) -> bool {
    for raw in issues {
        let ticket = raw.into_ticket();
        if let (Some(since), Some(updated)) = (since, ticket.updated.as_deref()) {
            if updated <= since {
                return true;
            }
        }
        out.push(ticket);
    }
    false
}

/// Whether an `Http` error message marks a missing endpoint (404/410). The transport formats
/// errors as `"<status> for <url>"`, so the status is the leading token.
fn endpoint_absent(msg: &str) -> bool {
    msg.starts_with("404 ") || msg.starts_with("410 ")
}

/// Minimal percent-encoding for a query-parameter value: everything but unreserved chars.
fn percent_encode(s: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A fake transport that returns a queued body per call, recording the requested paths.
    struct FakeHttp {
        bodies: RefCell<Vec<String>>,
        paths: RefCell<Vec<String>>,
    }
    impl FakeHttp {
        fn new(bodies: Vec<&str>) -> Self {
            Self {
                bodies: RefCell::new(bodies.into_iter().rev().map(String::from).collect()),
                paths: RefCell::new(Vec::new()),
            }
        }
    }
    // RefCell is not Sync, but the test executor is single-threaded; the RefCell is only touched
    // synchronously before the async block.
    impl JiraHttp for FakeHttp {
        fn get(&self, path: &str) -> impl Future<Output = Result<String, JiraError>> + Send {
            self.paths.borrow_mut().push(path.to_string());
            let body = self
                .bodies
                .borrow_mut()
                .pop()
                .unwrap_or_else(|| "{}".into());
            async move { Ok(body) }
        }
    }

    // A search response captured from issues.jenkins.io (trimmed to two issues, one assigned).
    const JENKINS_SEARCH: &str = r#"{
      "startAt": 0, "maxResults": 50, "total": 2,
      "issues": [
        {"key": "JENKINS-68355", "fields": {
          "summary": "Getting a problem when logging",
          "issuetype": {"name": "Bug"},
          "status": {"name": "Open", "statusCategory": {"key": "new", "name": "To Do"}},
          "assignee": null,
          "created": "2022-04-27T10:40:25.000+0000",
          "updated": "2026-06-12T14:25:16.000+0000",
          "resolutiondate": null
        }},
        {"key": "JENKINS-76480", "fields": {
          "summary": "Add retry to the agent",
          "issuetype": {"name": "Improvement"},
          "status": {"name": "In Progress", "statusCategory": {"key": "indeterminate", "name": "In Progress"}},
          "assignee": {"name": "yfangsl", "displayName": "Yeh Fang"},
          "created": "2025-01-01T00:00:00.000+0000",
          "updated": "2026-06-10T09:00:00.000+0000",
          "resolutiondate": null
        }}
      ]
    }"#;

    // A synthetic Agile sprint response (the Jenkins instance gates this behind auth).
    const SPRINTS: &str = r#"{"maxResults":50,"startAt":0,"total":1,"values":[
      {"id":42,"state":"active","name":"Sprint 7","startDate":"2026-06-01T00:00:00.000Z","endDate":"2026-06-15T00:00:00.000Z"}
    ]}"#;

    #[test]
    fn maps_real_jenkins_search_json() {
        let client = JiraClient::new(FakeHttp::new(vec![JENKINS_SEARCH]));
        let tickets = tokio_block(client.search_issues("JENKINS", None)).unwrap();
        assert_eq!(tickets.len(), 2);
        assert_eq!(tickets[0].key, "JENKINS-68355");
        assert_eq!(tickets[0].issue_type, "Bug");
        assert_eq!(tickets[0].status, "Open");
        assert_eq!(tickets[0].status_category.as_deref(), Some("new"));
        assert_eq!(tickets[0].assignee, None);
        assert_eq!(tickets[1].assignee.as_deref(), Some("Yeh Fang"));
        assert_eq!(tickets[1].status_category.as_deref(), Some("indeterminate"));
    }

    // The enhanced-search (/rest/api/3/search/jql) page shape: issues + token pagination, no total.
    const JQL_PAGE_1: &str = r#"{
      "isLast": false, "nextPageToken": "TOK2",
      "issues": [
        {"key": "SCRUM-9", "fields": {"summary": "Newer", "issuetype": {"name": "Task"},
          "status": {"name": "Done", "statusCategory": {"key": "done"}},
          "updated": "2026-06-13T00:00:00.000+0000"}}
      ]
    }"#;
    const JQL_PAGE_2: &str = r#"{
      "isLast": true,
      "issues": [
        {"key": "SCRUM-8", "fields": {"summary": "Older", "issuetype": {"name": "Bug"},
          "status": {"name": "Open", "statusCategory": {"key": "new"}},
          "updated": "2026-06-12T00:00:00.000+0000"}}
      ]
    }"#;

    // A child issue carrying `fields.parent` (a team-managed-project epic / sub-task parent).
    const JQL_PARENT: &str = r#"{
      "isLast": true,
      "issues": [
        {"key": "SCRUM-2", "fields": {"summary": "A child", "issuetype": {"name": "Story"},
          "status": {"name": "Done", "statusCategory": {"key": "done"}},
          "parent": {"key": "SCRUM-1"},
          "updated": "2026-06-13T00:00:00.000+0000"}},
        {"key": "SCRUM-3", "fields": {"summary": "No parent", "issuetype": {"name": "Task"},
          "status": {"name": "Open", "statusCategory": {"key": "new"}},
          "updated": "2026-06-12T00:00:00.000+0000"}}
      ]
    }"#;

    #[test]
    fn parses_parent_key() {
        let client = JiraClient::new(FakeHttp::new(vec![JQL_PARENT]));
        let tickets = tokio_block(client.search_issues("SCRUM", None)).unwrap();
        let child = tickets.iter().find(|t| t.key == "SCRUM-2").unwrap();
        assert_eq!(child.parent_key.as_deref(), Some("SCRUM-1"));
        let orphan = tickets.iter().find(|t| t.key == "SCRUM-3").unwrap();
        assert_eq!(orphan.parent_key, None);
    }

    #[test]
    fn cloud_search_pages_by_token() {
        // Two pages: the first carries nextPageToken, the second is isLast.
        let client = JiraClient::new(FakeHttp::new(vec![JQL_PAGE_1, JQL_PAGE_2]));
        let tickets = tokio_block(client.search_issues("SCRUM", None)).unwrap();
        assert_eq!(tickets.len(), 2);
        assert_eq!(tickets[0].key, "SCRUM-9");
        assert_eq!(tickets[0].status_category.as_deref(), Some("done"));
        assert_eq!(tickets[1].key, "SCRUM-8");
        // The second request carried the page token from the first response.
        assert!(client.http.paths.borrow()[1].contains("nextPageToken=TOK2"));
    }

    #[test]
    fn legacy_search_parses_offset_shape() {
        // The Server/DC offset/total shape, exercised directly (the Cloud path is the default).
        let client = JiraClient::new(FakeHttp::new(vec![JENKINS_SEARCH]));
        let tickets = tokio_block(client.search_issues_legacy("JENKINS", None)).unwrap();
        assert_eq!(tickets.len(), 2);
        assert_eq!(tickets[1].assignee.as_deref(), Some("Yeh Fang"));
    }

    #[test]
    fn endpoint_absent_matches_404_and_410() {
        assert!(endpoint_absent("410 for https://x/rest/api/2/search"));
        assert!(endpoint_absent("404 for https://x/rest/api/2/search"));
        assert!(!endpoint_absent("401 for https://x/rest/api/2/search"));
        assert!(!endpoint_absent("500 for https://x"));
    }

    #[test]
    fn search_stops_at_the_cursor() {
        let client = JiraClient::new(FakeHttp::new(vec![JENKINS_SEARCH]));
        // Cursor newer than the second issue's updated -> only the first comes back.
        let tickets =
            tokio_block(client.search_issues("JENKINS", Some("2026-06-11T00:00:00.000+0000")))
                .unwrap();
        assert_eq!(tickets.len(), 1);
        assert_eq!(tickets[0].key, "JENKINS-68355");
    }

    #[test]
    fn maps_sprint_json() {
        let client = JiraClient::new(FakeHttp::new(vec![SPRINTS]));
        let sprints = tokio_block(client.sprints(1)).unwrap();
        assert_eq!(sprints.len(), 1);
        assert_eq!(sprints[0].name, "Sprint 7");
        assert_eq!(sprints[0].state.as_deref(), Some("active"));
    }

    // A real /rest/agile/1.0/sprint/{id}/issue response captured from your-site.atlassian.net.
    const SPRINT_ISSUES: &str = r#"{
      "maxResults": 50, "startAt": 0, "total": 2,
      "issues": [
        {"key": "SCRUM-1", "fields": {
          "summary": "Test Task",
          "issuetype": {"name": "Task"},
          "status": {"name": "In Progress", "statusCategory": {"key": "indeterminate"}}
        }},
        {"key": "SCRUM-2", "fields": {
          "summary": "Ship it",
          "issuetype": {"name": "Story"},
          "status": {"name": "Done", "statusCategory": {"key": "done"}}
        }}
      ]
    }"#;

    #[test]
    fn maps_sprint_issue_membership() {
        let client = JiraClient::new(FakeHttp::new(vec![SPRINT_ISSUES]));
        let keys = tokio_block(client.sprint_issues(1)).unwrap();
        assert_eq!(keys, vec!["SCRUM-1".to_string(), "SCRUM-2".to_string()]);
    }

    #[tokio::test]
    async fn sync_then_link_writes_work_items_and_links() {
        let store = Store::open_in_memory().await.unwrap();
        // A repo + a PR whose title carries the issue key.
        store
            .upsert_repo(&core_store::Repo {
                id: "repo:gh/acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        store
            .upsert_pull_request(&core_store::PullRequest {
                id: "pr1".into(),
                repo_id: "repo:gh/acme/widget".into(),
                number: 5,
                title: "JENKINS-68355 fix logging".into(),
                state: "open".into(),
                author_login: Some("alice".into()),
                body: None,
                created_at: "2026-06-01T00:00:00Z".into(),
                merged_at: None,
                html_url: None,
            })
            .await
            .unwrap();
        store
            .create_board(
                "board:1",
                "B",
                core_store::BoardKind::Team,
                "2026-06-01T00:00:00Z",
            )
            .await
            .unwrap();
        store
            .upsert_tracker(&core_store::Tracker {
                id: "jiraA".into(),
                name: "Jenkins".into(),
                kind: "jira".into(),
                base_url: "https://issues.jenkins.io".into(),
                email: None,
            })
            .await
            .unwrap();
        store
            .link_board_tracker("board:1", "jiraA", "JENKINS")
            .await
            .unwrap();

        let client = JiraClient::new(FakeHttp::new(vec![JENKINS_SEARCH]));
        let report = sync_project_issues(
            &client,
            &store,
            "jiraA",
            "https://issues.jenkins.io/",
            "JENKINS",
            None,
        )
        .await
        .unwrap();
        assert_eq!(report.issues, 2);
        assert_eq!(
            report.cursor.as_deref(),
            Some("2026-06-12T14:25:16.000+0000")
        );

        let n = link_keys(&store, "jiraA", &["repo:gh/acme/widget".to_string()])
            .await
            .unwrap();
        assert_eq!(
            n, 1,
            "the PR title's JENKINS-68355 links to the synced issue"
        );

        let linked = store.jira_issues_for_pr("pr1").await.unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].issue_key, "JENKINS-68355");
        assert_eq!(
            linked[0].url.as_deref(),
            Some("https://issues.jenkins.io/browse/JENKINS-68355")
        );

        let board = store.jira_issues_for_board("board:1").await.unwrap();
        assert_eq!(board.len(), 2);
        // newest-updated first
        assert_eq!(board[0].issue_key, "JENKINS-68355");
    }

    // Tiny single-thread block_on so the non-tokio unit tests can drive the async client.
    fn tokio_block<F: Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(f)
    }
}
