//! Retrospective rituals: standup agenda and sprint review.
//! Deterministic readouts of past activity over the store, scoped to a team or sprint.
//! Timestamp comparisons use lexicographic ordering of RFC-3339 Zulu strings, which is correct for
//! the normalized `...Z` timestamps the sync engine stores.

use core_store::{Store, StoreError};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum RitualError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Summary(#[from] core_summary::SummaryError),
}

/// A compact reference to a pull request for ritual output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PrRef {
    pub id: String,
    pub number: i64,
    pub title: String,
    /// The forge web URL of the PR. `None` if the forge did not report one.
    pub url: Option<String>,
}

impl From<&core_store::PullRequest> for PrRef {
    fn from(p: &core_store::PullRequest) -> Self {
        Self {
            id: p.id.clone(),
            number: p.number,
            title: p.title.clone(),
            url: p.html_url.clone(),
        }
    }
}

/// A reference to a tracked issue on the standup: the work-item view of a GitHub issue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IssueRef {
    pub id: String,
    pub number: i64,
    pub title: String,
}

impl From<&core_store::Issue> for IssueRef {
    fn from(i: &core_store::Issue) -> Self {
        Self {
            id: i.id.clone(),
            number: i.number,
            title: i.title.clone(),
        }
    }
}

/// A standup agenda for a repo: what moved in the `since..until` window, what is waiting on
/// review, and what needs attention now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StandupAgenda {
    pub repo_id: String,
    pub since: String,
    pub until: String,
    /// Commits committed within `since..until`.
    pub moved_commits: usize,
    /// PRs merged within `since..until`.
    pub merged_prs: Vec<PrRef>,
    /// Open PRs with no review yet (waiting on a reviewer).
    pub waiting_on_review: Vec<PrRef>,
    /// Issues that moved to done within `since..until`.
    pub delivered_issues: Vec<IssueRef>,
    /// Open issues with a linked open PR.
    pub in_progress_issues: Vec<IssueRef>,
    /// What needs attention as of `now`: stale PRs, merged-without-review, failing CI,
    /// done-not-done.
    pub needs_attention: Vec<core_summary::AttentionItem>,
}

/// Build the standup agenda from activity. The half-open window `since..until` bounds "what
/// moved"; `now` (RFC-3339) anchors the attention watchdogs (e.g. PR staleness). For an open-ended
/// window, pass `now` as `until`. `bands` are the board's WIP aging bands, computed once by the
/// caller across the board's repos; pass `WipAgingBands::default()` with no board context.
pub async fn standup(
    store: &Store,
    repo_id: &str,
    since: &str,
    until: &str,
    now: &str,
    bands: &core_summary::WipAgingBands,
) -> Result<StandupAgenda, RitualError> {
    let in_window = |at: &str| at >= since && at < until;
    let commits = store.commits_for_repo(repo_id).await?;
    let moved_commits = commits
        .iter()
        .filter(|c| in_window(c.committed_at.as_str()))
        .count();

    let prs = store.pull_requests(repo_id).await?;
    let reviews = store.reviews_for_repo(repo_id).await?;
    let reviewed: std::collections::HashSet<&str> =
        reviews.iter().map(|r| r.pr_id.as_str()).collect();

    let merged_prs = prs
        .iter()
        .filter(|p| p.merged_at.as_deref().map(&in_window).unwrap_or(false))
        .map(PrRef::from)
        .collect();

    let waiting_on_review = prs
        .iter()
        .filter(|p| p.state == "open" && p.merged_at.is_none() && !reviewed.contains(p.id.as_str()))
        .map(PrRef::from)
        .collect();

    // Delivered: closed in the window. In progress: open issues whose work item is in progress
    // (an open PR links them).
    let issues = store.issues_for_repo(repo_id).await?;
    let delivered_issues = issues
        .iter()
        .filter(|i| i.state == "closed" && i.closed_at.as_deref().map(&in_window).unwrap_or(false))
        .map(IssueRef::from)
        .collect();
    let mut in_progress_issues = Vec::new();
    for issue in issues.iter().filter(|i| i.state == "open") {
        if let Some(wi) = store.work_item(&issue.id).await? {
            if wi.status_category.as_deref() == Some("indeterminate") {
                in_progress_issues.push(IssueRef::from(issue));
            }
        }
    }

    let needs_attention = core_summary::attention_enriched(store, repo_id, now, bands).await?;

    Ok(StandupAgenda {
        repo_id: repo_id.to_string(),
        since: since.to_string(),
        until: until.to_string(),
        moved_commits,
        merged_prs,
        waiting_on_review,
        delivered_issues,
        in_progress_issues,
        needs_attention,
    })
}

/// Render a repo's standup agenda as markdown: attention items (with their next-step action),
/// merged, waiting on review, delivered and in-progress issues. A repo with no content renders the
/// empty string. Producing it performs no egress.
pub fn render_agenda_markdown(agenda: &StandupAgenda, repo_label: &str) -> String {
    let has_content = agenda.moved_commits > 0
        || !agenda.merged_prs.is_empty()
        || !agenda.waiting_on_review.is_empty()
        || !agenda.needs_attention.is_empty()
        || !agenda.delivered_issues.is_empty()
        || !agenda.in_progress_issues.is_empty();
    if !has_content {
        return String::new();
    }
    let mut out = format!("## {repo_label}\n\n");

    if !agenda.needs_attention.is_empty() {
        out.push_str(&format!(
            "**Needs attention ({})**\n",
            agenda.needs_attention.len()
        ));
        // The WIP-aging band is left out of the copy-paste digest: a bare "> P75" has no stated
        // basis there. It travels on the item for the UI.
        for a in &agenda.needs_attention {
            if a.action.is_empty() {
                out.push_str(&format!("- {}\n", a.summary));
            } else {
                out.push_str(&format!("- {} -> {}\n", a.summary, a.action));
            }
        }
        out.push('\n');
    }
    if !agenda.merged_prs.is_empty() {
        out.push_str(&format!(
            "**Merged ({}), {} commit(s)**\n",
            agenda.merged_prs.len(),
            agenda.moved_commits
        ));
        for p in &agenda.merged_prs {
            out.push_str(&format!("- #{} {}\n", p.number, p.title));
        }
        out.push('\n');
    }
    if !agenda.waiting_on_review.is_empty() {
        out.push_str(&format!(
            "**Waiting on review ({})**\n",
            agenda.waiting_on_review.len()
        ));
        for p in &agenda.waiting_on_review {
            out.push_str(&format!("- #{} {}\n", p.number, p.title));
        }
        out.push('\n');
    }
    if !agenda.delivered_issues.is_empty() {
        out.push_str(&format!(
            "**Delivered ({})**\n",
            agenda.delivered_issues.len()
        ));
        for i in &agenda.delivered_issues {
            out.push_str(&format!("- #{} {}\n", i.number, i.title));
        }
        out.push('\n');
    }
    if !agenda.in_progress_issues.is_empty() {
        out.push_str(&format!(
            "**In progress ({})**\n",
            agenda.in_progress_issues.len()
        ));
        for i in &agenda.in_progress_issues {
            out.push_str(&format!("- #{} {}\n", i.number, i.title));
        }
        out.push('\n');
    }
    out
}

/// A shipped work item: the item and the merged PRs that delivered it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShippedItem {
    pub work_item_id: String,
    pub title: String,
    pub merged_pr_ids: Vec<String>,
}

/// What shipped this sprint: the sprint's work items whose linked PRs merged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SprintReview {
    pub sprint_id: String,
    pub shipped: Vec<ShippedItem>,
}

/// Build the sprint review by walking the sprint's work items to their merged linked PRs. An item
/// with no merged linked PR is not shipped.
pub async fn sprint_review(store: &Store, sprint_id: &str) -> Result<SprintReview, RitualError> {
    let items = store.work_items_for_sprint(sprint_id).await?;
    let mut shipped = Vec::new();
    for item in items {
        let links = store.links_from("work_item", &item.id).await?;
        let mut merged_pr_ids = Vec::new();
        for link in links.iter().filter(|l| l.dst_kind == "pull_request") {
            if let Some(pr) = store.pull_request(&link.dst_id).await? {
                if pr.merged_at.is_some() {
                    merged_pr_ids.push(pr.id);
                }
            }
        }
        if !merged_pr_ids.is_empty() {
            shipped.push(ShippedItem {
                work_item_id: item.id,
                title: item.title,
                merged_pr_ids,
            });
        }
    }
    Ok(SprintReview {
        sprint_id: sprint_id.to_string(),
        shipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_store::{Commit, Issue, PullRequest, Repo, Review, Sprint, WorkItem};

    fn repo() -> Repo {
        Repo {
            id: "repo:acme/widget".into(),
            owner: "acme".into(),
            name: "widget".into(),
            full_name: "acme/widget".into(),
            ownership: "owned".into(),
        }
    }

    fn pr(id: &str, state: &str, merged_at: Option<&str>) -> PullRequest {
        PullRequest {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number: 1,
            title: format!("PR {id}"),
            state: state.into(),
            author_login: Some("octocat".into()),
            body: None,
            created_at: "2026-06-01T00:00:00Z".into(),
            merged_at: merged_at.map(Into::into),
            html_url: None,
        }
    }

    #[tokio::test]
    async fn standup_reports_moved_and_waiting() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo()).await.unwrap();
        // one commit before `since`, one after
        for (sha, at) in [
            ("old", "2026-06-01T00:00:00Z"),
            ("new", "2026-06-10T00:00:00Z"),
        ] {
            store
                .upsert_commit(&Commit {
                    sha: sha.into(),
                    repo_id: "repo:acme/widget".into(),
                    author_login: None,
                    message: "m".into(),
                    committed_at: at.into(),
                })
                .await
                .unwrap();
        }
        // one merged after `since`, one open with no review
        store
            .upsert_pull_request(&pr("merged", "closed", Some("2026-06-09T00:00:00Z")))
            .await
            .unwrap();
        store
            .upsert_pull_request(&pr("waiting", "open", None))
            .await
            .unwrap();

        let agenda = standup(
            &store,
            "repo:acme/widget",
            "2026-06-05T00:00:00Z",
            "2026-06-20T00:00:00Z",
            "2026-06-20T00:00:00Z",
            &Default::default(),
        )
        .await
        .unwrap();
        assert_eq!(agenda.moved_commits, 1);
        assert_eq!(agenda.merged_prs.len(), 1);
        assert_eq!(agenda.merged_prs[0].id, "merged");
        assert_eq!(agenda.waiting_on_review.len(), 1);
        assert_eq!(agenda.waiting_on_review[0].id, "waiting");
        // The open PR is stale and the merged one had no review.
        let kinds: std::collections::HashSet<&str> = agenda
            .needs_attention
            .iter()
            .map(|i| i.kind.as_str())
            .collect();
        assert!(kinds.contains("stale_pr"));
        assert!(kinds.contains("merged_without_review"));
    }

    #[tokio::test]
    async fn render_agenda_markdown_composes_sections_with_actions() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo()).await.unwrap();
        // A stale open PR (-> needs attention with an action) + a merged PR in the window.
        store
            .upsert_pull_request(&pr("stale", "open", None))
            .await
            .unwrap();
        store
            .upsert_pull_request(&pr("merged", "closed", Some("2026-06-10T00:00:00Z")))
            .await
            .unwrap();
        let agenda = standup(
            &store,
            "repo:acme/widget",
            "2026-06-05T00:00:00Z",
            "2026-06-20T00:00:00Z",
            "2026-06-20T00:00:00Z",
            &Default::default(),
        )
        .await
        .unwrap();

        let md = render_agenda_markdown(&agenda, "acme/widget");
        assert!(md.starts_with("## acme/widget\n"));
        assert!(md.contains("**Needs attention"));
        // Every needs-attention line carries its envelope action.
        for a in &agenda.needs_attention {
            assert!(
                md.contains(&a.action),
                "the action `{}` must appear in the digest",
                a.action
            );
        }
        assert!(md.contains("**Merged ("));

        // A quiet repo renders nothing.
        let empty = StandupAgenda {
            repo_id: "r".into(),
            since: "s".into(),
            until: "u".into(),
            moved_commits: 0,
            merged_prs: vec![],
            waiting_on_review: vec![],
            delivered_issues: vec![],
            in_progress_issues: vec![],
            needs_attention: vec![],
        };
        assert_eq!(render_agenda_markdown(&empty, "quiet/repo"), "");
    }

    #[tokio::test]
    async fn standup_reports_delivered_and_in_progress_issues() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo()).await.unwrap();
        let issue = |id: &str, number: i64, state: &str, closed_at: Option<&str>| Issue {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number,
            title: format!("issue {number}"),
            state: state.into(),
            author_login: None,
            body: None,
            created_at: "2026-06-01T00:00:00Z".into(),
            closed_at: closed_at.map(Into::into),
            labels: String::new(),
            html_url: None,
        };
        // delivered: closed within the window. in-progress: open with an indeterminate work item.
        store
            .upsert_issue(&issue("i1", 1, "closed", Some("2026-06-10T00:00:00Z")))
            .await
            .unwrap();
        store
            .upsert_issue(&issue("i2", 2, "open", None))
            .await
            .unwrap();
        store
            .upsert_github_work_item("i2", "issue 2", "issue", "open", "indeterminate")
            .await
            .unwrap();

        let agenda = standup(
            &store,
            "repo:acme/widget",
            "2026-06-05T00:00:00Z",
            "2026-06-20T00:00:00Z",
            "2026-06-20T00:00:00Z",
            &Default::default(),
        )
        .await
        .unwrap();
        assert_eq!(agenda.delivered_issues.len(), 1);
        assert_eq!(agenda.delivered_issues[0].number, 1);
        assert_eq!(agenda.in_progress_issues.len(), 1);
        assert_eq!(agenda.in_progress_issues[0].number, 2);
    }

    #[tokio::test]
    async fn standup_excludes_reviewed_prs_from_waiting() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo()).await.unwrap();
        store
            .upsert_pull_request(&pr("reviewed", "open", None))
            .await
            .unwrap();
        store
            .upsert_review(&Review {
                id: "rv1".into(),
                pr_id: "reviewed".into(),
                reviewer_login: Some("alice".into()),
                state: "approved".into(),
                submitted_at: Some("2026-06-02T00:00:00Z".into()),
            })
            .await
            .unwrap();
        let agenda = standup(
            &store,
            "repo:acme/widget",
            "2026-06-01T00:00:00Z",
            "2026-06-20T00:00:00Z",
            "2026-06-20T00:00:00Z",
            &Default::default(),
        )
        .await
        .unwrap();
        assert!(agenda.waiting_on_review.is_empty());
    }

    #[tokio::test]
    async fn standup_window_bounds_what_moved_at_both_ends() {
        // Two PRs merged a week apart; a "last week" window includes the older only.
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo()).await.unwrap();
        store
            .upsert_pull_request(&pr("lastweek", "closed", Some("2026-06-08T00:00:00Z")))
            .await
            .unwrap();
        store
            .upsert_pull_request(&pr("thisweek", "closed", Some("2026-06-16T00:00:00Z")))
            .await
            .unwrap();

        // Window = [2026-06-06, 2026-06-13): catches lastweek, not thisweek.
        let last = standup(
            &store,
            "repo:acme/widget",
            "2026-06-06T00:00:00Z",
            "2026-06-13T00:00:00Z",
            "2026-06-20T00:00:00Z",
            &Default::default(),
        )
        .await
        .unwrap();
        let last_ids: Vec<&str> = last.merged_prs.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(last_ids, vec!["lastweek"]);

        // Window = [2026-06-13, 2026-06-20): catches thisweek, not lastweek.
        let this = standup(
            &store,
            "repo:acme/widget",
            "2026-06-13T00:00:00Z",
            "2026-06-20T00:00:00Z",
            "2026-06-20T00:00:00Z",
            &Default::default(),
        )
        .await
        .unwrap();
        let this_ids: Vec<&str> = this.merged_prs.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(this_ids, vec!["thisweek"]);
    }

    #[tokio::test]
    async fn sprint_review_lists_only_shipped_items() {
        let store = Store::open_in_memory().await.unwrap();
        store.upsert_repo(&repo()).await.unwrap();
        store
            .add_sprint(&Sprint {
                id: "s1".into(),
                team_id: None,
                name: "Sprint 1".into(),
                starts_on: None,
                ends_on: None,
                source: "manual".into(),
            })
            .await
            .unwrap();
        for id in ["wi-shipped", "wi-open"] {
            store
                .add_work_item(&WorkItem {
                    id: id.into(),
                    title: format!("item {id}"),
                    kind: "story".into(),
                    state: "open".into(),
                    source: "manual".into(),
                    status_category: None,
                })
                .await
                .unwrap();
            store.add_sprint_work_item("s1", id).await.unwrap();
        }
        store
            .upsert_pull_request(&pr("p-merged", "closed", Some("2026-06-03T00:00:00Z")))
            .await
            .unwrap();
        store
            .upsert_pull_request(&pr("p-open", "open", None))
            .await
            .unwrap();
        store
            .add_link(
                "work_item",
                "wi-shipped",
                "pull_request",
                "p-merged",
                "tracks",
            )
            .await
            .unwrap();
        store
            .add_link("work_item", "wi-open", "pull_request", "p-open", "tracks")
            .await
            .unwrap();

        let review = sprint_review(&store, "s1").await.unwrap();
        assert_eq!(review.shipped.len(), 1);
        assert_eq!(review.shipped[0].work_item_id, "wi-shipped");
        assert_eq!(
            review.shipped[0].merged_pr_ids,
            vec!["p-merged".to_string()]
        );
    }
}
