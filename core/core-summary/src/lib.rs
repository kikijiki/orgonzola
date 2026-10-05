//! Precomputed digests, assembled deterministically from the store and `core-metrics`.
//! Useful as structured data with the LLM disabled; the prose line comes from a `Summarizer`
//! whose default uses no LLM. Digests are stored via `core-store`.

use core_model::{AttentionEntity, LinkedEntity};
use core_store::{MetricSnapshot, Store, StoreError};
use serde::{Deserialize, Serialize};

/// Fallback staleness threshold (days) if settings cannot be read.
const DEFAULT_STALE_DAYS: i64 = 7;

/// The configured stale-open-PR threshold in days, or the default if settings are unreadable.
async fn stale_threshold(store: &Store) -> i64 {
    store
        .settings()
        .await
        .map(|s| s.stale_pr_days)
        .unwrap_or(DEFAULT_STALE_DAYS)
}

#[derive(Debug, thiserror::Error)]
pub enum SummaryError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// A per-scope digest: structured facts plus a prose line derived from them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Digest {
    pub scope_kind: String,
    pub scope_id: String,
    pub generated_at: String,
    pub median_cycle_time_secs: Option<i64>,
    /// Cycle-time phase medians: pickup = created -> first review, review = first review -> merge.
    pub median_pickup_secs: Option<i64>,
    pub median_review_secs: Option<i64>,
    pub wip: usize,
    pub stale_open_prs: usize,
    pub merged_without_review: usize,
    /// Open, unmerged PRs still waiting for a first review past the review-wait threshold.
    pub review_wait: usize,
    pub review_concentration: f64,
    /// Human-readable summary derived from the facts above.
    pub prose: String,
}

/// Days an open PR may wait for its first review before it counts as "review-wait". Shorter
/// than the stale-PR threshold.
const REVIEW_WAIT_DAYS: i64 = 2;

/// Days a work item may sit in progress before it counts as `aging_wip`. Longer than the
/// stale-PR threshold.
const WIP_AGING_DAYS: i64 = 14;

/// Days a file may go without a merged change before touching it counts as a `risky_change`.
/// About 6 months.
const DORMANT_DAYS: i64 = 180;

/// How many dormant files a `risky_change` item carries as evidence entities. The summary
/// still states the true total; the kept files are the longest-dormant ones.
const MAX_DORMANT_EVIDENCE: usize = 5;

/// Turns the structured facts of a digest into prose. The default impl uses no LLM.
/// `Send + Sync` so a digest can be built inside an async task that must be `Send`.
pub trait Summarizer: Send + Sync {
    fn summarize(&self, digest: &Digest) -> String;
}

/// The default no-LLM summarizer: a templated line from the facts.
pub struct RuleSummarizer;

impl Summarizer for RuleSummarizer {
    fn summarize(&self, d: &Digest) -> String {
        let cycle = match d.median_cycle_time_secs {
            Some(s) => format!("{} h median cycle time", s / 3600),
            None => "no merged PRs yet".to_string(),
        };
        format!(
            "{} {}: {} PR(s) in flight, {}, {} stale, {} merged without review.",
            d.scope_kind, d.scope_id, d.wip, cycle, d.stale_open_prs, d.merged_without_review
        )
    }
}

/// Assemble a repo digest from stored activity at time `now` (RFC-3339), default summarizer.
pub async fn repo_digest(store: &Store, repo_id: &str, now: &str) -> Result<Digest, SummaryError> {
    repo_digest_with(store, repo_id, now, &RuleSummarizer).await
}

/// As [`repo_digest`], but with a caller-supplied summarizer (e.g. a future LLM one).
pub async fn repo_digest_with(
    store: &Store,
    repo_id: &str,
    now: &str,
    summarizer: &dyn Summarizer,
) -> Result<Digest, SummaryError> {
    let prs = store.pull_requests(repo_id).await?;
    let reviews = store.reviews_for_repo(repo_id).await?;

    let phases = core_metrics::median_cycle_phases(&prs, &reviews);
    let mut digest = Digest {
        scope_kind: "repo".to_string(),
        scope_id: repo_id.to_string(),
        generated_at: now.to_string(),
        median_cycle_time_secs: core_metrics::median_cycle_time_secs(&prs),
        median_pickup_secs: phases.pickup_secs,
        median_review_secs: phases.review_secs,
        wip: core_metrics::wip_count(&prs),
        stale_open_prs: core_metrics::stale_open_prs(&prs, now, stale_threshold(store).await).len(),
        merged_without_review: core_metrics::merged_without_review(&prs, &reviews).0.len(),
        review_wait: core_metrics::review_wait_prs(&prs, &reviews, now, REVIEW_WAIT_DAYS * 86_400)
            .len(),
        review_concentration: core_metrics::review_concentration(&reviews),
        prose: String::new(),
    };
    digest.prose = summarizer.summarize(&digest);
    Ok(digest)
}

/// Store a digest so it serves instantly later.
pub async fn save_digest(store: &Store, digest: &Digest) -> Result<(), SummaryError> {
    let body = serde_json::to_string(digest)?;
    store
        .save_digest(
            &digest.scope_kind,
            &digest.scope_id,
            &digest.generated_at,
            &body,
        )
        .await?;
    Ok(())
}

/// One issue a change links to, with how (closes | mentions).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkedIssue {
    pub id: String,
    pub number: i64,
    pub title: String,
    pub relation: String,
}

/// One file a change touched: the diff behind the digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangedFile {
    pub filename: String,
    pub status: String,
    pub additions: i64,
    pub deletions: i64,
    pub patch: Option<String>,
}

/// A per-change (per-PR) digest: the PR's facts, body, review state, and linked issues,
/// composed from the store. The prose line uses no LLM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChangeDigest {
    pub pr_id: String,
    pub number: i64,
    pub title: String,
    pub state: String,
    pub author_login: Option<String>,
    pub body: Option<String>,
    pub review_count: usize,
    pub approved: bool,
    pub linked_issues: Vec<LinkedIssue>,
    /// The files the PR changed, with their patches. Empty if none are stored.
    pub files: Vec<ChangedFile>,
    /// Total additions / deletions across the changed files.
    pub additions: i64,
    pub deletions: i64,
    pub prose: String,
}

/// Assemble a change digest for `pr_id`, or `None` if no such PR.
pub async fn change_digest(
    store: &Store,
    pr_id: &str,
) -> Result<Option<ChangeDigest>, SummaryError> {
    let Some(pr) = store.pull_request(pr_id).await? else {
        return Ok(None);
    };
    let reviews = store.reviews_for_pr(pr_id).await?;
    let approved = reviews
        .iter()
        .any(|r| r.state.eq_ignore_ascii_case("APPROVED"));

    // Resolve the PR's work-item links to stored issues in the same repo. A GitHub issue's
    // work-item id is the issue id.
    let links = store.links_from("pull_request", pr_id).await?;
    let issues = store.issues_for_repo(&pr.repo_id).await?;
    let linked_issues: Vec<LinkedIssue> = links
        .iter()
        .filter(|l| l.dst_kind == "work_item")
        .filter_map(|l| {
            issues
                .iter()
                .find(|i| i.id == l.dst_id)
                .map(|i| LinkedIssue {
                    id: i.id.clone(),
                    number: i.number,
                    title: i.title.clone(),
                    relation: l.relation.clone(),
                })
        })
        .collect();

    let files: Vec<ChangedFile> = store
        .pr_files(pr_id)
        .await?
        .into_iter()
        .map(|f| ChangedFile {
            filename: f.filename,
            status: f.status,
            additions: f.additions,
            deletions: f.deletions,
            patch: f.patch,
        })
        .collect();
    let additions = files.iter().map(|f| f.additions).sum();
    let deletions = files.iter().map(|f| f.deletions).sum();

    let closes: Vec<i64> = linked_issues
        .iter()
        .filter(|l| l.relation == "closes")
        .map(|l| l.number)
        .collect();
    let prose = format!(
        "PR #{} \"{}\" ({}){}{}, {} review(s).",
        pr.number,
        pr.title,
        pr.state,
        if approved { ", approved" } else { "" },
        if closes.is_empty() {
            String::new()
        } else {
            format!(
                ", closes {}",
                closes
                    .iter()
                    .map(|n| format!("#{n}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        },
        reviews.len(),
    );

    Ok(Some(ChangeDigest {
        pr_id: pr.id,
        number: pr.number,
        title: pr.title,
        state: pr.state,
        author_login: pr.author_login,
        body: pr.body,
        review_count: reviews.len(),
        approved,
        linked_issues,
        files,
        additions,
        deletions,
        prose,
    }))
}

/// One thing that needs a manager's attention, surfaced by the deterministic watchdogs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionItem {
    /// The signal kind, e.g. "stale_pr" | "review_wait" | "merged_without_review" | "failing_ci" |
    /// "flaky_ci" | "risky_change" | "aging_wip" | "done_not_done" | "orphan_pr".
    pub kind: String,
    /// The typed envelope: what this item points at. Exactly one entity has the `subject` role,
    /// the flagged thing and the item's identity. `actor` names who the next step points at;
    /// `evidence` carries the fact that makes the flag true. Each entity carries its own forge URL
    /// where the forge reported one.
    pub entities: Vec<LinkedEntity>,
    /// The human one-liner. Always rendered, even for an item with no URL; also read by the
    /// markdown digest and the assistant's grounding.
    pub summary: String,
    /// When the flagged thing happened: the PR's open time (stale), its merge time (merged
    /// without review), or the run's completion (failing CI). `None` if the forge reported no
    /// timestamp.
    pub ts: Option<String>,
    /// The mechanical next step: a deterministic one-line imperative keyed to `kind`, computed
    /// without the LLM. Empty when there is no sensible action.
    pub action: String,
    /// For `aging_wip` items: which band of the board's in-progress-time distribution the item
    /// falls into, "over_p50" | "over_p75" | "over_p90". `None` with too little completion
    /// history or when the age is inside P50. Never set for other signal kinds.
    pub wip_percentile: Option<String>,
    /// The one-line basis for `wip_percentile`: what the distribution measures, its window, and
    /// how many completed items it is built from. Set when `wip_percentile` is set.
    pub wip_percentile_basis: Option<String>,
}

impl AttentionItem {
    /// The flagged thing: the item's one `subject` entity. `None` only if the envelope is
    /// malformed.
    pub fn subject(&self) -> Option<&AttentionEntity> {
        core_model::subject_of(&self.entities)
    }

    /// The people the next step points at, by login. Empty when the forge reported no author.
    pub fn actor_logins(&self) -> impl Iterator<Item = &str> {
        self.entities
            .iter()
            .filter_map(|e| match (&e.role, &e.target) {
                (core_model::AttentionRole::Actor, AttentionEntity::Person { login }) => {
                    Some(login.as_str())
                }
                _ => None,
            })
    }
}

/// Shorten a commit sha for a summary sentence (7 hex chars). `str::get` is a checked slice,
/// so a sha shorter than 7 bytes, or one where byte 7 splits a multi-byte character, passes
/// through unchanged instead of panicking.
fn short_sha(sha: &str) -> &str {
    const SHORT_LEN: usize = 7;
    sha.get(..SHORT_LEN).unwrap_or(sha)
}

/// The typed entity for a CI run, carrying the run's own forge URL.
fn ci_run_entity(run: &core_store::CiRun) -> AttentionEntity {
    AttentionEntity::CiRun {
        id: run.id.clone(),
        commit_sha: run.commit_sha.clone(),
        attempt: run.run_attempt,
        url: run.html_url.clone(),
    }
}

/// Build the entity list for a pull-request-subject item: the PR, plus its author when the
/// forge reported one.
fn pr_entities(pr: &core_store::PullRequest) -> Vec<LinkedEntity> {
    let mut out = vec![LinkedEntity::subject(AttentionEntity::PullRequest {
        id: pr.id.clone(),
        number: pr.number,
        title: pr.title.clone(),
        url: pr.html_url.clone(),
    })];
    if let Some(login) = pr.author_login.clone() {
        out.push(LinkedEntity::actor(AttentionEntity::Person { login }));
    }
    out
}

/// The board's WIP aging bands: percentiles of the active time (entered in progress -> done)
/// of the work items it completed recently. `sample` is how many completed items they are
/// built from.
/// Active time, not lead time: an in-progress item's age is measured from when it entered
/// `indeterminate`, so the band must measure the same span. A percentile is `None` when the
/// sample is too small (see `MIN_BAND_HISTORY` and `MIN_TAIL_HISTORY`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WipAgingBands {
    pub p50_secs: Option<i64>,
    pub p75_secs: Option<i64>,
    pub p90_secs: Option<i64>,
    pub sample: usize,
}

/// Minimum completed work items before any band is reported.
const MIN_BAND_HISTORY: usize = 8;

/// Minimum completed work items before the P75/P90 tail bands are reported. Nearest-rank puts
/// P90 at the maximum until n = 10 and leaves one observation above it until n = 20. Between
/// the two thresholds only P50 is reported.
const MIN_TAIL_HISTORY: usize = 20;

/// How far back the band history reaches, in days. Matches the dashboard flow card's default
/// window so the two baselines agree.
pub const BAND_WINDOW_DAYS: i64 = 30;

/// Compute the board's WIP aging bands from recent work-item completions: percentiles of
/// `indeterminate -> done` durations for items that reached `done` within `BAND_WINDOW_DAYS`
/// of `now`. All-None when fewer than `MIN_BAND_HISTORY` such items exist.
pub async fn board_wip_aging_bands(
    store: &Store,
    repo_ids: &[String],
    now: &str,
) -> Result<WipAgingBands, SummaryError> {
    let window_secs = BAND_WINDOW_DAYS * 86_400;
    let mut actives: Vec<i64> = Vec::new();
    for repo_id in repo_ids {
        for issue in store.issues_for_repo(repo_id).await? {
            let history = store.work_item_status_history(&issue.id).await?;
            let entered = |status: &str| {
                history
                    .iter()
                    .find(|(k, _)| k == status)
                    .map(|(_, at)| at.as_str())
            };
            let (Some(ip_at), Some(done_at)) = (entered("indeterminate"), entered("done")) else {
                continue;
            };
            // Recency window, so a team that got faster is not judged against an old pace.
            let Some(completed_ago) = core_metrics::secs_between(done_at, now) else {
                continue;
            };
            if !(0..=window_secs).contains(&completed_ago) {
                continue;
            }
            if let Some(secs) = core_metrics::secs_between(ip_at, done_at) {
                if secs >= 0 {
                    actives.push(secs);
                }
            }
        }
    }
    let sample = actives.len();
    if sample < MIN_BAND_HISTORY {
        return Ok(WipAgingBands::default());
    }
    let tail = sample >= MIN_TAIL_HISTORY;
    Ok(WipAgingBands {
        p50_secs: core_metrics::percentile(actives.clone(), 0.50),
        p75_secs: tail
            .then(|| core_metrics::percentile(actives.clone(), 0.75))
            .flatten(),
        p90_secs: tail
            .then(|| core_metrics::percentile(actives.clone(), 0.90))
            .flatten(),
        sample,
    })
}

/// Classify an in-progress age against the board's bands: the machine label plus the
/// sentence stating what the band is measured over. `None` when there is no usable history or
/// the age is inside P50.
fn wip_band(bands: &WipAgingBands, age_secs: i64) -> Option<(String, String)> {
    let p50 = bands.p50_secs?;
    let (label, pct) = if bands.p90_secs.is_some_and(|p| age_secs > p) {
        ("over_p90", "P90")
    } else if bands.p75_secs.is_some_and(|p| age_secs > p) {
        ("over_p75", "P75")
    } else if age_secs > p50 {
        ("over_p50", "P50")
    } else {
        return None;
    };
    let basis = format!(
        "Older than the {pct} of in-progress time (entered in progress -> done) across the {} work items \
         completed in the last {} days.",
        bands.sample, BAND_WINDOW_DAYS
    );
    Some((label.to_string(), basis))
}

/// The deterministic mechanical next step for an attention `kind`. Empty string when there is none.
pub fn attention_action(kind: &str) -> &'static str {
    match kind {
        "review_wait" => "Assign a reviewer.",
        "aging_wip" => "Check if it is blocked, or split it.",
        "stale_pr" => "Nudge a reviewer, or close it.",
        "merged_without_review" => "Review it post-merge, or confirm it was trivial.",
        "failing_ci" => "Open the failing run.",
        "flaky_ci" => "Quarantine or fix the flaky test; the suite passed only after a re-run.",
        "done_not_done" => "Reopen the issue, or confirm the PR closes it.",
        "orphan_pr" => "Link the PR to its issue, or confirm it is untracked work.",
        "risky_change" => "Review carefully - it changes long-stable code.",
        "upstream" => "Check the upstream alert.",
        _ => "",
    }
}

/// What needs attention in a repo at time `now` (RFC-3339): the union of the watchdog signals
/// as a flat list. Deterministic, no LLM. Aging-WIP items carry no percentile band from this
/// call. Callers that render items use `attention_enriched`; this one is for callers that only
/// count them, avoiding a full completion-history scan.
pub async fn attention(
    store: &Store,
    repo_id: &str,
    now: &str,
) -> Result<Vec<AttentionItem>, SummaryError> {
    attention_impl(store, repo_id, now, &WipAgingBands::default()).await
}

/// As `attention`, but labels `aging_wip` items with their band and its basis. `bands` comes
/// from `board_wip_aging_bands`; pass `WipAgingBands::default()` to opt out.
pub async fn attention_enriched(
    store: &Store,
    repo_id: &str,
    now: &str,
    bands: &WipAgingBands,
) -> Result<Vec<AttentionItem>, SummaryError> {
    attention_impl(store, repo_id, now, bands).await
}

async fn attention_impl(
    store: &Store,
    repo_id: &str,
    now: &str,
    bands: &WipAgingBands,
) -> Result<Vec<AttentionItem>, SummaryError> {
    let prs = store.pull_requests(repo_id).await?;
    let reviews = store.reviews_for_repo(repo_id).await?;
    let runs = store.ci_runs_for_repo(repo_id).await?;
    let issues = store.issues_for_repo(repo_id).await?;
    let threshold = stale_threshold(store).await;
    // Risky-change basis: when each file was last changed by a merged PR. A file absent from
    // this map has no merged history, so it is new, not dormant.
    let last_changed: std::collections::HashMap<String, String> = store
        .repo_file_last_changed(repo_id)
        .await?
        .into_iter()
        .collect();

    // Build an item from its typed envelope, stamping the next-step action. The subject entity
    // carries the id and the forge URL.
    let item = |kind: &str, entities: Vec<LinkedEntity>, summary: String, ts: Option<String>| {
        AttentionItem {
            kind: kind.to_string(),
            entities,
            summary,
            ts,
            action: attention_action(kind).to_string(),
            wip_percentile: None,
            wip_percentile_basis: None,
        }
    };

    let mut items = Vec::new();
    for pr in core_metrics::stale_open_prs(&prs, now, threshold) {
        items.push(item(
            "stale_pr",
            pr_entities(pr),
            format!("PR #{} \"{}\" is open and stale", pr.number, pr.title),
            Some(pr.created_at.clone()),
        ));
    }
    // review_wait: open with no first review past the review-wait threshold. Fires before
    // stale_pr; an old unreviewed PR raises both.
    for pr in core_metrics::review_wait_prs(&prs, &reviews, now, REVIEW_WAIT_DAYS * 86_400) {
        items.push(item(
            "review_wait",
            pr_entities(pr),
            format!(
                "PR #{} \"{}\" is waiting for its first review",
                pr.number, pr.title
            ),
            Some(pr.created_at.clone()),
        ));
    }
    // risky_change: an open PR whose changed files were last merged past the dormancy threshold.
    // A file with no merged history is new, not dormant.
    let dormant_secs = DORMANT_DAYS * 86_400;
    for pr in prs
        .iter()
        .filter(|p| p.state == "open" && p.merged_at.is_none())
    {
        let files = store.pr_files(&pr.id).await?;
        let mut dormant: Vec<(&str, i64)> = files
            .iter()
            .filter_map(|f| {
                let age = core_metrics::secs_between(last_changed.get(&f.filename)?, now)?;
                (age > dormant_secs).then_some((f.filename.as_str(), age))
            })
            .collect();
        if dormant.is_empty() {
            continue;
        }
        // Longest-dormant file first, so the summary and the evidence entities share an order.
        dormant.sort_by_key(|f| std::cmp::Reverse(f.1));
        let (worst_file, worst_age) = dormant[0];
        let mut entities = pr_entities(pr);
        // Cap the evidence; the summary keeps stating the true total.
        entities.extend(dormant.iter().take(MAX_DORMANT_EVIDENCE).map(|(name, _)| {
            LinkedEntity::evidence(AttentionEntity::SourceFile {
                path: (*name).to_string(),
                last_changed_at: last_changed.get(*name).cloned(),
            })
        }));
        items.push(item(
            "risky_change",
            entities,
            format!(
                "PR #{} \"{}\" changes {} long-stable file(s) (e.g. {}, last changed {}d ago)",
                pr.number,
                pr.title,
                dormant.len(),
                worst_file,
                worst_age / 86_400
            ),
            Some(pr.created_at.clone()),
        ));
    }
    for pr in core_metrics::merged_without_review(&prs, &reviews).0 {
        items.push(item(
            "merged_without_review",
            pr_entities(pr),
            format!(
                "PR #{} \"{}\" was merged without a review",
                pr.number, pr.title
            ),
            pr.merged_at.clone(),
        ));
    }
    for run in core_metrics::failing_ci_runs(&runs) {
        items.push(item(
            "failing_ci",
            vec![LinkedEntity::subject(ci_run_entity(run))],
            format!(
                "CI run failed{}",
                run.commit_sha
                    .as_deref()
                    .map(|s| format!(" on {}", short_sha(s)))
                    .unwrap_or_default()
            ),
            run.completed_at.clone(),
        ));
    }
    // flaky_ci: green only after a re-run on the same commit (attempt > 1), so the failure was
    // non-deterministic. The attempt number is named in the summary.
    for run in core_metrics::flaky_ci_runs(&runs) {
        items.push(item(
            "flaky_ci",
            vec![LinkedEntity::subject(ci_run_entity(run))],
            format!(
                "CI passed only after a re-run{}{}",
                run.commit_sha
                    .as_deref()
                    .map(|s| format!(" on {}", short_sha(s)))
                    .unwrap_or_default(),
                run.run_attempt
                    .map(|a| format!(" (attempt {a})"))
                    .unwrap_or_default(),
            ),
            run.completed_at.clone(),
        ));
    }
    // Work-item-graph signals over merged PRs, one link lookup per PR:
    // - done_not_done: a closes-linked work item is still open.
    // - orphan_pr: the merged PR has no work-item link at all.
    // The linker (`core_sync::link_references`) runs as one pass over the whole repo, so a repo it
    // has never touched has zero `pull_request -> work_item` links (an org board never runs it).
    // That must not read as "checked, none linked", so the repo-wide presence of any link is
    // checked once up front. While it is false, orphan_pr does not fire and the count is reported
    // by `attention_coverage`.
    let has_link_data = store.repo_has_work_item_links(repo_id).await?;
    let issue_by_id: std::collections::HashMap<&str, &core_store::Issue> =
        issues.iter().map(|i| (i.id.as_str(), i)).collect();
    for pr in prs.iter().filter(|p| p.merged_at.is_some()) {
        let links = store.links_from("pull_request", &pr.id).await?;
        let work_links: Vec<_> = links.iter().filter(|l| l.dst_kind == "work_item").collect();
        if work_links.is_empty() {
            if !has_link_data {
                continue;
            }
            items.push(item(
                "orphan_pr",
                pr_entities(pr),
                format!(
                    "PR #{} \"{}\" merged with no tracked work item",
                    pr.number, pr.title
                ),
                pr.merged_at.clone(),
            ));
            continue;
        }
        for link in work_links.iter().filter(|l| l.relation == "closes") {
            let Some(wi) = store.work_item(&link.dst_id).await? else {
                continue;
            };
            if wi.status_category.as_deref() == Some("done") {
                continue;
            }
            // The still-open work item is the evidence for this flag.
            let open_issue = issue_by_id.get(link.dst_id.as_str()).copied();
            let issue_ref = open_issue
                .map(|i| format!("#{}", i.number))
                .unwrap_or_else(|| wi.title.clone());
            let mut entities = pr_entities(pr);
            entities.push(LinkedEntity::evidence(AttentionEntity::WorkItem {
                id: wi.id.clone(),
                number: open_issue.map(|i| i.number).unwrap_or_default(),
                title: wi.title.clone(),
                url: open_issue.and_then(|i| i.html_url.clone()),
            }));
            items.push(item(
                "done_not_done",
                entities,
                format!(
                    "PR #{} \"{}\" merged but {} is still open",
                    pr.number, pr.title, issue_ref
                ),
                pr.merged_at.clone(),
            ));
        }
    }
    // aging_wip: a work item in progress (indeterminate) longer than the WIP-aging threshold,
    // from the derived status history. The threshold decides what is flagged; when band history
    // is available the item is also labeled with where its age sits in the board's distribution.
    let aging_secs = WIP_AGING_DAYS * 86_400;
    for issue in issues.iter().filter(|i| i.state == "open") {
        let Some(wi) = store.work_item(&issue.id).await? else {
            continue;
        };
        if wi.status_category.as_deref() != Some("indeterminate") {
            continue;
        }
        let history = store.work_item_status_history(&issue.id).await?;
        let Some((_, entered)) = history.iter().find(|(k, _)| k == "indeterminate") else {
            continue;
        };
        let Some(age) = core_metrics::secs_between(entered, now) else {
            continue;
        };
        if age <= aging_secs {
            continue;
        }
        let (wip_percentile, wip_percentile_basis) = match wip_band(bands, age) {
            Some((label, basis)) => (Some(label), Some(basis)),
            None => (None, None),
        };
        items.push(AttentionItem {
            kind: "aging_wip".to_string(),
            // No actor: the store has the issue's author, not its assignee.
            entities: vec![LinkedEntity::subject(AttentionEntity::WorkItem {
                id: issue.id.clone(),
                number: issue.number,
                title: issue.title.clone(),
                url: issue.html_url.clone(),
            })],
            summary: format!(
                "Issue #{} \"{}\" has been in progress {} days",
                issue.number,
                issue.title,
                age / 86_400
            ),
            ts: Some(entered.clone()),
            action: attention_action("aging_wip").to_string(),
            wip_percentile,
            wip_percentile_basis,
        });
    }
    Ok(items)
}

/// The evidence shortfall behind two watchdogs that can turn absent data into a finding:
/// `merged_without_review` and `orphan_pr`. Either can read as "checked, and bad" when the
/// truth is "never checked": a merged PR older than the synced review window, or a repo where
/// the linker has never run. It travels alongside the item list, not on the items.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionCoverage {
    /// See `core_metrics::MergedWithoutReviewCoverage`.
    pub merged_without_review_unjudged: usize,
    /// Merged PRs this repo could not judge for a tracked work item, because it has no
    /// `pull_request -> work_item` link data synced at all. 0 whenever the repo has any link data.
    pub orphan_pr_unjudged: usize,
}

/// Computes `AttentionCoverage` for one repo at `now`, from the same store rows
/// `attention_enriched` reads for these two watchdogs.
pub async fn attention_coverage(
    store: &Store,
    repo_id: &str,
) -> Result<AttentionCoverage, SummaryError> {
    let prs = store.pull_requests(repo_id).await?;
    let reviews = store.reviews_for_repo(repo_id).await?;
    let (_, mwr) = core_metrics::merged_without_review(&prs, &reviews);
    let has_link_data = store.repo_has_work_item_links(repo_id).await?;
    let orphan_pr_unjudged = if has_link_data {
        0
    } else {
        prs.iter().filter(|p| p.merged_at.is_some()).count()
    };
    Ok(AttentionCoverage {
        merged_without_review_unjudged: mwr.unjudged,
        orphan_pr_unjudged,
    })
}

/// How well a board's merged PRs are linked to the work-item graph. `linked` counts merged PRs
/// with any work-item link; `closes` counts those with a `closes` edge (vs a weaker
/// body-parsed `mentions`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkCoverage {
    pub merged_total: usize,
    pub linked: usize,
    pub closes: usize,
}

/// Compute the work-item link coverage across a board's repos. A board with no merged PRs
/// reports zeros.
pub async fn board_link_coverage(
    store: &Store,
    repo_ids: &[String],
) -> Result<LinkCoverage, SummaryError> {
    let mut cov = LinkCoverage {
        merged_total: 0,
        linked: 0,
        closes: 0,
    };
    for repo_id in repo_ids {
        for pr in store.pull_requests(repo_id).await? {
            if pr.merged_at.is_none() {
                continue;
            }
            cov.merged_total += 1;
            let links = store.links_from("pull_request", &pr.id).await?;
            let work_links: Vec<_> = links.iter().filter(|l| l.dst_kind == "work_item").collect();
            if !work_links.is_empty() {
                cov.linked += 1;
            }
            if work_links.iter().any(|l| l.relation == "closes") {
                cov.closes += 1;
            }
        }
    }
    Ok(cov)
}

/// Work-item flow over a board, computed from the derived `work_item_status_history`. Empty on
/// a board with no linked GitHub issues. `lead_time` is new -> done for items that reached
/// done in the window; `in_progress_age` is now - entered-in-progress for items still in
/// progress. Medians, not means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowMetrics {
    /// Work items that reached done within the window.
    pub completed: usize,
    pub median_lead_time_secs: Option<i64>,
    /// The cycle decomposition over completed items that passed through in progress: wait =
    /// new -> in progress, active = in progress -> done. `None` when no completed item had an
    /// in-progress entry.
    pub median_wait_secs: Option<i64>,
    pub median_active_secs: Option<i64>,
    /// Work items currently in progress (entered indeterminate, no done yet).
    pub in_progress: usize,
    pub median_in_progress_age_secs: Option<i64>,
}

/// Compute a board's work-item flow from the status history.
pub async fn board_flow_metrics(
    store: &Store,
    repo_ids: &[String],
    since: &str,
    until: &str,
    now: &str,
) -> Result<FlowMetrics, SummaryError> {
    let mut leads = Vec::new();
    let mut waits = Vec::new();
    let mut actives = Vec::new();
    let mut ages = Vec::new();
    for repo_id in repo_ids {
        for issue in store.issues_for_repo(repo_id).await? {
            let history = store.work_item_status_history(&issue.id).await?;
            let entered = |status: &str| {
                history
                    .iter()
                    .find(|(k, _)| k == status)
                    .map(|(_, at)| at.as_str())
            };
            // Completed in the window: a `done` in [since, until); lead = done - new.
            if let Some(done_at) = entered("done") {
                if done_at >= since && done_at < until {
                    if let Some(new_at) = entered("new") {
                        if let Some(secs) = core_metrics::secs_between(new_at, done_at) {
                            leads.push(secs);
                        }
                        // Cycle decomposition for items that passed through in progress.
                        if let Some(ip_at) = entered("indeterminate") {
                            if let Some(w) = core_metrics::secs_between(new_at, ip_at) {
                                waits.push(w);
                            }
                            if let Some(a) = core_metrics::secs_between(ip_at, done_at) {
                                actives.push(a);
                            }
                        }
                    }
                }
            } else if let Some(in_progress_at) = entered("indeterminate") {
                // Still in progress: age = now - entered indeterminate.
                if let Some(secs) = core_metrics::secs_between(in_progress_at, now) {
                    ages.push(secs);
                }
            }
        }
    }
    let completed = leads.len();
    let in_progress = ages.len();
    Ok(FlowMetrics {
        completed,
        median_lead_time_secs: core_metrics::median(leads),
        median_wait_secs: core_metrics::median(waits),
        median_active_secs: core_metrics::median(actives),
        in_progress,
        median_in_progress_age_secs: core_metrics::median(ages),
    })
}

/// A board's PR delivery-cycle phase medians: pickup = PR open -> first review, review =
/// first review -> merge. A true median over all the board's PRs, not a median of medians.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardCyclePhases {
    pub median_pickup_secs: Option<i64>,
    pub median_review_secs: Option<i64>,
}

/// Compute the board's PR cycle phase medians over all its PRs and reviews.
pub async fn board_cycle_phases(
    store: &Store,
    repo_ids: &[String],
) -> Result<BoardCyclePhases, SummaryError> {
    let mut prs = Vec::new();
    let mut reviews = Vec::new();
    for repo_id in repo_ids {
        prs.extend(store.pull_requests(repo_id).await?);
        reviews.extend(store.reviews_for_repo(repo_id).await?);
    }
    let phases = core_metrics::median_cycle_phases(&prs, &reviews);
    Ok(BoardCyclePhases {
        median_pickup_secs: phases.pickup_secs,
        median_review_secs: phases.review_secs,
    })
}

/// Bug inflow vs outflow over a board in a window: bug-labeled issues opened vs closed.
/// `opened > closed` is a rework early-warning. Empty when no issue carries a bug-like label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BugFlow {
    pub opened: usize,
    pub closed: usize,
}

/// True if any of the issue's newline-joined labels reads like a bug label (case-insensitive).
fn is_bug(labels: &str) -> bool {
    labels
        .split('\n')
        .any(|l| l.to_ascii_lowercase().contains("bug"))
}

/// Count bug-labeled issues opened (created) and closed within `[since, until)` across a board.
pub async fn board_bug_flow(
    store: &Store,
    repo_ids: &[String],
    since: &str,
    until: &str,
) -> Result<BugFlow, SummaryError> {
    let in_win = |ts: &str| ts >= since && ts < until;
    let mut flow = BugFlow {
        opened: 0,
        closed: 0,
    };
    for repo_id in repo_ids {
        for issue in store.issues_for_repo(repo_id).await? {
            if !is_bug(&issue.labels) {
                continue;
            }
            if in_win(&issue.created_at) {
                flow.opened += 1;
            }
            if issue.state == "closed" && issue.closed_at.as_deref().is_some_and(in_win) {
                flow.closed += 1;
            }
        }
    }
    Ok(flow)
}

/// A sprint's say-do: what was committed at the start vs what was delivered, plus scope churn
/// after commit. Count-based (no story points). `ratio` is `None` when nothing was committed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SprintSayDo {
    pub sprint_id: String,
    pub sprint_name: String,
    /// "active" | "closed" | "future".
    pub state: Option<String>,
    /// The date the commitment snapshot was frozen (first seen active), `None` if not yet.
    pub committed_on: Option<String>,
    /// Items in the frozen start-of-sprint set.
    pub committed: usize,
    /// Committed items whose status category is now `done`.
    pub delivered: usize,
    /// Committed items not yet done (`committed - delivered`).
    pub carryover: usize,
    /// Items in the live set that were not committed (scope added after commit).
    pub added: usize,
    /// Committed items no longer in the live set (pulled out after commit).
    pub removed: usize,
    /// `delivered / committed`, or `None` when nothing was committed.
    pub ratio: Option<f64>,
}

/// Compute say-do for one sprint: join the frozen commitment against current statuses and
/// live membership. A sprint with no snapshot reports `committed = 0` and `ratio = None`.
pub async fn sprint_say_do(
    store: &Store,
    sprint: &core_store::JiraSprintRow,
) -> Result<SprintSayDo, SummaryError> {
    use std::collections::BTreeSet;
    let committed_items = store.sprint_committed_items(&sprint.sprint_id).await?;
    let live: BTreeSet<String> = store
        .work_items_for_sprint(&sprint.sprint_id)
        .await?
        .into_iter()
        .map(|w| w.id)
        .collect();
    let committed_ids: BTreeSet<String> = committed_items.iter().map(|w| w.id.clone()).collect();
    let committed = committed_items.len();
    let delivered = committed_items
        .iter()
        .filter(|w| w.status_category.as_deref() == Some("done"))
        .count();
    let added = live.difference(&committed_ids).count();
    let removed = committed_ids.difference(&live).count();
    let ratio = (committed > 0).then(|| delivered as f64 / committed as f64);
    Ok(SprintSayDo {
        sprint_id: sprint.sprint_id.clone(),
        sprint_name: sprint.name.clone(),
        state: sprint.state.clone(),
        committed_on: sprint.committed_at.clone(),
        committed,
        delivered,
        carryover: committed - delivered,
        added,
        removed,
        ratio,
    })
}

/// Say-do for every Jira sprint a board watches, newest first. Sprints with no commitment
/// snapshot are included with `committed = 0` and `ratio = None`.
pub async fn board_say_do(store: &Store, board_id: &str) -> Result<Vec<SprintSayDo>, SummaryError> {
    let mut out = Vec::new();
    for sprint in store.jira_sprints_for_board(board_id).await? {
        out.push(sprint_say_do(store, &sprint).await?);
    }
    Ok(out)
}

/// Derive the velocity trend from say-do results: sprints with a frozen commitment and a
/// measurable ratio, oldest first. Ordered by `committed_on`, not by the store's order:
/// `board_say_do` is `starts_on DESC` and a sprint with no start date sorts last there, so a
/// plain reverse would plot it as the oldest point.
pub fn velocity_trend(sprints: &[SprintSayDo]) -> Vec<&SprintSayDo> {
    let mut with_commit: Vec<&SprintSayDo> = sprints
        .iter()
        .filter(|s| s.committed_on.is_some() && s.ratio.is_some())
        .collect();
    with_commit.sort_by(|a, b| a.committed_on.cmp(&b.committed_on));
    with_commit
}

/// One day of a board's flow/quality trend, read from the board-scoped daily snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowTrendPoint {
    pub captured_on: String,
    pub lead_time_secs: Option<i64>,
    pub wip: i64,
    pub bug_net: i64,
}

/// A board's flow and bug trend over the recorded daily snapshots: lead time, WIP, and bug
/// net (opened - closed) per day. Reads the `board`-scoped metric series.
pub async fn board_flow_trend(
    store: &Store,
    board_id: &str,
) -> Result<Vec<FlowTrendPoint>, SummaryError> {
    use std::collections::BTreeMap;
    let mut by_day: BTreeMap<String, FlowTrendPoint> = BTreeMap::new();
    let point = |day: &str| FlowTrendPoint {
        captured_on: day.to_string(),
        lead_time_secs: None,
        wip: 0,
        bug_net: 0,
    };
    for (day, v) in store
        .metric_series("board", board_id, "flow_lead_secs")
        .await?
    {
        by_day
            .entry(day.clone())
            .or_insert_with(|| point(&day))
            .lead_time_secs = Some(v);
    }
    for (day, v) in store.metric_series("board", board_id, "flow_wip").await? {
        by_day.entry(day.clone()).or_insert_with(|| point(&day)).wip = v;
    }
    for (day, v) in store.metric_series("board", board_id, "bug_net").await? {
        by_day
            .entry(day.clone())
            .or_insert_with(|| point(&day))
            .bug_net = v;
    }
    Ok(by_day.into_values().collect())
}

/// A flagged pull request with the columns a per-type attention table shows: the watchdog
/// `kind` plus the PR's own facts and its review tally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionPr {
    /// "stale_pr" | "merged_without_review".
    pub kind: String,
    pub id: String,
    pub number: i64,
    pub title: String,
    pub state: String,
    pub author_login: Option<String>,
    pub created_at: String,
    pub merged_at: Option<String>,
    pub review_count: usize,
    pub approved: bool,
    /// The PR's forge web URL, so the UI can open it. `None` if the forge reported none.
    pub url: Option<String>,
}

/// A flagged CI run with the columns a per-type attention table shows. `kind` is always
/// "failing_ci", so it is not repeated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionCi {
    pub id: String,
    pub commit_sha: Option<String>,
    pub status: String,
    pub conclusion: Option<String>,
    pub completed_at: Option<String>,
    /// The run's forge web URL, so the UI can open it. `None` if the forge reported none.
    pub url: Option<String>,
}

/// A repo's flagged items grouped by type for the attention detail tables: the same signals
/// as `attention`, as typed rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AttentionDetail {
    pub prs: Vec<AttentionPr>,
    pub ci: Vec<AttentionCi>,
}

/// The detailed, per-type form of `attention`: flagged PRs (stale and merged-without-review,
/// tagged with which) and failing CI runs. Same rules and `now` semantics as `attention`.
pub async fn attention_detail(
    store: &Store,
    repo_id: &str,
    now: &str,
) -> Result<AttentionDetail, SummaryError> {
    let prs = store.pull_requests(repo_id).await?;
    let reviews = store.reviews_for_repo(repo_id).await?;
    let runs = store.ci_runs_for_repo(repo_id).await?;
    let threshold = stale_threshold(store).await;

    // Review tally per PR (count + whether any review approved).
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut approved: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for r in &reviews {
        *counts.entry(r.pr_id.as_str()).or_default() += 1;
        if r.state.eq_ignore_ascii_case("APPROVED") {
            approved.insert(r.pr_id.as_str());
        }
    }
    let row = |pr: &core_store::PullRequest, kind: &str| AttentionPr {
        kind: kind.to_string(),
        id: pr.id.clone(),
        number: pr.number,
        title: pr.title.clone(),
        state: pr.state.clone(),
        author_login: pr.author_login.clone(),
        created_at: pr.created_at.clone(),
        merged_at: pr.merged_at.clone(),
        review_count: counts.get(pr.id.as_str()).copied().unwrap_or(0),
        approved: approved.contains(pr.id.as_str()),
        url: pr.html_url.clone(),
    };

    let mut out_prs = Vec::new();
    for pr in core_metrics::stale_open_prs(&prs, now, threshold) {
        out_prs.push(row(pr, "stale_pr"));
    }
    for pr in core_metrics::merged_without_review(&prs, &reviews).0 {
        out_prs.push(row(pr, "merged_without_review"));
    }
    let ci = core_metrics::failing_ci_runs(&runs)
        .into_iter()
        .map(|r| AttentionCi {
            id: r.id.clone(),
            commit_sha: r.commit_sha.clone(),
            status: r.status.clone(),
            conclusion: r.conclusion.clone(),
            completed_at: r.completed_at.clone(),
            url: r.html_url.clone(),
        })
        .collect();
    Ok(AttentionDetail { prs: out_prs, ci })
}

/// A board-level attention item: a per-repo `AttentionItem` plus its repo and whether it
/// involves one of the board's people (`by_team`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardAttentionItem {
    pub repo_id: String,
    pub kind: String,
    /// The item's typed envelope, carried through from the per-repo item.
    pub entities: Vec<LinkedEntity>,
    pub summary: String,
    /// True when one of the item's `actor` entities is one of the board's people.
    pub by_team: bool,
    /// When the flagged thing happened, from `AttentionItem.ts`. `None` if the forge reported no
    /// timestamp.
    pub ts: Option<String>,
    /// For `aging_wip` items: the band of the board's in-progress-time distribution. From
    /// `AttentionItem.wip_percentile`.
    pub wip_percentile: Option<String>,
    /// The basis sentence for `wip_percentile`, from `AttentionItem.wip_percentile_basis`.
    pub wip_percentile_basis: Option<String>,
}

/// Attention rolled up across a board's repos: each repo's items tagged with the source repo
/// and a `by_team` flag for items whose actor is one of `people` (may be empty). Labels
/// `aging_wip` items with the board's in-progress-time percentile bands.
pub async fn board_attention(
    store: &Store,
    repo_ids: &[String],
    people: &[String],
    now: &str,
) -> Result<Vec<BoardAttentionItem>, SummaryError> {
    let bands = board_wip_aging_bands(store, repo_ids, now).await?;
    let team: std::collections::HashSet<&str> = people.iter().map(String::as_str).collect();
    let mut out = Vec::new();
    for repo_id in repo_ids {
        for item in attention_impl(store, repo_id, now, &bands).await? {
            let by_team = !team.is_empty() && item.actor_logins().any(|login| team.contains(login));
            out.push(BoardAttentionItem {
                repo_id: repo_id.clone(),
                kind: item.kind,
                entities: item.entities,
                summary: item.summary,
                by_team,
                ts: item.ts,
                wip_percentile: item.wip_percentile,
                wip_percentile_basis: item.wip_percentile_basis,
            });
        }
    }
    Ok(out)
}

/// A human label for an attention kind, for prose. Mirrors the UI's `attentionStyle` labels.
fn attention_label(kind: &str) -> &str {
    match kind {
        "failing_ci" => "failing CI",
        "flaky_ci" => "flaky CI",
        "review_wait" => "awaiting review",
        "stale_pr" => "stale PR",
        "aging_wip" => "aging work in progress",
        "risky_change" => "risky change",
        "merged_without_review" => "merged without review",
        "done_not_done" => "merged but issue open",
        "orphan_pr" => "merged, untracked",
        _ => kind,
    }
}

/// Severity order (most urgent first). Shared by the count brief and the LLM focus input.
const ATTENTION_ORDER: [&str; 9] = [
    "failing_ci",
    "review_wait",
    "stale_pr",
    "aging_wip",
    "risky_change",
    "merged_without_review",
    "flaky_ci",
    "done_not_done",
    "orphan_pr",
];

/// Where `kind` sits in `ATTENTION_ORDER`: lower is more urgent. An unknown kind sorts after
/// every known one. Exposed so a payload boundary can keep the same priority order.
pub fn attention_severity_rank(kind: &str) -> usize {
    ATTENTION_ORDER
        .iter()
        .position(|k| *k == kind)
        .unwrap_or(ATTENTION_ORDER.len())
}

/// The present kinds with their counts in severity order, then unknown kinds alphabetically.
fn attention_counts(items: &[BoardAttentionItem]) -> Vec<(&str, usize)> {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for i in items {
        *counts.entry(i.kind.as_str()).or_default() += 1;
    }
    let mut present: Vec<(&str, usize)> = ATTENTION_ORDER
        .iter()
        .filter_map(|k| counts.get(k).map(|&n| (*k, n)))
        .collect();
    for (&k, &n) in &counts {
        if !ATTENTION_ORDER.contains(&k) {
            present.push((k, n));
        }
    }
    present
}

pub fn attention_brief(items: &[BoardAttentionItem]) -> String {
    if items.is_empty() {
        return "Nothing needs attention right now.".to_string();
    }
    let present = attention_counts(items);
    let parts: Vec<String> = present
        .iter()
        .map(|(k, n)| format!("{n} {}", attention_label(k)))
        .collect();
    let mut brief = format!(
        "{} item(s) need attention: {}.",
        items.len(),
        parts.join(", ")
    );
    if let Some((lead, _)) = present.first() {
        let action = attention_action(lead);
        if !action.is_empty() {
            brief.push_str(&format!(
                " Start with the {}: {action}",
                attention_label(lead)
            ));
        }
    }
    brief
}

/// How long ago `ts` was, relative to `now`, as a short phrase for the narrator input. `None`
/// when `ts` is absent, fails to parse, or is after `now`.
fn age_phrase(ts: Option<&str>, now: &str) -> Option<String> {
    let secs = core_metrics::secs_between(ts?, now)?;
    if secs < 0 {
        return None;
    }
    let days = secs / 86_400;
    if days >= 1 {
        return Some(format!("{days}d old"));
    }
    let hours = secs / 3_600;
    if hours >= 1 {
        return Some(format!("{hours}h old"));
    }
    Some("under 1h old".to_string())
}

/// A richer narration input for the optional LLM narrator: the per-kind counts, then the most
/// pressing items as ranked facts. Unlike [`attention_brief`], this hands the model concrete
/// items with the kind and age that rank them. An item with no timestamp gets no age phrase.
/// Items are in severity order, team items first within a kind, capped at `max`, and rendered
/// as one flowing line with no leading `- ` per item.
pub fn attention_focus_input(items: &[BoardAttentionItem], max: usize, now: &str) -> String {
    if items.is_empty() {
        return "Nothing needs attention right now.".to_string();
    }
    let present = attention_counts(items);
    let parts: Vec<String> = present
        .iter()
        .map(|(k, n)| format!("{n} {}", attention_label(k)))
        .collect();
    let mut ranked: Vec<String> = Vec::new();
    let mut shown = 0;
    'kinds: for (kind, _) in &present {
        // Team items first within a kind.
        for team_first in [true, false] {
            for it in items
                .iter()
                .filter(|i| i.kind == *kind && i.by_team == team_first)
            {
                if shown >= max {
                    break 'kinds;
                }
                // Keep each line short so the prompt stays small.
                let s = it.summary.trim();
                let s: String = if s.chars().count() > 160 {
                    format!("{}...", s.chars().take(157).collect::<String>())
                } else {
                    s.to_string()
                };
                shown += 1;
                let age = age_phrase(it.ts.as_deref(), now)
                    .map(|a| format!(", {a}"))
                    .unwrap_or_default();
                let team = if it.by_team { ", your team" } else { "" };
                ranked.push(format!(
                    "{shown}. {s} [{}{age}{team}]",
                    attention_label(kind)
                ));
            }
        }
    }
    format!(
        "{} item(s) need attention: {}.\nRanked most severe first (rank 1 = act on first), with age \
where known: {}",
        items.len(),
        parts.join(", "),
        ranked.join("; ")
    )
}

/// A board-wide pull request row for the Changes grid: a PR with its repo and review tally.
/// Not a digest; just the columns the list needs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardPr {
    pub repo_id: String,
    pub id: String,
    pub number: i64,
    pub title: String,
    pub state: String,
    pub author_login: Option<String>,
    pub created_at: String,
    pub merged_at: Option<String>,
    pub review_count: usize,
    pub approved: bool,
    /// The PR's forge web URL. `None` if the forge reported none.
    pub url: Option<String>,
}

/// Every pull request across a board's repos, each tagged with its repo and review tally,
/// for the Changes grid. `repo_ids` are the board's effective repos (pinned, else discovered).
pub async fn board_pull_requests(
    store: &Store,
    repo_ids: &[String],
) -> Result<Vec<BoardPr>, SummaryError> {
    let mut out = Vec::new();
    for repo_id in repo_ids {
        let prs = store.pull_requests(repo_id).await?;
        let reviews = store.reviews_for_repo(repo_id).await?;
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        let mut approved: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for r in &reviews {
            *counts.entry(r.pr_id.as_str()).or_default() += 1;
            if r.state.eq_ignore_ascii_case("APPROVED") {
                approved.insert(r.pr_id.as_str());
            }
        }
        for pr in prs {
            out.push(BoardPr {
                review_count: counts.get(pr.id.as_str()).copied().unwrap_or(0),
                approved: approved.contains(pr.id.as_str()),
                repo_id: repo_id.clone(),
                id: pr.id,
                number: pr.number,
                title: pr.title,
                state: pr.state,
                author_login: pr.author_login,
                created_at: pr.created_at,
                merged_at: pr.merged_at,
                url: pr.html_url,
            });
        }
    }
    Ok(out)
}

/// A pull request a person has open, for the per-person activity view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersonPr {
    pub repo_id: String,
    pub number: i64,
    pub title: String,
}

/// One person's activity across a board's repos: what is in flight and the counts behind it.
/// Descriptive only, no ranking or score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersonActivity {
    pub login: String,
    /// The person's open PRs across the repos (what is in flight).
    pub open_prs: Vec<PersonPr>,
    /// How many of the person's PRs have merged.
    pub merged_prs: usize,
    /// Reviews the person submitted.
    pub reviews_given: usize,
    /// Commits the person authored.
    pub commits: usize,
    /// Which of the repos the person appears in at all.
    pub repos: Vec<String>,
}

/// Compose one person's activity across `repo_ids` from the stored data, filtered by `login`.
/// An unknown login yields empty/zeroes.
pub async fn person_activity(
    store: &Store,
    repo_ids: &[String],
    login: &str,
) -> Result<PersonActivity, SummaryError> {
    let mut open_prs = Vec::new();
    let mut merged_prs = 0;
    let mut reviews_given = 0;
    let mut commits = 0;
    let mut repos = std::collections::BTreeSet::new();
    for repo_id in repo_ids {
        for pr in store.pull_requests(repo_id).await? {
            if pr.author_login.as_deref() == Some(login) {
                repos.insert(repo_id.clone());
                if pr.merged_at.is_some() {
                    merged_prs += 1;
                } else if pr.state == "open" {
                    open_prs.push(PersonPr {
                        repo_id: repo_id.clone(),
                        number: pr.number,
                        title: pr.title,
                    });
                }
            }
        }
        for review in store.reviews_for_repo(repo_id).await? {
            if review.reviewer_login.as_deref() == Some(login) {
                reviews_given += 1;
                repos.insert(repo_id.clone());
            }
        }
        for commit in store.commits_for_repo(repo_id).await? {
            if commit.author_login.as_deref() == Some(login) {
                commits += 1;
                repos.insert(repo_id.clone());
            }
        }
    }
    Ok(PersonActivity {
        login: login.to_string(),
        open_prs,
        merged_prs,
        reviews_given,
        commits,
        repos: repos.into_iter().collect(),
    })
}

/// One teammate's operational stats across a board's repos over a window, for the People
/// comparison. Window counts cover `[since, until)`; `open_prs` is current load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersonStats {
    pub login: String,
    pub commits: usize,
    pub prs_opened: usize,
    pub prs_merged: usize,
    pub reviews_given: usize,
    /// Open PRs the person has authored right now (work in flight).
    pub open_prs: usize,
    /// Reviews the person submitted that approved a PR.
    pub approvals_given: usize,
    /// PRs the person authored and merged in the window that had no approving review.
    pub self_merges: usize,
    /// Median time (seconds) from open to merge over the person's merged-in-window PRs. `None`
    /// when they merged nothing in the window.
    pub median_cycle_time_secs: Option<i64>,
    /// Distinct UTC calendar days in the window with a timestamped work event (commit, PR opened,
    /// review). A cadence read, not hours worked. Merges are excluded (they can be automated).
    /// Same event basis as the off-hours figures.
    pub active_days: usize,
    /// Median time (seconds) from a PR's open to this person's first review on it, over their
    /// reviews in the window. `None` when they reviewed nothing datable.
    pub median_review_latency_secs: Option<i64>,
    /// Average change size (additions + deletions) over the person's merged-in-window PRs that
    /// have synced file data. `None` when none do.
    pub avg_pr_churn: Option<i64>,
    /// Issues closed by closing links on the person's merged-in-window PRs; `0` when none.
    pub issues_closed: usize,
    /// Failing CI runs in the window whose head commit the person authored. Branch is not
    /// recorded, so a shared or flaky suite is attributed to the head-commit author.
    pub ci_failures: usize,
    /// The person's timestamped work events (commits, PRs opened, reviews) that fell off-hours
    /// (weekend or outside 07:00-20:00 UTC), and the total. UTC, no per-person timezone.
    pub off_hours_events: usize,
    pub total_events: usize,
}

/// The board's People comparison: per-person rows plus the same off-hours numbers rolled up
/// across everyone. The off-hours basis is weekend or outside 07:00-20:00 UTC, over timestamped
/// work events (commits, PRs opened, reviews; merges excluded).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardPeopleStats {
    pub people: Vec<PersonStats>,
    /// Board-wide work events that fell off-hours: the sum of the per-person figures.
    pub off_hours_events: usize,
    /// Board-wide timestamped work events in the window, the off-hours denominator.
    pub total_events: usize,
}

/// Comparison stats for each of `logins` across `repo_ids` over `[since, until)` (RFC-3339).
/// Everyone in `logins` gets a row (zeros if quiet). Deterministic; the host orders them.
pub async fn board_people_stats(
    store: &Store,
    repo_ids: &[String],
    logins: &[String],
    since: &str,
    until: &str,
) -> Result<BoardPeopleStats, SummaryError> {
    use std::collections::HashMap;
    // Accumulator per login. Pre-seeded so a person with no activity still appears.
    let mut acc: HashMap<&str, PersonStats> = logins
        .iter()
        .map(|l| {
            (
                l.as_str(),
                PersonStats {
                    login: l.clone(),
                    commits: 0,
                    prs_opened: 0,
                    prs_merged: 0,
                    reviews_given: 0,
                    open_prs: 0,
                    ci_failures: 0,
                    off_hours_events: 0,
                    total_events: 0,
                    approvals_given: 0,
                    self_merges: 0,
                    median_cycle_time_secs: None,
                    active_days: 0,
                    median_review_latency_secs: None,
                    avg_pr_churn: None,
                    issues_closed: 0,
                },
            )
        })
        .collect();
    // Team-level off-hours rollup.
    let mut team_off_hours = 0usize;
    let mut team_total = 0usize;
    // Per-login merged-in-window PRs, for the median PR cycle time after the pass.
    let mut merged_by: HashMap<String, Vec<core_store::PullRequest>> = HashMap::new();
    // Distinct UTC days a person had a work event.
    let mut active_days: HashMap<String, std::collections::HashSet<(i32, u32, u32)>> =
        HashMap::new();
    // Per (login, pr_id) earliest review latency, for the review-latency median. Keyed per PR so
    // multiple reviews on one PR count once.
    let mut review_latency: HashMap<(String, String), i64> = HashMap::new();
    // Per-login change sizes of their merged PRs that have synced file data.
    let mut churn_by: HashMap<String, Vec<i64>> = HashMap::new();
    // Apply `f` to a pre-seeded login (unknown logins are ignored: `get_mut` never inserts),
    // count the event toward off-hours figures, and record its UTC day.
    let mut bump =
        |acc: &mut HashMap<&str, PersonStats>, login: &str, ts: &str, f: fn(&mut PersonStats)| {
            if let Some(s) = acc.get_mut(login) {
                f(s);
                s.total_events += 1;
                team_total += 1;
                if is_off_hours(ts) {
                    s.off_hours_events += 1;
                    team_off_hours += 1;
                }
                if let Some((y, m, d, _)) = ymd_hour(ts) {
                    active_days
                        .entry(login.to_string())
                        .or_default()
                        .insert((y, m, d));
                }
            }
        };
    for repo_id in repo_ids {
        let prs = store.pull_requests(repo_id).await?;
        let reviews = store.reviews_for_repo(repo_id).await?;
        let commits = store.commits_for_repo(repo_id).await?;
        let runs = store.ci_runs_for_repo(repo_id).await?;
        // pr_id -> change size, only for PRs with synced files.
        let pr_churn: HashMap<String, i64> =
            store.repo_pr_churn(repo_id).await?.into_iter().collect();
        // How many issues each PR closes.
        let mut closing_count: HashMap<String, usize> = HashMap::new();
        for (pr_id, _issue) in store.repo_pr_closing_issues(repo_id).await? {
            *closing_count.entry(pr_id).or_default() += 1;
        }
        // pr_id -> open time, for review latency.
        let pr_created: HashMap<&str, &str> = prs
            .iter()
            .map(|p| (p.id.as_str(), p.created_at.as_str()))
            .collect();

        // PRs that got an approving review from anyone; a merge with none is a self-merge.
        let approved_prs: std::collections::HashSet<&str> = reviews
            .iter()
            .filter(|r| r.state.eq_ignore_ascii_case("approved"))
            .map(|r| r.pr_id.as_str())
            .collect();

        for pr in &prs {
            let Some(login) = pr.author_login.as_deref() else {
                continue;
            };
            // Open PRs are current load, not windowed.
            if pr.merged_at.is_none() && pr.state == "open" {
                if let Some(s) = acc.get_mut(login) {
                    s.open_prs += 1;
                }
            }
            if in_window(&pr.created_at, since, until) {
                bump(&mut acc, login, &pr.created_at, |s| s.prs_opened += 1);
            }
            if let Some(merged) = &pr.merged_at {
                if in_window(merged, since, until) {
                    // A merge is not a "when do they work" event; count it, skip off-hours.
                    if let Some(s) = acc.get_mut(login) {
                        s.prs_merged += 1;
                        // Merged with no approving review = a self-merge.
                        if !approved_prs.contains(pr.id.as_str()) {
                            s.self_merges += 1;
                        }
                        // Issues this merged PR closed.
                        s.issues_closed += closing_count.get(pr.id.as_str()).copied().unwrap_or(0);
                    }
                    // Change size, only when the PR's files are synced.
                    if let Some(churn) = pr_churn.get(pr.id.as_str()) {
                        if acc.contains_key(login) {
                            churn_by.entry(login.to_string()).or_default().push(*churn);
                        }
                    }
                    // Keep the merged PR for this author's cycle-time median.
                    if acc.contains_key(login) {
                        merged_by
                            .entry(login.to_string())
                            .or_default()
                            .push(pr.clone());
                    }
                }
            }
        }
        for r in &reviews {
            let Some(login) = r.reviewer_login.as_deref() else {
                continue;
            };
            if let Some(ts) = &r.submitted_at {
                if in_window(ts, since, until) {
                    let approved = r.state.eq_ignore_ascii_case("approved");
                    bump(&mut acc, login, ts, |s| s.reviews_given += 1);
                    if approved {
                        if let Some(s) = acc.get_mut(login) {
                            s.approvals_given += 1;
                        }
                    }
                    // Latency from PR open to this review; keep the smallest per (person, PR).
                    if acc.contains_key(login) {
                        if let Some(opened) = pr_created.get(r.pr_id.as_str()) {
                            if let Some(secs) = core_metrics::secs_between(opened, ts) {
                                if secs >= 0 {
                                    let key = (login.to_string(), r.pr_id.clone());
                                    let e = review_latency.entry(key).or_insert(secs);
                                    if secs < *e {
                                        *e = secs;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // sha -> author, so a failing CI run is attributed to its head commit's author.
        let mut author_of: HashMap<&str, &str> = HashMap::new();
        for c in &commits {
            if let Some(login) = c.author_login.as_deref() {
                author_of.insert(c.sha.as_str(), login);
                if in_window(&c.committed_at, since, until) {
                    bump(&mut acc, login, &c.committed_at, |s| s.commits += 1);
                }
            }
        }
        for run in core_metrics::failing_ci_runs(&runs) {
            let Some(completed) = &run.completed_at else {
                continue;
            };
            if !in_window(completed, since, until) {
                continue;
            }
            let Some(sha) = run.commit_sha.as_deref() else {
                continue;
            };
            if let Some(login) = author_of.get(sha) {
                if let Some(s) = acc.get_mut(*login) {
                    s.ci_failures += 1;
                }
            }
        }
    }
    // Each person's median PR cycle time (open -> merge) over merged-in-window PRs.
    for (login, prs) in &merged_by {
        if let Some(s) = acc.get_mut(login.as_str()) {
            s.median_cycle_time_secs = core_metrics::median_cycle_time_secs(prs);
        }
    }
    for (login, days) in &active_days {
        if let Some(s) = acc.get_mut(login.as_str()) {
            s.active_days = days.len();
        }
    }
    // Median review latency per person, over their reviewed PRs.
    let mut latencies: HashMap<&str, Vec<i64>> = HashMap::new();
    for ((login, _pr), secs) in &review_latency {
        latencies.entry(login.as_str()).or_default().push(*secs);
    }
    for (login, secs) in latencies {
        if let Some(s) = acc.get_mut(login) {
            s.median_review_latency_secs = core_metrics::median(secs);
        }
    }
    // Average change size per person over merged PRs that had file data.
    for (login, churns) in &churn_by {
        if churns.is_empty() {
            continue;
        }
        if let Some(s) = acc.get_mut(login.as_str()) {
            let sum: i64 = churns.iter().sum();
            s.avg_pr_churn = Some(sum / churns.len() as i64);
        }
    }
    // Return in the given `logins` order.
    let people = logins
        .iter()
        .filter_map(|l| acc.remove(l.as_str()))
        .collect();
    Ok(BoardPeopleStats {
        people,
        off_hours_events: team_off_hours,
        total_events: team_total,
    })
}

/// One area (directory) a person has authored changes in. `changes` is how many of their PRs
/// touched files in it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersonArea {
    pub area: String,
    pub changes: i64,
}

/// A person's pickup profile: current in-flight load, last activity, and the areas of the
/// codebase they have context in. Descriptive, not a score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersonPickup {
    pub login: String,
    /// Currently-open authored PRs (current, not windowed).
    pub open_prs: usize,
    /// Most recent of their commits, opened PRs and reviews (RFC-3339); `None` if none.
    pub last_active: Option<String>,
    /// The directories they have authored changes in, most-touched first, capped by the caller.
    pub areas: Vec<PersonArea>,
}

/// The directory an area path belongs to: everything before the last `/`, or "(root)".
fn dir_of(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[..i],
        None => "(root)",
    }
}

/// Per-person pickup profiles for a board: open-PR load, last-active timestamp, and the top
/// `area_limit` areas (directories they have authored in). Returned in `logins` order, which
/// is not a ranking. Everyone in `logins` gets a row.
pub async fn board_pickup(
    store: &Store,
    repo_ids: &[String],
    logins: &[String],
    area_limit: usize,
) -> Result<Vec<PersonPickup>, SummaryError> {
    use std::collections::{BTreeMap, HashMap};
    let team: std::collections::HashSet<&str> = logins.iter().map(String::as_str).collect();
    // Per login (only those on the board): open-PR load, last-active, directory -> changes.
    let mut open: HashMap<String, usize> = HashMap::new();
    let mut last: HashMap<String, String> = HashMap::new();
    let mut areas: HashMap<String, BTreeMap<String, i64>> = HashMap::new();
    // Keep the most recent timestamp per person (RFC-3339 is fixed-width, so string order is
    // time order).
    let mut bump_last = |login: &str, ts: &str| {
        let entry = last.entry(login.to_string()).or_default();
        if ts > entry.as_str() {
            *entry = ts.to_string();
        }
    };
    for repo_id in repo_ids {
        let prs = store.pull_requests(repo_id).await?;
        let reviews = store.reviews_for_repo(repo_id).await?;
        let commits = store.commits_for_repo(repo_id).await?;
        for pr in &prs {
            if let Some(login) = pr.author_login.as_deref().filter(|l| team.contains(l)) {
                if pr.merged_at.is_none() && pr.state == "open" {
                    *open.entry(login.to_string()).or_default() += 1;
                }
                bump_last(login, &pr.created_at);
            }
        }
        for r in &reviews {
            if let (Some(login), Some(ts)) =
                (r.reviewer_login.as_deref(), r.submitted_at.as_deref())
            {
                if team.contains(login) {
                    bump_last(login, ts);
                }
            }
        }
        for c in &commits {
            if let Some(login) = c.author_login.as_deref().filter(|l| team.contains(l)) {
                bump_last(login, &c.committed_at);
            }
        }
        for fa in store.repo_file_authorship(repo_id).await? {
            if team.contains(fa.author.as_str()) {
                *areas
                    .entry(fa.author)
                    .or_default()
                    .entry(dir_of(&fa.path).to_string())
                    .or_default() += fa.changes;
            }
        }
    }
    Ok(logins
        .iter()
        .map(|login| {
            // Top areas by changes (ties by name), capped.
            let mut by_dir: Vec<PersonArea> = areas
                .get(login)
                .map(|m| {
                    m.iter()
                        .map(|(area, &changes)| PersonArea {
                            area: area.clone(),
                            changes,
                        })
                        .collect()
                })
                .unwrap_or_default();
            by_dir.sort_by(|a, b| b.changes.cmp(&a.changes).then_with(|| a.area.cmp(&b.area)));
            by_dir.truncate(area_limit);
            PersonPickup {
                login: login.clone(),
                open_prs: open.get(login).copied().unwrap_or(0),
                last_active: last.get(login).cloned(),
                areas: by_dir,
            }
        })
        .collect())
}

/// The distinct logins active across `repo_ids` over `[since, until)`: anyone who authored a
/// commit, opened or merged a PR, or left a review, plus authors of currently-open PRs. Sorted
/// and deduplicated. Populates the People tab for boards with no configured roster.
pub async fn contributors(
    store: &Store,
    repo_ids: &[String],
    since: &str,
    until: &str,
) -> Result<Vec<String>, SummaryError> {
    use std::collections::BTreeSet;
    let mut logins: BTreeSet<String> = BTreeSet::new();
    for repo_id in repo_ids {
        for pr in store.pull_requests(repo_id).await? {
            let Some(login) = pr.author_login.as_deref() else {
                continue;
            };
            let open_now = pr.merged_at.is_none() && pr.state == "open";
            let opened = in_window(&pr.created_at, since, until);
            let merged = pr
                .merged_at
                .as_deref()
                .is_some_and(|m| in_window(m, since, until));
            if open_now || opened || merged {
                logins.insert(login.to_string());
            }
        }
        for c in store.commits_for_repo(repo_id).await? {
            if let Some(login) = c.author_login.as_deref() {
                if in_window(&c.committed_at, since, until) {
                    logins.insert(login.to_string());
                }
            }
        }
        for r in store.reviews_for_repo(repo_id).await? {
            if let Some(login) = r.reviewer_login.as_deref() {
                if r.submitted_at
                    .as_deref()
                    .is_some_and(|ts| in_window(ts, since, until))
                {
                    logins.insert(login.to_string());
                }
            }
        }
    }
    Ok(logins.into_iter().collect())
}

/// The investment category a conventional-commit PR title maps to, or `None` when the title
/// has no recognized `type:` / `type(scope):` / `type!:` prefix.
fn conventional_category(title: &str) -> Option<&'static str> {
    // Take the token before the first ':'. A conventional type is `type`, `type(scope)` or
    // `type!`, where `type` is lowercase letters; strip an optional `(scope)` and a trailing `!`.
    let head = title.split(':').next()?;
    let head = head.trim();
    let ty = head
        .split_once('(')
        .map(|(t, _)| t)
        .unwrap_or(head)
        .trim_end_matches('!');
    if ty.is_empty() || !ty.bytes().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    match ty {
        "feat" => Some("feature"),
        "fix" => Some("bug"),
        "docs" => Some("docs"),
        "test" | "tests" => Some("test"),
        "refactor" | "perf" | "chore" | "build" | "ci" | "style" | "revert" => Some("maintenance"),
        _ => None,
    }
}

/// The investment categories in display order. "other" is the unclassified bucket.
pub const INVESTMENT_CATEGORIES: [&str; 6] =
    ["feature", "bug", "maintenance", "docs", "test", "other"];

/// One category's share of the investment distribution: merged-PR count (primary) and churn
/// (additions + deletions, secondary).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvestmentBucket {
    pub category: String,
    pub count: usize,
    pub churn: i64,
}

/// Where a board's delivered effort went over a window: merged PRs grouped into investment
/// categories by count and churn, with the totals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvestmentDistribution {
    pub buckets: Vec<InvestmentBucket>,
    pub total_count: usize,
    pub total_churn: i64,
}

/// Investment distribution across a board's repos over `[since, until)`: each merged PR is
/// classified by its conventional-commit title type, then by a linked `bug` work item, then
/// "other". Board-level only.
pub async fn board_investment(
    store: &Store,
    repo_ids: &[String],
    since: &str,
    until: &str,
) -> Result<InvestmentDistribution, SummaryError> {
    use std::collections::HashMap;
    let mut count: HashMap<&str, usize> = HashMap::new();
    let mut churn: HashMap<&str, i64> = HashMap::new();
    for repo_id in repo_ids {
        let prs = store.pull_requests(repo_id).await?;
        let pr_churn: HashMap<String, i64> =
            store.repo_pr_churn(repo_id).await?.into_iter().collect();
        for pr in prs.iter().filter(|p| {
            p.merged_at
                .as_deref()
                .is_some_and(|m| in_window(m, since, until))
        }) {
            let category = match conventional_category(&pr.title) {
                Some(c) => c,
                // No title prefix: a PR closing a bug-kind work item is bug-fixing, else other.
                None => {
                    let mut bug = false;
                    for link in store.links_from("pull_request", &pr.id).await? {
                        if link.dst_kind == "work_item" {
                            if let Some(wi) = store.work_item(&link.dst_id).await? {
                                if wi.kind == "bug" {
                                    bug = true;
                                    break;
                                }
                            }
                        }
                    }
                    if bug {
                        "bug"
                    } else {
                        "other"
                    }
                }
            };
            *count.entry(category).or_default() += 1;
            *churn.entry(category).or_default() += pr_churn.get(&pr.id).copied().unwrap_or(0);
        }
    }
    let buckets: Vec<InvestmentBucket> = INVESTMENT_CATEGORIES
        .iter()
        .map(|&c| InvestmentBucket {
            category: c.to_string(),
            count: count.get(c).copied().unwrap_or(0),
            churn: churn.get(c).copied().unwrap_or(0),
        })
        .collect();
    let total_count = buckets.iter().map(|b| b.count).sum();
    let total_churn = buckets.iter().map(|b| b.churn).sum();
    Ok(InvestmentDistribution {
        buckets,
        total_count,
        total_churn,
    })
}

/// One epic/initiative's progress: how many of its children are done or in progress, and an
/// at-risk flag. Child-count based, not story points.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpicProgress {
    /// The parent (epic) issue key.
    pub key: String,
    /// The epic's title when the parent issue is synced, else the key.
    pub title: String,
    pub total: usize,
    pub done: usize,
    pub in_progress: usize,
    /// At risk: it has remaining children and none are in progress.
    pub at_risk: bool,
}

/// Epic progress for a board's Jira issues: group children by parent key and report, per epic,
/// the children total / done / in-progress and an at-risk flag. At-risk epics first, then
/// least-complete. Empty when the board has no Jira parents.
pub async fn board_epics(store: &Store, board_id: &str) -> Result<Vec<EpicProgress>, SummaryError> {
    use std::collections::{BTreeMap, HashMap};
    let issues = store.jira_issues_for_board(board_id).await?;
    let title_by_key: HashMap<&str, &str> = issues
        .iter()
        .map(|i| (i.issue_key.as_str(), i.title.as_str()))
        .collect();
    // parent key -> (total, done, in_progress)
    let mut by_parent: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    for i in &issues {
        let Some(parent) = i.parent_key.as_deref() else {
            continue;
        };
        let e = by_parent.entry(parent.to_string()).or_default();
        e.0 += 1;
        match i.status_category.as_deref() {
            Some("done") => e.1 += 1,
            Some("indeterminate") => e.2 += 1,
            _ => {}
        }
    }
    let mut epics: Vec<EpicProgress> = by_parent
        .into_iter()
        .map(|(key, (total, done, in_progress))| {
            let title = title_by_key
                .get(key.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| key.clone());
            EpicProgress {
                at_risk: total > done && in_progress == 0,
                key,
                title,
                total,
                done,
                in_progress,
            }
        })
        .collect();
    // At-risk first, then least-complete (cross-multiply the done ratio), then key.
    epics.sort_by(|a, b| {
        b.at_risk
            .cmp(&a.at_risk)
            .then_with(|| (a.done * b.total).cmp(&(b.done * a.total)))
            .then_with(|| a.key.cmp(&b.key))
    });
    Ok(epics)
}

/// When a person works, as a 7x24 heatmap: `buckets[weekday*24 + hour]` counts their commits,
/// PRs opened and reviews in `[since, until)`, weekday 0 = Sunday .. 6 = Saturday, hour 0..23
/// in UTC. Timezone-naive, so the shape is the signal, not the absolute clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkPattern {
    pub buckets: Vec<u32>,
}

/// Build [`WorkPattern`] for `login` across `repo_ids` over `[since, until)`. Merges are
/// excluded (they can be automated).
pub async fn person_work_pattern(
    store: &Store,
    repo_ids: &[String],
    login: &str,
    since: &str,
    until: &str,
) -> Result<WorkPattern, SummaryError> {
    let mut buckets = vec![0u32; 7 * 24];
    let mut tally = |ts: &str| {
        if !in_window(ts, since, until) {
            return;
        }
        if let Some((y, m, d, h)) = ymd_hour(ts) {
            let idx = (weekday(y, m, d) * 24 + h) as usize;
            if idx < buckets.len() {
                buckets[idx] += 1;
            }
        }
    };
    for repo_id in repo_ids {
        for c in store.commits_for_repo(repo_id).await? {
            if c.author_login.as_deref() == Some(login) {
                tally(&c.committed_at);
            }
        }
        for pr in store.pull_requests(repo_id).await? {
            if pr.author_login.as_deref() == Some(login) {
                tally(&pr.created_at);
            }
        }
        for r in store.reviews_for_repo(repo_id).await? {
            if r.reviewer_login.as_deref() == Some(login) {
                if let Some(ts) = &r.submitted_at {
                    tally(ts);
                }
            }
        }
    }
    Ok(WorkPattern { buckets })
}

/// `since <= ts < until` for RFC-3339 timestamps. The fixed-width format makes string order
/// chronological, so no date parsing is needed.
fn in_window(ts: &str, since: &str, until: &str) -> bool {
    ts >= since && ts < until
}

/// Pull (year, month, day, hour) out of an RFC-3339 timestamp like `2026-06-12T14:30:00Z`.
/// `None` if it does not have that fixed shape.
fn ymd_hour(ts: &str) -> Option<(i32, u32, u32, u32)> {
    let b = ts.as_bytes();
    if ts.len() < 13 || b[4] != b'-' || b[7] != b'-' || (b[10] != b'T' && b[10] != b' ') {
        return None;
    }
    let y = ts.get(0..4)?.parse().ok()?;
    let m = ts.get(5..7)?.parse().ok()?;
    let d = ts.get(8..10)?.parse().ok()?;
    let h = ts.get(11..13)?.parse().ok()?;
    // Range-check before any caller indexes a month-keyed table (`weekday` does `t[m - 1]`);
    // month 00/13 would panic. Out of range yields `None`.
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || h > 23 {
        return None;
    }
    Some((y, m, d, h))
}

/// Day of week for a Gregorian date via Sakamoto's algorithm: 0 = Sunday .. 6 = Saturday.
fn weekday(y: i32, m: u32, d: u32) -> u32 {
    let t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let yy = if m < 3 { y - 1 } else { y };
    let w = (yy + yy / 4 - yy / 100 + yy / 400 + t[(m - 1) as usize] + d as i32) % 7;
    ((w + 7) % 7) as u32
}

/// Whether an RFC-3339 timestamp falls outside normal working hours (UTC): a weekend, or
/// before 07:00 or at/after 20:00. A rough cut.
fn is_off_hours(ts: &str) -> bool {
    match ymd_hour(ts) {
        Some((y, m, d, h)) => {
            let wd = weekday(y, m, d);
            wd == 0 || wd == 6 || !(7..20).contains(&h)
        }
        None => false,
    }
}

/// Record a metric snapshot for `scope_id` dated by `now` (RFC-3339; the snapshot day is its
/// date part). Composes the digest and attention count and upserts one row for the day. The
/// host supplies `now` so the core stays clock-free.
pub async fn record_snapshot(
    store: &Store,
    scope_kind: &str,
    scope_id: &str,
    now: &str,
) -> Result<(), SummaryError> {
    let digest = repo_digest(store, scope_id, now).await?;
    // Only the count is used, so take the band-free path to skip a full completion-history scan.
    let attention_count = attention(store, scope_id, now).await?.len();
    let captured_on = now.get(..10).unwrap_or(now).to_string();
    store
        .upsert_metric_snapshot(&MetricSnapshot {
            scope_kind: scope_kind.to_string(),
            scope_id: scope_id.to_string(),
            captured_on,
            wip: digest.wip as i64,
            stale_open_prs: digest.stale_open_prs as i64,
            merged_without_review: digest.merged_without_review as i64,
            attention_count: attention_count as i64,
            median_cycle_time_secs: digest.median_cycle_time_secs,
            median_pickup_secs: digest.median_pickup_secs,
            median_review_secs: digest.median_review_secs,
        })
        .await?;
    // Bound the daily history: keep the most recent window per scope.
    store
        .trim_metric_snapshots(scope_kind, scope_id, SNAPSHOT_RETENTION_DAYS)
        .await?;
    Ok(())
}

/// How many daily metric snapshots to keep per scope; older days are trimmed.
const SNAPSHOT_RETENTION_DAYS: i64 = 180;

/// One day on a board's trend: the totals across the board's repos for that day.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrendPoint {
    pub captured_on: String,
    pub wip: i64,
    pub stale_open_prs: i64,
    pub merged_without_review: i64,
    pub attention_count: i64,
}

/// A board's metric trend: per day present, the summed snapshot values across `repo_ids`,
/// oldest day first. Empty when there are no snapshots.
pub async fn board_trend(
    store: &Store,
    repo_ids: &[String],
) -> Result<Vec<TrendPoint>, SummaryError> {
    let mut by_day: std::collections::BTreeMap<String, TrendPoint> =
        std::collections::BTreeMap::new();
    for repo_id in repo_ids {
        for s in store.metric_snapshots("repo", repo_id).await? {
            let point = by_day
                .entry(s.captured_on.clone())
                .or_insert_with(|| TrendPoint {
                    captured_on: s.captured_on.clone(),
                    wip: 0,
                    stale_open_prs: 0,
                    merged_without_review: 0,
                    attention_count: 0,
                });
            point.wip += s.wip;
            point.stale_open_prs += s.stale_open_prs;
            point.merged_without_review += s.merged_without_review;
            point.attention_count += s.attention_count;
        }
    }
    Ok(by_day.into_values().collect())
}

/// The lowest module bus factor across a repo, or `None` if there is no module data.
async fn min_bus_factor(store: &Store, repo_id: &str) -> Result<Option<i64>, SummaryError> {
    const MIN_CHANGES: i64 = 3;
    let rows = store.repo_file_authorship(repo_id).await?;
    let files: Vec<core_codehealth::FileAuthorship> = rows
        .into_iter()
        .map(|r| core_codehealth::FileAuthorship {
            path: r.path,
            author: r.author,
            changes: r.changes,
        })
        .collect();
    Ok(core_codehealth::module_ownership(&files, MIN_CHANGES)
        .iter()
        .map(|m| m.bus_factor)
        .min())
}

/// A repo's deterministic scorecard: Bronze/Silver/Gold from the repo's flow, CI and ownership
/// facts.
pub async fn repo_scorecard(
    store: &Store,
    repo_id: &str,
    now: &str,
) -> Result<core_scorecard::Scorecard, SummaryError> {
    let digest = repo_digest(store, repo_id, now).await?;
    let ci_runs = store.ci_runs_for_repo(repo_id).await?;
    let reviews = store.reviews_for_repo(repo_id).await?;
    let releases = store.releases_for_repo(repo_id).await?;
    let facts = core_scorecard::RepoFacts {
        failing_ci: core_metrics::failing_ci_runs(&ci_runs).len() as i64,
        merged_without_review: digest.merged_without_review as i64,
        stale_open_prs: digest.stale_open_prs as i64,
        review_wait: digest.review_wait as i64,
        median_cycle_secs: digest.median_cycle_time_secs,
        releases: releases.len() as i64,
        min_bus_factor: min_bus_factor(store, repo_id).await?,
        has_reviews: !reviews.is_empty(),
    };
    Ok(core_scorecard::scorecard(repo_id, &facts))
}

/// A board's scorecard composite: each repo's scorecard plus a roll-up. The composite tier is
/// the weakest repo's tier; the counts break the repos down by tier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardScorecard {
    /// "none" | "bronze" | "silver" | "gold". "none" for an empty board.
    pub composite_tier: String,
    pub gold: usize,
    pub silver: usize,
    pub bronze: usize,
    pub none: usize,
    pub repos: Vec<core_scorecard::Scorecard>,
}

fn tier_rank(tier: &str) -> u8 {
    match tier {
        "gold" => 3,
        "silver" => 2,
        "bronze" => 1,
        _ => 0,
    }
}

pub async fn board_scorecard(
    store: &Store,
    repo_ids: &[String],
    now: &str,
) -> Result<BoardScorecard, SummaryError> {
    let mut repos = Vec::new();
    for repo_id in repo_ids {
        repos.push(repo_scorecard(store, repo_id, now).await?);
    }
    let count = |t: &str| repos.iter().filter(|s| s.tier == t).count();
    // Weakest tier across repos (empty -> none).
    let composite = repos
        .iter()
        .map(|s| s.tier.as_str())
        .min_by_key(|t| tier_rank(t))
        .unwrap_or("none")
        .to_string();
    Ok(BoardScorecard {
        composite_tier: composite,
        gold: count("gold"),
        silver: count("silver"),
        bronze: count("bronze"),
        none: count("none"),
        repos,
    })
}

/// A dependency used across a board: ecosystem and name, and which of the board's repos
/// declare it. `repos.len() > 1` means shared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DepUsage {
    pub ecosystem: String,
    pub name: String,
    /// Repo ids that declare this dependency.
    pub repos: Vec<String>,
}

/// The dependencies across a board's repos, each with the repos that use it, sorted by usage
/// (most-shared first) then name.
pub async fn board_dependencies(
    store: &Store,
    repo_ids: &[String],
) -> Result<Vec<DepUsage>, SummaryError> {
    use std::collections::BTreeMap;
    // (ecosystem, name) -> set of repo ids (BTreeSet for dedup and determinism).
    let mut by_dep: BTreeMap<(String, String), std::collections::BTreeSet<String>> =
        BTreeMap::new();
    for repo_id in repo_ids {
        for dep in store.dependencies_for_repo(repo_id).await? {
            by_dep
                .entry((dep.ecosystem, dep.name))
                .or_default()
                .insert(repo_id.clone());
        }
    }
    let mut out: Vec<DepUsage> = by_dep
        .into_iter()
        .map(|((ecosystem, name), repos)| DepUsage {
            ecosystem,
            name,
            repos: repos.into_iter().collect(),
        })
        .collect();
    // Most-shared first, then ecosystem, then name.
    out.sort_by(|a, b| {
        b.repos
            .len()
            .cmp(&a.repos.len())
            .then_with(|| a.ecosystem.cmp(&b.ecosystem))
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(out)
}

/// A code hotspot for a board: a file that changes often and churns a lot, tagged with its repo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardHotspot {
    pub repo_id: String,
    pub path: String,
    pub changes: i64,
    pub churn: i64,
    pub authors: i64,
    pub score: f64,
}

/// The hottest files across a board's repos, ranked by change-frequency x churn. Generated and
/// vendored paths are excluded by `core_codehealth`. Returns the top `limit` overall.
pub async fn board_hotspots(
    store: &Store,
    repo_ids: &[String],
    limit: usize,
) -> Result<Vec<BoardHotspot>, SummaryError> {
    let mut all: Vec<BoardHotspot> = Vec::new();
    for repo_id in repo_ids {
        let stats = store.repo_file_stats(repo_id).await?;
        let acts: Vec<core_codehealth::FileActivity> = stats
            .into_iter()
            .map(|s| core_codehealth::FileActivity {
                path: s.path,
                changes: s.changes,
                additions: s.additions,
                deletions: s.deletions,
                authors: s.authors,
            })
            .collect();
        for h in core_codehealth::hotspots(&acts, limit) {
            all.push(BoardHotspot {
                repo_id: repo_id.clone(),
                path: h.path,
                changes: h.changes,
                churn: h.churn,
                authors: h.authors,
                score: h.score,
            });
        }
    }
    all.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    all.truncate(limit);
    Ok(all)
}

/// A module's ownership risk for a board, tagged with its repo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardOwnershipRisk {
    pub repo_id: String,
    pub module: String,
    pub changes: i64,
    pub authors: i64,
    pub top_author: String,
    pub top_share: f64,
    pub bus_factor: i64,
}

/// At-risk modules across a board's repos: low bus factor or high concentration, riskiest
/// first. Returns up to `limit`.
pub async fn board_ownership_risks(
    store: &Store,
    repo_ids: &[String],
    limit: usize,
) -> Result<Vec<BoardOwnershipRisk>, SummaryError> {
    const MIN_CHANGES: i64 = 3;
    let mut all: Vec<BoardOwnershipRisk> = Vec::new();
    for repo_id in repo_ids {
        let rows = store.repo_file_authorship(repo_id).await?;
        let files: Vec<core_codehealth::FileAuthorship> = rows
            .into_iter()
            .map(|r| core_codehealth::FileAuthorship {
                path: r.path,
                author: r.author,
                changes: r.changes,
            })
            .collect();
        for m in core_codehealth::module_ownership(&files, MIN_CHANGES) {
            all.push(BoardOwnershipRisk {
                repo_id: repo_id.clone(),
                module: m.module,
                changes: m.changes,
                authors: m.authors,
                top_author: m.top_author,
                top_share: m.top_share,
                bus_factor: m.bus_factor,
            });
        }
    }
    // Riskiest first: lowest bus factor, then most active.
    all.sort_by(|a, b| {
        a.bus_factor
            .cmp(&b.bus_factor)
            .then_with(|| b.changes.cmp(&a.changes))
            .then_with(|| a.module.cmp(&b.module))
    });
    all.truncate(limit);
    Ok(all)
}

/// A PR that touched a code-risk file, for the drill-down.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeRiskPr {
    pub number: i64,
    pub title: String,
    pub url: Option<String>,
}

/// A board-level code-risk row: a file's fused risk tagged with its repo, plus the recent PRs
/// that touched it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardCodeRisk {
    pub repo_id: String,
    pub path: String,
    pub changes: i64,
    pub churn: i64,
    pub authors: i64,
    pub top_author: String,
    pub top_share: f64,
    pub bus_factor: i64,
    pub merged_touches: i64,
    pub unreviewed: i64,
    pub review_gap: f64,
    pub score: f64,
    pub reasons: Vec<String>,
    pub recent_prs: Vec<CodeRiskPr>,
}

/// How many recent touching PRs to attach to each code-risk file.
const CODE_RISK_DRILL_PRS: usize = 5;

/// The riskiest files across a board's repos: per-file hotspot x ownership x review fused by
/// `core_codehealth::code_risk`, each tagged with its repo and its most recent touching PRs.
/// Returns up to `limit`, riskiest first.
pub async fn board_code_risk(
    store: &Store,
    repo_ids: &[String],
    limit: usize,
) -> Result<Vec<BoardCodeRisk>, SummaryError> {
    let mut all: Vec<BoardCodeRisk> = Vec::new();
    for repo_id in repo_ids {
        let activity: Vec<core_codehealth::FileActivity> = store
            .repo_file_stats(repo_id)
            .await?
            .into_iter()
            .map(|s| core_codehealth::FileActivity {
                path: s.path,
                changes: s.changes,
                additions: s.additions,
                deletions: s.deletions,
                authors: s.authors,
            })
            .collect();
        let authorship: Vec<core_codehealth::FileAuthorship> = store
            .repo_file_authorship(repo_id)
            .await?
            .into_iter()
            .map(|r| core_codehealth::FileAuthorship {
                path: r.path,
                author: r.author,
                changes: r.changes,
            })
            .collect();
        let reviews: Vec<core_codehealth::FileReview> = store
            .repo_file_review_gaps(repo_id)
            .await?
            .into_iter()
            .map(
                |(path, merged_touches, unreviewed)| core_codehealth::FileReview {
                    path,
                    merged_touches,
                    unreviewed,
                },
            )
            .collect();
        // Drill map: file -> its recent touching PRs (the query is newest-first), capped per file.
        let mut prs_by_file: std::collections::HashMap<String, Vec<CodeRiskPr>> =
            std::collections::HashMap::new();
        for (filename, number, title, url, _created) in store.repo_file_prs(repo_id).await? {
            let entry = prs_by_file.entry(filename).or_default();
            if entry.len() < CODE_RISK_DRILL_PRS {
                entry.push(CodeRiskPr { number, title, url });
            }
        }

        for r in core_codehealth::code_risk(&activity, &authorship, &reviews, limit) {
            let recent_prs = prs_by_file.get(&r.path).cloned().unwrap_or_default();
            all.push(BoardCodeRisk {
                repo_id: repo_id.clone(),
                path: r.path,
                changes: r.changes,
                churn: r.churn,
                authors: r.authors,
                top_author: r.top_author,
                top_share: r.top_share,
                bus_factor: r.bus_factor,
                merged_touches: r.merged_touches,
                unreviewed: r.unreviewed,
                review_gap: r.review_gap,
                score: r.score,
                reasons: r.reasons,
                recent_prs,
            });
        }
    }
    all.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    all.truncate(limit);
    Ok(all)
}

/// A change-coupling pair for a board: two files in the same repo that frequently co-appear
/// in the same merged PR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardCouplingPair {
    pub repo_id: String,
    pub path_a: String,
    pub path_b: String,
    /// Merged PRs where both files changed together.
    pub together: i64,
    /// Jaccard coupling strength: `together / (prs_a + prs_b - together)`.
    pub coupling: f64,
}

/// Top change-coupling pairs across a board's repos: file pairs that often co-appear in the
/// same merged PR. Generated/vendored paths and pairs with fewer than 2 co-occurrences are
/// excluded. Returns up to `limit`, ranked by Jaccard coupling.
pub async fn board_coupling(
    store: &Store,
    repo_ids: &[String],
    limit: usize,
) -> Result<Vec<BoardCouplingPair>, SummaryError> {
    const MIN_TOGETHER: i64 = 2;
    let mut all: Vec<BoardCouplingPair> = Vec::new();
    for repo_id in repo_ids {
        let raw = store.repo_file_coupling(repo_id).await?;
        let inputs: Vec<core_codehealth::FilePairCo> = raw
            .into_iter()
            .map(
                |(path_a, path_b, together, prs_a, prs_b)| core_codehealth::FilePairCo {
                    path_a,
                    path_b,
                    together,
                    prs_a,
                    prs_b,
                },
            )
            .collect();
        for p in core_codehealth::coupling_pairs(&inputs, MIN_TOGETHER, limit) {
            all.push(BoardCouplingPair {
                repo_id: repo_id.clone(),
                path_a: p.path_a,
                path_b: p.path_b,
                together: p.together,
                coupling: p.coupling,
            });
        }
    }
    all.sort_by(|a, b| {
        b.coupling
            .partial_cmp(&a.coupling)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.together.cmp(&a.together))
            .then_with(|| a.path_a.cmp(&b.path_a))
    });
    all.truncate(limit);
    Ok(all)
}

/// A file's AST health for the board Code tab. Score 1-10 (10 = simplest).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardFileHealth {
    pub repo_id: String,
    pub path: String,
    pub loc: i64,
    pub functions: i64,
    pub branches: i64,
    /// Health score 1-10 derived from branch density + LOC (10 = simplest).
    pub score: i64,
}

/// Per-file AST health scores across a board's repos, weakest first, capped at `limit`. Only
/// parsed files are returned: a file not yet indexed, or whose language has no tree-sitter
/// grammar, is left out rather than reported as healthy.
pub async fn board_code_health(
    store: &Store,
    repo_ids: &[String],
    limit: usize,
) -> Result<Vec<BoardFileHealth>, SummaryError> {
    let mut all = Vec::new();
    for repo_id in repo_ids {
        for (path, loc, functions, branches, score) in store.repo_file_health(repo_id).await? {
            all.push(BoardFileHealth {
                repo_id: repo_id.clone(),
                path,
                loc,
                functions,
                branches,
                score,
            });
        }
    }
    // Weakest first; tie-break by path.
    all.sort_by(|a, b| a.score.cmp(&b.score).then_with(|| a.path.cmp(&b.path)));
    all.truncate(limit);
    Ok(all)
}

/// One metric's week-over-week movement: the latest value, the value ~7 days ago, the delta,
/// and whether the latest value is anomalous (more than ~2 standard deviations from its
/// trailing baseline).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricDelta {
    pub metric: String,
    pub current: i64,
    pub previous: i64,
    pub delta: i64,
    pub anomaly: bool,
    /// True when the anomaly flag was decided against the same-weekday rolling baseline rather
    /// than the all-days one. Always false when not anomalous.
    pub seasonal: bool,
}

/// A board's "what changed" summary over its daily snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrendDeltas {
    /// Days of history the series spanned.
    pub days: i64,
    pub deltas: Vec<MetricDelta>,
}

/// How many standard deviations from the baseline mean counts as anomalous.
const ANOMALY_SIGMA: f64 = 2.0;
/// Minimum prior same-weekday points before the seasonal baseline is used (~3 weeks). Below
/// this the all-days baseline is used.
const MIN_SEASONAL_POINTS: usize = 3;
/// Minimum prior points for the all-days fallback baseline.
const MIN_BASELINE_POINTS: usize = 4;

/// The day of week for a `YYYY-MM-DD` date (0 = Sunday .. 6 = Saturday), via Sakamoto's
/// algorithm. `None` when the prefix does not parse as a calendar date.
fn weekday_index(date: &str) -> Option<u32> {
    let b = date.as_bytes();
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let num = |s: &str| s.parse::<i64>().ok();
    let (mut y, m, d) = (num(&date[0..4])?, num(&date[5..7])?, num(&date[8..10])?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // Sakamoto: month offset table; Jan/Feb count as months of the prior year.
    const T: [i64; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    if m < 3 {
        y -= 1;
    }
    let w = (y + y / 4 - y / 100 + y / 400 + T[(m - 1) as usize] + d).rem_euclid(7);
    Some(w as u32)
}

/// Whether `current` is anomalous against prior values: more than `ANOMALY_SIGMA` standard
/// deviations from the mean, given enough points and a non-zero spread. `None` baseline means
/// not enough data.
fn is_anomalous(current: i64, base: &[i64], min_points: usize) -> bool {
    if base.len() < min_points {
        return false;
    }
    let mean = base.iter().sum::<i64>() as f64 / base.len() as f64;
    let var = base.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / base.len() as f64;
    let std = var.sqrt();
    std > 0.0 && (current as f64 - mean).abs() > ANOMALY_SIGMA * std
}

/// Compute current / previous(~7d) / delta / anomaly for a daily series (oldest first). The
/// latest value is compared against the rolling history of the same weekday, so weekly
/// seasonality does not false-flag; with thin same-weekday history it falls back to the
/// all-days baseline. `dates` aligns with `values`.
fn metric_delta(metric: &str, dates: &[&str], values: &[i64]) -> MetricDelta {
    let n = values.len();
    let current = values.last().copied().unwrap_or(0);
    // ~7 days back by position (one row per day present); fall back to the oldest.
    let previous = if n >= 8 {
        values[n - 8]
    } else {
        values.first().copied().unwrap_or(0)
    };
    // Prefer the same-weekday baseline; fall back to all prior days when too thin.
    let mut anomaly = false;
    let mut seasonal = false;
    if n >= 2 {
        let prior = &values[..n - 1];
        let today = dates.last().and_then(|d| weekday_index(d));
        let same: Vec<i64> = today
            .map(|wd| {
                dates[..n - 1]
                    .iter()
                    .zip(prior)
                    .filter(|(d, _)| weekday_index(d) == Some(wd))
                    .map(|(_, v)| *v)
                    .collect()
            })
            .unwrap_or_default();
        if is_anomalous(current, &same, MIN_SEASONAL_POINTS) {
            anomaly = true;
            seasonal = true;
        } else if same.len() < MIN_SEASONAL_POINTS
            && is_anomalous(current, prior, MIN_BASELINE_POINTS)
        {
            anomaly = true;
        }
    }
    MetricDelta {
        metric: metric.to_string(),
        current,
        previous,
        delta: current - previous,
        anomaly,
        seasonal,
    }
}

/// A board's week-over-week deltas and anomaly flags over its daily snapshots.
pub async fn trend_deltas(store: &Store, repo_ids: &[String]) -> Result<TrendDeltas, SummaryError> {
    let series = board_trend(store, repo_ids).await?; // oldest day first
    let days = series.len() as i64;
    let dates: Vec<&str> = series.iter().map(|t| t.captured_on.as_str()).collect();
    let col = |f: fn(&TrendPoint) -> i64| series.iter().map(f).collect::<Vec<i64>>();
    let deltas = vec![
        metric_delta("wip", &dates, &col(|t| t.wip)),
        metric_delta("attention", &dates, &col(|t| t.attention_count)),
        metric_delta("stale_prs", &dates, &col(|t| t.stale_open_prs)),
        metric_delta(
            "merged_without_review",
            &dates,
            &col(|t| t.merged_without_review),
        ),
    ];
    Ok(TrendDeltas { days, deltas })
}

/// DORA-lite metrics for a scope. Deploy frequency comes from releases; lead time and change
/// failure rate are proxies (median cycle time, revert share). Recovery time is omitted (needs
/// incidents). Each `*_tier` is "elite" / "high" / "medium" / "low" / "unknown".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoraMetrics {
    pub window_days: i64,
    /// Releases per week over the window.
    pub deploy_frequency_per_week: Option<f64>,
    pub deploy_tier: String,
    /// Lead time for changes in seconds: merge -> deploy when `lead_from_deploys`, else the median
    /// cycle-time proxy.
    pub lead_time_secs: Option<i64>,
    pub lead_tier: String,
    /// True when `lead_time_secs` is the deploy-based lead time, false for the cycle-time proxy.
    pub lead_from_deploys: bool,
    /// Revert commits / merged PRs, a proxy for change failure rate.
    pub change_failure_rate: Option<f64>,
    pub cfr_tier: String,
}

/// The default window for DORA-lite rates.
pub const DORA_WINDOW_DAYS: i64 = 30;

fn dora_from(
    prs: &[core_store::PullRequest],
    commits: &[core_store::Commit],
    releases: &[core_store::Release],
    now: &str,
    window_days: i64,
) -> DoraMetrics {
    let deploy = core_metrics::release_frequency_per_week(releases, now, window_days);
    // Prefer the merge->deploy lead time; fall back to the cycle-time proxy when the board has
    // no releases.
    let deploy_lead = core_metrics::lead_time_to_deploy(prs, releases);
    let lead_from_deploys = deploy_lead.is_some();
    let lead = deploy_lead.or_else(|| core_metrics::median_cycle_time_secs(prs));
    let cfr = core_metrics::change_failure_rate_proxy(commits, prs);
    DoraMetrics {
        window_days,
        deploy_frequency_per_week: deploy,
        deploy_tier: core_metrics::tier_deploy_frequency(deploy)
            .as_str()
            .to_string(),
        lead_time_secs: lead,
        lead_tier: core_metrics::tier_lead_time(lead).as_str().to_string(),
        lead_from_deploys,
        change_failure_rate: cfr,
        cfr_tier: core_metrics::tier_change_failure_rate(cfr)
            .as_str()
            .to_string(),
    }
}

/// DORA-lite metrics for one repo.
pub async fn dora_metrics(
    store: &Store,
    repo_id: &str,
    now: &str,
    window_days: i64,
) -> Result<DoraMetrics, SummaryError> {
    let prs = store.pull_requests(repo_id).await?;
    let commits = store.commits_for_repo(repo_id).await?;
    let releases = store.releases_for_repo(repo_id).await?;
    Ok(dora_from(&prs, &commits, &releases, now, window_days))
}

/// DORA-lite metrics for a board, pooling activity across its repos.
pub async fn board_dora(
    store: &Store,
    repo_ids: &[String],
    now: &str,
    window_days: i64,
) -> Result<DoraMetrics, SummaryError> {
    let mut prs = Vec::new();
    let mut commits = Vec::new();
    let mut releases = Vec::new();
    for id in repo_ids {
        prs.extend(store.pull_requests(id).await?);
        commits.extend(store.commits_for_repo(id).await?);
        releases.extend(store.releases_for_repo(id).await?);
    }
    Ok(dora_from(&prs, &commits, &releases, now, window_days))
}

/// An upstream repo that needs attention, surfaced to a repo that depends on it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpstreamAlert {
    /// The depended-on repo's id.
    pub repo_id: String,
    /// How many attention items it currently has.
    pub attention_count: usize,
}

/// Upstream alerts for `repo_id` at time `now` (RFC-3339): each repo it links to via
/// `depends_on` that currently has attention items, with the count. Direct dependencies only.
pub async fn upstream_alerts(
    store: &Store,
    repo_id: &str,
    now: &str,
) -> Result<Vec<UpstreamAlert>, SummaryError> {
    let links = store.links_from("repo", repo_id).await?;
    let mut alerts = Vec::new();
    for link in links.iter().filter(|l| l.relation == "depends_on") {
        // Only a count, so take the band-free path (see `attention`).
        let count = attention(store, &link.dst_id, now).await?.len();
        if count > 0 {
            alerts.push(UpstreamAlert {
                repo_id: link.dst_id.clone(),
                attention_count: count,
            });
        }
    }
    Ok(alerts)
}

/// Read back a stored digest for a scope, if present.
pub async fn load_digest(
    store: &Store,
    scope_kind: &str,
    scope_id: &str,
) -> Result<Option<Digest>, SummaryError> {
    match store.get_digest(scope_kind, scope_id).await? {
        Some(body) => Ok(Some(serde_json::from_str(&body)?)),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_store::{CiRun, Commit, Issue, PrFile, PullRequest, Repo, Review, Settings};

    /// The store id of an item's subject entity. Panics if the envelope has no subject.
    fn subject_id(entities: &[LinkedEntity]) -> &str {
        match core_model::subject_of(entities).expect("every attention item has a subject entity") {
            AttentionEntity::PullRequest { id, .. }
            | AttentionEntity::WorkItem { id, .. }
            | AttentionEntity::CiRun { id, .. } => id,
            AttentionEntity::Person { login } => login,
            AttentionEntity::SourceFile { path, .. } => path,
        }
    }

    fn pr(id: &str, state: &str, created_at: &str, merged_at: Option<&str>) -> PullRequest {
        pr_with_body(id, state, created_at, merged_at, None)
    }

    fn pr_with_body(
        id: &str,
        state: &str,
        created_at: &str,
        merged_at: Option<&str>,
        body: Option<&str>,
    ) -> PullRequest {
        PullRequest {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number: 1,
            title: format!("PR {id}"),
            state: state.into(),
            author_login: Some("octocat".into()),
            body: body.map(Into::into),
            created_at: created_at.into(),
            merged_at: merged_at.map(Into::into),
            html_url: None,
        }
    }

    async fn seeded() -> Store {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        // merged 2-day cycle, no review -> merged_without_review = 1
        store
            .upsert_pull_request(&pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-03T00:00:00Z"),
            ))
            .await
            .unwrap();
        // open and old -> stale at now=2026-06-20 with threshold 7
        store
            .upsert_pull_request(&pr("p2", "open", "2026-06-01T00:00:00Z", None))
            .await
            .unwrap();
        // open and fresh
        store
            .upsert_pull_request(&pr("p3", "open", "2026-06-19T00:00:00Z", None))
            .await
            .unwrap();
        store
            .upsert_review(&Review {
                id: "rv1".into(),
                pr_id: "p2".into(),
                reviewer_login: Some("alice".into()),
                state: "approved".into(),
                submitted_at: Some("2026-06-02T00:00:00Z".into()),
            })
            .await
            .unwrap();
        store
    }

    #[tokio::test]
    async fn digest_reports_metrics_and_watchdogs() {
        let store = seeded().await;
        let d = repo_digest(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        assert_eq!(d.median_cycle_time_secs, Some(2 * 86_400));
        assert_eq!(d.wip, 2);
        assert_eq!(d.stale_open_prs, 1);
        assert_eq!(d.merged_without_review, 1);
        assert!(!d.prose.is_empty());
        // The WIP count appears in the prose.
        assert!(d.prose.contains("2 PR(s) in flight"));
    }

    #[tokio::test]
    async fn attention_unions_the_watchdog_signals() {
        let store = seeded().await; // p1 merged-no-review, p2 open+stale (has a review), p3 fresh
        store
            .upsert_ci_run(&CiRun {
                id: "run1".into(),
                repo_id: "repo:acme/widget".into(),
                commit_sha: Some("aaa".into()),
                status: "completed".into(),
                conclusion: Some("failure".into()),
                completed_at: None,
                html_url: None,
                run_attempt: None,
            })
            .await
            .unwrap();

        let items = attention(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        let kinds: std::collections::HashSet<&str> =
            items.iter().map(|i| i.kind.as_str()).collect();
        assert!(kinds.contains("stale_pr"));
        assert!(kinds.contains("merged_without_review"));
        assert!(kinds.contains("failing_ci"));
        // The failing-CI item points at the run.
        assert!(items
            .iter()
            .any(|i| i.kind == "failing_ci" && subject_id(&i.entities) == "run1"));
    }

    #[tokio::test]
    async fn attention_flags_flaky_ci_only_on_rerun_then_green() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let run = |id: &str, conclusion: &str, attempt: Option<i64>| CiRun {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            commit_sha: Some("aaa".into()),
            status: "completed".into(),
            conclusion: Some(conclusion.into()),
            completed_at: Some("2026-06-19T00:00:00Z".into()),
            html_url: None,
            run_attempt: attempt,
        };
        // green first try (not flaky), green after a re-run (flaky), still failing after re-runs.
        for r in [
            run("first", "success", Some(1)),
            run("flaky", "success", Some(2)),
            run("broken", "failure", Some(3)),
        ] {
            store.upsert_ci_run(&r).await.unwrap();
        }

        let items = attention(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        let flaky: Vec<_> = items.iter().filter(|i| i.kind == "flaky_ci").collect();
        assert_eq!(flaky.len(), 1, "only the re-run-then-green run is flaky");
        assert_eq!(subject_id(&flaky[0].entities), "flaky");
        assert!(flaky[0].summary.contains("attempt 2"));
        assert!(flaky[0].action.contains("flaky"));
        // The still-failing re-run is failing_ci, not flaky_ci; the first-try green is neither.
        assert!(items
            .iter()
            .any(|i| i.kind == "failing_ci" && subject_id(&i.entities) == "broken"));
        assert!(!items.iter().any(|i| subject_id(&i.entities) == "first"));
    }

    /// The summary shortens a 40-char sha; the entity keeps the full value.
    #[tokio::test]
    async fn failing_and_flaky_ci_summaries_shorten_the_sha_but_entities_keep_it_full() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let full_sha = "605e94460502826bf3bcfa0f111e42484136ec86";
        store
            .upsert_ci_run(&CiRun {
                id: "run_red".into(),
                repo_id: "repo:acme/widget".into(),
                commit_sha: Some(full_sha.into()),
                status: "completed".into(),
                conclusion: Some("failure".into()),
                completed_at: Some("2026-06-19T00:00:00Z".into()),
                html_url: Some("https://forge/acme/widget/actions/runs/run_red".into()),
                run_attempt: Some(1),
            })
            .await
            .unwrap();
        store
            .upsert_ci_run(&CiRun {
                id: "run_flaky".into(),
                repo_id: "repo:acme/widget".into(),
                commit_sha: Some(full_sha.into()),
                status: "completed".into(),
                conclusion: Some("success".into()),
                completed_at: Some("2026-06-19T00:00:00Z".into()),
                html_url: Some("https://forge/acme/widget/actions/runs/run_flaky".into()),
                run_attempt: Some(2),
            })
            .await
            .unwrap();

        let items = attention(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();

        let red = items.iter().find(|i| i.kind == "failing_ci").unwrap();
        assert!(
            red.summary.contains("605e944"),
            "summary was: {}",
            red.summary
        );
        assert!(
            !red.summary.contains(full_sha),
            "summary must not carry the full sha: {}",
            red.summary
        );
        assert_eq!(
            red.subject(),
            Some(&AttentionEntity::CiRun {
                id: "run_red".into(),
                commit_sha: Some(full_sha.into()),
                attempt: Some(1),
                url: Some("https://forge/acme/widget/actions/runs/run_red".into()),
            }),
            "the entity must keep the full sha and the run's forge url"
        );

        let flaky = items.iter().find(|i| i.kind == "flaky_ci").unwrap();
        assert!(
            flaky.summary.contains("605e944"),
            "summary was: {}",
            flaky.summary
        );
        assert!(
            !flaky.summary.contains(full_sha),
            "summary must not carry the full sha: {}",
            flaky.summary
        );
        assert_eq!(
            flaky.subject(),
            Some(&AttentionEntity::CiRun {
                id: "run_flaky".into(),
                commit_sha: Some(full_sha.into()),
                attempt: Some(2),
                url: Some("https://forge/acme/widget/actions/runs/run_flaky".into()),
            }),
            "the entity must keep the full sha and the run's forge url"
        );
    }

    // ---- the typed attention envelope --------------------------------------------------

    /// Every attention kind from one store, so the subject invariant is asserted over the whole
    /// signal set: a kind added without a subject entity fails here.
    #[tokio::test]
    async fn every_attention_kind_carries_exactly_one_subject_and_its_pr_author() {
        use std::collections::HashSet;
        let store = seeded().await; // p1 merged-no-review, p2 open+stale (reviewed), p3 fresh
                                    // review_wait: open PR past the threshold, no reviews.
        store
            .upsert_pull_request(&pr("p4", "open", "2026-06-10T00:00:00Z", None))
            .await
            .unwrap();
        // A failing run and a flaky one.
        for (id, conclusion, attempt) in [
            ("run_red", "failure", Some(1)),
            ("run_ok", "success", Some(2)),
        ] {
            store
                .upsert_ci_run(&CiRun {
                    id: id.into(),
                    repo_id: "repo:acme/widget".into(),
                    commit_sha: Some("aaa".into()),
                    status: "completed".into(),
                    conclusion: Some(conclusion.into()),
                    completed_at: Some("2026-06-19T00:00:00Z".into()),
                    html_url: Some(format!("https://forge/acme/widget/actions/runs/{id}")),
                    run_attempt: attempt,
                })
                .await
                .unwrap();
        }
        // done_not_done: p1 closes an issue whose work item is still open.
        store
            .upsert_issue(&Issue {
                id: "i_open".into(),
                repo_id: "repo:acme/widget".into(),
                number: 42,
                title: "a bug".into(),
                state: "open".into(),
                author_login: Some("dana".into()),
                body: None,
                created_at: "2026-05-01T00:00:00Z".into(),
                closed_at: None,
                labels: String::new(),
                html_url: Some("https://forge/acme/widget/issues/42".into()),
            })
            .await
            .unwrap();
        store
            .upsert_github_work_item("i_open", "a bug", "issue", "open", "indeterminate")
            .await
            .unwrap();
        store
            .add_link("pull_request", "p1", "work_item", "i_open", "closes")
            .await
            .unwrap();
        // aging_wip: the same issue entered "indeterminate" long ago and never reached done.
        store
            .replace_work_item_status_history(
                "i_open",
                &[("indeterminate", "2026-05-01T00:00:00Z".to_string())],
            )
            .await
            .unwrap();
        // risky_change: p2 disturbs a file last merged long ago.
        let file = |pr_id: &str, name: &str| PrFile {
            pr_id: pr_id.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: 1,
            deletions: 0,
            patch: None,
        };
        store
            .upsert_pull_request(&pr(
                "m_old",
                "closed",
                "2025-01-01T00:00:00Z",
                Some("2025-01-02T00:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .replace_pr_files("m_old", &[file("m_old", "core/db.rs")])
            .await
            .unwrap();
        store
            .replace_pr_files("p2", &[file("p2", "core/db.rs")])
            .await
            .unwrap();

        let items = attention(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        let kinds: HashSet<&str> = items.iter().map(|i| i.kind.as_str()).collect();
        // The fixture exercises the whole signal set.
        for expected in [
            "stale_pr",
            "review_wait",
            "risky_change",
            "merged_without_review",
            "failing_ci",
            "flaky_ci",
            "done_not_done",
            "orphan_pr",
            "aging_wip",
        ] {
            assert!(
                kinds.contains(expected),
                "fixture must produce {expected}: {kinds:?}"
            );
        }

        for item in &items {
            let subjects: Vec<_> = item
                .entities
                .iter()
                .filter(|e| e.role == core_model::AttentionRole::Subject)
                .collect();
            assert_eq!(
                subjects.len(),
                1,
                "{} must carry exactly one subject entity, got {:?}",
                item.kind,
                item.entities
            );
            // A PR-subject item names its author as the actor; every fixture PR has one.
            if matches!(item.subject(), Some(AttentionEntity::PullRequest { .. })) {
                assert_eq!(
                    item.actor_logins().collect::<Vec<_>>(),
                    vec!["octocat"],
                    "{} must name the PR author as its actor",
                    item.kind
                );
            }
        }

        // The subject payload is the store's own record; the CI subject carries the synced URL.
        let red = items.iter().find(|i| i.kind == "failing_ci").unwrap();
        assert_eq!(
            red.subject(),
            Some(&AttentionEntity::CiRun {
                id: "run_red".into(),
                commit_sha: Some("aaa".into()),
                attempt: Some(1),
                url: Some("https://forge/acme/widget/actions/runs/run_red".into()),
            })
        );

        // aging_wip names no actor: the store has the issue's author, not its assignee.
        let aging = items.iter().find(|i| i.kind == "aging_wip").unwrap();
        assert_eq!(
            aging.actor_logins().count(),
            0,
            "aging_wip must not guess an actor"
        );
        assert!(matches!(
            aging.subject(),
            Some(AttentionEntity::WorkItem { number: 42, .. })
        ));
    }

    /// The still-open work item behind a `done_not_done` is linkable evidence carrying the
    /// issue's own synced `html_url`.
    #[tokio::test]
    async fn done_not_done_carries_the_open_work_item_as_linkable_evidence() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        store
            .upsert_pull_request(&pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-03T00:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .upsert_issue(&Issue {
                id: "i1".into(),
                repo_id: "repo:acme/widget".into(),
                number: 42,
                title: "a bug".into(),
                state: "open".into(),
                author_login: Some("dana".into()),
                body: None,
                created_at: "2026-05-01T00:00:00Z".into(),
                closed_at: None,
                labels: String::new(),
                html_url: Some("https://forge/acme/widget/issues/42".into()),
            })
            .await
            .unwrap();
        store
            .upsert_github_work_item("i1", "a bug", "issue", "open", "new")
            .await
            .unwrap();
        store
            .add_link("pull_request", "p1", "work_item", "i1", "closes")
            .await
            .unwrap();

        let items = attention(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        let item = items.iter().find(|i| i.kind == "done_not_done").unwrap();
        let evidence: Vec<_> = item
            .entities
            .iter()
            .filter(|e| e.role == core_model::AttentionRole::Evidence)
            .map(|e| &e.target)
            .collect();
        assert_eq!(
            evidence,
            vec![&AttentionEntity::WorkItem {
                id: "i1".into(),
                number: 42,
                title: "a bug".into(),
                url: Some("https://forge/acme/widget/issues/42".into()),
            }],
            "the still-open work item is the evidence, with the URL the forge reported"
        );
        // The PR stays the subject.
        assert_eq!(subject_id(&item.entities), "p1");
    }

    /// `risky_change` evidence lists the dormant files, longest-dormant first, capped, with the
    /// summary keeping the true total.
    #[tokio::test]
    async fn risky_change_carries_dormant_files_longest_first_and_capped() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let file = |pr_id: &str, name: &str| PrFile {
            pr_id: pr_id.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: 1,
            deletions: 0,
            patch: None,
        };
        // Seven files, each last merged in a different (dormant) year, oldest first in `expected`.
        let years = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"];
        for (n, year) in years.iter().enumerate() {
            let merged = format!("m{n}");
            store
                .upsert_pull_request(&pr(
                    &merged,
                    "closed",
                    &format!("{year}-01-01T00:00:00Z"),
                    Some(&format!("{year}-02-01T00:00:00Z")),
                ))
                .await
                .unwrap();
            store
                .replace_pr_files(&merged, &[file(&merged, &format!("f{n}.rs"))])
                .await
                .unwrap();
        }
        // One open PR touching all seven.
        store
            .upsert_pull_request(&pr("p_open", "open", "2026-06-18T00:00:00Z", None))
            .await
            .unwrap();
        let touched: Vec<PrFile> = (0..years.len())
            .map(|n| file("p_open", &format!("f{n}.rs")))
            .collect();
        store.replace_pr_files("p_open", &touched).await.unwrap();

        let items = attention(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        let risky = items.iter().find(|i| i.kind == "risky_change").unwrap();
        let files: Vec<(&str, Option<&str>)> = risky
            .entities
            .iter()
            .filter(|e| e.role == core_model::AttentionRole::Evidence)
            .filter_map(|e| match &e.target {
                AttentionEntity::SourceFile {
                    path,
                    last_changed_at,
                } => Some((path.as_str(), last_changed_at.as_deref())),
                _ => None,
            })
            .collect();
        // Longest-dormant first, capped at MAX_DORMANT_EVIDENCE of the seven.
        assert_eq!(files.len(), MAX_DORMANT_EVIDENCE);
        assert_eq!(
            files.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            vec!["f0.rs", "f1.rs", "f2.rs", "f3.rs", "f4.rs"]
        );
        assert_eq!(files[0].1, Some("2019-02-01T00:00:00Z"));
        // The summary still states all seven.
        assert!(
            risky.summary.contains("7 long-stable file(s)"),
            "the summary must keep the true total: {}",
            risky.summary
        );
    }

    /// `by_team` reads the actor entity, and an item whose PR has no author is not flagged.
    #[tokio::test]
    async fn board_attention_flags_by_team_from_the_actor_entity() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:a".into(),
                owner: "acme".into(),
                name: "a".into(),
                full_name: "acme/a".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let stale = |id: &str, who: Option<&str>| PullRequest {
            id: id.into(),
            repo_id: "repo:a".into(),
            number: 1,
            title: "T".into(),
            state: "open".into(),
            author_login: who.map(Into::into),
            body: None,
            created_at: "2026-06-01T00:00:00Z".into(),
            merged_at: None,
            html_url: None,
        };
        for p in [
            stale("p_alice", Some("alice")),
            stale("p_carol", Some("carol")),
            stale("p_ghost", None),
        ] {
            store.upsert_pull_request(&p).await.unwrap();
        }

        let items = board_attention(
            &store,
            &["repo:a".to_string()],
            &["alice".into()],
            "2026-06-20T00:00:00Z",
        )
        .await
        .unwrap();
        let by_team = |id: &str| {
            items
                .iter()
                .find(|i| i.kind == "stale_pr" && subject_id(&i.entities) == id)
                .unwrap()
                .by_team
        };
        assert!(by_team("p_alice"), "alice is on the board");
        assert!(!by_team("p_carol"), "carol is not");
        assert!(
            !by_team("p_ghost"),
            "a PR the forge reported no author for is not the team's by default"
        );
    }

    #[tokio::test]
    async fn attention_flags_done_not_done() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        // A merged PR that closes an issue whose work item is still open.
        store
            .upsert_pull_request(&pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-03T00:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .upsert_github_work_item("i1", "a bug", "issue", "open", "new")
            .await
            .unwrap();
        store
            .add_link("pull_request", "p1", "work_item", "i1", "closes")
            .await
            .unwrap();

        let now = "2026-06-20T00:00:00Z";
        let items = attention(&store, "repo:acme/widget", now).await.unwrap();
        assert!(
            items
                .iter()
                .any(|i| i.kind == "done_not_done" && subject_id(&i.entities) == "p1"),
            "a merged PR whose linked work item is still open is flagged"
        );

        // Closing the work item clears the signal.
        store
            .upsert_github_work_item("i1", "a bug", "issue", "closed", "done")
            .await
            .unwrap();
        let items = attention(&store, "repo:acme/widget", now).await.unwrap();
        assert!(items.iter().all(|i| i.kind != "done_not_done"));
    }

    #[test]
    fn every_emitted_attention_kind_has_an_action() {
        for kind in [
            "review_wait",
            "stale_pr",
            "aging_wip",
            "merged_without_review",
            "failing_ci",
            "done_not_done",
            "orphan_pr",
            "upstream",
        ] {
            assert!(
                !attention_action(kind).is_empty(),
                "{kind} must carry a mechanical next step"
            );
        }
        // An unknown kind has no action.
        assert_eq!(attention_action("something_else"), "");
    }

    #[test]
    fn short_sha_truncates_to_seven_chars_and_never_panics_on_short_input() {
        assert_eq!(
            short_sha("605e94460502826bf3bcfa0f111e42484136ec86"),
            "605e944"
        );
        // Shorter than the prefix length: passed through, no byte-slice panic.
        assert_eq!(short_sha("aaa"), "aaa");
        assert_eq!(short_sha(""), "");
        // Exactly 7 chars: unchanged.
        assert_eq!(short_sha("1234567"), "1234567");
        // Non-hex, non-ASCII input shorter than 7 bytes: passed through.
        assert_eq!(short_sha("\u{1F600}"), "\u{1F600}");
    }

    #[tokio::test]
    async fn attention_flags_review_wait_until_reviewed() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        // An open PR created 3 days before `now`, no review yet (threshold is 2 days).
        store
            .upsert_pull_request(&pr("p1", "open", "2026-06-01T00:00:00Z", None))
            .await
            .unwrap();

        let now = "2026-06-04T00:00:00Z";
        let items = attention(&store, "repo:acme/widget", now).await.unwrap();
        let rw = items.iter().find(|i| i.kind == "review_wait").unwrap();
        assert_eq!(subject_id(&rw.entities), "p1");
        assert_eq!(rw.action, "Assign a reviewer.");

        // A review clears it.
        store
            .upsert_review(&Review {
                id: "rv1".into(),
                pr_id: "p1".into(),
                reviewer_login: Some("octo".into()),
                state: "COMMENTED".into(),
                submitted_at: Some("2026-06-02T00:00:00Z".into()),
            })
            .await
            .unwrap();
        let items = attention(&store, "repo:acme/widget", now).await.unwrap();
        assert!(items.iter().all(|i| i.kind != "review_wait"));
    }

    #[tokio::test]
    async fn attention_flags_orphan_pr_only_when_the_repo_has_link_data() {
        // A repo where the linker has never run (e.g. an org board) has zero `pull_request ->
        // work_item` links for every merged PR; that must not read as "checked, none track one".
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        // Two merged PRs, neither linked, and no link data anywhere in the repo yet.
        store
            .upsert_pull_request(&pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-03T00:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .upsert_pull_request(&pr(
                "p2",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-03T00:00:00Z"),
            ))
            .await
            .unwrap();
        let now = "2026-06-20T00:00:00Z";

        let items = attention(&store, "repo:acme/widget", now).await.unwrap();
        assert!(
            items.iter().all(|i| i.kind != "orphan_pr"),
            "no link data anywhere in the repo -> neither PR is flagged"
        );

        // The linker has run for this repo (p2 got a link); an unlinked merged PR is a finding.
        store
            .add_link("pull_request", "p2", "work_item", "i2", "closes")
            .await
            .unwrap();
        let items = attention(&store, "repo:acme/widget", now).await.unwrap();
        let orphan = items.iter().find(|i| i.kind == "orphan_pr").unwrap();
        assert_eq!(subject_id(&orphan.entities), "p1");
        assert!(
            !orphan.action.is_empty(),
            "the envelope carries a next step"
        );
        assert!(items
            .iter()
            .all(|i| !(i.kind == "orphan_pr" && subject_id(&i.entities) == "p2")));

        // Linking p1 too clears the orphan signal entirely.
        store
            .add_link("pull_request", "p1", "work_item", "i1", "closes")
            .await
            .unwrap();
        let items = attention(&store, "repo:acme/widget", now).await.unwrap();
        assert!(items.iter().all(|i| i.kind != "orphan_pr"));
    }

    #[tokio::test]
    async fn attention_coverage_reports_both_shortfalls() {
        // Composes the merged-without-review window and the orphan_pr link-data gate.
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        // Merged before any synced review, and with no link data anywhere -> unjudged on both.
        store
            .upsert_pull_request(&pr(
                "ancient",
                "closed",
                "2022-01-01T00:00:00Z",
                Some("2022-01-02T00:00:00Z"),
            ))
            .await
            .unwrap();
        // Merged inside the (future) review window, so a real merged_without_review flag.
        store
            .upsert_pull_request(&pr(
                "recent",
                "closed",
                "2026-08-10T00:00:00Z",
                Some("2026-08-11T00:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .upsert_review(&Review {
                id: "rv1".into(),
                pr_id: "recent".into(),
                reviewer_login: Some("alice".into()),
                state: "approved".into(),
                submitted_at: Some("2026-08-01T00:00:00Z".into()),
            })
            .await
            .unwrap();

        let coverage = attention_coverage(&store, "repo:acme/widget")
            .await
            .unwrap();
        assert_eq!(coverage.merged_without_review_unjudged, 1, "the ancient PR");
        assert_eq!(
            coverage.orphan_pr_unjudged, 2,
            "both PRs, no link data at all"
        );

        // Give the repo link data for one PR, so only "recent" stays unjudged.
        store
            .add_link("pull_request", "ancient", "work_item", "i1", "closes")
            .await
            .unwrap();
        let coverage = attention_coverage(&store, "repo:acme/widget")
            .await
            .unwrap();
        assert_eq!(
            coverage.orphan_pr_unjudged, 0,
            "the repo has link data now, so every merged PR was judged"
        );
    }

    #[tokio::test]
    async fn attention_flags_aging_wip_past_the_threshold() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        store
            .upsert_issue(&core_store::Issue {
                id: "i1".into(),
                repo_id: "repo:acme/widget".into(),
                number: 7,
                title: "big feature".into(),
                state: "open".into(),
                author_login: None,
                body: None,
                created_at: "2026-06-01T00:00:00Z".into(),
                closed_at: None,
                labels: String::new(),
                html_url: None,
            })
            .await
            .unwrap();
        store
            .upsert_github_work_item("i1", "big feature", "issue", "open", "indeterminate")
            .await
            .unwrap();
        store
            .replace_work_item_status_history(
                "i1",
                &[
                    ("new", "2026-06-01T00:00:00Z".to_string()),
                    ("indeterminate", "2026-06-01T00:00:00Z".to_string()),
                ],
            )
            .await
            .unwrap();

        // 19 days in progress (> 14) -> flagged.
        let items = attention(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        assert!(items
            .iter()
            .any(|i| i.kind == "aging_wip" && subject_id(&i.entities) == "i1"));
        // 9 days in progress (< 14) -> not yet.
        let items = attention(&store, "repo:acme/widget", "2026-06-10T00:00:00Z")
            .await
            .unwrap();
        assert!(items.iter().all(|i| i.kind != "aging_wip"));
    }

    // ---- aging_wip percentile bands ----
    // Every completed item has a non-zero wait stage (new -> indeterminate). A band built from
    // lead time (new -> done) would be wider than the in-progress age by that wait, so the 20-day
    // wait makes the two spans disagree on every item.

    const BAND_NOW: &str = "2026-06-30T00:00:00Z";

    /// `BAND_NOW` minus `days_before_now`, as RFC-3339.
    fn band_ts(days_before_now: i64) -> String {
        use time::{format_description::well_known::Rfc3339, OffsetDateTime};
        (OffsetDateTime::parse(BAND_NOW, &Rfc3339).unwrap() - time::Duration::days(days_before_now))
            .format(&Rfc3339)
            .unwrap()
    }

    async fn band_store() -> Store {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        store
    }

    /// Seed one work item with an explicit status history. Day arguments are days before
    /// `BAND_NOW`; `done_days` `None` leaves the item in progress.
    async fn seed_work_item(
        store: &Store,
        id: &str,
        number: i64,
        new_days: i64,
        in_progress_days: i64,
        done_days: Option<i64>,
    ) {
        let state = if done_days.is_some() {
            "closed"
        } else {
            "open"
        };
        let status = if done_days.is_some() {
            "done"
        } else {
            "indeterminate"
        };
        store
            .upsert_issue(&core_store::Issue {
                id: id.into(),
                repo_id: "repo:acme/widget".into(),
                number,
                title: format!("issue {id}"),
                state: state.into(),
                author_login: None,
                body: None,
                created_at: band_ts(new_days),
                closed_at: done_days.map(band_ts),
                labels: String::new(),
                html_url: None,
            })
            .await
            .unwrap();
        store
            .upsert_github_work_item(id, &format!("issue {id}"), "issue", state, status)
            .await
            .unwrap();
        let mut history = vec![
            ("new", band_ts(new_days)),
            ("indeterminate", band_ts(in_progress_days)),
        ];
        if let Some(d) = done_days {
            history.push(("done", band_ts(d)));
        }
        store
            .replace_work_item_status_history(id, &history)
            .await
            .unwrap();
    }

    /// Seed `n` completed items with active times 2, 4, ... 2n days, each preceded by a 20-day
    /// wait and finished 10 days ago (inside the band window).
    async fn seed_band_history(store: &Store, n: i64) {
        const DONE_DAYS_AGO: i64 = 10;
        const WAIT_DAYS: i64 = 20;
        for k in 1..=n {
            let active = 2 * k;
            seed_work_item(
                store,
                &format!("done_{k}"),
                k,
                DONE_DAYS_AGO + active + WAIT_DAYS,
                DONE_DAYS_AGO + active,
                Some(DONE_DAYS_AGO),
            )
            .await;
        }
    }

    fn repo_ids() -> Vec<String> {
        vec!["repo:acme/widget".to_string()]
    }

    #[tokio::test]
    async fn aging_wip_bands_measure_active_time_not_lead_time() {
        let store = band_store().await;
        // 20 completions, active 2..40 days -> P50 20d, P75 30d, P90 36d (nearest rank 10/15/18).
        // Their lead times are 22..60 days, so a lead-time band would label items differently.
        seed_band_history(&store, 20).await;
        // In-progress items, all past the 14-day flag threshold.
        seed_work_item(&store, "wip_p50", 101, 45, 25, None).await;
        seed_work_item(&store, "wip_p75", 102, 53, 33, None).await;
        seed_work_item(&store, "wip_p90", 103, 65, 45, None).await;
        seed_work_item(&store, "wip_inside", 104, 38, 18, None).await;

        let bands = board_wip_aging_bands(&store, &repo_ids(), BAND_NOW)
            .await
            .unwrap();
        assert_eq!(bands.sample, 20);
        assert_eq!(bands.p50_secs, Some(20 * 86_400), "P50 of active time");
        assert_eq!(bands.p75_secs, Some(30 * 86_400), "P75 of active time");
        assert_eq!(bands.p90_secs, Some(36 * 86_400), "P90 of active time");

        let items = attention_enriched(&store, "repo:acme/widget", BAND_NOW, &bands)
            .await
            .unwrap();
        let band = |id: &str| {
            items
                .iter()
                .find(|i| subject_id(&i.entities) == id)
                .unwrap_or_else(|| panic!("{id} was not flagged aging_wip"))
                .wip_percentile
                .clone()
        };
        assert_eq!(band("wip_p50").as_deref(), Some("over_p50"), "25d active");
        assert_eq!(band("wip_p75").as_deref(), Some("over_p75"), "33d active");
        assert_eq!(band("wip_p90").as_deref(), Some("over_p90"), "45d active");
        assert_eq!(band("wip_inside"), None, "18d active is inside P50");

        // Every band states what it is measured over: the span, the sample, the window.
        let flagged = items
            .iter()
            .find(|i| subject_id(&i.entities) == "wip_p90")
            .unwrap();
        let basis = flagged.wip_percentile_basis.as_deref().expect("basis set");
        assert!(basis.contains("P90"), "names the percentile: {basis}");
        assert!(
            basis.contains("in progress -> done"),
            "names the span: {basis}"
        );
        assert!(basis.contains("20 work items"), "names the sample: {basis}");
        assert!(basis.contains("30 days"), "names the window: {basis}");
        assert!(
            items
                .iter()
                .find(|i| subject_id(&i.entities) == "wip_inside")
                .unwrap()
                .wip_percentile_basis
                .is_none(),
            "no band, no basis"
        );

        // Without bands the items are still flagged, unlabelled.
        let plain = attention(&store, "repo:acme/widget", BAND_NOW)
            .await
            .unwrap();
        let unlabelled = plain
            .iter()
            .find(|i| subject_id(&i.entities) == "wip_p90")
            .unwrap();
        assert_eq!(unlabelled.kind, "aging_wip");
        assert!(unlabelled.wip_percentile.is_none());
    }

    #[tokio::test]
    async fn aging_wip_bands_ignore_completions_outside_the_window() {
        let store = band_store().await;
        seed_band_history(&store, 20).await;
        // 20 more completions, five times slower, finished 200 days ago: outside the 30-day
        // window, so they must not move the baseline.
        for k in 1..=20 {
            seed_work_item(&store, &format!("old_{k}"), 200 + k, 320, 300, Some(200)).await;
        }
        let bands = board_wip_aging_bands(&store, &repo_ids(), BAND_NOW)
            .await
            .unwrap();
        assert_eq!(bands.sample, 20, "only in-window completions count");
        assert_eq!(bands.p90_secs, Some(36 * 86_400));
    }

    #[tokio::test]
    async fn aging_wip_bands_gate_the_tail_on_sample_size() {
        // Below MIN_BAND_HISTORY there is no distribution.
        let store = band_store().await;
        seed_band_history(&store, MIN_BAND_HISTORY as i64 - 1).await;
        let bands = board_wip_aging_bands(&store, &repo_ids(), BAND_NOW)
            .await
            .unwrap();
        assert_eq!(bands, WipAgingBands::default(), "too few completions");

        // Between the thresholds only P50 is stated, so nothing claims "> P90".
        let store = band_store().await;
        seed_band_history(&store, MIN_TAIL_HISTORY as i64 - 1).await;
        seed_work_item(&store, "wip_slowest", 101, 120, 100, None).await;
        let bands = board_wip_aging_bands(&store, &repo_ids(), BAND_NOW)
            .await
            .unwrap();
        assert_eq!(bands.sample, MIN_TAIL_HISTORY - 1);
        assert!(bands.p50_secs.is_some());
        assert_eq!(bands.p75_secs, None, "P75 needs a bigger sample");
        assert_eq!(bands.p90_secs, None, "P90 needs a bigger sample");
        let items = attention_enriched(&store, "repo:acme/widget", BAND_NOW, &bands)
            .await
            .unwrap();
        let slowest = items
            .iter()
            .find(|i| subject_id(&i.entities) == "wip_slowest")
            .unwrap();
        assert_eq!(slowest.wip_percentile.as_deref(), Some("over_p50"));
    }

    #[tokio::test]
    async fn attention_flags_risky_change_for_dormant_files_only() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let file = |pr: &str, name: &str| core_store::PrFile {
            pr_id: pr.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: 1,
            deletions: 0,
            patch: None,
        };
        // core/db.rs was last changed by a PR merged ~384 days before `now` -> dormant.
        store
            .upsert_pull_request(&pr(
                "m_old",
                "closed",
                "2025-05-01T00:00:00Z",
                Some("2025-06-01T00:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .replace_pr_files("m_old", &[file("m_old", "core/db.rs")])
            .await
            .unwrap();
        // An open PR that disturbs the dormant file.
        store
            .upsert_pull_request(&pr("p_open", "open", "2026-06-18T00:00:00Z", None))
            .await
            .unwrap();
        store
            .replace_pr_files("p_open", &[file("p_open", "core/db.rs")])
            .await
            .unwrap();
        // An open PR that only adds a brand-new file (no merged history) -> not dormant.
        store
            .upsert_pull_request(&pr("p_new", "open", "2026-06-18T00:00:00Z", None))
            .await
            .unwrap();
        store
            .replace_pr_files("p_new", &[file("p_new", "brand_new.rs")])
            .await
            .unwrap();

        let now = "2026-06-20T00:00:00Z";
        let items = attention(&store, "repo:acme/widget", now).await.unwrap();
        let risky: Vec<_> = items.iter().filter(|i| i.kind == "risky_change").collect();
        assert_eq!(risky.len(), 1, "only the dormant-file PR is flagged");
        assert_eq!(subject_id(&risky[0].entities), "p_open");
        assert!(risky[0].summary.contains("core/db.rs"));
        assert!(!risky[0].action.is_empty());

        // A fresh merged change to the same file clears the dormancy.
        store
            .upsert_pull_request(&pr(
                "m_recent",
                "closed",
                "2026-06-10T00:00:00Z",
                Some("2026-06-15T00:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .replace_pr_files("m_recent", &[file("m_recent", "core/db.rs")])
            .await
            .unwrap();
        let items = attention(&store, "repo:acme/widget", now).await.unwrap();
        assert!(
            items.iter().all(|i| i.kind != "risky_change"),
            "a recent merged change to the file clears the risk"
        );
    }

    #[tokio::test]
    async fn board_cycle_phases_aggregates_pr_phases() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        // PR opened day 1, first review day 3, merged day 5 -> pickup 2d, review 2d.
        store
            .upsert_pull_request(&pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-05T00:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .upsert_review(&Review {
                id: "rv1".into(),
                pr_id: "p1".into(),
                reviewer_login: Some("octo".into()),
                state: "COMMENTED".into(),
                submitted_at: Some("2026-06-03T00:00:00Z".into()),
            })
            .await
            .unwrap();

        let p = board_cycle_phases(&store, &["repo:acme/widget".to_string()])
            .await
            .unwrap();
        assert_eq!(p.median_pickup_secs, Some(2 * 86_400));
        assert_eq!(p.median_review_secs, Some(2 * 86_400));
    }

    #[tokio::test]
    async fn board_flow_trend_merges_recorded_series() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .record_metric("board", "b1", "2026-06-10", "flow_lead_secs", 5 * 86_400)
            .await
            .unwrap();
        store
            .record_metric("board", "b1", "2026-06-10", "flow_wip", 3)
            .await
            .unwrap();
        store
            .record_metric("board", "b1", "2026-06-11", "bug_net", 2)
            .await
            .unwrap();

        let trend = board_flow_trend(&store, "b1").await.unwrap();
        assert_eq!(trend.len(), 2, "two distinct days");
        assert_eq!(trend[0].captured_on, "2026-06-10");
        assert_eq!(trend[0].lead_time_secs, Some(5 * 86_400));
        assert_eq!(trend[0].wip, 3);
        assert_eq!(trend[1].captured_on, "2026-06-11");
        assert_eq!(trend[1].bug_net, 2);
        assert!(
            trend[1].lead_time_secs.is_none(),
            "no lead recorded that day"
        );
    }

    #[tokio::test]
    async fn board_bug_flow_counts_bug_labeled_issues() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let issue =
            |id: &str, n: i64, state: &str, created: &str, closed: Option<&str>, labels: &str| {
                core_store::Issue {
                    id: id.into(),
                    repo_id: "repo:acme/widget".into(),
                    number: n,
                    title: format!("i{n}"),
                    state: state.into(),
                    author_login: None,
                    body: None,
                    created_at: created.into(),
                    closed_at: closed.map(Into::into),
                    labels: labels.into(),
                    html_url: None,
                }
            };
        // b1: bug opened in window. b2: bug closed in window. f1: a feature (not counted).
        store
            .upsert_issue(&issue(
                "b1",
                1,
                "open",
                "2026-06-10T00:00:00Z",
                None,
                "Type: Bug",
            ))
            .await
            .unwrap();
        store
            .upsert_issue(&issue(
                "b2",
                2,
                "closed",
                "2026-05-01T00:00:00Z",
                Some("2026-06-12T00:00:00Z"),
                "bug",
            ))
            .await
            .unwrap();
        store
            .upsert_issue(&issue(
                "f1",
                3,
                "open",
                "2026-06-11T00:00:00Z",
                None,
                "enhancement",
            ))
            .await
            .unwrap();

        let flow = board_bug_flow(
            &store,
            &["repo:acme/widget".to_string()],
            "2026-06-05T00:00:00Z",
            "2026-06-20T00:00:00Z",
        )
        .await
        .unwrap();
        assert_eq!(
            flow.opened, 1,
            "b1 opened in window; the feature is not a bug"
        );
        assert_eq!(flow.closed, 1, "b2 closed in window");
    }

    #[tokio::test]
    async fn board_flow_metrics_from_status_history() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        // Two issues need rows in `issues` for the board scan to find them.
        let issue = |id: &str, number: i64| core_store::Issue {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number,
            title: format!("issue {number}"),
            state: "open".into(),
            author_login: None,
            body: None,
            created_at: "2026-06-01T00:00:00Z".into(),
            closed_at: None,
            labels: String::new(),
            html_url: None,
        };
        store.upsert_issue(&issue("i1", 1)).await.unwrap();
        store.upsert_issue(&issue("i2", 2)).await.unwrap();
        // i1: completed in window: new day 1, in progress day 3, done day 6 -> lead 5d, wait 2d,
        // active 3d.
        store
            .replace_work_item_status_history(
                "i1",
                &[
                    ("new", "2026-06-01T00:00:00Z".to_string()),
                    ("indeterminate", "2026-06-03T00:00:00Z".to_string()),
                    ("done", "2026-06-06T00:00:00Z".to_string()),
                ],
            )
            .await
            .unwrap();
        // i2: still in progress, entered indeterminate on day 4, no done.
        store
            .replace_work_item_status_history(
                "i2",
                &[
                    ("new", "2026-06-01T00:00:00Z".to_string()),
                    ("indeterminate", "2026-06-04T00:00:00Z".to_string()),
                ],
            )
            .await
            .unwrap();

        let m = board_flow_metrics(
            &store,
            &["repo:acme/widget".to_string()],
            "2026-06-05T00:00:00Z",
            "2026-06-20T00:00:00Z",
            "2026-06-10T00:00:00Z",
        )
        .await
        .unwrap();
        assert_eq!(m.completed, 1);
        assert_eq!(m.median_lead_time_secs, Some(5 * 86_400)); // 5 days
        assert_eq!(m.median_wait_secs, Some(2 * 86_400)); // new day1 -> in-progress day3
        assert_eq!(m.median_active_secs, Some(3 * 86_400)); // in-progress day3 -> done day6
        assert_eq!(m.in_progress, 1);
        assert_eq!(m.median_in_progress_age_secs, Some(6 * 86_400)); // day 4 -> day 10
    }

    #[tokio::test]
    async fn board_link_coverage_counts_linked_merged_prs() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let t = "2026-06-01T00:00:00Z";
        let m = Some("2026-06-03T00:00:00Z");
        store
            .upsert_pull_request(&pr("p1", "closed", t, m))
            .await
            .unwrap();
        store
            .upsert_pull_request(&pr("p2", "closed", t, m))
            .await
            .unwrap();
        store
            .upsert_pull_request(&pr("p3", "open", t, None))
            .await
            .unwrap(); // not merged
        store
            .add_link("pull_request", "p1", "work_item", "i1", "closes")
            .await
            .unwrap();

        let cov = board_link_coverage(&store, &["repo:acme/widget".into()])
            .await
            .unwrap();
        assert_eq!(cov.merged_total, 2, "p1 + p2 merged, p3 open");
        assert_eq!(cov.linked, 1, "only p1 has a work-item link");
        assert_eq!(cov.closes, 1);
    }

    #[tokio::test]
    async fn attention_detail_groups_flagged_items_with_their_facts() {
        let store = seeded().await; // p1 merged-no-review, p2 open+stale (1 approved review), p3 fresh
        store
            .upsert_ci_run(&CiRun {
                id: "run1".into(),
                repo_id: "repo:acme/widget".into(),
                commit_sha: Some("aaa".into()),
                status: "completed".into(),
                conclusion: Some("failure".into()),
                completed_at: Some("2026-06-19T00:00:00Z".into()),
                html_url: None,
                run_attempt: None,
            })
            .await
            .unwrap();

        let d = attention_detail(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        // Two flagged PRs: p2 (stale) and p1 (merged without review); p3 is fresh and absent.
        assert_eq!(d.prs.len(), 2);
        let stale = d.prs.iter().find(|p| p.kind == "stale_pr").unwrap();
        assert_eq!(stale.id, "p2");
        assert_eq!(stale.review_count, 1); // its one approved review is counted
        assert!(stale.approved);
        let mwr = d
            .prs
            .iter()
            .find(|p| p.kind == "merged_without_review")
            .unwrap();
        assert_eq!(mwr.id, "p1");
        assert_eq!(mwr.review_count, 0);
        assert!(!mwr.approved);
        // One failing CI run, carrying its own facts.
        assert_eq!(d.ci.len(), 1);
        assert_eq!(d.ci[0].id, "run1");
        assert_eq!(d.ci[0].conclusion.as_deref(), Some("failure"));
    }

    #[tokio::test]
    async fn board_attention_unions_repos_and_flags_team() {
        let store = Store::open_in_memory().await.unwrap();
        for (id, name) in [("repo:a", "a"), ("repo:b", "b")] {
            store
                .upsert_repo(&Repo {
                    id: id.into(),
                    owner: "acme".into(),
                    name: name.into(),
                    full_name: format!("acme/{name}"),
                    ownership: "owned".into(),
                })
                .await
                .unwrap();
        }
        // One stale open PR in each repo; alice authors a's, carol authors b's.
        let stale = |id: &str, repo: &str, who: &str| PullRequest {
            id: id.into(),
            repo_id: repo.into(),
            number: 1,
            title: "T".into(),
            state: "open".into(),
            author_login: Some(who.into()),
            body: None,
            created_at: "2026-06-01T00:00:00Z".into(),
            merged_at: None,
            html_url: None,
        };
        store
            .upsert_pull_request(&stale("pa", "repo:a", "alice"))
            .await
            .unwrap();
        store
            .upsert_pull_request(&stale("pb", "repo:b", "carol"))
            .await
            .unwrap();

        let repos = vec!["repo:a".to_string(), "repo:b".to_string()];
        let items = board_attention(&store, &repos, &["alice".into()], "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        // Union across both repos.
        assert!(items
            .iter()
            .any(|i| i.repo_id == "repo:a" && i.kind == "stale_pr"));
        assert!(items
            .iter()
            .any(|i| i.repo_id == "repo:b" && i.kind == "stale_pr"));
        // alice is on the board, carol is not.
        assert!(
            items
                .iter()
                .find(|i| subject_id(&i.entities) == "pa")
                .unwrap()
                .by_team
        );
        assert!(
            !items
                .iter()
                .find(|i| subject_id(&i.entities) == "pb")
                .unwrap()
                .by_team
        );

        // The Changes grid draws PRs from both repos, each tagged with its repo.
        let prs = board_pull_requests(&store, &repos).await.unwrap();
        assert_eq!(prs.len(), 2);
        assert!(prs.iter().any(|p| p.repo_id == "repo:a" && p.id == "pa"));
        assert!(prs.iter().any(|p| p.repo_id == "repo:b" && p.id == "pb"));
    }

    #[tokio::test]
    async fn attention_respects_configured_stale_threshold() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:a".into(),
                owner: "acme".into(),
                name: "a".into(),
                full_name: "acme/a".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        store
            .upsert_pull_request(&PullRequest {
                id: "p1".into(),
                repo_id: "repo:a".into(),
                number: 1,
                title: "t".into(),
                state: "open".into(),
                author_login: None,
                body: None,
                created_at: "2026-06-01T00:00:00Z".into(),
                merged_at: None,
                html_url: None,
            })
            .await
            .unwrap();
        let now = "2026-06-11T00:00:00Z"; // PR is 10 days old

        // Default threshold (7) -> stale.
        let items = attention(&store, "repo:a", now).await.unwrap();
        assert!(items.iter().any(|i| i.kind == "stale_pr"));

        // Raise the threshold to 30 -> not stale.
        store
            .update_settings(&Settings {
                sync_period_secs: 300,
                stale_pr_days: 30,
                digest_webhook_url: None,
                digest_schedule_hours: None,
                llm_enabled: true,
                llm_model: None,
                storage_budget_mb: None,
            })
            .await
            .unwrap();
        let items = attention(&store, "repo:a", now).await.unwrap();
        assert!(!items.iter().any(|i| i.kind == "stale_pr"));
    }

    #[tokio::test]
    async fn board_trend_sums_snapshots_per_day_across_repos() {
        use core_store::MetricSnapshot;
        let store = Store::open_in_memory().await.unwrap();
        let snap = |repo: &str, day: &str, wip: i64, attn: i64| MetricSnapshot {
            scope_kind: "repo".into(),
            scope_id: repo.into(),
            captured_on: day.into(),
            wip,
            stale_open_prs: 0,
            merged_without_review: 0,
            attention_count: attn,
            median_cycle_time_secs: None,
            median_pickup_secs: None,
            median_review_secs: None,
        };
        // Day 1: A wip 2, B wip 3 -> total 5. Day 2: A wip 1.
        store
            .upsert_metric_snapshot(&snap("repo:a", "2026-06-10", 2, 1))
            .await
            .unwrap();
        store
            .upsert_metric_snapshot(&snap("repo:b", "2026-06-10", 3, 0))
            .await
            .unwrap();
        store
            .upsert_metric_snapshot(&snap("repo:a", "2026-06-11", 1, 0))
            .await
            .unwrap();

        let repos = vec!["repo:a".to_string(), "repo:b".to_string()];
        let trend = board_trend(&store, &repos).await.unwrap();
        assert_eq!(trend.len(), 2);
        assert_eq!(trend[0].captured_on, "2026-06-10");
        assert_eq!(trend[0].wip, 5);
        assert_eq!(trend[0].attention_count, 1);
        assert_eq!(trend[1].captured_on, "2026-06-11");
        assert_eq!(trend[1].wip, 1);

        // No snapshots -> empty trend, not an error.
        assert!(board_trend(&store, &["repo:none".to_string()])
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn person_activity_aggregates_across_repos_for_a_login() {
        let store = Store::open_in_memory().await.unwrap();
        for (id, name) in [("repo:a", "a"), ("repo:b", "b")] {
            store
                .upsert_repo(&Repo {
                    id: id.into(),
                    owner: "acme".into(),
                    name: name.into(),
                    full_name: format!("acme/{name}"),
                    ownership: "owned".into(),
                })
                .await
                .unwrap();
        }
        let mkpr = |id: &str, repo: &str, n: i64, state: &str, who: &str, merged: Option<&str>| {
            PullRequest {
                id: id.into(),
                repo_id: repo.into(),
                number: n,
                title: format!("PR {id}"),
                state: state.into(),
                author_login: Some(who.into()),
                body: None,
                created_at: "2026-06-01T00:00:00Z".into(),
                merged_at: merged.map(Into::into),
                html_url: None,
            }
        };
        // alice: open PR in a, merged PR in b. bob: an open PR in a (excluded).
        store
            .upsert_pull_request(&mkpr("pa", "repo:a", 1, "open", "alice", None))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr(
                "pb",
                "repo:b",
                2,
                "closed",
                "alice",
                Some("2026-06-05T00:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr("pc", "repo:a", 3, "open", "bob", None))
            .await
            .unwrap();
        store
            .upsert_review(&Review {
                id: "r1".into(),
                pr_id: "pa".into(),
                reviewer_login: Some("alice".into()),
                state: "approved".into(),
                submitted_at: Some("2026-06-02T00:00:00Z".into()),
            })
            .await
            .unwrap();
        store
            .upsert_commit(&Commit {
                sha: "c1".into(),
                repo_id: "repo:b".into(),
                author_login: Some("alice".into()),
                message: "m".into(),
                committed_at: "2026-06-03T00:00:00Z".into(),
            })
            .await
            .unwrap();

        let repos = vec!["repo:a".to_string(), "repo:b".to_string()];
        let act = person_activity(&store, &repos, "alice").await.unwrap();
        assert_eq!(act.open_prs.len(), 1);
        assert_eq!(act.open_prs[0].number, 1);
        assert_eq!(act.merged_prs, 1);
        assert_eq!(act.reviews_given, 1);
        assert_eq!(act.commits, 1);
        assert_eq!(act.repos, vec!["repo:a".to_string(), "repo:b".to_string()]);

        // Unknown login -> empty, not an error.
        let none = person_activity(&store, &repos, "nobody").await.unwrap();
        assert!(none.open_prs.is_empty());
        assert_eq!(none.commits, 0);
        assert!(none.repos.is_empty());
    }

    #[tokio::test]
    async fn contributors_discovers_logins_active_in_the_window() {
        let store = Store::open_in_memory().await.unwrap();
        for (id, name) in [("repo:a", "a"), ("repo:b", "b")] {
            store
                .upsert_repo(&Repo {
                    id: id.into(),
                    owner: "acme".into(),
                    name: name.into(),
                    full_name: format!("acme/{name}"),
                    ownership: "owned".into(),
                })
                .await
                .unwrap();
        }
        let mkpr =
            |id: &str, repo: &str, state: &str, who: &str, created: &str, merged: Option<&str>| {
                PullRequest {
                    id: id.into(),
                    repo_id: repo.into(),
                    number: 1,
                    title: format!("PR {id}"),
                    state: state.into(),
                    author_login: Some(who.into()),
                    body: None,
                    created_at: created.into(),
                    merged_at: merged.map(Into::into),
                    html_url: None,
                }
            };
        // alice: a currently-open PR (counts regardless of the window via current load).
        store
            .upsert_pull_request(&mkpr(
                "pa",
                "repo:a",
                "open",
                "alice",
                "2020-01-01T00:00:00Z",
                None,
            ))
            .await
            .unwrap();
        // bob: a PR opened inside the window. eve: a closed PR opened before it -> excluded.
        store
            .upsert_pull_request(&mkpr(
                "pb",
                "repo:a",
                "closed",
                "bob",
                "2026-06-01T00:00:00Z",
                None,
            ))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr(
                "pe",
                "repo:b",
                "closed",
                "eve",
                "2026-01-01T00:00:00Z",
                None,
            ))
            .await
            .unwrap();
        // carol: a review inside the window. dave: a commit inside. frank: a commit before it.
        store
            .upsert_review(&Review {
                id: "r1".into(),
                pr_id: "pb".into(),
                reviewer_login: Some("carol".into()),
                state: "approved".into(),
                submitted_at: Some("2026-06-02T00:00:00Z".into()),
            })
            .await
            .unwrap();
        store
            .upsert_commit(&Commit {
                sha: "c1".into(),
                repo_id: "repo:b".into(),
                author_login: Some("dave".into()),
                message: "m".into(),
                committed_at: "2026-06-03T00:00:00Z".into(),
            })
            .await
            .unwrap();
        store
            .upsert_commit(&Commit {
                sha: "c2".into(),
                repo_id: "repo:b".into(),
                author_login: Some("frank".into()),
                message: "old".into(),
                committed_at: "2026-01-01T00:00:00Z".into(),
            })
            .await
            .unwrap();

        let repos = vec!["repo:a".to_string(), "repo:b".to_string()];
        let people = contributors(
            &store,
            &repos,
            "2026-05-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        )
        .await
        .unwrap();
        // Sorted + deduplicated; only those active in the window (plus the open-PR author).
        assert_eq!(people, vec!["alice", "bob", "carol", "dave"]);
    }

    #[test]
    fn weekday_and_off_hours_are_correct() {
        // 2026-06-12 is a Friday; the 13th a Saturday; the 14th a Sunday.
        assert_eq!(weekday(2026, 6, 12), 5);
        assert_eq!(weekday(2026, 6, 13), 6);
        assert_eq!(weekday(2026, 6, 14), 0);
        // Weekday daytime is on-hours; weekend any time and weekday night are off-hours.
        assert!(!is_off_hours("2026-06-12T10:00:00Z"));
        assert!(is_off_hours("2026-06-12T23:00:00Z")); // Friday 23:00
        assert!(is_off_hours("2026-06-12T05:00:00Z")); // Friday 05:00
        assert!(is_off_hours("2026-06-13T10:00:00Z")); // Saturday
        assert!(!is_off_hours("garbage")); // malformed -> ignored, not counted
                                           // Malformed month/day/hour is ignored, never panics.
        assert!(!is_off_hours("2026-00-12T10:00:00Z")); // month 00
        assert!(!is_off_hours("2026-13-12T10:00:00Z")); // month 13
        assert!(!is_off_hours("2026-06-32T10:00:00Z")); // day 32
        assert!(!is_off_hours("2026-06-12T99:00:00Z")); // hour 99
    }

    #[tokio::test]
    async fn board_people_stats_throughput_plus_team_off_hours() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:a".into(),
                owner: "acme".into(),
                name: "a".into(),
                full_name: "acme/a".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let mut pr = pr("p1", "open", "2026-06-10T10:00:00Z", None);
        pr.repo_id = "repo:a".into();
        pr.author_login = Some("alice".into());
        store.upsert_pull_request(&pr).await.unwrap();
        // alice: a weekday-daytime commit (on-hours) + a Saturday commit (off-hours). The PR was
        // opened weekday daytime, so off-hours is the one Saturday commit.
        for (sha, ts) in [
            ("c1", "2026-06-12T10:00:00Z"),
            ("c2", "2026-06-13T11:00:00Z"),
        ] {
            store
                .upsert_commit(&Commit {
                    sha: sha.into(),
                    repo_id: "repo:a".into(),
                    author_login: Some("alice".into()),
                    message: "m".into(),
                    committed_at: ts.into(),
                })
                .await
                .unwrap();
        }
        // A failing CI run on alice's Saturday commit c2 (in window) -> alice's ci_failures.
        store
            .upsert_ci_run(&CiRun {
                id: "r1".into(),
                repo_id: "repo:a".into(),
                commit_sha: Some("c2".into()),
                status: "completed".into(),
                conclusion: Some("failure".into()),
                completed_at: Some("2026-06-13T12:00:00Z".into()),
                html_url: None,
                run_attempt: None,
            })
            .await
            .unwrap();

        let repos = vec!["repo:a".to_string()];
        let logins = vec!["alice".to_string(), "bob".to_string()];
        let stats = board_people_stats(
            &store,
            &repos,
            &logins,
            "2026-06-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        )
        .await
        .unwrap();
        assert_eq!(
            stats.people.len(),
            2,
            "every board person gets a row, even quiet bob"
        );
        let alice = &stats.people[0];
        assert_eq!(alice.login, "alice");
        assert_eq!(alice.commits, 2);
        assert_eq!(alice.prs_opened, 1);
        assert_eq!(alice.open_prs, 1);
        // alice has 1 off-hours event (the Saturday commit) of 3, and 1 CI failure (run on c2).
        assert_eq!(alice.off_hours_events, 1, "the Saturday commit only");
        assert_eq!(alice.total_events, 3);
        assert_eq!(alice.ci_failures, 1, "the failing run on her commit c2");
        let bob = &stats.people[1];
        assert_eq!(bob.login, "bob");
        assert_eq!(bob.commits, 0);
        assert_eq!(bob.ci_failures, 0);
        // The team off-hours rollup is the sum: 3 events, 1 off-hours.
        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.off_hours_events, 1);
    }

    #[tokio::test]
    async fn board_people_stats_zero_activity_is_zero_not_misleading() {
        // A board with people but no timestamped work in the window reports zero events.
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:a".into(),
                owner: "acme".into(),
                name: "a".into(),
                full_name: "acme/a".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let repos = vec!["repo:a".to_string()];
        let logins = vec!["alice".to_string(), "bob".to_string()];
        let stats = board_people_stats(
            &store,
            &repos,
            &logins,
            "2026-06-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        )
        .await
        .unwrap();
        assert_eq!(stats.total_events, 0, "no activity -> no denominator");
        assert_eq!(stats.off_hours_events, 0);
        assert_eq!(stats.people.len(), 2, "every board person still gets a row");
        assert!(stats
            .people
            .iter()
            .all(|p| p.commits == 0 && p.prs_opened == 0));
    }

    #[tokio::test]
    async fn board_people_stats_aggregates_one_person_across_repos() {
        // Login-keyed accumulators aggregate a person across repos: active days are a union (the
        // same UTC day in two repos counts once), and CI failures sum.
        let store = Store::open_in_memory().await.unwrap();
        for (id, name) in [("repo:a", "a"), ("repo:b", "b")] {
            store
                .upsert_repo(&Repo {
                    id: id.into(),
                    owner: "acme".into(),
                    name: name.into(),
                    full_name: format!("acme/{name}"),
                    ownership: "owned".into(),
                })
                .await
                .unwrap();
        }
        // alice commits on the same day in BOTH repos (one active day), plus a day in repo:b.
        for (repo, sha, at) in [
            ("repo:a", "a1", "2026-06-10T10:00:00Z"),
            ("repo:b", "b1", "2026-06-10T11:00:00Z"),
            ("repo:b", "b2", "2026-06-11T10:00:00Z"),
        ] {
            store
                .upsert_commit(&Commit {
                    sha: sha.into(),
                    repo_id: repo.into(),
                    author_login: Some("alice".into()),
                    message: "c".into(),
                    committed_at: at.into(),
                })
                .await
                .unwrap();
        }
        // A failing CI run on alice's head commit in each repo.
        for (id, repo, sha) in [("r1", "repo:a", "a1"), ("r2", "repo:b", "b1")] {
            store
                .upsert_ci_run(&CiRun {
                    id: id.into(),
                    repo_id: repo.into(),
                    commit_sha: Some(sha.into()),
                    status: "completed".into(),
                    conclusion: Some("failure".into()),
                    completed_at: Some("2026-06-12T00:00:00Z".into()),
                    html_url: None,
                    run_attempt: None,
                })
                .await
                .unwrap();
        }
        let stats = board_people_stats(
            &store,
            &["repo:a".to_string(), "repo:b".to_string()],
            &["alice".to_string()],
            "2026-06-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        )
        .await
        .unwrap();
        let alice = &stats.people[0];
        assert_eq!(alice.commits, 3, "all three commits across both repos");
        assert_eq!(
            alice.active_days, 2,
            "06-10 (both repos, counted once) + 06-11"
        );
        assert_eq!(
            alice.ci_failures, 2,
            "a failing run attributed in each repo"
        );
    }

    #[tokio::test]
    async fn board_people_stats_cycle_time_approvals_and_self_merges() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:a".into(),
                owner: "acme".into(),
                name: "a".into(),
                full_name: "acme/a".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        // alice merges two PRs: pr1 (2-day cycle) approved by bob; pr2 (1-day cycle) unreviewed.
        for (id, merged, secs) in [
            ("pr1", "2026-06-12T10:00:00Z", "2026-06-10T10:00:00Z"),
            ("pr2", "2026-06-11T10:00:00Z", "2026-06-10T10:00:00Z"),
        ] {
            let mut p = pr(id, "merged", secs, Some(merged));
            p.repo_id = "repo:a".into();
            p.author_login = Some("alice".into());
            store.upsert_pull_request(&p).await.unwrap();
        }
        store
            .upsert_review(&Review {
                id: "rv1".into(),
                pr_id: "pr1".into(),
                reviewer_login: Some("bob".into()),
                state: "APPROVED".into(),
                // pr1 opened 2026-06-10T10:00 -> bob's review a day later = 86400s latency.
                submitted_at: Some("2026-06-11T10:00:00Z".into()),
            })
            .await
            .unwrap();
        // File data so PR size is measurable: pr1 = 10 lines, pr2 = 20 -> alice avg 15.
        store
            .replace_pr_files(
                "pr1",
                &[PrFile {
                    pr_id: "pr1".into(),
                    filename: "a.rs".into(),
                    status: "modified".into(),
                    additions: 7,
                    deletions: 3,
                    patch: None,
                }],
            )
            .await
            .unwrap();
        store
            .replace_pr_files(
                "pr2",
                &[PrFile {
                    pr_id: "pr2".into(),
                    filename: "b.rs".into(),
                    status: "modified".into(),
                    additions: 15,
                    deletions: 5,
                    patch: None,
                }],
            )
            .await
            .unwrap();
        // pr1 authoritatively closes one issue -> alice resolved 1.
        store.replace_pr_closing_issues("pr1", &[1]).await.unwrap();
        // A commit by alice on a third day, so active-days spans two distinct days.
        store
            .upsert_commit(&Commit {
                sha: "c1".into(),
                repo_id: "repo:a".into(),
                author_login: Some("alice".into()),
                message: "fix".into(),
                committed_at: "2026-06-13T10:00:00Z".into(),
            })
            .await
            .unwrap();

        let repos = vec!["repo:a".to_string()];
        let logins = vec!["alice".to_string(), "bob".to_string()];
        let stats = board_people_stats(
            &store,
            &repos,
            &logins,
            "2026-06-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        )
        .await
        .unwrap();
        let alice = &stats.people[0];
        assert_eq!(alice.prs_merged, 2);
        // pr2 had no approving review -> one self-merge; pr1 was approved by bob.
        assert_eq!(alice.self_merges, 1);
        // Median of a 2-day and a 1-day cycle = 1.5 days.
        assert_eq!(alice.median_cycle_time_secs, Some(86400 + 43200));
        // Two PRs opened on 06-10 + a commit on 06-13 = two distinct active days.
        assert_eq!(alice.active_days, 2);
        // Average of a 10-line and a 20-line PR.
        assert_eq!(alice.avg_pr_churn, Some(15));
        // pr1's one closing issue.
        assert_eq!(alice.issues_closed, 1);
        // alice gave no reviews, so no latency.
        assert_eq!(alice.median_review_latency_secs, None);
        let bob = &stats.people[1];
        assert_eq!(bob.reviews_given, 1);
        assert_eq!(bob.approvals_given, 1);
        assert_eq!(bob.self_merges, 0);
        // bob reviewed pr1 a day after it opened.
        assert_eq!(bob.median_review_latency_secs, Some(86400));
        // bob authored no merged PR with files / closing issues.
        assert_eq!(bob.avg_pr_churn, None);
        assert_eq!(bob.issues_closed, 0);
    }

    #[tokio::test]
    async fn upstream_alerts_flag_depended_on_repos_with_attention() {
        let store = Store::open_in_memory().await.unwrap();
        for (id, name) in [("repo:a", "a"), ("repo:b", "b")] {
            store
                .upsert_repo(&Repo {
                    id: id.into(),
                    owner: "acme".into(),
                    name: name.into(),
                    full_name: format!("acme/{name}"),
                    ownership: "owned".into(),
                })
                .await
                .unwrap();
        }
        store
            .add_link("repo", "repo:a", "repo", "repo:b", "depends_on")
            .await
            .unwrap();

        // B is clean -> A has no upstream alerts.
        let none = upstream_alerts(&store, "repo:a", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        assert!(none.is_empty());

        // B gets a failing CI run -> A is alerted about B.
        store
            .upsert_ci_run(&CiRun {
                id: "r1".into(),
                repo_id: "repo:b".into(),
                commit_sha: Some("x".into()),
                status: "completed".into(),
                conclusion: Some("failure".into()),
                completed_at: None,
                html_url: None,
                run_attempt: None,
            })
            .await
            .unwrap();
        let alerts = upstream_alerts(&store, "repo:a", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].repo_id, "repo:b");
        assert_eq!(alerts[0].attention_count, 1);
    }

    #[tokio::test]
    async fn change_digest_composes_pr_body_reviews_and_links() {
        let store = seeded().await;
        store
            .upsert_issue(&Issue {
                id: "i10".into(),
                repo_id: "repo:acme/widget".into(),
                number: 10,
                title: "a bug".into(),
                state: "open".into(),
                author_login: None,
                body: None,
                created_at: "2026-06-01T00:00:00Z".into(),
                closed_at: None,
                labels: String::new(),
                html_url: None,
            })
            .await
            .unwrap();
        // p2 already exists from seeded(); re-upsert it with a body (idempotent by id).
        store
            .upsert_pull_request(&pr_with_body(
                "p2",
                "open",
                "2026-06-01T00:00:00Z",
                None,
                Some("Fixes #10"),
            ))
            .await
            .unwrap();
        store
            .add_link("pull_request", "p2", "work_item", "i10", "closes")
            .await
            .unwrap();
        // The diff: two changed files (+10/-3 total).
        store
            .replace_pr_files(
                "p2",
                &[
                    core_store::PrFile {
                        pr_id: "p2".into(),
                        filename: "a.rs".into(),
                        status: "modified".into(),
                        additions: 6,
                        deletions: 1,
                        patch: Some("@@ a @@".into()),
                    },
                    core_store::PrFile {
                        pr_id: "p2".into(),
                        filename: "b.rs".into(),
                        status: "added".into(),
                        additions: 4,
                        deletions: 2,
                        patch: None,
                    },
                ],
            )
            .await
            .unwrap();

        let d = change_digest(&store, "p2").await.unwrap().unwrap();
        assert_eq!(d.review_count, 1);
        assert!(d.approved); // p2's review is APPROVED
        assert_eq!(d.body.as_deref(), Some("Fixes #10"));
        assert_eq!(d.linked_issues.len(), 1);
        assert_eq!(d.linked_issues[0].number, 10);
        assert_eq!(d.linked_issues[0].relation, "closes");
        assert!(d.prose.contains("closes #10"), "prose was: {}", d.prose);
        // The diff is composed in: two files, +10/-3.
        assert_eq!(d.files.len(), 2);
        assert_eq!(d.additions, 10);
        assert_eq!(d.deletions, 3);

        // Unknown PR -> None, not an error.
        assert!(change_digest(&store, "nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn digest_round_trips_through_the_store() {
        let store = seeded().await;
        let d = repo_digest(&store, "repo:acme/widget", "2026-06-20T00:00:00Z")
            .await
            .unwrap();
        save_digest(&store, &d).await.unwrap();
        let loaded = load_digest(&store, "repo", "repo:acme/widget")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded, d);
        assert!(load_digest(&store, "repo", "nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn board_scorecard_composite_is_the_weakest_repo() {
        let store = Store::open_in_memory().await.unwrap();
        for full in ["acme/good", "acme/bad"] {
            store
                .upsert_repo(&Repo {
                    id: format!("repo:{full}"),
                    owner: "acme".into(),
                    name: full.split('/').nth(1).unwrap().into(),
                    full_name: full.into(),
                    ownership: "owned".into(),
                })
                .await
                .unwrap();
        }
        // good repo: a merged PR with a review -> passes the Bronze rules.
        store
            .upsert_pull_request(&PullRequest {
                id: "g1".into(),
                repo_id: "repo:acme/good".into(),
                number: 1,
                title: "PR".into(),
                state: "closed".into(),
                author_login: Some("alice".into()),
                body: None,
                created_at: "2026-06-19T00:00:00Z".into(),
                merged_at: Some("2026-06-19T06:00:00Z".into()),
                html_url: None,
            })
            .await
            .unwrap();
        store
            .upsert_review(&Review {
                id: "rv1".into(),
                pr_id: "g1".into(),
                reviewer_login: Some("bob".into()),
                state: "APPROVED".into(),
                submitted_at: Some("2026-06-19T03:00:00Z".into()),
            })
            .await
            .unwrap();
        // bad repo: a failing CI run -> fails the Bronze CI rule -> none.
        store
            .upsert_ci_run(&CiRun {
                id: "c1".into(),
                repo_id: "repo:acme/bad".into(),
                commit_sha: Some("x".into()),
                status: "completed".into(),
                conclusion: Some("failure".into()),
                completed_at: Some("2026-06-19T00:00:00Z".into()),
                html_url: None,
                run_attempt: None,
            })
            .await
            .unwrap();

        let sc = board_scorecard(
            &store,
            &["repo:acme/good".to_string(), "repo:acme/bad".to_string()],
            "2026-06-20T00:00:00Z",
        )
        .await
        .unwrap();
        // The bad repo drags the composite to "none".
        assert_eq!(sc.composite_tier, "none");
        assert_eq!(sc.none, 1);
        assert_eq!(sc.repos.len(), 2);
        let bad = sc.repos.iter().find(|r| r.repo_id.contains("bad")).unwrap();
        assert_eq!(bad.tier, "none");
    }

    #[tokio::test]
    async fn board_dependencies_aggregate_and_flag_shared() {
        let store = Store::open_in_memory().await.unwrap();
        for full in ["acme/a", "acme/b"] {
            store
                .upsert_repo(&Repo {
                    id: format!("repo:{full}"),
                    owner: "acme".into(),
                    name: full.split('/').nth(1).unwrap().into(),
                    full_name: full.into(),
                    ownership: "owned".into(),
                })
                .await
                .unwrap();
        }
        let dep = |repo: &str, name: &str| core_store::DependencyRow {
            repo_id: repo.into(),
            ecosystem: "cargo".into(),
            name: name.into(),
            version_req: Some("1".into()),
            kind: "normal".into(),
            source: "Cargo.toml".into(),
        };
        // serde is shared by both repos; tokio only by a.
        store
            .upsert_dependency(&dep("repo:acme/a", "serde"))
            .await
            .unwrap();
        store
            .upsert_dependency(&dep("repo:acme/b", "serde"))
            .await
            .unwrap();
        store
            .upsert_dependency(&dep("repo:acme/a", "tokio"))
            .await
            .unwrap();

        let deps = board_dependencies(
            &store,
            &["repo:acme/a".to_string(), "repo:acme/b".to_string()],
        )
        .await
        .unwrap();
        // serde leads (shared by 2).
        assert_eq!(deps[0].name, "serde");
        assert_eq!(deps[0].repos.len(), 2);
        let tokio = deps.iter().find(|d| d.name == "tokio").unwrap();
        assert_eq!(tokio.repos, vec!["repo:acme/a".to_string()]);
    }

    #[tokio::test]
    async fn trend_deltas_compute_delta_and_flag_anomaly() {
        let store = Store::open_in_memory().await.unwrap();
        // 9 days of a stable WIP (2), then a spike to 20 on the last day. (One row per day.)
        let wips = [2, 2, 3, 2, 2, 3, 2, 2, 20];
        for (i, w) in wips.iter().enumerate() {
            store
                .upsert_metric_snapshot(&MetricSnapshot {
                    scope_kind: "repo".into(),
                    scope_id: "repo:a".into(),
                    captured_on: format!("2026-06-{:02}", i + 1),
                    wip: *w,
                    stale_open_prs: 0,
                    merged_without_review: 0,
                    attention_count: 0,
                    median_cycle_time_secs: None,
                    median_pickup_secs: None,
                    median_review_secs: None,
                })
                .await
                .unwrap();
        }
        let t = trend_deltas(&store, &["repo:a".to_string()]).await.unwrap();
        assert_eq!(t.days, 9);
        let wip = t.deltas.iter().find(|d| d.metric == "wip").unwrap();
        assert_eq!(wip.current, 20);
        assert_eq!(wip.previous, 2); // ~7 days back (index n-8)
        assert_eq!(wip.delta, 18);
        assert!(
            wip.anomaly,
            "a 20 against a stable ~2 baseline is anomalous"
        );
        // A flat metric is not anomalous.
        let stale = t.deltas.iter().find(|d| d.metric == "stale_prs").unwrap();
        assert!(!stale.anomaly);
        // A consecutive-day spike has no same-weekday history, so it flags via the fallback.
        assert!(!wip.seasonal);
    }

    #[test]
    fn weekday_index_matches_known_dates() {
        assert_eq!(weekday_index("2026-06-14"), Some(0)); // Sunday
        assert_eq!(weekday_index("2026-06-15"), Some(1)); // Monday
        assert_eq!(weekday_index("2026-06-16"), Some(2)); // Tuesday
        assert_eq!(weekday_index("1970-01-01"), Some(4)); // Thursday
        assert_eq!(weekday_index("2000-01-01"), Some(6)); // Saturday
        assert_eq!(weekday_index("2026-06-16T12:00:00Z"), Some(2)); // RFC-3339 prefix parses
        assert_eq!(weekday_index("not-a-date"), None);
        assert_eq!(weekday_index("2026-13-40"), None); // out-of-range month/day
    }

    #[test]
    fn metric_delta_weekday_baseline_suppresses_seasonal_false_positive() {
        // June 2026: Tuesdays are 06-02/09/16/23/30. Tuesday is the busy day (~20), other days
        // quiet (4), so a normal Tuesday looks like an all-days spike.
        let tuesday = |day: u32| match day {
            2 => Some(18),
            9 => Some(20),
            16 => Some(22),
            23 => Some(20),
            _ => None,
        };
        let build = |last: i64| {
            let mut dates = Vec::new();
            let mut vals = Vec::new();
            for day in 1..=30u32 {
                dates.push(format!("2026-06-{day:02}"));
                vals.push(if day == 30 {
                    last
                } else {
                    tuesday(day).unwrap_or(4)
                });
            }
            (dates, vals)
        };

        // A normal Tuesday (20): high against the all-days mean, normal for a Tuesday.
        let (d, v) = build(20);
        let refs: Vec<&str> = d.iter().map(String::as_str).collect();
        let a = metric_delta("wip", &refs, &v);
        assert!(!a.anomaly, "20 is normal for a Tuesday");
        // ...yet the all-days baseline alone would have flagged it.
        assert!(is_anomalous(20, &v[..v.len() - 1], MIN_BASELINE_POINTS));

        // An abnormal Tuesday (4): unremarkable against the all-days mean, low for a Tuesday ->
        // flagged and marked seasonal.
        let (d, v) = build(4);
        let refs: Vec<&str> = d.iter().map(String::as_str).collect();
        let b = metric_delta("wip", &refs, &v);
        assert!(
            b.anomaly && b.seasonal,
            "4 is anomalously low for a Tuesday"
        );
        assert!(
            !is_anomalous(4, &v[..v.len() - 1], MIN_BASELINE_POINTS),
            "the all-days baseline would have missed it"
        );
    }

    #[tokio::test]
    async fn board_code_risk_fuses_and_attaches_drill_prs() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&core_store::Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let mkpr = |id: &str, num: i64, merged: &str| core_store::PullRequest {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number: num,
            title: format!("PR {num}"),
            state: "closed".into(),
            author_login: Some("alice".into()),
            body: None,
            created_at: format!("2026-0{num}-01T00:00:00Z"),
            merged_at: Some(merged.into()),
            html_url: Some(format!("https://h/pr/{num}")),
        };
        let file = |pr: &str| core_store::PrFile {
            pr_id: pr.into(),
            filename: "core/auth.rs".into(),
            status: "modified".into(),
            additions: 100,
            deletions: 50,
            patch: None,
        };
        // Two merged PRs touch core/auth.rs, both by alice; m1 reviewed, m2 not.
        store
            .upsert_pull_request(&mkpr("m1", 1, "2026-02-01T00:00:00Z"))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr("m2", 2, "2026-03-01T00:00:00Z"))
            .await
            .unwrap();
        store.replace_pr_files("m1", &[file("m1")]).await.unwrap();
        store.replace_pr_files("m2", &[file("m2")]).await.unwrap();
        store
            .upsert_review(&core_store::Review {
                id: "rv1".into(),
                pr_id: "m1".into(),
                reviewer_login: Some("bob".into()),
                state: "APPROVED".into(),
                submitted_at: Some("2026-01-15T00:00:00Z".into()),
            })
            .await
            .unwrap();

        let risks = board_code_risk(&store, &["repo:acme/widget".to_string()], 10)
            .await
            .unwrap();
        let auth = risks.iter().find(|r| r.path == "core/auth.rs").unwrap();
        assert_eq!(auth.top_author, "alice");
        assert_eq!(auth.bus_factor, 1);
        assert!((auth.review_gap - 0.5).abs() < 1e-9);
        assert!(auth.reasons.iter().any(|r| r.contains("single owner")));
        assert!(auth.reasons.iter().any(|r| r.contains("unreviewed")));
        // Drill: the touching PRs, newest first (m2 then m1).
        let nums: Vec<i64> = auth.recent_prs.iter().map(|p| p.number).collect();
        assert_eq!(nums, vec![2, 1]);
    }

    #[tokio::test]
    async fn board_coupling_surfaces_co_changing_file_pairs() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&core_store::Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let mkpr = |id: &str, num: i64| core_store::PullRequest {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number: num,
            title: format!("PR {num}"),
            state: "closed".into(),
            author_login: Some("alice".into()),
            body: None,
            created_at: format!("2026-0{num}-01T00:00:00Z"),
            merged_at: Some("2026-06-01T00:00:00Z".into()),
            html_url: None,
        };
        let f = |pr: &str, name: &str| core_store::PrFile {
            pr_id: pr.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: 1,
            deletions: 0,
            patch: None,
        };
        // auth.rs + db.rs co-appear in 3 merged PRs (together=3, prs each=3 -> Jaccard=1.0).
        // auth.rs + api.rs co-appear in only 1 PR (below MIN_TOGETHER=2, filtered out).
        for (id, num) in [("m1", 1), ("m2", 2), ("m3", 3)] {
            store.upsert_pull_request(&mkpr(id, num)).await.unwrap();
            store
                .replace_pr_files(id, &[f(id, "core/auth.rs"), f(id, "core/db.rs")])
                .await
                .unwrap();
        }
        store.upsert_pull_request(&mkpr("m4", 4)).await.unwrap();
        store
            .replace_pr_files("m4", &[f("m4", "core/auth.rs"), f("m4", "api/handler.rs")])
            .await
            .unwrap();

        let pairs = board_coupling(&store, &["repo:acme/widget".to_string()], 10)
            .await
            .unwrap();
        // Only auth+db passes MIN_TOGETHER=2.
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].path_a, "core/auth.rs");
        assert_eq!(pairs[0].path_b, "core/db.rs");
        assert_eq!(pairs[0].together, 3);
        // auth.rs: 4 PRs (m1-m4); db.rs: 3 (m1-m3); together=3 -> Jaccard 3/(4+3-3) = 0.75.
        assert!((pairs[0].coupling - 0.75).abs() < 1e-9, "Jaccard coupling");
    }

    #[tokio::test]
    async fn board_code_health_returns_the_weakest_files_first_and_honours_the_limit() {
        let store = Store::open_in_memory().await.unwrap();
        for (id, name) in [("repo:acme/widget", "widget"), ("repo:acme/api", "api")] {
            store
                .upsert_repo(&core_store::Repo {
                    id: id.into(),
                    owner: "acme".into(),
                    name: name.into(),
                    full_name: format!("acme/{name}"),
                    ownership: "owned".into(),
                })
                .await
                .unwrap();
        }
        // Scores across two repos, out of order, plus a tie to pin the path tie-break and an
        // unanalyzed Kotlin file that must not appear.
        for (repo_id, path, score) in [
            ("repo:acme/widget", "src/ok.rs", 9),
            ("repo:acme/widget", "src/worst.rs", 2),
            ("repo:acme/widget", "src/tie_b.rs", 4),
            ("repo:acme/api", "src/tie_a.rs", 4),
            ("repo:acme/api", "src/mid.rs", 6),
        ] {
            store
                .upsert_file_health(repo_id, path, 100, 5, 10, Some(score))
                .await
                .unwrap();
        }
        store
            .upsert_file_health("repo:acme/widget", "src/App.kt", 900, 0, 0, None)
            .await
            .unwrap();

        let repo_ids = vec!["repo:acme/widget".to_string(), "repo:acme/api".to_string()];
        let all = board_code_health(&store, &repo_ids, 10).await.unwrap();
        assert_eq!(
            all.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(),
            vec![
                "src/worst.rs",
                "src/tie_a.rs",
                "src/tie_b.rs",
                "src/mid.rs",
                "src/ok.rs",
            ],
            "weakest first, ties broken by path; the unanalyzed Kotlin file is absent"
        );
        assert_eq!(all[0].score, 2);
        assert_eq!(all[0].repo_id, "repo:acme/widget");
        assert_eq!(
            (all[0].loc, all[0].functions, all[0].branches),
            (100, 5, 10)
        );

        // The limit keeps the weakest, not an arbitrary slice.
        let capped = board_code_health(&store, &repo_ids, 2).await.unwrap();
        assert_eq!(
            capped.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(),
            vec!["src/worst.rs", "src/tie_a.rs"]
        );
    }

    #[tokio::test]
    async fn board_pickup_reports_load_recency_and_areas() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&core_store::Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let mkpr =
            |id: &str, num: i64, who: &str, state: &str, merged: Option<&str>, created: &str| {
                core_store::PullRequest {
                    id: id.into(),
                    repo_id: "repo:acme/widget".into(),
                    number: num,
                    title: format!("PR {num}"),
                    state: state.into(),
                    author_login: Some(who.into()),
                    body: None,
                    created_at: created.into(),
                    merged_at: merged.map(Into::into),
                    html_url: None,
                }
            };
        let file = |pr: &str, name: &str| core_store::PrFile {
            pr_id: pr.into(),
            filename: name.into(),
            status: "modified".into(),
            additions: 1,
            deletions: 0,
            patch: None,
        };
        // alice: one OPEN PR touching payments/. bob: one MERGED PR touching ui/, nothing open.
        store
            .upsert_pull_request(&mkpr(
                "pa",
                1,
                "alice",
                "open",
                None,
                "2026-06-10T00:00:00Z",
            ))
            .await
            .unwrap();
        store
            .upsert_pull_request(&mkpr(
                "pb",
                2,
                "bob",
                "closed",
                Some("2026-05-01T00:00:00Z"),
                "2026-04-20T00:00:00Z",
            ))
            .await
            .unwrap();
        store
            .replace_pr_files("pa", &[file("pa", "payments/api.rs")])
            .await
            .unwrap();
        store
            .replace_pr_files("pb", &[file("pb", "ui/list.tsx")])
            .await
            .unwrap();

        let logins = vec!["alice".to_string(), "bob".to_string()];
        let rows = board_pickup(&store, &["repo:acme/widget".to_string()], &logins, 4)
            .await
            .unwrap();
        // Returned in roster order (not ranked).
        assert_eq!(
            rows.iter().map(|r| r.login.as_str()).collect::<Vec<_>>(),
            vec!["alice", "bob"]
        );
        let alice = &rows[0];
        assert_eq!(alice.open_prs, 1, "one open PR in flight");
        assert_eq!(alice.last_active.as_deref(), Some("2026-06-10T00:00:00Z"));
        assert!(alice.areas.iter().any(|a| a.area == "payments"));
        let bob = &rows[1];
        assert_eq!(bob.open_prs, 0, "nothing in flight");
        assert!(bob.areas.iter().any(|a| a.area == "ui"));

        // Reordering the roster reorders the output identically.
        let flipped = board_pickup(
            &store,
            &["repo:acme/widget".to_string()],
            &["bob".to_string(), "alice".to_string()],
            4,
        )
        .await
        .unwrap();
        assert_eq!(
            flipped.iter().map(|r| r.login.as_str()).collect::<Vec<_>>(),
            vec!["bob", "alice"]
        );
    }

    #[test]
    fn conventional_category_maps_known_types() {
        assert_eq!(conventional_category("feat: add x"), Some("feature"));
        assert_eq!(conventional_category("fix(api): y"), Some("bug"));
        assert_eq!(conventional_category("docs: z"), Some("docs"));
        assert_eq!(conventional_category("chore: w"), Some("maintenance"));
        assert_eq!(
            conventional_category("refactor(core)!: big"),
            Some("maintenance")
        );
        assert_eq!(conventional_category("feat!: breaking"), Some("feature"));
        assert_eq!(conventional_category("test: t"), Some("test"));
        assert_eq!(conventional_category("Update stuff"), None); // no colon
        assert_eq!(conventional_category("WIP"), None);
        assert_eq!(conventional_category("Foo: not lowercase type"), None);
    }

    #[tokio::test]
    async fn board_investment_classifies_merged_prs() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&core_store::Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        let mut n = 0i64;
        let mut mk = |title: &str| {
            n += 1;
            core_store::PullRequest {
                id: format!("p{n}"),
                repo_id: "repo:acme/widget".into(),
                number: n,
                title: title.into(),
                state: "closed".into(),
                author_login: Some("alice".into()),
                body: None,
                created_at: "2026-06-01T00:00:00Z".into(),
                merged_at: Some("2026-06-10T00:00:00Z".into()),
                html_url: None,
            }
        };
        for title in [
            "feat: x",
            "fix: y",
            "docs: z",
            "chore: w",
            "update stuff", // unprefixed, will link a bug -> bug
            "random thing", // unprefixed, no link -> other
        ] {
            store.upsert_pull_request(&mk(title)).await.unwrap();
        }
        // Churn for the feature PR, to check the secondary weight accumulates.
        store
            .replace_pr_files(
                "p1",
                &[core_store::PrFile {
                    pr_id: "p1".into(),
                    filename: "feature.rs".into(),
                    status: "modified".into(),
                    additions: 10,
                    deletions: 5,
                    patch: None,
                }],
            )
            .await
            .unwrap();
        // "update stuff" (p5) closes a bug-kind work item -> classified bug via the fallback.
        store
            .upsert_github_work_item("i1", "a bug", "bug", "closed", "done")
            .await
            .unwrap();
        store
            .add_link("pull_request", "p5", "work_item", "i1", "closes")
            .await
            .unwrap();

        let dist = board_investment(
            &store,
            &["repo:acme/widget".to_string()],
            "2026-06-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        )
        .await
        .unwrap();
        let get = |c: &str| dist.buckets.iter().find(|b| b.category == c).unwrap();
        assert_eq!(get("feature").count, 1);
        assert_eq!(get("feature").churn, 15);
        assert_eq!(
            get("bug").count,
            2,
            "fix: plus the bug-linked unprefixed PR"
        );
        assert_eq!(get("docs").count, 1);
        assert_eq!(get("maintenance").count, 1);
        assert_eq!(get("test").count, 0);
        assert_eq!(get("other").count, 1);
        assert_eq!(dist.total_count, 6);
        assert_eq!(dist.total_churn, 15);
    }

    #[test]
    fn attention_brief_leads_with_severity_and_action() {
        let item = |kind: &str| BoardAttentionItem {
            repo_id: "repo:a".into(),
            kind: kind.into(),
            entities: vec![LinkedEntity::subject(AttentionEntity::PullRequest {
                id: "p".into(),
                number: 1,
                title: "x".into(),
                url: None,
            })],
            summary: "x".into(),
            by_team: false,
            ts: None,
            wip_percentile: None,
            wip_percentile_basis: None,
        };
        let items = vec![
            item("stale_pr"),
            item("stale_pr"),
            item("review_wait"),
            item("review_wait"),
            item("review_wait"),
            item("failing_ci"),
        ];
        let brief = attention_brief(&items);
        assert!(brief.starts_with("6 item(s) need attention:"), "{brief}");
        // Severity order: failing CI leads, before awaiting-review and stale.
        let ci = brief.find("failing CI").unwrap();
        let rw = brief.find("awaiting review").unwrap();
        let stale = brief.find("stale PR").unwrap();
        assert!(ci < rw && rw < stale, "severity order: {brief}");
        // Leads with the most-severe kind's deterministic action.
        assert!(
            brief.contains("Start with the failing CI: Open the failing run."),
            "{brief}"
        );
        assert_eq!(attention_brief(&[]), "Nothing needs attention right now.");
    }

    #[test]
    fn attention_focus_input_ranks_by_severity_and_age_team_first_and_caps() {
        const NOW: &str = "2026-06-30T00:00:00Z";
        let item =
            |kind: &str, summary: &str, by_team: bool, ts: Option<&str>| BoardAttentionItem {
                repo_id: "repo:a".into(),
                kind: kind.into(),
                entities: vec![LinkedEntity::subject(AttentionEntity::PullRequest {
                    id: summary.into(),
                    number: 1,
                    title: summary.into(),
                    url: None,
                })],
                summary: summary.into(),
                by_team,
                ts: ts.map(String::from),
                wip_percentile: None,
                wip_percentile_basis: None,
            };
        let items = vec![
            item(
                "review_wait",
                "PR #1 other",
                false,
                Some("2026-06-28T00:00:00Z"),
            ),
            item(
                "review_wait",
                "PR #2 mine",
                true,
                Some("2026-06-29T06:00:00Z"),
            ),
            item(
                "stale_pr",
                "PR #3 stale",
                false,
                Some("2026-06-01T00:00:00Z"),
            ),
            item("failing_ci", "run #4 red", false, None),
        ];
        let out = attention_focus_input(&items, 3, NOW);
        // Header still carries the counts, in severity order.
        assert!(out.starts_with("4 item(s) need attention:"), "{out}");
        // Capped at 3 items, numbered 1 through 3.
        assert!(out.contains("1. "), "{out}");
        assert!(out.contains("2. "), "{out}");
        assert!(out.contains("3. "), "{out}");
        assert!(!out.contains("4. "), "capped at max: {out}");
        // Most severe kind (failing CI) leads the item list.
        let ci = out.find("[failing CI]").unwrap();
        let rw = out.find("[awaiting review, 18h old, your team]").unwrap();
        assert!(ci < rw, "severity order in items: {out}");
        // Within awaiting-review, the team item is flagged and comes first.
        let mine = out
            .find("PR #2 mine [awaiting review, 18h old, your team]")
            .unwrap();
        let other = out.find("PR #1 other [awaiting review, 2d old]").unwrap();
        assert!(mine < other, "team item first: {out}");
        // failing_ci has no ts: no age phrase.
        assert!(
            out.contains("run #4 red [failing CI]"),
            "no fabricated age: {out}"
        );
        assert_eq!(
            attention_focus_input(&[], 5, NOW),
            "Nothing needs attention right now."
        );
    }

    #[test]
    fn attention_focus_input_reads_as_data_not_a_list_to_echo() {
        const NOW: &str = "2026-06-30T00:00:00Z";
        let items = vec![BoardAttentionItem {
            repo_id: "repo:a".into(),
            kind: "failing_ci".into(),
            entities: vec![LinkedEntity::subject(AttentionEntity::PullRequest {
                id: "p".into(),
                number: 1,
                title: "x".into(),
                url: None,
            })],
            summary: "run #1 red".into(),
            by_team: false,
            ts: Some("2026-06-29T12:00:00Z".into()),
            wip_percentile: None,
            wip_percentile_basis: None,
        }];
        let out = attention_focus_input(&items, 6, NOW);
        // No markdown bullet list and no header naming itself a list.
        assert!(!out.contains("\n- "), "no bullet list: {out}");
        assert!(
            !out.to_lowercase().contains("most pressing items:"),
            "no list-shaped header: {out}"
        );
    }

    #[tokio::test]
    async fn board_epics_rolls_children_up_and_flags_stalled() {
        let store = Store::open_in_memory().await.unwrap();
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
                name: "Jira".into(),
                kind: "jira".into(),
                base_url: "https://j".into(),
                email: None,
            })
            .await
            .unwrap();
        store
            .link_board_tracker("board:1", "jiraA", "SCRUM")
            .await
            .unwrap();
        // Build a Jira issue row (sync); the caller awaits the upsert.
        let row =
            |key: &str, title: &str, cat: &str, parent: Option<&str>| core_store::JiraIssueRow {
                work_item_id: format!("jira:jiraA:{key}"),
                tracker_id: "jiraA".into(),
                project: "SCRUM".into(),
                issue_key: key.into(),
                title: title.into(),
                issue_type: if parent.is_none() {
                    "Epic".into()
                } else {
                    "Story".into()
                },
                status: cat.into(),
                status_category: Some(cat.into()),
                assignee: None,
                url: None,
                created_at: None,
                updated_at: Some("2026-06-10T00:00:00Z".into()),
                resolved_at: None,
                parent_key: parent.map(Into::into),
            };
        for r in [
            row("SCRUM-1", "The epic", "indeterminate", None),
            row("SCRUM-2", "child a", "done", Some("SCRUM-1")),
            row("SCRUM-3", "child b", "done", Some("SCRUM-1")),
            row("SCRUM-4", "child c", "new", Some("SCRUM-1")),
            row("SCRUM-5", "child d", "new", Some("SCRUM-1")),
        ] {
            store.upsert_jira_issue(&r).await.unwrap();
        }

        let epics = board_epics(&store, "board:1").await.unwrap();
        let e = epics.iter().find(|e| e.key == "SCRUM-1").unwrap();
        assert_eq!(e.title, "The epic");
        assert_eq!((e.total, e.done, e.in_progress), (4, 2, 0));
        assert!(e.at_risk, "2 of 4 done and nothing in progress -> stalled");

        // Move a child into progress -> not stalled.
        store
            .upsert_jira_issue(&row("SCRUM-4", "child c", "indeterminate", Some("SCRUM-1")))
            .await
            .unwrap();
        let epics = board_epics(&store, "board:1").await.unwrap();
        let e = epics.iter().find(|e| e.key == "SCRUM-1").unwrap();
        assert_eq!((e.total, e.done, e.in_progress), (4, 2, 1));
        assert!(!e.at_risk, "work is in progress -> not stalled");
    }

    #[tokio::test]
    async fn dora_metrics_over_store() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .upsert_repo(&Repo {
                id: "repo:acme/widget".into(),
                owner: "acme".into(),
                name: "widget".into(),
                full_name: "acme/widget".into(),
                ownership: "owned".into(),
            })
            .await
            .unwrap();
        // One merged PR (1-day cycle -> elite lead time), one revert commit (CFR proxy 1/1 = 1.0
        // -> low),
        // one release in the 30-day window ending at `now`.
        store
            .upsert_pull_request(&pr(
                "p1",
                "closed",
                "2026-06-19T00:00:00Z",
                Some("2026-06-19T12:00:00Z"),
            ))
            .await
            .unwrap();
        store
            .upsert_commit(&Commit {
                sha: "rev".into(),
                repo_id: "repo:acme/widget".into(),
                author_login: Some("a".into()),
                message: "Revert \"x\"".into(),
                committed_at: "2026-06-20T00:00:00Z".into(),
            })
            .await
            .unwrap();
        store
            .upsert_release(&core_store::Release {
                id: "rel1".into(),
                repo_id: "repo:acme/widget".into(),
                tag: "v1".into(),
                name: None,
                published_at: Some("2026-06-15T00:00:00Z".into()),
            })
            .await
            .unwrap();

        let d = dora_metrics(&store, "repo:acme/widget", "2026-06-29T00:00:00Z", 30)
            .await
            .unwrap();
        assert_eq!(d.lead_time_secs, Some(12 * 3600));
        assert_eq!(d.lead_tier, "elite");
        assert_eq!(d.change_failure_rate, Some(1.0));
        assert_eq!(d.cfr_tier, "low");
        assert_eq!(d.deploy_frequency_per_week, Some(1.0 / (30.0 / 7.0)));

        // Board pooling over a single repo matches the repo metrics.
        let b = board_dora(
            &store,
            &["repo:acme/widget".to_string()],
            "2026-06-29T00:00:00Z",
            30,
        )
        .await
        .unwrap();
        assert_eq!(b.lead_time_secs, d.lead_time_secs);
    }

    // --- sprint say-do ---------------------------------------------------------------

    fn jira_issue(key: &str, cat: &str) -> core_store::JiraIssueRow {
        core_store::JiraIssueRow {
            work_item_id: format!("jira:trk:{key}"),
            tracker_id: "trk".into(),
            project: "SCRUM".into(),
            issue_key: key.into(),
            title: format!("Issue {key}"),
            issue_type: "Task".into(),
            status: cat.into(),
            status_category: Some(cat.into()),
            assignee: None,
            url: None,
            created_at: None,
            updated_at: None,
            resolved_at: None,
            parent_key: None,
        }
    }

    async fn jira_board() -> Store {
        let store = Store::open_in_memory().await.unwrap();
        store
            .create_board(
                "board:t",
                "Team",
                core_store::BoardKind::Team,
                "2026-06-01T00:00:00Z",
            )
            .await
            .unwrap();
        store
            .upsert_tracker(&core_store::Tracker {
                id: "trk".into(),
                name: "Jira".into(),
                kind: "jira".into(),
                base_url: "https://x.atlassian.net".into(),
                email: Some("a@b.c".into()),
            })
            .await
            .unwrap();
        store
            .link_board_tracker("board:t", "trk", "SCRUM")
            .await
            .unwrap();
        store
    }

    fn ids(keys: &[&str]) -> Vec<String> {
        keys.iter().map(|k| format!("jira:trk:{k}")).collect()
    }

    #[tokio::test]
    async fn say_do_committed_vs_delivered_with_scope_churn() {
        let store = jira_board().await;
        // Five issues: 1,2 done; 3 in progress; 4,5 not started.
        for (k, c) in [
            ("SCRUM-1", "done"),
            ("SCRUM-2", "done"),
            ("SCRUM-3", "indeterminate"),
            ("SCRUM-4", "new"),
            ("SCRUM-5", "new"),
        ] {
            store.upsert_jira_issue(&jira_issue(k, c)).await.unwrap();
        }
        store
            .upsert_jira_sprint(&core_store::JiraSprintRow {
                sprint_id: "jira:trk:sprint:1".into(),
                tracker_id: "trk".into(),
                project: "SCRUM".into(),
                name: "Sprint 1".into(),
                state: Some("active".into()),
                starts_on: Some("2026-06-13".into()),
                ends_on: Some("2026-06-27".into()),
                committed_at: None,
            })
            .await
            .unwrap();
        // Committed at start: 1..4. Then 4 is pulled and 5 is added mid-sprint -> live = 1,2,3,5.
        store
            .record_sprint_commitment(
                "jira:trk:sprint:1",
                &ids(&["SCRUM-1", "SCRUM-2", "SCRUM-3", "SCRUM-4"]),
                "2026-06-13",
            )
            .await
            .unwrap();
        store
            .replace_sprint_work_items(
                "jira:trk:sprint:1",
                &ids(&["SCRUM-1", "SCRUM-2", "SCRUM-3", "SCRUM-5"]),
            )
            .await
            .unwrap();

        let rows = board_say_do(&store, "board:t").await.unwrap();
        assert_eq!(rows.len(), 1);
        let sd = &rows[0];
        assert_eq!(sd.committed, 4);
        assert_eq!(sd.delivered, 2); // 1,2 done
        assert_eq!(sd.carryover, 2); // 3,4 not done
        assert_eq!(sd.added, 1); // 5 added
        assert_eq!(sd.removed, 1); // 4 pulled
        assert_eq!(sd.ratio, Some(0.5));
        assert_eq!(sd.committed_on.as_deref(), Some("2026-06-13"));
        // The snapshot is frozen: a second observation must not re-snapshot.
        assert!(store
            .sprint_commitment_taken("jira:trk:sprint:1")
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn say_do_honest_empty_without_commitment() {
        let store = jira_board().await;
        store
            .upsert_jira_sprint(&core_store::JiraSprintRow {
                sprint_id: "jira:trk:sprint:9".into(),
                tracker_id: "trk".into(),
                project: "SCRUM".into(),
                name: "Future".into(),
                state: Some("future".into()),
                starts_on: None,
                ends_on: None,
                committed_at: None,
            })
            .await
            .unwrap();
        // No commitment snapshot -> no ratio, and committed_on stays None.
        let rows = board_say_do(&store, "board:t").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].committed, 0);
        assert_eq!(rows[0].ratio, None);
        assert_eq!(rows[0].committed_on, None);
    }

    #[tokio::test]
    async fn say_do_velocity_trend_across_sprints() {
        let store = jira_board().await;
        for (k, c) in [
            ("SCRUM-1", "done"),
            ("SCRUM-2", "done"),
            ("SCRUM-3", "indeterminate"),
        ] {
            store.upsert_jira_issue(&jira_issue(k, c)).await.unwrap();
        }
        // Sprint 1: closed, committed 2026-05-01, 2/2 done = 100%.
        store
            .upsert_jira_sprint(&core_store::JiraSprintRow {
                sprint_id: "jira:trk:sprint:1".into(),
                tracker_id: "trk".into(),
                project: "SCRUM".into(),
                name: "Sprint 1".into(),
                state: Some("closed".into()),
                starts_on: Some("2026-05-01".into()),
                ends_on: Some("2026-05-14".into()),
                committed_at: None,
            })
            .await
            .unwrap();
        store
            .record_sprint_commitment(
                "jira:trk:sprint:1",
                &ids(&["SCRUM-1", "SCRUM-2"]),
                "2026-05-01",
            )
            .await
            .unwrap();
        store
            .replace_sprint_work_items("jira:trk:sprint:1", &ids(&["SCRUM-1", "SCRUM-2"]))
            .await
            .unwrap();
        // Sprint 2: closed, committed 2026-05-15, 1/2 done = 50%.
        store
            .upsert_jira_sprint(&core_store::JiraSprintRow {
                sprint_id: "jira:trk:sprint:2".into(),
                tracker_id: "trk".into(),
                project: "SCRUM".into(),
                name: "Sprint 2".into(),
                state: Some("closed".into()),
                starts_on: Some("2026-05-15".into()),
                ends_on: Some("2026-05-28".into()),
                committed_at: None,
            })
            .await
            .unwrap();
        store
            .record_sprint_commitment(
                "jira:trk:sprint:2",
                &ids(&["SCRUM-1", "SCRUM-3"]),
                "2026-05-15",
            )
            .await
            .unwrap();
        store
            .replace_sprint_work_items("jira:trk:sprint:2", &ids(&["SCRUM-1", "SCRUM-3"]))
            .await
            .unwrap();
        // Sprint 3: future, no commitment yet, so excluded from the trend.
        store
            .upsert_jira_sprint(&core_store::JiraSprintRow {
                sprint_id: "jira:trk:sprint:3".into(),
                tracker_id: "trk".into(),
                project: "SCRUM".into(),
                name: "Sprint 3".into(),
                state: Some("future".into()),
                starts_on: Some("2026-05-29".into()),
                ends_on: None,
                committed_at: None,
            })
            .await
            .unwrap();

        let rows = board_say_do(&store, "board:t").await.unwrap();
        let trend = velocity_trend(&rows);
        // Only committed sprints, oldest-first.
        assert_eq!(trend.len(), 2);
        assert_eq!(trend[0].sprint_name, "Sprint 1");
        assert_eq!(trend[0].ratio, Some(1.0));
        assert_eq!(trend[1].sprint_name, "Sprint 2");
        assert_eq!(trend[1].ratio, Some(0.5));
    }

    #[tokio::test]
    async fn say_do_velocity_trend_orders_by_commit_date_not_store_order() {
        // A sprint with no start date sorts last in the store's `starts_on DESC` order; the trend
        // must place it by `committed_on`, not as the oldest point.
        let store = jira_board().await;
        store
            .upsert_jira_issue(&jira_issue("SCRUM-1", "done"))
            .await
            .unwrap();
        for (id, name, starts_on, committed_on) in [
            (
                "jira:trk:sprint:1",
                "Sprint 1",
                Some("2026-05-01"),
                "2026-05-01",
            ),
            ("jira:trk:sprint:2", "Sprint 2", None, "2026-05-15"),
            (
                "jira:trk:sprint:3",
                "Sprint 3",
                Some("2026-05-29"),
                "2026-05-29",
            ),
        ] {
            store
                .upsert_jira_sprint(&core_store::JiraSprintRow {
                    sprint_id: id.into(),
                    tracker_id: "trk".into(),
                    project: "SCRUM".into(),
                    name: name.into(),
                    state: Some("closed".into()),
                    starts_on: starts_on.map(str::to_string),
                    ends_on: None,
                    committed_at: None,
                })
                .await
                .unwrap();
            store
                .record_sprint_commitment(id, &ids(&["SCRUM-1"]), committed_on)
                .await
                .unwrap();
            store
                .replace_sprint_work_items(id, &ids(&["SCRUM-1"]))
                .await
                .unwrap();
        }

        let rows = board_say_do(&store, "board:t").await.unwrap();
        let trend = velocity_trend(&rows);
        let names: Vec<&str> = trend.iter().map(|s| s.sprint_name.as_str()).collect();
        assert_eq!(names, ["Sprint 1", "Sprint 2", "Sprint 3"]);
    }
}
