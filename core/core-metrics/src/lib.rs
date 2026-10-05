//! Flow metrics and watchdog rules.
//! Pure, deterministic functions over stored activity. They describe the system (flow,
//! bottlenecks, review distribution) and do not rank individuals. Timestamps are RFC-3339 strings.

use std::collections::HashMap;

use core_store::{CiRun, Commit, PullRequest, Release, Review};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

pub use core_model::{CyclePhases, DoraTier, PhaseMedians};

fn parse(ts: &str) -> Option<OffsetDateTime> {
    OffsetDateTime::parse(ts, &Rfc3339).ok()
}

/// Whole seconds from `a` to `b` (b - a), if both parse.
pub fn secs_between(a: &str, b: &str) -> Option<i64> {
    Some((parse(b)? - parse(a)?).whole_seconds())
}

/// Median of a set of values, or None if empty.
pub fn median(mut xs: Vec<i64>) -> Option<i64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_unstable();
    let n = xs.len();
    Some(if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2
    })
}

/// Percentile of a set of values using the nearest-rank method. `p` is in `0.0..=1.0`. Returns
/// `None` on an empty input. The input need not be sorted; it is sorted internally.
pub fn percentile(mut xs: Vec<i64>, p: f64) -> Option<i64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_unstable();
    let n = xs.len();
    // Nearest-rank: ordinal rank = ceil(p * n), clamped to [1, n], converted to 0-based index.
    let rank = ((p * n as f64).ceil() as usize).max(1).min(n);
    Some(xs[rank - 1])
}

// ---- flow metrics ----------------------------------------------------------

/// Cycle time of a single PR in seconds: creation to merge. None unless merged.
pub fn cycle_time_secs(pr: &PullRequest) -> Option<i64> {
    let merged = pr.merged_at.as_deref()?;
    secs_between(&pr.created_at, merged)
}

/// Median cycle time over the merged PRs in the set.
pub fn median_cycle_time_secs(prs: &[PullRequest]) -> Option<i64> {
    median(prs.iter().filter_map(cycle_time_secs).collect())
}

/// Lead time for changes: median over merged PRs of the gap from merge to the first release
/// published at or after it. `None` if no merged PR has a later release.
pub fn lead_time_to_deploy(prs: &[PullRequest], releases: &[Release]) -> Option<i64> {
    let mut deploys: Vec<OffsetDateTime> = releases
        .iter()
        .filter_map(|r| r.published_at.as_deref())
        .filter_map(parse)
        .collect();
    if deploys.is_empty() {
        return None;
    }
    deploys.sort();
    let gaps: Vec<i64> = prs
        .iter()
        .filter_map(|p| {
            let merged = parse(p.merged_at.as_deref()?)?;
            let deploy = deploys.iter().find(|d| **d >= merged)?;
            Some((*deploy - merged).whole_seconds())
        })
        .collect();
    median(gaps)
}

/// Work in progress: open, unmerged PRs.
pub fn wip_count(prs: &[PullRequest]) -> usize {
    prs.iter()
        .filter(|p| p.state == "open" && p.merged_at.is_none())
        .count()
}

/// Time from a PR's creation to its first review, in seconds. None if it has no dated review.
pub fn time_to_first_review_secs(pr: &PullRequest, reviews: &[Review]) -> Option<i64> {
    let first = first_review_at(pr, reviews)?;
    Some((first - parse(&pr.created_at)?).whole_seconds())
}

/// The earliest dated review of a PR. `reviews` may be the whole repo set.
fn first_review_at(pr: &PullRequest, reviews: &[Review]) -> Option<OffsetDateTime> {
    reviews
        .iter()
        .filter(|r| r.pr_id == pr.id)
        .filter_map(|r| r.submitted_at.as_deref())
        .filter_map(parse)
        .min()
}

/// Pickup and review phases for one PR. `reviews` may be the whole repo set.
pub fn cycle_phases(pr: &PullRequest, reviews: &[Review]) -> CyclePhases {
    let Some(merged) = pr.merged_at.as_deref().and_then(parse) else {
        return CyclePhases::default();
    };
    let created = parse(&pr.created_at);
    match first_review_at(pr, reviews) {
        // pickup = created -> first review, review = first review -> merge.
        Some(first) => CyclePhases {
            pickup_secs: created.map(|c| (first - c).whole_seconds()),
            review_secs: Some((merged - first).whole_seconds()),
        },
        // Never reviewed: the whole span counts as pickup.
        None => CyclePhases {
            pickup_secs: created.map(|c| (merged - c).whole_seconds()),
            review_secs: None,
        },
    }
}

/// Median pickup and review phase durations over the merged PRs in a set.
pub fn median_cycle_phases(prs: &[PullRequest], reviews: &[Review]) -> PhaseMedians {
    let phases: Vec<CyclePhases> = prs.iter().map(|p| cycle_phases(p, reviews)).collect();
    PhaseMedians {
        pickup_secs: median(phases.iter().filter_map(|p| p.pickup_secs).collect()),
        review_secs: median(phases.iter().filter_map(|p| p.review_secs).collect()),
    }
}

/// Open, unmerged PRs with no review that are older than `threshold_secs` at `now`.
pub fn review_wait_prs<'a>(
    prs: &'a [PullRequest],
    reviews: &[Review],
    now: &str,
    threshold_secs: i64,
) -> Vec<&'a PullRequest> {
    let Some(now) = parse(now) else {
        return Vec::new();
    };
    prs.iter()
        .filter(|p| {
            p.state == "open"
                && p.merged_at.is_none()
                && first_review_at(p, reviews).is_none()
                && parse(&p.created_at)
                    .map(|c| (now - c).whole_seconds() > threshold_secs)
                    .unwrap_or(false)
        })
        .collect()
}

// ---- watchdog rules --------------------------------------------------------

/// What the merged-without-review watchdog could not judge. Reviews are only fetched for PRs that
/// changed in a sync pass, so the earliest synced review marks where review evidence starts. A PR
/// merged before that has no review data either way, so it is counted here instead of flagged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MergedWithoutReviewCoverage {
    /// Merged PRs older than the earliest synced review for this repo. Always 0 when no reviews
    /// are synced.
    pub unjudged: usize,
}

/// Merged PRs that received no review, plus the coverage shortfall. Matching is by `pr_id`.
/// A PR merged before the earliest synced review is counted in `unjudged`, not flagged.
/// With no reviews synced for the repo there is no window to establish, so every merged PR is
/// flagged.
pub fn merged_without_review<'a>(
    prs: &'a [PullRequest],
    reviews: &[Review],
) -> (Vec<&'a PullRequest>, MergedWithoutReviewCoverage) {
    let reviewed: std::collections::HashSet<&str> =
        reviews.iter().map(|r| r.pr_id.as_str()).collect();
    let window_start = reviews
        .iter()
        .filter_map(|r| r.submitted_at.as_deref())
        .filter_map(parse)
        .min();
    let mut unjudged = 0usize;
    let flagged = prs
        .iter()
        .filter(|p| {
            if reviewed.contains(p.id.as_str()) {
                return false;
            }
            let Some(merged_at) = p.merged_at.as_deref().and_then(parse) else {
                return false;
            };
            if let Some(start) = window_start {
                if merged_at < start {
                    unjudged += 1;
                    return false;
                }
            }
            true
        })
        .collect();
    (flagged, MergedWithoutReviewCoverage { unjudged })
}

/// Open PRs older than `threshold_days` at `now` (RFC-3339).
pub fn stale_open_prs<'a>(
    prs: &'a [PullRequest],
    now: &str,
    threshold_days: i64,
) -> Vec<&'a PullRequest> {
    let Some(now) = parse(now) else {
        return Vec::new();
    };
    let limit = threshold_days * 86_400;
    prs.iter()
        .filter(|p| {
            p.state == "open"
                && p.merged_at.is_none()
                && parse(&p.created_at)
                    .map(|c| (now - c).whole_seconds() > limit)
                    .unwrap_or(false)
        })
        .collect()
}

/// CI runs that completed with a failing conclusion. In-progress runs are not failures.
pub fn failing_ci_runs(runs: &[CiRun]) -> Vec<&CiRun> {
    runs.iter()
        .filter(|r| r.status == "completed" && r.conclusion.as_deref() == Some("failure"))
        .collect()
}

/// CI runs that went green only after a re-run on the same commit: completed, conclusion
/// `success`, latest attempt past the first. The code did not change between attempts, so the
/// failure was flaky. Runs with no reported attempt are excluded.
pub fn flaky_ci_runs(runs: &[CiRun]) -> Vec<&CiRun> {
    runs.iter()
        .filter(|r| {
            r.status == "completed"
                && r.conclusion.as_deref() == Some("success")
                && r.run_attempt.is_some_and(|a| a > 1)
        })
        .collect()
}

/// The largest share of reviews by a single reviewer, in `0.0..=1.0`. 0.0 when there are none.
pub fn review_concentration(reviews: &[Review]) -> f64 {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for r in reviews {
        if let Some(login) = r.reviewer_login.as_deref() {
            *counts.entry(login).or_default() += 1;
        }
    }
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    counts.values().copied().max().unwrap_or(0) as f64 / total as f64
}

// ---- DORA-lite ----
// Some DORA keys are computable from repo data, others are proxies. Recovery time needs an
// incident source and is omitted.

/// Revert commits, the change-failure signal: a message starting with "revert" (case-insensitive)
/// or containing git's "This reverts commit".
pub fn revert_commits(commits: &[Commit]) -> Vec<&Commit> {
    commits
        .iter()
        .filter(|c| {
            let m = c.message.trim_start();
            m.len() >= 6 && m[..6].eq_ignore_ascii_case("revert")
                || c.message.contains("This reverts commit")
        })
        .collect()
}

/// Change-failure-rate proxy: revert commits as a share of merged PRs, in `0.0..=1.0`. `None`
/// with no merged PRs. True CFR needs an incident source.
pub fn change_failure_rate_proxy(commits: &[Commit], prs: &[PullRequest]) -> Option<f64> {
    let merges = prs.iter().filter(|p| p.merged_at.is_some()).count();
    if merges == 0 {
        return None;
    }
    Some(revert_commits(commits).len() as f64 / merges as f64)
}

/// Releases per week over the `window_days` ending at `now` (RFC-3339), counting `published_at`
/// in the window. `None` if `now` or the window is unusable.
pub fn release_frequency_per_week(
    releases: &[Release],
    now: &str,
    window_days: i64,
) -> Option<f64> {
    if window_days <= 0 {
        return None;
    }
    let now = parse(now)?;
    let window_secs = window_days * 86_400;
    let count = releases
        .iter()
        .filter_map(|r| r.published_at.as_deref())
        .filter_map(parse)
        .filter(|p| {
            let age = (now - *p).whole_seconds();
            (0..=window_secs).contains(&age)
        })
        .count();
    Some(count as f64 / (window_days as f64 / 7.0))
}

/// Tier deploy frequency (releases/week): elite >= daily, high >= weekly, medium >= ~monthly.
pub fn tier_deploy_frequency(per_week: Option<f64>) -> DoraTier {
    match per_week {
        Some(f) if f >= 7.0 => DoraTier::Elite,
        Some(f) if f >= 1.0 => DoraTier::High,
        Some(f) if f >= 0.23 => DoraTier::Medium, // ~once a month
        Some(_) => DoraTier::Low,
        None => DoraTier::Unknown,
    }
}

/// Tier lead time (seconds): elite < 1 day, high < 1 week, medium < 1 month, low above.
pub fn tier_lead_time(secs: Option<i64>) -> DoraTier {
    match secs {
        Some(s) if s < 86_400 => DoraTier::Elite,
        Some(s) if s < 7 * 86_400 => DoraTier::High,
        Some(s) if s < 30 * 86_400 => DoraTier::Medium,
        Some(_) => DoraTier::Low,
        None => DoraTier::Unknown,
    }
}

/// Tier change failure rate (0.0..=1.0): elite <= 15%, high <= 30%, medium <= 45%, low above.
pub fn tier_change_failure_rate(rate: Option<f64>) -> DoraTier {
    match rate {
        Some(r) if r <= 0.15 => DoraTier::Elite,
        Some(r) if r <= 0.30 => DoraTier::High,
        Some(r) if r <= 0.45 => DoraTier::Medium,
        Some(_) => DoraTier::Low,
        None => DoraTier::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pr(id: &str, state: &str, created_at: &str, merged_at: Option<&str>) -> PullRequest {
        PullRequest {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            number: 1,
            title: "t".into(),
            state: state.into(),
            author_login: Some("octocat".into()),
            body: None,
            created_at: created_at.into(),
            merged_at: merged_at.map(Into::into),
            html_url: None,
        }
    }

    fn ci(id: &str, status: &str, conclusion: Option<&str>, attempt: Option<i64>) -> CiRun {
        CiRun {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            commit_sha: Some("aaa".into()),
            status: status.into(),
            conclusion: conclusion.map(Into::into),
            completed_at: Some("2026-06-02T00:00:00Z".into()),
            html_url: None,
            run_attempt: attempt,
        }
    }

    #[test]
    fn percentile_nearest_rank() {
        let xs = vec![5 * 86_400, 10 * 86_400, 15 * 86_400];
        assert_eq!(percentile(xs.clone(), 0.50), Some(10 * 86_400)); // rank 2 -> xs[1]
        assert_eq!(percentile(xs.clone(), 0.75), Some(15 * 86_400)); // rank 3 -> xs[2]
        assert_eq!(percentile(xs.clone(), 0.90), Some(15 * 86_400)); // rank 3 -> xs[2]
        assert_eq!(percentile(xs.clone(), 0.0), Some(5 * 86_400)); // rank 0 clamped to 1 -> xs[0]
        assert_eq!(percentile(xs.clone(), 1.0), Some(15 * 86_400));
        assert_eq!(
            percentile(vec![15 * 86_400, 5 * 86_400, 10 * 86_400], 0.50),
            Some(10 * 86_400)
        );
        assert_eq!(percentile(vec![], 0.50), None);
        assert_eq!(percentile(vec![42], 0.50), Some(42));
    }

    #[test]
    fn failing_ci_runs_flags_only_completed_failures() {
        let runs = vec![
            ci("1", "completed", Some("success"), None),
            ci("2", "completed", Some("failure"), None),
            ci("3", "in_progress", None, None),
            ci("4", "completed", Some("failure"), None),
        ];
        let failing = failing_ci_runs(&runs);
        assert_eq!(failing.len(), 2);
        assert!(failing
            .iter()
            .all(|r| r.conclusion.as_deref() == Some("failure")));
    }

    #[test]
    fn flaky_ci_runs_flags_only_rerun_then_green() {
        let runs = vec![
            ci("1", "completed", Some("success"), Some(1)),
            ci("2", "completed", Some("success"), Some(2)),
            ci("3", "completed", Some("failure"), Some(3)),
            ci("4", "in_progress", None, Some(2)),
            // No attempt reported: never flaky.
            ci("5", "completed", Some("success"), None),
        ];
        let flaky = flaky_ci_runs(&runs);
        assert_eq!(flaky.len(), 1);
        assert_eq!(flaky[0].id, "2");
    }

    fn review(id: &str, pr_id: &str, who: Option<&str>, at: Option<&str>) -> Review {
        Review {
            id: id.into(),
            pr_id: pr_id.into(),
            reviewer_login: who.map(Into::into),
            state: "approved".into(),
            submitted_at: at.map(Into::into),
        }
    }

    #[test]
    fn median_cycle_time_is_two_days() {
        let prs = vec![
            pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-02T00:00:00Z"),
            ), // 1 day
            pr(
                "p2",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-04T00:00:00Z"),
            ), // 3 days
            pr("p3", "open", "2026-06-01T00:00:00Z", None), // ignored
        ];
        assert_eq!(median_cycle_time_secs(&prs), Some(2 * 86_400));
    }

    #[test]
    fn lead_time_to_deploy_is_merge_to_next_release() {
        // Merged on day 1 and day 2, release on day 3: gaps 2d and 1d.
        let prs = vec![
            pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-01T00:00:00Z"),
            ),
            pr(
                "p2",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-02T00:00:00Z"),
            ),
            pr("p3", "open", "2026-06-01T00:00:00Z", None), // not merged -> ignored
        ];
        let releases = vec![release("r1", Some("2026-06-03T00:00:00Z"))];
        assert_eq!(
            lead_time_to_deploy(&prs, &releases),
            Some(((2 * 86_400) + 86_400) / 2)
        );
        // A merge after the only release is excluded; no releases at all gives None.
        assert_eq!(lead_time_to_deploy(&prs, &[]), None);
    }

    #[test]
    fn wip_counts_open_unmerged() {
        let prs = vec![
            pr("p1", "open", "2026-06-01T00:00:00Z", None),
            pr("p2", "open", "2026-06-01T00:00:00Z", None),
            pr(
                "p3",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-02T00:00:00Z"),
            ),
        ];
        assert_eq!(wip_count(&prs), 2);
    }

    #[test]
    fn merged_without_review_flags_the_unreviewed() {
        let prs = vec![
            pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-02T00:00:00Z"),
            ),
            pr(
                "p2",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-02T00:00:00Z"),
            ),
        ];
        let reviews = vec![review(
            "r1",
            "p2",
            Some("alice"),
            Some("2026-06-01T12:00:00Z"),
        )];
        let (flagged, coverage) = merged_without_review(&prs, &reviews);
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].id, "p1");
        // p1 merged after the only synced review, so it is inside the window.
        assert_eq!(coverage.unjudged, 0);
    }

    #[test]
    fn merged_without_review_does_not_flag_prs_merged_before_the_synced_review_window() {
        // The oldest synced review is 2026-08-01, so a PR merged before it has no review evidence.
        let prs = vec![
            // Before the window: unjudged, not flagged.
            pr(
                "ancient",
                "closed",
                "2022-01-01T00:00:00Z",
                Some("2022-01-02T00:00:00Z"),
            ),
            // Inside the window, no review: flagged.
            pr(
                "recent_unreviewed",
                "closed",
                "2026-08-10T00:00:00Z",
                Some("2026-08-11T00:00:00Z"),
            ),
            // Inside the window, reviewed: not flagged.
            pr(
                "recent_reviewed",
                "closed",
                "2026-08-12T00:00:00Z",
                Some("2026-08-13T00:00:00Z"),
            ),
        ];
        let reviews = vec![
            review(
                "r1",
                "recent_reviewed",
                Some("alice"),
                Some("2026-08-12T06:00:00Z"),
            ),
            // Earliest synced review; sets the window boundary.
            review(
                "r2",
                "some_other_pr",
                Some("bob"),
                Some("2026-08-01T00:00:00Z"),
            ),
        ];
        let (flagged, coverage) = merged_without_review(&prs, &reviews);
        let ids: std::collections::HashSet<&str> = flagged.iter().map(|p| p.id.as_str()).collect();
        assert!(
            !ids.contains("ancient"),
            "a PR older than the synced review window must not be flagged"
        );
        assert!(ids.contains("recent_unreviewed"));
        assert!(!ids.contains("recent_reviewed"));
        assert_eq!(
            coverage.unjudged, 1,
            "the one too-old PR is counted, not silently dropped"
        );
    }

    #[test]
    fn merged_without_review_flags_every_merged_pr_with_zero_synced_reviews() {
        // No reviews synced: no window to establish, so every merged PR is flagged.
        let prs = vec![pr(
            "p1",
            "closed",
            "2022-01-01T00:00:00Z",
            Some("2022-01-02T00:00:00Z"),
        )];
        let (flagged, coverage) = merged_without_review(&prs, &[]);
        assert_eq!(flagged.len(), 1);
        assert_eq!(coverage.unjudged, 0);
    }

    #[test]
    fn stale_flags_old_open_prs() {
        let prs = vec![
            pr("old", "open", "2026-06-01T00:00:00Z", None),
            pr("fresh", "open", "2026-06-10T00:00:00Z", None),
        ];
        let now = "2026-06-11T00:00:00Z";
        let stale = stale_open_prs(&prs, now, 7);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].id, "old");
    }

    #[test]
    fn concentration_is_max_reviewer_share() {
        let reviews = vec![
            review("r1", "p1", Some("alice"), Some("2026-06-01T00:00:00Z")),
            review("r2", "p1", Some("alice"), Some("2026-06-01T00:00:00Z")),
            review("r3", "p2", Some("alice"), Some("2026-06-01T00:00:00Z")),
            review("r4", "p2", Some("bob"), Some("2026-06-01T00:00:00Z")),
        ];
        assert_eq!(review_concentration(&reviews), 0.75);
        assert_eq!(review_concentration(&[]), 0.0);
    }

    #[test]
    fn cycle_phases_split_pickup_and_review() {
        let p = pr(
            "p1",
            "closed",
            "2026-06-01T00:00:00Z",
            Some("2026-06-04T00:00:00Z"),
        );
        let reviews = vec![review(
            "r1",
            "p1",
            Some("alice"),
            Some("2026-06-02T00:00:00Z"),
        )];
        let phases = cycle_phases(&p, &reviews);
        assert_eq!(phases.pickup_secs, Some(86_400));
        assert_eq!(phases.review_secs, Some(2 * 86_400));
    }

    #[test]
    fn cycle_phases_unreviewed_merge_is_all_pickup() {
        // Merged with no review: the whole 2-day span is pickup.
        let p = pr(
            "p1",
            "closed",
            "2026-06-01T00:00:00Z",
            Some("2026-06-03T00:00:00Z"),
        );
        let phases = cycle_phases(&p, &[]);
        assert_eq!(phases.pickup_secs, Some(2 * 86_400));
        assert_eq!(phases.review_secs, None);
        let open = pr("p2", "open", "2026-06-01T00:00:00Z", None);
        assert_eq!(cycle_phases(&open, &[]), CyclePhases::default());
    }

    #[test]
    fn median_cycle_phases_over_a_set() {
        let prs = vec![
            pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-04T00:00:00Z"),
            ),
            pr(
                "p2",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-06T00:00:00Z"),
            ),
        ];
        let reviews = vec![
            review("r1", "p1", Some("a"), Some("2026-06-02T00:00:00Z")), // pickup 1d, review 2d
            review("r2", "p2", Some("a"), Some("2026-06-03T00:00:00Z")), // pickup 2d, review 3d
        ];
        let m = median_cycle_phases(&prs, &reviews);
        assert_eq!(m.pickup_secs, Some((1 + 2) * 86_400 / 2)); // median of 1d,2d
        assert_eq!(m.review_secs, Some((2 + 3) * 86_400 / 2)); // median of 2d,3d
    }

    #[test]
    fn review_wait_flags_unreviewed_open_prs_past_threshold() {
        let prs = vec![
            pr("waiting", "open", "2026-06-01T00:00:00Z", None),
            pr("fresh", "open", "2026-06-10T00:00:00Z", None),
            pr("reviewed", "open", "2026-06-01T00:00:00Z", None),
        ];
        let reviews = vec![review(
            "r1",
            "reviewed",
            Some("a"),
            Some("2026-06-02T00:00:00Z"),
        )];
        let now = "2026-06-11T00:00:00Z";
        let waiting = review_wait_prs(&prs, &reviews, now, 2 * 86_400);
        assert_eq!(waiting.len(), 1);
        assert_eq!(waiting[0].id, "waiting");
    }

    fn commit(sha: &str, msg: &str, at: &str) -> Commit {
        Commit {
            sha: sha.into(),
            repo_id: "repo:acme/widget".into(),
            author_login: Some("octocat".into()),
            message: msg.into(),
            committed_at: at.into(),
        }
    }

    fn release(id: &str, at: Option<&str>) -> Release {
        Release {
            id: id.into(),
            repo_id: "repo:acme/widget".into(),
            tag: "v1".into(),
            name: None,
            published_at: at.map(Into::into),
        }
    }

    #[test]
    fn revert_detection_and_cfr_proxy() {
        let commits = vec![
            commit("a", "Add feature", "2026-06-01T00:00:00Z"),
            commit("b", "Revert \"Add feature\"", "2026-06-02T00:00:00Z"),
            commit(
                "c",
                "fix\n\nThis reverts commit abc.",
                "2026-06-03T00:00:00Z",
            ),
        ];
        assert_eq!(revert_commits(&commits).len(), 2);
        let prs = vec![
            pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-02T00:00:00Z"),
            ),
            pr(
                "p2",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-02T00:00:00Z"),
            ),
            pr("p3", "open", "2026-06-01T00:00:00Z", None),
        ];
        // 2 reverts / 2 merges = 1.0.
        assert_eq!(change_failure_rate_proxy(&commits, &prs), Some(1.0));
        // No merges -> None.
        assert_eq!(change_failure_rate_proxy(&commits, &[]), None);
    }

    #[test]
    fn release_frequency_counts_window_and_tiers() {
        let now = "2026-06-29T00:00:00Z";
        let releases = vec![
            release("r1", Some("2026-06-08T00:00:00Z")), // in window
            release("r2", Some("2026-06-22T00:00:00Z")), // in window
            release("r3", Some("2026-01-01T00:00:00Z")), // too old
            release("r4", None),                         // undated
        ];
        // 2 releases over 28 days = 2 / 4 weeks = 0.5/week.
        let f = release_frequency_per_week(&releases, now, 28).unwrap();
        assert!((f - 0.5).abs() < 1e-9);
        assert_eq!(tier_deploy_frequency(Some(0.5)), DoraTier::Medium);
        assert_eq!(tier_deploy_frequency(Some(8.0)), DoraTier::Elite);
        assert_eq!(tier_deploy_frequency(None), DoraTier::Unknown);
    }

    #[test]
    fn dora_tiers_classify() {
        assert_eq!(tier_lead_time(Some(3600)), DoraTier::Elite);
        assert_eq!(tier_lead_time(Some(10 * 86_400)), DoraTier::Medium);
        assert_eq!(tier_change_failure_rate(Some(0.05)), DoraTier::Elite);
        assert_eq!(tier_change_failure_rate(Some(0.5)), DoraTier::Low);
        assert_eq!(DoraTier::Elite.as_str(), "elite");
    }

    // Runs over real store reads.
    #[tokio::test]
    async fn metrics_over_store_reads() {
        use core_store::Store;
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
        store
            .upsert_pull_request(&pr(
                "p1",
                "closed",
                "2026-06-01T00:00:00Z",
                Some("2026-06-03T00:00:00Z"),
            ))
            .await
            .unwrap();
        let prs = store.pull_requests("repo:acme/widget").await.unwrap();
        assert_eq!(median_cycle_time_secs(&prs), Some(2 * 86_400));
    }
}
