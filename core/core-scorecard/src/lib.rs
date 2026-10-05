//! Deterministic repo scorecards: a fixed set of tier-tagged boolean rules over a repo's facts,
//! yielding a Bronze / Silver / Gold grade. Explainable, no LLM, no people ranking. Pure: no
//! store, no network.

use serde::{Deserialize, Serialize};

/// The facts a scorecard is computed from, gathered from the store by the caller (`core-summary`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepoFacts {
    pub failing_ci: i64,
    pub merged_without_review: i64,
    pub stale_open_prs: i64,
    pub review_wait: i64,
    pub median_cycle_secs: Option<i64>,
    pub releases: i64,
    /// Lowest bus factor across the repo's modules (`None` if no data). 1 is key-person risk.
    pub min_bus_factor: Option<i64>,
    /// Whether the repo has any reviews at all (review is happening).
    pub has_reviews: bool,
}

/// A maturity tier. Ordered None < Bronze < Silver < Gold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    None,
    Bronze,
    Silver,
    Gold,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::None => "none",
            Tier::Bronze => "bronze",
            Tier::Silver => "silver",
            Tier::Gold => "gold",
        }
    }
}

/// A rule's evaluated outcome. `Unknown` means the underlying fact was never observed (not
/// synced, or nothing to measure), distinct from `Fail`, where the fact was observed and the
/// rule's threshold was not met. A `passed: bool` cannot represent "unknown" without reading it
/// as "false", hence three explicit values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleStatus {
    Pass,
    Fail,
    Unknown,
}

impl RuleStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RuleStatus::Pass => "pass",
            RuleStatus::Fail => "fail",
            RuleStatus::Unknown => "unknown",
        }
    }

    fn from_bool(passed: bool) -> Self {
        if passed {
            RuleStatus::Pass
        } else {
            RuleStatus::Fail
        }
    }
}

/// One rule's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleResult {
    pub rule: String,
    /// The tier this rule is required for ("bronze" | "silver" | "gold").
    pub tier: String,
    pub status: RuleStatus,
    /// A short human reason (the fact that decided it).
    pub detail: String,
}

/// A repo's scorecard: the achieved tier + every rule's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scorecard {
    pub repo_id: String,
    /// "none" | "bronze" | "silver" | "gold".
    pub tier: String,
    pub rules: Vec<RuleResult>,
}

/// Healthy cycle-time bar for the Gold rule: under one week.
const HEALTHY_CYCLE_SECS: i64 = 7 * 86_400;

/// Evaluate the fixed rule set. A repo reaches a tier only when every rule at that tier and
/// all lower tiers passes.
pub fn scorecard(repo_id: &str, f: &RepoFacts) -> Scorecard {
    let rule = |name: &str, tier: Tier, status: RuleStatus, detail: String| RuleResult {
        rule: name.to_string(),
        tier: tier.as_str().to_string(),
        status,
        detail,
    };
    // A rule over an absent fact is Unknown, not Pass: no merged PRs means no cycle time to
    // measure, and no module data means bus factor was never computed.
    let cycle_status = match f.median_cycle_secs {
        Some(s) => RuleStatus::from_bool(s < HEALTHY_CYCLE_SECS),
        None => RuleStatus::Unknown,
    };
    let bus_status = match f.min_bus_factor {
        Some(b) => RuleStatus::from_bool(b >= 2),
        None => RuleStatus::Unknown,
    };

    let rules = vec![
        // Bronze: basic hygiene.
        rule(
            "CI is green on the default branch",
            Tier::Bronze,
            RuleStatus::from_bool(f.failing_ci == 0),
            format!("{} failing CI run(s)", f.failing_ci),
        ),
        rule(
            "Review is happening",
            Tier::Bronze,
            RuleStatus::from_bool(f.has_reviews),
            if f.has_reviews {
                "reviews present".into()
            } else {
                "no reviews recorded".into()
            },
        ),
        // Silver: healthy flow.
        rule(
            "Changes are reviewed before merge",
            Tier::Silver,
            RuleStatus::from_bool(f.merged_without_review == 0),
            format!("{} merged without review", f.merged_without_review),
        ),
        rule(
            "No PRs left waiting on review",
            Tier::Silver,
            RuleStatus::from_bool(f.review_wait == 0),
            format!("{} awaiting first review", f.review_wait),
        ),
        rule(
            "No stale open PRs",
            Tier::Silver,
            RuleStatus::from_bool(f.stale_open_prs == 0),
            format!("{} stale PR(s)", f.stale_open_prs),
        ),
        // Gold: excellence.
        rule(
            "Healthy cycle time (< 1 week)",
            Tier::Gold,
            cycle_status,
            match f.median_cycle_secs {
                Some(s) => format!("{} h median cycle", s / 3600),
                None => "no merged PRs".into(),
            },
        ),
        rule(
            "Is shipping (has releases)",
            Tier::Gold,
            RuleStatus::from_bool(f.releases > 0),
            format!("{} release(s)", f.releases),
        ),
        rule(
            "No single-maintainer critical module",
            Tier::Gold,
            bus_status,
            match f.min_bus_factor {
                Some(b) => format!("lowest module bus factor {b}"),
                None => "no module data".into(),
            },
        ),
    ];

    // A tier requires every rule at that tier to be Pass. Fail and Unknown both block it, so a
    // tier is never granted on evidence the tool does not have.
    let passes = |tier: Tier| {
        rules
            .iter()
            .filter(|r| r.tier == tier.as_str())
            .all(|r| r.status == RuleStatus::Pass)
    };
    let tier = if passes(Tier::Bronze) && passes(Tier::Silver) && passes(Tier::Gold) {
        Tier::Gold
    } else if passes(Tier::Bronze) && passes(Tier::Silver) {
        Tier::Silver
    } else if passes(Tier::Bronze) {
        Tier::Bronze
    } else {
        Tier::None
    };

    Scorecard {
        repo_id: repo_id.to_string(),
        tier: tier.as_str().to_string(),
        rules,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect() -> RepoFacts {
        RepoFacts {
            failing_ci: 0,
            merged_without_review: 0,
            stale_open_prs: 0,
            review_wait: 0,
            median_cycle_secs: Some(3600),
            releases: 3,
            min_bus_factor: Some(3),
            has_reviews: true,
        }
    }

    #[test]
    fn all_pass_is_gold() {
        assert_eq!(scorecard("r", &perfect()).tier, "gold");
    }

    #[test]
    fn failing_bronze_rule_is_none() {
        let mut f = perfect();
        f.failing_ci = 2; // a Bronze rule fails
        let sc = scorecard("r", &f);
        assert_eq!(sc.tier, "none");
        assert!(sc
            .rules
            .iter()
            .any(|r| r.rule.contains("CI is green") && r.status == RuleStatus::Fail));
    }

    #[test]
    fn failing_gold_rule_caps_at_silver() {
        let mut f = perfect();
        f.min_bus_factor = Some(1); // a Gold rule genuinely fails: measured, and below the threshold
        let sc = scorecard("r", &f);
        assert_eq!(sc.tier, "silver");
        let bus_rule = sc
            .rules
            .iter()
            .find(|r| r.rule.contains("single-maintainer"))
            .unwrap();
        assert_eq!(bus_rule.status, RuleStatus::Fail);
    }

    #[test]
    fn failing_silver_rule_caps_at_bronze() {
        let mut f = perfect();
        f.stale_open_prs = 4; // a Silver rule fails
        assert_eq!(scorecard("r", &f).tier, "bronze");
    }

    /// A rule over an absent fact is Unknown, not Pass, and an Unknown rule blocks its tier like a
    /// Fail would, even though nothing here failed a threshold.
    #[test]
    fn unknown_facts_are_unknown_and_block_the_tier() {
        let mut f = perfect();
        f.median_cycle_secs = None;
        f.min_bus_factor = None;
        let sc = scorecard("r", &f);
        assert_eq!(sc.tier, "silver");
        let cycle_rule = sc
            .rules
            .iter()
            .find(|r| r.rule.contains("Healthy cycle time"))
            .unwrap();
        assert_eq!(cycle_rule.status, RuleStatus::Unknown);
        assert_eq!(cycle_rule.detail, "no merged PRs");
        let bus_rule = sc
            .rules
            .iter()
            .find(|r| r.rule.contains("single-maintainer"))
            .unwrap();
        assert_eq!(bus_rule.status, RuleStatus::Unknown);
        assert_eq!(bus_rule.detail, "no module data");
    }

    /// A passing rule (fact present, threshold met) still reports Pass and grants its tier.
    #[test]
    fn present_facts_that_meet_the_threshold_still_pass() {
        let sc = scorecard("r", &perfect());
        for r in &sc.rules {
            assert_eq!(r.status, RuleStatus::Pass, "{} should pass", r.rule);
        }
        assert_eq!(sc.tier, "gold");
    }

    /// A repo we measured and found wanting (bus factor 1, single maintainer) must not score worse
    /// than a repo we never measured at all (no module data): both are capped at the same tier.
    #[test]
    fn unmeasured_repo_does_not_outscore_a_measured_failing_repo() {
        let mut measured_failing = perfect();
        measured_failing.min_bus_factor = Some(1);
        let mut never_measured = perfect();
        never_measured.min_bus_factor = None;

        let measured_sc = scorecard("measured", &measured_failing);
        let unmeasured_sc = scorecard("unmeasured", &never_measured);

        assert_eq!(measured_sc.tier, "silver");
        assert_eq!(unmeasured_sc.tier, "silver");
        assert_eq!(measured_sc.tier, unmeasured_sc.tier);
    }
}
