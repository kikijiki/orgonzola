//! Shared domain types that cross the UI-core boundary. Pure types only: serde for the wire
//! format, specta for the generated TypeScript bindings. No host dependency.

use serde::{Deserialize, Serialize};
use specta::Type;

/// Result of the health/ping command. Proves a typed round trip from the UI into core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct Health {
    /// Always "ok" while the core is responsive.
    pub status: String,
    /// The core crate version, so the UI can show what it is talking to.
    pub core_version: String,
}

// ---- sources ----

/// What kind of GitHub entity a source points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    Org,
    Repo,
    User,
}

/// Whether we own a source (full detail + alerting) or only observe it (low-noise, higher
/// salience bar).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum Ownership {
    Owned,
    Observed,
}

/// Per-source thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct Thresholds {
    /// A PR with no movement for this many days counts as stale for this source.
    pub stale_pr_days: u32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self { stale_pr_days: 7 }
    }
}

/// A watched GitHub source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct Source {
    /// Stable id, e.g. "repo:owner/name" or "org:acme". The registry key.
    pub id: String,
    pub kind: SourceKind,
    /// "owner/name" for a repo, the login for an org or user.
    pub name: String,
    pub ownership: Ownership,
    /// Coarse include filters (labels, path prefixes); refined in later phases.
    pub filters: Vec<String>,
    pub thresholds: Thresholds,
}

/// Errors from mutating the source registry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    #[error("a source with id `{0}` is already registered")]
    DuplicateId(String),
}

/// The set of sources orgonzola watches.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct SourceRegistry {
    sources: Vec<Source>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a source, rejecting a duplicate id.
    pub fn add(&mut self, source: Source) -> Result<(), RegistryError> {
        if self.sources.iter().any(|s| s.id == source.id) {
            return Err(RegistryError::DuplicateId(source.id));
        }
        self.sources.push(source);
        Ok(())
    }

    pub fn all(&self) -> &[Source] {
        &self.sources
    }

    pub fn get(&self, id: &str) -> Option<&Source> {
        self.sources.iter().find(|s| s.id == id)
    }

    pub fn owned(&self) -> impl Iterator<Item = &Source> {
        self.sources
            .iter()
            .filter(|s| s.ownership == Ownership::Owned)
    }

    pub fn observed(&self) -> impl Iterator<Item = &Source> {
        self.sources
            .iter()
            .filter(|s| s.ownership == Ownership::Observed)
    }
}

// ---- analytics types ----

/// The two phases of a merged PR's cycle time: pickup (creation to first review) and review (first
/// review to merge). Both None for an unmerged PR or a PR that was never reviewed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CyclePhases {
    pub pickup_secs: Option<i64>,
    pub review_secs: Option<i64>,
}

/// Median pickup and review phase durations over a set of merged PRs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhaseMedians {
    pub pickup_secs: Option<i64>,
    pub review_secs: Option<i64>,
}

/// A DORA performance tier for a metric. Unknown when there is nothing to classify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoraTier {
    Elite,
    High,
    Medium,
    Low,
    Unknown,
}

impl DoraTier {
    /// A stable lowercase label for the boundary / UI coloring.
    pub fn as_str(self) -> &'static str {
        match self {
            DoraTier::Elite => "elite",
            DoraTier::High => "high",
            DoraTier::Medium => "medium",
            DoraTier::Low => "low",
            DoraTier::Unknown => "unknown",
        }
    }
}

// ---- attention envelope ----

/// Why an entity is attached to an attention item: the flagged thing, the person to nudge, or
/// the fact that makes the flag true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum AttentionRole {
    /// The thing the signal is about. Exactly one per item.
    Subject,
    /// Who the next step points at (the PR's author). Omitted if unknown.
    Actor,
    /// The fact that makes the flag true, e.g. the open work item behind `done_not_done`.
    Evidence,
}

/// One concrete thing an attention item points at, with everything a surface needs to name and
/// link it. `url` is whatever web URL the forge reported; `None` renders as plain text. Nothing
/// is composed from a URL convention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AttentionEntity {
    PullRequest {
        /// The store id, e.g. "p1". Stable across syncs; the key for any further lookup.
        id: String,
        number: i64,
        title: String,
        url: Option<String>,
    },
    /// A tracked work item (a forge issue, or a Jira issue).
    WorkItem {
        id: String,
        number: i64,
        title: String,
        url: Option<String>,
    },
    CiRun {
        id: String,
        commit_sha: Option<String>,
        /// The run attempt: >1 means the suite went green only on a re-run.
        attempt: Option<i64>,
        url: Option<String>,
    },
    /// A person, by forge login. No URL: a profile URL is not synced, and the surface that shows a
    /// login already composes one from the board's forge web root.
    Person { login: String },
    /// A file in the repo the item belongs to.
    SourceFile {
        path: String,
        /// When a merged PR last changed this file, RFC-3339. `None` when it has no merged
        /// history at all.
        last_changed_at: Option<String>,
    },
}

impl AttentionEntity {
    /// The entity's stable identity as `<type>:<id>`, for a set key or dedup. Not a wire field.
    pub fn key(&self) -> String {
        match self {
            Self::PullRequest { id, .. } => format!("pull_request:{id}"),
            Self::WorkItem { id, .. } => format!("work_item:{id}"),
            Self::CiRun { id, .. } => format!("ci_run:{id}"),
            Self::Person { login } => format!("person:{login}"),
            Self::SourceFile { path, .. } => format!("source_file:{path}"),
        }
    }
}

/// An entity attached to an attention item, with the reason it is attached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type)]
pub struct LinkedEntity {
    pub role: AttentionRole,
    pub target: AttentionEntity,
}

impl LinkedEntity {
    pub fn subject(target: AttentionEntity) -> Self {
        Self {
            role: AttentionRole::Subject,
            target,
        }
    }

    pub fn actor(target: AttentionEntity) -> Self {
        Self {
            role: AttentionRole::Actor,
            target,
        }
    }

    pub fn evidence(target: AttentionEntity) -> Self {
        Self {
            role: AttentionRole::Evidence,
            target,
        }
    }
}

/// The one `subject` entity of an attention item's envelope, or `None` if it has none.
pub fn subject_of(entities: &[LinkedEntity]) -> Option<&AttentionEntity> {
    entities
        .iter()
        .find(|e| e.role == AttentionRole::Subject)
        .map(|e| &e.target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(id: &str, ownership: Ownership) -> Source {
        Source {
            id: id.to_string(),
            kind: SourceKind::Repo,
            name: "acme/widget".to_string(),
            ownership,
            filters: vec![],
            thresholds: Thresholds::default(),
        }
    }

    #[test]
    fn health_serde_roundtrips() {
        let h = Health {
            status: "ok".into(),
            core_version: "0.1.0".into(),
        };
        let json = serde_json::to_string(&h).unwrap();
        let back: Health = serde_json::from_str(&json).unwrap();
        assert_eq!(h, back);
    }

    #[test]
    fn attention_entity_serializes_as_a_tagged_union() {
        // The generated TypeScript union discriminates on `type`. Pin the wire shape so a serde
        // attribute change is caught here.
        let pr = AttentionEntity::PullRequest {
            id: "p1".into(),
            number: 12,
            title: "Fix the thing".into(),
            url: Some("https://forge/acme/widget/pull/12".into()),
        };
        let json: serde_json::Value = serde_json::to_value(&pr).unwrap();
        assert_eq!(json["type"], "pull_request");
        assert_eq!(json["number"], 12);
        let back: AttentionEntity = serde_json::from_value(json).unwrap();
        assert_eq!(pr, back);

        let person = AttentionEntity::Person {
            login: "octocat".into(),
        };
        assert_eq!(serde_json::to_value(&person).unwrap()["type"], "person");
        assert_eq!(person.key(), "person:octocat");

        let link = LinkedEntity::evidence(person);
        assert_eq!(serde_json::to_value(&link).unwrap()["role"], "evidence");
    }

    #[test]
    fn subject_of_finds_the_one_subject() {
        let entities = vec![
            LinkedEntity::actor(AttentionEntity::Person {
                login: "octocat".into(),
            }),
            LinkedEntity::subject(AttentionEntity::CiRun {
                id: "run1".into(),
                commit_sha: Some("aaa".into()),
                attempt: None,
                url: None,
            }),
        ];
        assert_eq!(subject_of(&entities).unwrap().key(), "ci_run:run1");
        assert!(subject_of(&[]).is_none());
    }

    #[test]
    fn registry_rejects_duplicate_id_and_leaves_state_unchanged() {
        let mut reg = SourceRegistry::new();
        reg.add(source("repo:acme/widget", Ownership::Owned))
            .unwrap();
        let err = reg
            .add(source("repo:acme/widget", Ownership::Observed))
            .unwrap_err();
        assert_eq!(err, RegistryError::DuplicateId("repo:acme/widget".into()));
        assert_eq!(reg.all().len(), 1);
        // The original (owned) source is kept, not replaced by the observed duplicate.
        assert_eq!(
            reg.get("repo:acme/widget").unwrap().ownership,
            Ownership::Owned
        );
    }

    #[test]
    fn registry_partitions_owned_and_observed() {
        let mut reg = SourceRegistry::new();
        reg.add(source("repo:a", Ownership::Owned)).unwrap();
        reg.add(source("repo:b", Ownership::Observed)).unwrap();
        reg.add(source("repo:c", Ownership::Owned)).unwrap();
        assert_eq!(reg.owned().count(), 2);
        assert_eq!(reg.observed().count(), 1);
    }
}
