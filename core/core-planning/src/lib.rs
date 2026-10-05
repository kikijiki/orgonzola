//! Sprint-planning support. Informs rather than derives: structures the inputs (backlog,
//! carry-over, capacity) into a draft the manager edits, never an authoritative plan.
//! Deterministic over the store, no LLM. Capacity is never invented; the manager supplies
//! availability and the tool only pairs it with team membership.

use std::collections::HashMap;

use core_store::{Store, StoreError, WorkItem};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum PlanningError {
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// A work item is "finished" when its state is done or closed.
fn is_finished(state: &str) -> bool {
    matches!(state, "done" | "closed")
}

/// A backlog entry: an unfinished work item plus how many outgoing links it has.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BacklogItem {
    pub id: String,
    pub title: String,
    pub state: String,
    pub link_count: usize,
}

/// Unfinished work items with their link counts.
pub async fn backlog(store: &Store) -> Result<Vec<BacklogItem>, PlanningError> {
    let mut out = Vec::new();
    for wi in store.work_items().await? {
        if is_finished(&wi.state) {
            continue;
        }
        let links = store.links_from("work_item", &wi.id).await?;
        out.push(BacklogItem {
            id: wi.id,
            title: wi.title,
            state: wi.state,
            link_count: links.len(),
        });
    }
    Ok(out)
}

/// The sprint's work items that did not finish.
pub async fn carry_over(store: &Store, sprint_id: &str) -> Result<Vec<WorkItem>, PlanningError> {
    let items = store.work_items_for_sprint(sprint_id).await?;
    Ok(items
        .into_iter()
        .filter(|w| !is_finished(&w.state))
        .collect())
}

/// A team member paired with the manager-supplied availability (days). `None` when none was given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemberCapacity {
    pub identity_id: String,
    pub available_days: Option<u32>,
}

/// Pair each team member with the manager-supplied availability.
pub async fn team_capacity(
    store: &Store,
    team_id: &str,
    availability: &HashMap<String, u32>,
) -> Result<Vec<MemberCapacity>, PlanningError> {
    let members = store.team_members(team_id).await?;
    Ok(members
        .into_iter()
        .map(|id| {
            let available_days = availability.get(&id).copied();
            MemberCapacity {
                identity_id: id,
                available_days,
            }
        })
        .collect())
}

/// A planning draft: a candidate the manager edits, not an authoritative plan.
#[derive(Debug, Clone, Serialize)]
pub struct PlanningDraft {
    pub sprint_id: String,
    pub team_id: String,
    pub carry_over: Vec<WorkItem>,
    pub backlog: Vec<BacklogItem>,
    pub capacity: Vec<MemberCapacity>,
}

/// Bundle backlog + carry-over + capacity into a draft.
pub async fn planning_draft(
    store: &Store,
    sprint_id: &str,
    team_id: &str,
    availability: &HashMap<String, u32>,
) -> Result<PlanningDraft, PlanningError> {
    Ok(PlanningDraft {
        sprint_id: sprint_id.to_string(),
        team_id: team_id.to_string(),
        carry_over: carry_over(store, sprint_id).await?,
        backlog: backlog(store).await?,
        capacity: team_capacity(store, team_id, availability).await?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_store::{Sprint, Team, WorkItem};

    fn work_item(id: &str, state: &str) -> WorkItem {
        WorkItem {
            id: id.into(),
            title: format!("item {id}"),
            kind: "story".into(),
            state: state.into(),
            source: "manual".into(),
            status_category: None,
        }
    }

    #[tokio::test]
    async fn backlog_returns_only_unfinished() {
        let store = Store::open_in_memory().await.unwrap();
        store.add_work_item(&work_item("a", "open")).await.unwrap();
        store.add_work_item(&work_item("b", "open")).await.unwrap();
        store.add_work_item(&work_item("c", "done")).await.unwrap();
        let bl = backlog(&store).await.unwrap();
        assert_eq!(bl.len(), 2);
        assert!(bl.iter().all(|i| i.state == "open"));
    }

    #[tokio::test]
    async fn carry_over_returns_sprint_unfinished() {
        let store = Store::open_in_memory().await.unwrap();
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
        store
            .add_work_item(&work_item("open", "open"))
            .await
            .unwrap();
        store
            .add_work_item(&work_item("done", "done"))
            .await
            .unwrap();
        store.add_sprint_work_item("s1", "open").await.unwrap();
        store.add_sprint_work_item("s1", "done").await.unwrap();
        let co = carry_over(&store, "s1").await.unwrap();
        assert_eq!(co.len(), 1);
        assert_eq!(co[0].id, "open");
    }

    #[tokio::test]
    async fn capacity_pairs_members_without_inventing() {
        let store = Store::open_in_memory().await.unwrap();
        store
            .add_team(&Team {
                id: "t1".into(),
                name: "Platform".into(),
                source: "manual".into(),
            })
            .await
            .unwrap();
        store.add_identity("u1", "alice").await.unwrap();
        store.add_identity("u2", "bob").await.unwrap();
        store.add_team_member("t1", "u1").await.unwrap();
        store.add_team_member("t1", "u2").await.unwrap();

        let mut availability = HashMap::new();
        availability.insert("u1".to_string(), 8);
        let cap = team_capacity(&store, "t1", &availability).await.unwrap();
        assert_eq!(cap.len(), 2);
        let u1 = cap.iter().find(|m| m.identity_id == "u1").unwrap();
        let u2 = cap.iter().find(|m| m.identity_id == "u2").unwrap();
        assert_eq!(u1.available_days, Some(8));
        // The manager gave nothing for u2; the tool does not invent a number.
        assert_eq!(u2.available_days, None);
    }
}
