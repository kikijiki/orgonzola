import type { AttentionEntity, AttentionView, LinkedEntity } from "@/bindings"

// Selectors over an attention item's typed envelope: exactly one `subject` entity (the flagged
// thing), zero or more `actor` entities (who the next step points at) and zero or more
// `evidence` entities (the fact that makes the flag true).

// An item-like object carrying the envelope; both `AttentionView` and `BoardAttentionView` do.
type Enveloped = Pick<AttentionView, "entities">

// The flagged thing. `null` only for a malformed envelope; callers handle it anyway.
export function subjectOf(item: Enveloped): AttentionEntity | null {
  return item.entities.find((e) => e.role === "subject")?.target ?? null
}

// The flagged thing when it is a pull request, else null.
export function prSubjectOf(item: Enveloped) {
  const s = subjectOf(item)
  return s?.type === "pull_request" ? s : null
}

export function actorLoginsOf(item: Enveloped): string[] {
  return item.entities
    .filter((e) => e.role === "actor" && e.target.type === "person")
    .map((e) => (e.target as Extract<AttentionEntity, { type: "person" }>).login)
}

// What makes the flag true, e.g. the open work item behind a `done_not_done`. Empty for kinds
// that need no supporting fact.
export function evidenceOf(item: Enveloped): AttentionEntity[] {
  return item.entities.filter((e) => e.role === "evidence").map((e: LinkedEntity) => e.target)
}

// An entity's forge URL, or null where the forge reported none or the type has no URL (a
// person, a source file). Null means render as plain text.
export function entityHref(entity: AttentionEntity): string | null {
  switch (entity.type) {
    case "pull_request":
    case "work_item":
    case "ci_run":
      return entity.url
    case "person":
    case "source_file":
      return null
  }
}

// Stable set key for an item's subject, e.g. "pull_request:p1". Mirrors the core's
// `entity.key()` so both sides of a `by_team` lookup agree.
export function subjectKey(item: Enveloped): string | null {
  const s = subjectOf(item)
  if (!s) return null
  switch (s.type) {
    case "pull_request":
      return `pull_request:${s.id}`
    case "work_item":
      return `work_item:${s.id}`
    case "ci_run":
      return `ci_run:${s.id}`
    case "person":
      return `person:${s.login}`
    case "source_file":
      return `source_file:${s.path}`
  }
}
