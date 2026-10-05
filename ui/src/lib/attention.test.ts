import type { AttentionEntity, LinkedEntity } from "@/bindings"
import { describe, expect, it } from "vitest"
import {
  actorLoginsOf,
  entityHref,
  evidenceOf,
  prSubjectOf,
  subjectKey,
  subjectOf,
} from "./attention"

const pr: AttentionEntity = {
  type: "pull_request",
  id: "p1",
  number: 7,
  title: "Fix the thing",
  url: "https://github.com/acme/widgets/pull/7",
}
const person: AttentionEntity = { type: "person", login: "octocat" }
const file: AttentionEntity = { type: "source_file", path: "src/main.rs", last_changed_at: null }

const link = (role: LinkedEntity["role"], target: AttentionEntity): LinkedEntity => ({
  role,
  target,
})
const item = (...entities: LinkedEntity[]) => ({ entities })

describe("subjectOf", () => {
  it("returns the one subject entity", () => {
    expect(subjectOf(item(link("subject", pr), link("actor", person)))).toEqual(pr)
  })

  it("is null on a malformed envelope rather than throwing", () => {
    expect(subjectOf(item(link("actor", person)))).toBeNull()
    expect(subjectOf(item())).toBeNull()
  })
})

describe("prSubjectOf", () => {
  it("narrows a pull-request subject", () => {
    expect(prSubjectOf(item(link("subject", pr)))?.number).toBe(7)
  })

  it("is null when the subject is some other type", () => {
    expect(prSubjectOf(item(link("subject", file)))).toBeNull()
  })
})

describe("actorLoginsOf", () => {
  it("collects the logins of person actors only", () => {
    const i = item(link("subject", pr), link("actor", person), link("evidence", file))
    expect(actorLoginsOf(i)).toEqual(["octocat"])
  })

  it("is empty when the forge reported no author", () => {
    expect(actorLoginsOf(item(link("subject", pr)))).toEqual([])
  })
})

describe("evidenceOf", () => {
  it("returns the supporting facts, unwrapped", () => {
    expect(evidenceOf(item(link("subject", pr), link("evidence", file)))).toEqual([file])
  })

  it("is empty for kinds that need no supporting fact", () => {
    expect(evidenceOf(item(link("subject", pr)))).toEqual([])
  })
})

describe("entityHref", () => {
  it("gives the forge URL for linkable types", () => {
    expect(entityHref(pr)).toBe("https://github.com/acme/widgets/pull/7")
  })

  it("is null for types with no URL, so they render as plain text", () => {
    expect(entityHref(person)).toBeNull()
    expect(entityHref(file)).toBeNull()
  })
})

describe("subjectKey", () => {
  it("mirrors the core's key format so both sides of a lookup agree", () => {
    expect(subjectKey(item(link("subject", pr)))).toBe("pull_request:p1")
    expect(subjectKey(item(link("subject", person)))).toBe("person:octocat")
    expect(subjectKey(item(link("subject", file)))).toBe("source_file:src/main.rs")
  })

  it("is null when there is no subject", () => {
    expect(subjectKey(item())).toBeNull()
  })
})
