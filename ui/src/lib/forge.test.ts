import { describe, expect, it } from "vitest"
import {
  avatarUrl,
  commitUrl,
  dirUrl,
  fileUrl,
  forgeWebBase,
  ownerOf,
  repoUrl,
  userUrl,
} from "./forge"

describe("forgeWebBase", () => {
  it("is null with no forge, so callers degrade to plain text", () => {
    expect(forgeWebBase(null)).toBeNull()
    expect(forgeWebBase(undefined)).toBeNull()
  })

  it("maps github.com's API host to its web host", () => {
    expect(forgeWebBase({ base_url: "https://api.github.com", kind: "github" })).toBe(
      "https://github.com",
    )
  })

  it("strips a trailing slash before matching", () => {
    expect(forgeWebBase({ base_url: "https://api.github.com/", kind: "github" })).toBe(
      "https://github.com",
    )
  })

  it("strips the GitHub Enterprise and Gitea API suffixes", () => {
    expect(forgeWebBase({ base_url: "https://ghe.acme.com/api/v3", kind: "github" })).toBe(
      "https://ghe.acme.com",
    )
    expect(forgeWebBase({ base_url: "https://gitea.acme.com/api/v1", kind: "gitea" })).toBe(
      "https://gitea.acme.com",
    )
  })

  it("falls back to the base URL as-is for an unrecognized shape", () => {
    expect(forgeWebBase({ base_url: "https://forge.acme.com", kind: "gitea" })).toBe(
      "https://forge.acme.com",
    )
  })
})

describe("url builders", () => {
  const web = "https://github.com"

  it("all return null when the web base is unknown", () => {
    expect(repoUrl(null, "acme/widgets")).toBeNull()
    expect(userUrl(null, "octocat")).toBeNull()
    expect(fileUrl(null, "acme/widgets", "src/main.rs")).toBeNull()
    expect(dirUrl(null, "acme/widgets", "src")).toBeNull()
    expect(commitUrl(null, "acme/widgets", "abc123")).toBeNull()
    expect(avatarUrl(null, "octocat")).toBeNull()
  })

  it("build the browsable forge URLs", () => {
    expect(repoUrl(web, "acme/widgets")).toBe("https://github.com/acme/widgets")
    expect(userUrl(web, "octocat")).toBe("https://github.com/octocat")
    expect(fileUrl(web, "acme/widgets", "src/main.rs")).toBe(
      "https://github.com/acme/widgets/blob/HEAD/src/main.rs",
    )
    expect(dirUrl(web, "acme/widgets", "src")).toBe("https://github.com/acme/widgets/tree/HEAD/src")
    expect(commitUrl(web, "acme/widgets", "abc123")).toBe(
      "https://github.com/acme/widgets/commit/abc123",
    )
    expect(avatarUrl(web, "octocat")).toBe("https://github.com/octocat.png?size=48")
  })
})

describe("ownerOf", () => {
  it("takes the owner half of owner/name", () => {
    expect(ownerOf("acme/widgets")).toBe("acme")
  })

  it("falls back to the whole string when there is no slash", () => {
    expect(ownerOf("acme")).toBe("acme")
  })
})
