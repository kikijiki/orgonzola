import { describe, expect, it, vi } from "vitest"
import {
  attentionMeaning,
  attentionStyle,
  codeRef,
  formatBytes,
  formatDate,
  formatDuration,
  relativeAge,
  repoLabel,
  storageGroupLabel,
} from "./format"

describe("repoLabel", () => {
  it("strips the repo: prefix from an id", () => {
    expect(repoLabel("repo:acme/widgets")).toBe("acme/widgets")
  })

  it("leaves a bare full_name alone", () => {
    expect(repoLabel("acme/widgets")).toBe("acme/widgets")
  })
})

describe("codeRef", () => {
  it("splits a ref_id into repo and path", () => {
    expect(codeRef("repo:acme/widgets#src/main.rs")).toEqual({
      repo: "acme/widgets",
      path: "src/main.rs",
    })
  })

  it("keeps a path containing a later # intact", () => {
    expect(codeRef("repo:acme/widgets#src/a#b.rs").path).toBe("src/a#b.rs")
  })

  it("falls back to the whole id as the path when there is no separator", () => {
    expect(codeRef("repo:acme/widgets")).toEqual({ repo: "", path: "acme/widgets" })
  })
})

describe("formatDuration", () => {
  it("shows a dash for an unknown duration", () => {
    expect(formatDuration(null)).toBe("-")
  })

  it("scales to days, hours, then minutes", () => {
    expect(formatDuration(86_400 * 2)).toBe("2.0d")
    expect(formatDuration(3_600 * 5)).toBe("5.0h")
    expect(formatDuration(600)).toBe("10m")
  })

  it("never rounds a non-zero duration down to 0m", () => {
    expect(formatDuration(5)).toBe("1m")
  })

  it("switches unit exactly at the boundary", () => {
    expect(formatDuration(3_600)).toBe("1.0h")
    expect(formatDuration(3_599)).toBe("60m")
    expect(formatDuration(86_400)).toBe("1.0d")
  })
})

describe("relativeAge", () => {
  it("is empty for a missing or unparseable timestamp", () => {
    expect(relativeAge(null)).toBe("")
    expect(relativeAge(undefined)).toBe("")
    expect(relativeAge("not a date")).toBe("")
  })

  it("reads under a minute as 'just now'", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-01-01T00:00:30Z"))
    expect(relativeAge("2026-01-01T00:00:00Z")).toBe("just now")
    vi.useRealTimers()
  })

  it("formats an older timestamp as a duration", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-01-03T00:00:00Z"))
    expect(relativeAge("2026-01-01T00:00:00Z")).toBe("2.0d")
    vi.useRealTimers()
  })
})

describe("formatDate", () => {
  it("shows a dash for a missing or unparseable timestamp", () => {
    expect(formatDate(null)).toBe("-")
    expect(formatDate("not a date")).toBe("-")
  })

  it("renders a parseable timestamp as a short date", () => {
    // Locale-dependent, so assert it produced something date-like rather than one exact string.
    expect(formatDate("2026-06-03T12:00:00Z")).not.toBe("-")
  })
})

describe("formatBytes", () => {
  it("shows a dash when the size is unknown", () => {
    expect(formatBytes(null)).toBe("-")
    expect(formatBytes(undefined)).toBe("-")
  })

  it("keeps whole bytes whole and scales by 1024", () => {
    expect(formatBytes(0)).toBe("0 B")
    expect(formatBytes(512)).toBe("512 B")
    expect(formatBytes(1024)).toBe("1.0 KB")
    expect(formatBytes(1024 * 1024 * 1.5)).toBe("1.5 MB")
  })

  it("drops the decimal once the number is big enough not to need it", () => {
    expect(formatBytes(1024 * 200)).toBe("200 KB")
  })

  it("stops scaling at TB and keeps the sign", () => {
    expect(formatBytes(-2048)).toBe("-2.0 KB")
    expect(formatBytes(1024 ** 5)).toBe("1024 TB")
  })
})

describe("attentionStyle", () => {
  it("labels a known kind", () => {
    expect(attentionStyle("failing_ci").label.length).toBeGreaterThan(0)
  })

  it("falls back rather than throwing on an unknown kind", () => {
    expect(() => attentionStyle("not_a_real_kind")).not.toThrow()
  })
})

describe("attentionMeaning", () => {
  it("explains every kind the style function knows about", () => {
    for (const kind of ["failing_ci", "stale_pr", "merged_without_review", "orphan_pr"]) {
      expect(attentionMeaning(kind)).not.toBe("")
    }
  })

  it("is empty for an unknown kind, so the UI can omit the line", () => {
    expect(attentionMeaning("not_a_real_kind")).toBe("")
  })
})

describe("storageGroupLabel", () => {
  it("names a known group in plain language", () => {
    expect(storageGroupLabel("search_index")).toBe("Search index (vector + full text)")
  })

  it("falls back to the raw value rather than hiding an unknown group", () => {
    expect(storageGroupLabel("brand_new_group")).toBe("brand_new_group")
  })
})
