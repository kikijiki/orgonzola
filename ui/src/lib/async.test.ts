import { describe, expect, it, vi } from "vitest"
import { settledResult, withTimeout } from "./async"

describe("withTimeout", () => {
  it("passes a settled result through unchanged", async () => {
    const r = await withTimeout(Promise.resolve({ status: "ok" as const, data: 42 }), 1000)
    expect(r).toEqual({ status: "ok", data: 42 })
  })

  it("passes an application-level error through unchanged", async () => {
    const r = await withTimeout(Promise.resolve({ status: "error" as const, error: "nope" }), 1000)
    expect(r).toEqual({ status: "error", error: "nope" })
  })

  it("rejects when the underlying call rejects, so a transport failure is not masked", async () => {
    await expect(withTimeout(Promise.reject(new Error("ipc died")), 1000)).rejects.toThrow(
      "ipc died",
    )
  })

  it("resolves with a synthetic error rather than hanging when the clock wins", async () => {
    vi.useFakeTimers()
    const pending = withTimeout(new Promise<never>(() => {}), 20_000)
    await vi.advanceTimersByTimeAsync(20_000)
    expect(await pending).toEqual({ status: "error", error: "timed out after 20s" })
    vi.useRealTimers()
  })
})

describe("settledResult", () => {
  it("unwraps a fulfilled slot", () => {
    expect(settledResult({ status: "fulfilled", value: { status: "ok", data: 1 } })).toEqual({
      status: "ok",
      data: 1,
    })
  })

  it("turns a rejection into the same error shape the UI already branches on", () => {
    expect(settledResult({ status: "rejected", reason: new Error("boom") })).toEqual({
      status: "error",
      error: "boom",
    })
  })

  it("stringifies a non-Error rejection reason", () => {
    expect(settledResult({ status: "rejected", reason: "plain string" })).toEqual({
      status: "error",
      error: "plain string",
    })
  })
})
