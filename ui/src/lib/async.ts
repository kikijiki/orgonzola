import type { Result } from "@/bindings"

// A Tauri command promise can reject (IPC/transport error, e.g. a payload that fails to
// deserialize) or never settle. These helpers turn both into the ordinary `Result` error shape.

// How long a command call may take before it counts as failed. A large org's payload can take
// low seconds to serialize; 20s avoids tripping on a large board.
export const COMMAND_TIMEOUT_MS = 20_000

// Races a command call against a clock. If the call settles first, its result (or rejection)
// passes through. If the clock wins, resolves with a synthetic error `Result`.
export function withTimeout<T>(
  promise: Promise<Result<T, string>>,
  ms: number,
): Promise<Result<T, string>> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      resolve({ status: "error", error: `timed out after ${Math.round(ms / 1000)}s` })
    }, ms)
    promise.then(
      (r) => {
        clearTimeout(timer)
        resolve(r)
      },
      (e) => {
        clearTimeout(timer)
        reject(e)
      },
    )
  })
}

// Normalizes one `Promise.allSettled` slot into the plain `Result` callers check with
// `status === "ok"`, so a rejected promise reads like an application-level error.
export function settledResult<T>(r: PromiseSettledResult<Result<T, string>>): Result<T, string> {
  return r.status === "fulfilled"
    ? r.value
    : { status: "error", error: r.reason instanceof Error ? r.reason.message : String(r.reason) }
}
