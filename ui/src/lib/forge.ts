import type { ForgeView } from "@/bindings"

// Forge web links: turn API base URLs and `owner/name` / login / path identifiers into browsable
// URLs. Builders return `null` when the web base is unknown, so callers fall back to plain text.

// The forge's web root, derived from its API `base_url`: api.github.com maps to github.com; GHE and
// Gitea put the API under `/api/v3` or `/api/v1`. An unrecognized shape returns the base URL as-is.
export function forgeWebBase(
  forge: Pick<ForgeView, "base_url" | "kind"> | null | undefined,
): string | null {
  if (!forge) return null
  const b = forge.base_url.replace(/\/+$/, "")
  if (b === "https://api.github.com") return "https://github.com"
  for (const suffix of ["/api/v3", "/api/v1"]) {
    if (b.endsWith(suffix)) return b.slice(0, -suffix.length)
  }
  return b
}

export const repoUrl = (web: string | null, fullName: string): string | null =>
  web ? `${web}/${fullName}` : null

export const userUrl = (web: string | null, login: string): string | null =>
  web ? `${web}/${login}` : null

// A file at the repo's default branch (`HEAD` resolves to it on GitHub/Gitea).
export const fileUrl = (web: string | null, fullName: string, path: string): string | null =>
  web ? `${web}/${fullName}/blob/HEAD/${path}` : null

export const dirUrl = (web: string | null, fullName: string, path: string): string | null =>
  web ? `${web}/${fullName}/tree/HEAD/${path}` : null

export const commitUrl = (web: string | null, fullName: string, sha: string): string | null =>
  web ? `${web}/${fullName}/commit/${sha}` : null

// An owner/user/org avatar. `${web}/${login}.png` redirects to the avatar on GitHub (and GHE);
// other forges may 404, and Avatar falls back to initials.
export const avatarUrl = (web: string | null, login: string): string | null =>
  web ? `${web}/${login}.png?size=48` : null

export const ownerOf = (fullName: string): string => fullName.split("/")[0] ?? fullName
