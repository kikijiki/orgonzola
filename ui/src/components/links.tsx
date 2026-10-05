import type { PrRefView } from "@/bindings"
import { Avatar, ExtLink } from "@/components/primitives"
import { avatarUrl, ownerOf, repoUrl } from "@/lib/forge"
import { repoLabel } from "@/lib/format"
import { cn } from "@/lib/utils"

// A pull-request reference: a muted mono `#number` prefix and the title linked to the PR on the
// forge. Plain text when the forge reported no URL.
export function PrRefLink({ pr }: { pr: PrRefView }) {
  return (
    <span>
      <span className="font-mono text-muted-foreground">#{pr.number}</span>{" "}
      <ExtLink href={pr.url}>{pr.title}</ExtLink>
    </span>
  )
}

export function RepoLink({
  webBase,
  fullName,
  className,
}: {
  webBase: string | null
  fullName: string
  className?: string
}) {
  return (
    <span className={cn("inline-flex min-w-0 items-center gap-1.5", className)}>
      <Avatar login={ownerOf(fullName)} src={avatarUrl(webBase, ownerOf(fullName))} />
      <ExtLink href={repoUrl(webBase, fullName)}>
        {/* block: overflow/text-overflow do not apply to a plain inline span. */}
        <span className="block truncate">{repoLabel(fullName)}</span>
      </ExtLink>
    </span>
  )
}

// A user mention: avatar + login. With `onOpenPerson` it opens the People tab in-app; otherwise it
// links to the forge profile, or is plain text if neither.
export function UserLink({
  login,
  webBase,
  onOpenPerson,
  className,
}: {
  login: string
  webBase: string | null
  onOpenPerson?: (login: string) => void
  className?: string
}) {
  const inner = (
    <span className={cn("inline-flex items-center gap-1", className)}>
      <Avatar login={login} src={avatarUrl(webBase, login)} />
      <span className="hover:underline">{login}</span>
    </span>
  )
  if (onOpenPerson) {
    return (
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          onOpenPerson(login)
        }}
        className="inline-flex max-w-full items-center hover:text-foreground"
        title={`Open ${login} in People`}
      >
        {inner}
      </button>
    )
  }
  return inner
}
