import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

// Render GitHub-flavored Markdown (PR/issue bodies) as styled prose. `prose` comes from the
// Tailwind typography plugin.
export function Markdown({ children }: { children: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  )
}
