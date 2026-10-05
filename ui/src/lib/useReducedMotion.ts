import { useEffect, useState } from "react"

const QUERY = "(prefers-reduced-motion: reduce)"

// Whether the OS asks for reduced motion. The stylesheet covers CSS; this hook is for Recharts,
// whose entry and tooltip animations run in JavaScript. WebKitGTK re-evaluates
// `prefers-reduced-motion` when GTK's `gtk-enable-animations` flips, so the value can change at
// runtime.
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => window.matchMedia(QUERY).matches)

  useEffect(() => {
    const mq = window.matchMedia(QUERY)
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches)
    mq.addEventListener("change", onChange)
    // Re-read on mount: the setting may have changed between the initial state and this effect.
    setReduced(mq.matches)
    return () => mq.removeEventListener("change", onChange)
  }, [])

  return reduced
}
