// Helper to resolve public asset URLs respecting Vite base in all modes (dev and build).
// Our dev server is mounted at BASE_URL (e.g., '/simulatio_next/'), so always join with BASE_URL.
export function asset(path: string): string {
  const isDev: boolean = !!((import.meta as any).env?.DEV);
  const configuredBase: string = (import.meta as any).env?.BASE_URL ?? '/';
  const baseNorm = configuredBase.endsWith('/') ? configuredBase : configuredBase + '/';
  const clean = path.replace(/^\/+/, '');
  // In Vite dev, public/ is served at '/', not under base. Use root there.
  if (isDev) return `/${clean}`;
  return `${baseNorm}${clean}`;
}
