// Helper to resolve public asset URLs respecting Vite base on build, but use root in dev.
// In Vite dev, public/ is served at '/', not under base. If we keep BASE_URL (e.g. '/simulatio_next/'),
// requests like '/simulatio_next/data/neos.json' 404 and return index.html, causing JSON parse errors.
export function asset(path: string): string {
  // Prefer plain root in dev to avoid subpath issues
  const isDev = (import.meta as any).env?.DEV ?? false;
  const base: string = isDev ? '/' : ((import.meta as any).env?.BASE_URL ?? '/');
  const baseNorm = base.endsWith('/') ? base : base + '/';
  const clean = path.replace(/^\/+/, '');
  return `${baseNorm}${clean}`;
}
