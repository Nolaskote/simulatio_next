// Helper to resolve public asset URLs respecting Vite base (e.g., /Simulation/ on GitHub Pages)
// Avoid using new URL with a relative base (which throws in browsers). Do simple path join.
export function asset(path: string): string {
  const base: string = (import.meta as any).env?.BASE_URL ?? (import.meta as any).env?.VITE_BASE ?? '/';
  const baseNorm = base.endsWith('/') ? base : base + '/';
  const clean = path.replace(/^\/+/, '');
  return `${baseNorm}${clean}`;
}
