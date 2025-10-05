// Create a copy of index.html as 404.html for GitHub Pages SPA fallback
import { copyFile, mkdir, readdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const dist = resolve(__dirname, '..', 'dist');

try {
  await copyFile(resolve(dist, 'index.html'), resolve(dist, '404.html'));
  // Duplicate neos.json to neos.jsn to avoid Git LFS JSON rules on gh-pages
  try {
    const dataDir = resolve(dist, 'data');
    await copyFile(resolve(dataDir, 'neos.json'), resolve(dataDir, 'neos.jsn'));
  } catch {}
  // Copy TFJS WASM assets from public/tfwasm and if missing, from node_modules
  const tfSrc = resolve(__dirname, '..', 'public', 'tfwasm');
  const tfDst = resolve(dist, 'tfwasm');
  try {
    await mkdir(tfDst, { recursive: true });
    let files = []
    try {
      files = await readdir(tfSrc);
    } catch {}
    if (!files.length) {
      // fallback: try to read from node_modules
      const nm = resolve(__dirname, '..', 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist');
      try {
        files = (await readdir(nm)).filter(f => f.endsWith('.wasm'))
        await Promise.all(files.map(f => copyFile(resolve(nm, f), resolve(tfDst, f))));
      } catch {}
    } else {
      await Promise.all(files.map(f => copyFile(resolve(tfSrc, f), resolve(tfDst, f))));
    }
    console.log('Copied TFJS WASM assets');
  } catch (e) {
    // If missing, skip silently
  }
  console.log('Created dist/404.html for SPA fallback');
} catch (err) {
  console.error('Failed to create 404.html', err);
  process.exitCode = 1;
}
