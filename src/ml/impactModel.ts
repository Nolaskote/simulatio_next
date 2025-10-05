// Lightweight TensorFlow.js integration for impact probability predictions in the browser.
// - If a TFJS model is present at public/models/impact_model/model.json, it will be loaded and used.
// - Otherwise, a documented heuristic fallback is used. This is NOT a scientific model, only a placeholder.

import * as tf from '@tensorflow/tfjs'

export type ImpactFeatures = {
  // Minimum Orbit Intersection Distance (AU)
  moidAU: number
  // Miss distance at closest approach (km)
  missDistanceKm: number
  // Relative velocity at close approach (km/s)
  relativeVelocityKmS: number
  // Estimated diameter (km), optional
  diameterKm?: number
  // Absolute magnitude H, optional (lower is brighter/larger)
  hMagnitude?: number
  // Inclination (deg), optional
  inclinationDeg?: number
  // Days until (or since) the close approach epoch (days); optional
  daysToCloseApproach?: number
}

type ModelMeta = {
  targetType?: 'probability' | 'severity'
  modelType?: 'shallow' | 'deep'
  normalizeInModel?: boolean
  featureOrder?: string[]
}

export type LoadedImpactModel = {
  type: 'tfjs'
  model: tf.LayersModel
  meta?: ModelMeta
  calib?: { type: 'platt'; coef: number; intercept: number }
} | {
  type: 'heuristic'
}

let cachedModel: LoadedImpactModel | null = null

// Try to choose an efficient backend. We prefer WebGL; WASM is optional if installed.
export async function initTFBackend(prefer: Array<'wasm' | 'webgl' | 'cpu'> = ['webgl', 'wasm', 'cpu']): Promise<string> {
  // Attempt to dynamically register the WASM backend if the package is available.
  // This import is optional; if the dependency isn't installed, we silently skip it.
  if (prefer.includes('wasm')) {
    try {
      // Optional dependency: avoid static resolution by bundler
      const wasmPkg = '@tensorflow/tfjs-backend-wasm'
      // @ts-ignore
      const wasmMod: any = await import(/* @vite-ignore */ wasmPkg)
      if (wasmMod && typeof wasmMod.setWasmPaths === 'function') {
        const base = (typeof import.meta !== 'undefined' && (import.meta as any).env?.BASE_URL) || '/'
        const normBase = base.endsWith('/') ? base : base + '/'
        wasmMod.setWasmPaths(`${normBase}tfwasm/`)
      }
    } catch {
      // ignore if not available
    }
  }

  for (const backend of prefer) {
    try {
      const ok = await tf.setBackend(backend)
      if (ok) {
        await tf.ready()
        return tf.getBackend()
      }
    } catch {
      // try next backend
    }
  }
  await tf.ready()
  return tf.getBackend()
}

export async function loadImpactModel(modelUrl = '/models/impact_model/model.json'): Promise<LoadedImpactModel> {
  if (cachedModel) return cachedModel
  try {
    const model = await tf.loadLayersModel(modelUrl)
    // Try to load optional metadata and calibration placed next to model.json
    let meta: ModelMeta | undefined
    let calib: { type: 'platt'; coef: number; intercept: number } | undefined
    try {
      const base = modelUrl.replace(/model\.json$/, '')
      const [m, c] = await Promise.all([
        fetch(base + 'metadata.json').then(r => r.ok ? r.json() : undefined).catch(() => undefined),
        fetch(base + 'calibration.json').then(r => r.ok ? r.json() : undefined).catch(() => undefined),
      ])
      meta = m
      calib = c
    } catch {}
    cachedModel = { type: 'tfjs', model, meta, calib }
    return cachedModel
  } catch (err) {
    // No model available; fallback to heuristic
    cachedModel = { type: 'heuristic' }
    return cachedModel
  }
}

function featuresToTensor(f: ImpactFeatures): tf.Tensor {
  // Simple feature engineering and normalization to keep values roughly in [0, 1].
  const moidAU = isFinite(f.moidAU) ? Math.max(0, Math.min(1, f.moidAU / 0.5)) : 1 // cap at 0.5 AU
  const missKm = isFinite(f.missDistanceKm) ? Math.max(0, Math.min(1, f.missDistanceKm / 5_000_000)) : 1 // 5M km cap
  const relV = isFinite(f.relativeVelocityKmS) ? Math.max(0, Math.min(1, f.relativeVelocityKmS / 40)) : 0.5 // 40 km/s cap
  const diam = f.diameterKm != null && isFinite(f.diameterKm) ? Math.max(0, Math.min(1, f.diameterKm / 1)) : 0 // 1 km cap
  const H = f.hMagnitude != null && isFinite(f.hMagnitude) ? Math.max(0, Math.min(1, (35 - f.hMagnitude) / 20)) : 0.5 // lower H -> larger
  const inc = f.inclinationDeg != null && isFinite(f.inclinationDeg) ? Math.max(0, Math.min(1, f.inclinationDeg / 60)) : 0.3 // 60 deg cap
  const days = f.daysToCloseApproach != null && isFinite(f.daysToCloseApproach) ? Math.max(0, Math.min(1, 1 - Math.tanh(Math.abs(f.daysToCloseApproach) / 365))) : 0.5

  // Order matters if a trained model expects a specific feature order.
  return tf.tensor2d([[moidAU, missKm, relV, diam, H, inc, days]])
}

// Fallback heuristic: a simple logistic over hand-crafted weights.
function heuristicProbability(f: ImpactFeatures): number {
  // IMPORTANT: This is a placeholder. It is NOT validated and should be replaced by a trained model.
  const x = {
    moidAU: f.moidAU,
    missKm: f.missDistanceKm,
    relV: f.relativeVelocityKmS,
    diam: f.diameterKm ?? 0,
    H: f.hMagnitude ?? 22,
    inc: f.inclinationDeg ?? 10,
    days: f.daysToCloseApproach ?? 365,
  }

  // Normalize similarly to featuresToTensor
  const moid = Math.max(0, Math.min(1, (0.5 - Math.min(0.5, x.moidAU)) / 0.5)) // smaller MOID => higher risk
  const miss = Math.max(0, Math.min(1, (5_000_000 - Math.min(5_000_000, x.missKm)) / 5_000_000)) // smaller miss => higher
  const v = Math.max(0, Math.min(1, x.relV / 40))
  const d = Math.max(0, Math.min(1, (x.diam) / 1))
  const h = Math.max(0, Math.min(1, (35 - x.H) / 20)) // lower H -> larger
  const i = Math.max(0, Math.min(1, (60 - Math.min(60, x.inc)) / 60)) // lower inclination => slightly higher
  const t = Math.max(0, Math.min(1, 1 - Math.tanh(Math.abs(x.days) / 365))) // sooner => higher

  // Weights chosen for illustration only.
  const z = (
    3.0 * moid +
    3.0 * miss +
    1.0 * v +
    2.0 * d +
    1.5 * h +
    0.5 * i +
    1.5 * t -
    4.0 // bias
  )
  const prob = 1 / (1 + Math.exp(-z))
  return Math.max(0, Math.min(1, prob))
}

export async function predictImpactProbability(features: ImpactFeatures, options?: { modelUrl?: string }): Promise<{ probability: number, source: 'model' | 'heuristic' }>{
  const backend = tf.getBackend()
  if (!backend) {
    await initTFBackend()
  }
  const modelRef = await loadImpactModel(options?.modelUrl)

  if (modelRef.type === 'tfjs') {
    const input = featuresToTensor(features)
    const out = modelRef.model.predict(input) as tf.Tensor | tf.Tensor[]
    const tensor = Array.isArray(out) ? out[0] : out
    const data = await tensor.data()
    let val = Number.isFinite(data[0]) ? data[0] : heuristicProbability(features)
    // Optional Platt calibration
    if (modelRef.calib && isFinite(val)) {
      const { coef, intercept } = modelRef.calib
      const z = coef * val + intercept
      val = 1 / (1 + Math.exp(-z))
    }
    tf.dispose([input, out])
    return { probability: Math.max(0, Math.min(1, val)), source: 'model' }
  }

  return { probability: heuristicProbability(features), source: 'heuristic' }
}

export function clearCachedImpactModel() {
  if (cachedModel && cachedModel.type === 'tfjs') {
    cachedModel.model.dispose()
  }
  cachedModel = null
}
