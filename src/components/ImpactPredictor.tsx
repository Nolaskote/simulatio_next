import React, { useEffect, useMemo, useState } from 'react'
import { NEOData } from '../utils/orbitalMath'
import { predictImpactProbability } from '../ml/impactModel'
import { asset } from '../utils/asset'
import CesiumGlobe from './CesiumGlobe'
// Note: Legacy Three.js globe removed in favor of Cesium globe-only view

export default function ImpactPredictor({ neos, onExit }: { neos: NEOData[]; onExit: () => void }) {
  const [selectedId, setSelectedId] = useState<string>('')
  const selectedNeo = useMemo(() => neos.find(n => String(n.id) === String(selectedId)) || null, [neos, selectedId])
  const [prob, setProb] = useState<{ probability: number, source: 'model' | 'heuristic' } | null>(null)
  // Default location: Riobamba, Chimborazo, Ecuador
  // Approx coords: Lat -1.664, Lon -78.654
  const [markerLat, setMarkerLat] = useState<number>(-1.664)
  const [markerLon, setMarkerLon] = useState<number>(-78.654)
  // Do not override marker based on NEO selection; user controls the marker.

  useEffect(() => {
    let canceled = false
    async function run() {
      if (!selectedNeo) { if (!canceled) setProb(null); return }
      // Map limited orbital data to features; placeholders where unknown.
      const a = Number(selectedNeo.a)
      const e = Number(selectedNeo.e)
      const i = Number(selectedNeo.i)
      const q = (Number.isFinite(a) && Number.isFinite(e)) ? a * (1 - e) : NaN // perihelion distance ~ lower => riskier
      const features = {
        moidAU: Number.isFinite(q) ? q : 0.3,
        missDistanceKm: 3_000_000, // placeholder
        relativeVelocityKmS: 20,   // placeholder
        diameterKm: undefined,
        hMagnitude: undefined,
        inclinationDeg: Number.isFinite(i) ? i : 10,
        daysToCloseApproach: 180,
      }
      const out = await predictImpactProbability(features)
      if (!canceled) setProb(out)
    }
    run()
    return () => { canceled = true }
  }, [selectedNeo])

  // All picking handled within CesiumGlobe via onPick

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#000', zIndex: 1000, display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px', background: 'rgba(0,0,0,0.8)', borderBottom: '1px solid #222' }}>
        <button className="btn" onClick={onExit}>← Volver</button>
        <div style={{ color: '#fff', fontWeight: 700 }}>Predecir Impactos</div>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <label style={{ color: '#ddd' }}>NEO/PHA:</label>
          <select value={selectedId} onChange={e => setSelectedId(e.target.value)} style={{ padding: '4px 8px' }}>
            <option value="">— Selecciona —</option>
            {neos.map((n, idx) => (
              <option key={`${n.id}-${idx}`} value={n.id}>{n.name} ({n.type})</option>
            ))}
          </select>
          {selectedNeo && (
            <div style={{ color: '#fff' }}>
              Probabilidad: <strong>{prob ? (prob.probability * 100).toFixed(2) : '—'}%</strong> <span style={{ color: '#aaa' }}>({prob?.source || '—'})</span>
            </div>
          )}
        </div>
      </div>
      <div style={{ flex: 1, position: 'relative' }}>
        <CesiumGlobe
          markerLat={markerLat}
          markerLon={markerLon}
          onPick={(lat, lon) => { setMarkerLat(lat); setMarkerLon(lon) }}
        />
        {/* Panel flotante de coordenadas (no bloquea interacción de widgets) */}
        <div style={{ position: 'absolute', bottom: 12, right: 12, background: 'rgba(0,0,0,0.6)', color: '#fff', padding: '8px 10px', borderRadius: 8, border: '1px solid #333', pointerEvents: 'none' }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>Coordenadas</div>
          <div>Lat: {markerLat.toFixed(6)}°</div>
          <div>Lon: {markerLon.toFixed(6)}°</div>
        </div>
      </div>
      {/* footer removido */}
    </div>
  )
}
