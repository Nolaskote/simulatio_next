import React from 'react';

export default function StatsAndControls({
  stats,
  showAsteroids,
  setShowAsteroids,
  showOrbits,
  setShowOrbits,
  haloEnabled,
  setHaloEnabled,
  bgEnabled,
  setBgEnabled,
  onOpenPredictor,
  pointSizePx,
  setPointSizePx,
  visible = true,
  onRequestHide,
}: {
  stats: { total: number; phas: number; neosOnly: number };
  showAsteroids: boolean;
  setShowAsteroids: (v: boolean) => void;
  showOrbits: boolean;
  setShowOrbits: (v: boolean) => void;
  haloEnabled: boolean;
  setHaloEnabled: (v: boolean) => void;
  bgEnabled: boolean;
  setBgEnabled: (v: boolean) => void;
  onOpenPredictor?: () => void;
  pointSizePx: number;
  setPointSizePx: (v: number) => void;
  visible?: boolean;
  onRequestHide?: () => void;
}) {
  return (
    <div className={`panel panel--left ${visible ? 'is-visible' : 'is-hidden'}`}>
      <button className="panel-close" onClick={onRequestHide} aria-label="Hide panel">
        <i className="fa-solid fa-chevron-left" aria-hidden />
        <span style={{ marginLeft: 6 }}>Hide</span>
      </button>

      {/* Sección: Estadísticas */}
      <section className="section section--stats">
        <h3 className="title"><i className="fa-solid fa-chart-line" style={{ marginRight: 6 }} />Statistics</h3>
        <p className="stat-line">Total in DB: <strong>{stats.total.toLocaleString()}</strong></p>
        <p className="stat-line">PHA: <strong className="pha">{stats.phas.toLocaleString()}</strong></p>
        <p className="stat-line">NEO: <strong className="neo">{stats.neosOnly.toLocaleString()}</strong></p>
        <div style={{ marginTop: 10 }}>
          <label htmlFor="size-slider" style={{ display: 'block', marginBottom: 8 }}>
            <i className="fa-solid fa-circle-dot" style={{ marginRight: 6 }} />
            Asteroid size: <strong>{pointSizePx}px</strong>
          </label>
          <input
            id="size-slider"
            type="range"
            min={2}
            max={8}
            step={1}
            value={pointSizePx}
            onChange={(e) => setPointSizePx(parseInt(e.target.value, 10))}
            style={{ width: '100%' }}
            aria-label="Asteroid size in pixels"
          />
        </div>
        <div className="separator" />
      </section>

      {/* Sección: Controles */}
      <section className="section section--controls">
        <h3 className="title"><i className="fa-solid fa-sliders" style={{ marginRight: 6 }} />Controls</h3>
        <div className="grid">
          <label className="check">
            <input type="checkbox" checked={showAsteroids} onChange={e => setShowAsteroids(e.target.checked)} />
            <span><i className="fa-solid fa-circle-dot" style={{ marginRight: 6 }} />Show Asteroids</span>
          </label>
          <label className="check">
            <input type="checkbox" checked={showOrbits} onChange={e => setShowOrbits(e.target.checked)} />
            <span><i className="fa-solid fa-ring" style={{ marginRight: 6 }} />Show Orbits</span>
          </label>
          <label className="check">
            <input type="checkbox" checked={bgEnabled} onChange={e => setBgEnabled(e.target.checked)} />
            <span><i className="fa-solid fa-star" style={{ marginRight: 6 }} />Space background</span>
          </label>
          <label className="check">
            <input type="checkbox" checked={haloEnabled} onChange={e => setHaloEnabled(e.target.checked)} />
            <span><i className="fa-solid fa-sun" style={{ marginRight: 6 }} />Glow around</span>
          </label>
          {/* Sliders removidos por solicitud. Se mantiene configuración fija. */}
        </div>
        {/* Nueva línea blanca debajo de "Brillo alrededor" */}
        <div className="separator" />
      </section>

      {/* Sección: Indicaciones (leyenda) */}
      <section className="section section--legend">
        <h3 className="title"><i className="fa-solid fa-circle-question" style={{ marginRight: 6 }} />Tips</h3>
        <div className="legend">
          <p><i className="fa-solid fa-computer-mouse" style={{ marginRight: 6 }} />Drag to rotate</p>
          <p><i className="fa-solid fa-magnifying-glass" style={{ marginRight: 6 }} />Scroll to zoom</p>
          <p><i className="fa-solid fa-hand-pointer" style={{ marginRight: 6 }} />Click asteroids</p>
          <p><i className="fa-solid fa-circle" style={{ color: '#ff4444', marginRight: 6 }} />Red = PHA</p>
          <p><i className="fa-solid fa-circle" style={{ color: '#00e676', marginRight: 6 }} />Green = NEO</p>
        </div>
      </section>
    </div>
  );
}
