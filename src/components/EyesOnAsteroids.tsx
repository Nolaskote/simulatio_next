import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useProgress } from '@react-three/drei';
import { useRef } from 'react';
// Panel combinado de Estadísticas + Controles
import SolarSystem from '../three/SolarSystem';
import { NEOData } from '../utils/orbitalMath';
import { TimeState, createInitialTimeState, dateToJulianDay } from '../utils/timeSystem';
import './EyesOnAsteroids.css';
import './responsive.css';
import LoadingScreen from './LoadingScreen';
import { asset } from '../utils/asset';
import AsteroidInfo from './AsteroidInfo.tsx';
import StatsAndControls from './StatsAndControls.tsx';
import ImpactPredictor from './ImpactPredictor';
import SearchBox from './SearchBox';
import InfoPanel from './InfoPanel';

// Hook para cargar datos de asteroides
const useNEOSData = () => {
  const [neos, setNeos] = useState<NEOData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadNEOS = useCallback(async () => {
    try {
      setError(null);
      setLoading(true);
      // Try non-JSON extension first on Pages to bypass LFS rules there; fallback to .json for local/dev
      let text: string;
      try {
        const resJSN = await fetch(asset('data/neos.jsn'), { cache: 'no-cache' });
        if (!resJSN.ok) throw new Error('neos.jsn not found');
        text = await resJSN.text();
      } catch {
        const resJSON = await fetch(asset('data/neos.json'), { cache: 'no-cache' });
        if (!resJSON.ok) throw new Error('Failed to load NEO data');
        text = await resJSON.text();
      }
      // Parse manually to detect/throw if it's an LFS pointer
      if (/^\s*version https:\/\/git-lfs.github.com\/spec\/v1/m.test(text)) {
        throw new Error('NEO dataset is a Git LFS pointer on Pages. Redeploy needed.');
      }
      const data = JSON.parse(text);
      setNeos(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadNEOS(); }, [loadNEOS]);

  return { neos, loading, error, reload: loadNEOS };
};

// Components moved to separate files: AsteroidInfo.tsx and StatsAndControls.tsx

// Componente principal
export default function EyesOnAsteroids() {
  const { neos, loading, error, reload } = useNEOSData();
  // Pantalla de carga: espera mínima de 5s y datos + assets listos
  const [splashDone, setSplashDone] = useState(false);
  const dataReady = !loading && !error;
  // Progreso global de texturas/modelos vía Three LoadingManager
  const { progress } = useProgress();
  const assetsReady = progress >= 100;
  const [showAsteroids, setShowAsteroids] = useState(true);
  const [showOrbits, setShowOrbits] = useState(true);
  const [selectedAsteroid, setSelectedAsteroid] = useState<NEOData | null>(null);
  // When a search selection is active, only render that NEO
  const [searchSelectedId, setSearchSelectedId] = useState<string | null>(null);
  const [hoveredAsteroid, setHoveredAsteroid] = useState<NEOData | null>(null);
  // Toggle de brillo/halo alrededor (planetas y asteroides)
  const [haloEnabled, setHaloEnabled] = useState(false);
  // Iluminación fija, sin modificador (ahora controlada desde el sistema con un valor por defecto)
  // Parámetros de rendimiento fijos (mantenidos en SolarSystem): 60 Hz, tamaño 2.0
  const now = useMemo(() => new Date(), []);
  const [timeState, setTimeState] = useState<TimeState>({
    ...createInitialTimeState(),
    currentDate: now,
    julianDay: dateToJulianDay(now),
    isPlaying: true,
    timeScale: 1,
  });
  const [displayDate, setDisplayDate] = useState<Date>(now);
  const [displayJD, setDisplayJD] = useState<number>(dateToJulianDay(now));
  const [direction, setDirection] = useState<1 | -1>(1);
  // Velocidades predefinidas (días de simulación por segundo real)
  // 1s, 1min, 1h, 1d, 1w, 1m(~30d), 1y(365.25d)
  const speedPresets = useMemo(() => (
    [
      { label: '1s/s', daysPerSec: 1 / 86400 },
      { label: '1min/s', daysPerSec: 1 / 1440 },
      { label: '1h/s', daysPerSec: 1 / 24 },
      { label: '1d/s', daysPerSec: 1 },
      { label: '1w/s', daysPerSec: 7 },
      { label: '1m/s', daysPerSec: 30 },
      { label: '1y/s', daysPerSec: 365.25 },
    ]
  ), []);
  const [speedIndex, setSpeedIndex] = useState(0); // arranca en 1s/s
  const currentSpeed = speedPresets[speedIndex];
  // Mostrar/Ocultar paneles
  const [showLeftPanel, setShowLeftPanel] = useState(true);
  const [showTimeControls, setShowTimeControls] = useState(true);
  // Fondo espacial: por defecto desactivado, activable desde controles
  const [bgEnabled, setBgEnabled] = useState(false);
  const [showPredictor, setShowPredictor] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);
  // Toggle simple e inmediato: el fondo es un mesh que se monta/desmonta sin carreras
  const handleSetBgEnabled = useCallback((next: boolean) => {
    setBgEnabled(next);
  }, []);

  // Helper para clamping de timeScale
  const clampScale = (v: number) => Math.min(64, Math.max(0.125, v));

  // Mostrar todos los asteroides cuando showAsteroids está activo
  const filteredNeos = useMemo(() => {
    // Search selection has priority: show only that NEO if present
    if (searchSelectedId) {
      const one = neos.find(n => String(n.id) === String(searchSelectedId))
      return one ? [one] : []
    }
    // If no search selection: show all when toggle is on; none when off
    return showAsteroids ? neos : []
  }, [neos, showAsteroids, searchSelectedId]);

  // Estadísticas como antes: Totales de la BD
  const stats = useMemo(() => {
    const total = neos.length;
    const phas = neos.filter(neo => neo.type === 'PHA').length;
    const neosOnly = total - phas;
    return { total, phas, neosOnly };
  }, [neos]);

  // Se removieron los controles de tiempo; la simulación usa el tiempo inicial por defecto.

  if (error) {
    return (
      <div className="overlay overlay--error">
        <div>Error: {error}</div>
        <button className="btn btn--primary" onClick={reload}>Retry</button>
      </div>
    );
  }

  return (
    <div className="eoa-root">
      {/* Loader fijo arriba mientras se precargan datos + assets */}
      {!splashDone && (
        <LoadingScreen
          minDurationMs={5000}
          ready={dataReady && assetsReady}
          assetsProgressPct={progress}
          onComplete={() => setSplashDone(true)}
        />
      )}
      {/* Botones de mostrar (cuando están ocultos) */}
      {!showLeftPanel && (
        <button
          className="reveal-btn reveal-btn--left"
          onClick={() => setShowLeftPanel(true)}
          aria-label="Show Statistics and Controls"
        >
          Show Panel
        </button>
      )}
      {!showTimeControls && (
        <button
          className="reveal-btn reveal-btn--bottom"
          onClick={() => setShowTimeControls(true)}
          aria-label="Show Time Controls"
        >
          Show Time
        </button>
      )}
      {/* Simulación 3D: se pausa (se desmonta) cuando el predictor está abierto para evitar lag */}
      {!showPredictor && (
        <SolarSystem
          neos={filteredNeos}
          showAsteroids={showAsteroids}
          showOrbits={showOrbits}
          timeState={timeState}
          haloEnabled={haloEnabled}
          bgEnabled={bgEnabled}
          uiLightIntensity={2.0}
          uiPointSizePx={2}
          updateHz={60}
          pointSize={2.0}
          rateDaysPerSecond={currentSpeed.daysPerSec}
          onTimeUpdate={(jd, date) => { setDisplayDate(date); setDisplayJD(jd); }}
          direction={direction}
          onAsteroidHover={(neo) => setHoveredAsteroid(neo || null)}
          onAsteroidClick={(neo) => setSelectedAsteroid(curr => (curr && String(curr.id) === String(neo.id) ? null : neo))}
          selectedNeoId={selectedAsteroid?.id}
        />
      )}

  {/* Información del asteroide seleccionado movida junto al NEO (renderizada dentro de SolarSystem con <Html />) */}

      {/* Panel combinado: Estadísticas + Controles */}
      <StatsAndControls
        stats={stats}
        showAsteroids={showAsteroids}
        setShowAsteroids={(v: boolean) => { setShowAsteroids(v); if (!v) setSelectedAsteroid(null); }}
        showOrbits={showOrbits}
        setShowOrbits={setShowOrbits}
  bgEnabled={bgEnabled}
    setBgEnabled={handleSetBgEnabled}
        haloEnabled={haloEnabled}
        setHaloEnabled={setHaloEnabled}
  onOpenPredictor={() => { setShowPredictor(true); setTimeState(s => ({ ...s, isPlaying: false })); }}
        visible={showLeftPanel}
        onRequestHide={() => setShowLeftPanel(false)}
      />

      {/* Controles esquina superior derecha: buscador + botón info */}
      <div className="top-right-controls">
        <SearchBox
          neos={neos}
          placeholder="Search NEO/PHA…"
          maxSuggestions={15}
          onSelect={(neo) => {
            setSelectedAsteroid(neo);
            setSearchSelectedId(String(neo.id));
            if (!showAsteroids) setShowAsteroids(true);
          }}
          onClear={() => {
            setSearchSelectedId(null);
            setSelectedAsteroid(null);
            setShowAsteroids(true);
          }}
          compact
        />
  <button className="info-btn" data-open={infoOpen ? 'true' : 'false'} onClick={() => setInfoOpen(o => !o)} aria-label="Information">
    <i className="fa-solid fa-circle-info" aria-hidden />
    <span className="sr-only">Information</span>
  </button>
      </div>

      <InfoPanel visible={infoOpen} onClose={() => setInfoOpen(false)} />

      {/* Contenedor compacto con 3 columnas: izquierda, centro, derecha */}
      <div className={`time-controls time-controls--compact ${showTimeControls ? 'is-visible' : 'is-hidden'}`}>
        <div className="tc-badge">
          <div>Date: {displayDate.toLocaleDateString('en-US', { year: 'numeric', month: '2-digit', day: '2-digit' })}</div>
          <div>Time: {displayDate.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</div>
          <div>JD: {displayJD.toFixed(5)}</div>
        </div>
        <div className="tc-grid">
          {/* Columna izquierda: Velocidad + / etiqueta / Velocidad - */}
          <div className="tc-col tc-col--left">
            <button className="btn" onClick={() => setSpeedIndex(i => Math.min(speedPresets.length - 1, i + 1))}>Speed +</button>
            <div className="speed-label">Speed {currentSpeed.label}</div>
            <button className="btn" onClick={() => setSpeedIndex(i => Math.max(0, i - 1))}>Speed -</button>
          </div>

          {/* Columna centro: Reproducir/Pausar / Actual / Ocultar */}
          <div className="tc-col tc-col--center">
            <button className="btn btn--danger-outline" onClick={() => setTimeState(s => ({ ...s, isPlaying: !s.isPlaying }))}>
              {timeState.isPlaying ? 'Pause' : 'Play'}
            </button>
            <button className="btn btn--accent" onClick={() => {
              const now = new Date();
              const nowJD = dateToJulianDay(now);
              setTimeState(s => ({ ...s, julianDay: nowJD, currentDate: now }));
              setDisplayDate(now);
              setDisplayJD(nowJD);
            }}>
              Now
            </button>
            <button className="panel-close" onClick={() => setShowTimeControls(false)} aria-label="Hide time controls">Hide</button>
          </div>

          {/* Columna derecha: Futuro ▶ / Dirección / Pasado ◀ */}
          <div className="tc-col tc-col--right">
            <button className="btn" onClick={() => { setDirection(1); setTimeState(s => ({ ...s, isPlaying: true })); }}>Future ▶</button>
            <div className="dir-label">Direction: {direction === 1 ? 'Future' : 'Past'}</div>
            <button className="btn" onClick={() => { setDirection(-1); setTimeState(s => ({ ...s, isPlaying: true })); }}>Past ◀</button>
          </div>
        </div>
      </div>
      {showPredictor && (
        <ImpactPredictor
          neos={neos}
          onExit={() => { setShowPredictor(false); setTimeState(s => ({ ...s, isPlaying: true })); }}
        />
      )}
    </div>
  );
}
