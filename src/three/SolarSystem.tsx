import React, { useState, useMemo, Suspense, useCallback, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import Planet from "./Planet";
import Asteroid from "../components/Asteroid";
import AsteroidField from "../components/AsteroidField";
import OrbitPath from "../components/OrbitPath";
import { NEOData } from "../utils/orbitalMath";
import { TimeState, getTimeInDays, J2000_EPOCH, julianDayToDate } from "../utils/timeSystem";
import TimeTicker from './TimeTicker';
import SpaceBackground from './SpaceBackground';
import { AU_TO_UNITS } from "../utils/constants";
import { asset } from "../utils/asset";

// Datos reales del sistema solar con escalas apropiadas
const PLANETS = [
  {
    name: "Mercury",
  texture: asset("textures/mercury.jpg"),
    size: 0.38,
    distance: 5.8, // UA real: 0.387, escalado para visualización
    rotation: 58.6,
    retrograde: false,
    orbitalPeriod: 88, // días
    // Elementos keplerianos aproximados
    eccentricity: 0.2056,
    inclination: 7.0,
    longitudeAscendingNode: 48.3,
    argumentPerihelion: 29.1,
    meanAnomaly: 252.3
  },
  {
    name: "Venus",
  texture: asset("textures/venus.jpg"),
    size: 0.95,
    distance: 10.8, // UA real: 0.723
    rotation: -243,
    retrograde: true,
    orbitalPeriod: 225,
    eccentricity: 0.0067,
    inclination: 3.4,
    longitudeAscendingNode: 76.7,
    argumentPerihelion: 54.9,
    meanAnomaly: 181.0
  },
  {
    name: "Earth",
  texture: asset("textures/earth.jpg"),
    size: 1,
    distance: 15, // UA real: 1.0
    rotation: 1,
    retrograde: false,
    orbitalPeriod: 365.25,
    eccentricity: 0.0167,
    inclination: 0.0,
    longitudeAscendingNode: 0.0,
    argumentPerihelion: 102.9,
    meanAnomaly: 100.5
  },
  {
    name: "Mars",
  texture: asset("textures/mars.jpg"),
    size: 0.53,
    distance: 22.8, // UA real: 1.524
    rotation: 1.03,
    retrograde: false,
    orbitalPeriod: 687,
    eccentricity: 0.0934,
    inclination: 1.8,
    longitudeAscendingNode: 49.6,
    argumentPerihelion: 286.5,
    meanAnomaly: 355.4
  },
  {
    name: "Jupiter",
  texture: asset("textures/jupiter.jpg"),
    size: 11.2,
    distance: 77.8, // UA real: 5.203
    rotation: 0.41,
    retrograde: false,
    orbitalPeriod: 4333,
    eccentricity: 0.0484,
    inclination: 1.3,
    longitudeAscendingNode: 100.5,
    argumentPerihelion: 275.1,
    meanAnomaly: 34.4
  },
  {
    name: "Saturn",
  texture: asset("textures/saturn.jpg"),
    size: 9.45,
    distance: 143, // UA real: 9.537
    rotation: 0.45,
    retrograde: false,
    orbitalPeriod: 10759,
    eccentricity: 0.0542,
    inclination: 2.5,
    longitudeAscendingNode: 113.7,
    argumentPerihelion: 339.4,
    meanAnomaly: 50.1,
    hasRings: true,
  ringTexture: asset("textures/saturn_ring.png"),
  },
  {
    name: "Uranus",
  texture: asset("textures/uranus.jpg"),
    size: 4.0,
    distance: 287, // UA real: 19.191
    rotation: -0.72,
    retrograde: true,
    orbitalPeriod: 30687,
    eccentricity: 0.0472,
    inclination: 0.8,
    longitudeAscendingNode: 74.0,
    argumentPerihelion: 96.5,
    meanAnomaly: 142.2
  },
  {
    name: "Neptune",
  texture: asset("textures/neptune.jpg"),
    size: 3.88,
    distance: 450, // UA real: 30.069
    rotation: 0.67,
    retrograde: false,
    orbitalPeriod: 60190,
    eccentricity: 0.0086,
    inclination: 1.8,
    longitudeAscendingNode: 131.8,
    argumentPerihelion: 265.6,
    meanAnomaly: 260.2
  },
];

// Colores de órbita por planeta
const ORBIT_COLORS: Record<string, string> = {
  Mercury: '#c2b280',
  Venus: '#f0d9a7',
  Earth: '#6ab7ff',
  Mars: '#ff6f61',
  Jupiter: '#ffd27a',
  Saturn: '#ffe0a3',
  Uranus: '#98e1ff',
  Neptune: '#8bb6ff',
};

interface SolarSystemProps {
  neos?: NEOData[];
  showAsteroids?: boolean;
  showOrbits?: boolean;
  timeState: TimeState;
  onAsteroidHover?: (neo: NEOData | undefined) => void;
  onAsteroidClick?: (neo: NEOData) => void;
  onTimeUpdate?: (jd: number, date: Date) => void; // notify current JD/time
  direction?: 1 | -1; // forward or backward play
  haloEnabled?: boolean; // show halos around planets/asteroids
  bgEnabled?: boolean; // toggle background space image
  updateHz?: number; // target updates per second for asteroid positions
  pointSize?: number; // size of asteroid points
  rateDaysPerSecond?: number; // días simulados por segundo real
  uiLightIntensity?: number; // UI-controlled planet light intensity
  uiPointSizePx?: number; // UI-controlled asteroid point size in px
  selectedNeoId?: string; // externally selected neo id
}

export default function SolarSystem({ 
  neos = [], 
  showAsteroids = true, 
  showOrbits = true,
  timeState,
  onAsteroidHover,
  onAsteroidClick,
  onTimeUpdate,
  direction = 1,
  haloEnabled = true,
  bgEnabled = false,
  updateHz = 60,
  pointSize = 2.0,
  rateDaysPerSecond,
  uiLightIntensity = 1.2,
  uiPointSizePx = 7,
  selectedNeoId,
}: SolarSystemProps) {
  const [simulationTime, setSimulationTime] = useState(0);
  const [currentJD, setCurrentJD] = useState(timeState.julianDay);
  const [selectedNeo, setSelectedNeo] = useState<NEOData | null>(null);
  const [openPlanet, setOpenPlanet] = useState<string | null>(null);
  const disableBG = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('nobg');
  const minimal = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('minimal');
  const lowParam = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('low');
  const isMobile = typeof navigator !== 'undefined' && /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;
  const lowQuality = lowParam || isMobile || dpr > 1.5;

  // Quality knobs
  const planetSegments = lowQuality ? 24 : 64;
  const orbitSegments = lowQuality ? 64 : 128;
  const asteroidUpdateHz = lowQuality ? Math.min(30, updateHz) : updateHz;
  const asteroidGlowEnabled = lowQuality ? false : haloEnabled;

  // Sin filtrar: todos los asteroides (cap en modo low para rendimiento)
  const asteroids = useMemo(() => (showAsteroids ? neos : []), [neos, showAsteroids]);

  // Smoothly interpolate to external JD when parent changes it (e.g., clicking "Actual")
  const animatingRef = useRef(false);
  const animFrameRef = useRef<number | null>(null);
  const targetJDRef = useRef<number>(timeState.julianDay);

  useEffect(() => {
    const from = currentJD;
    const to = timeState.julianDay;
    targetJDRef.current = to;
    // If no significant change, just ensure exact sync and skip animation
    if (!isFinite(from) || Math.abs(to - from) < 1e-7) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      animatingRef.current = false;
      setCurrentJD(to);
      return;
    }
    // Cancel any previous animation
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    animatingRef.current = true;
    const duration = 600; // ms
    const start = performance.now();
    const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = easeInOutCubic(t);
      // If parent changed target again mid-flight, update 'to'
      const latestTo = targetJDRef.current;
      const value = from + (latestTo - from) * eased;
      setCurrentJD(value);
      if (t < 1) {
        animFrameRef.current = requestAnimationFrame(step);
      } else {
        animatingRef.current = false;
        animFrameRef.current = null;
        setCurrentJD(latestTo);
      }
    };
    animFrameRef.current = requestAnimationFrame(step);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    };
  }, [timeState.julianDay]);

  // Derive simulation time (days since J2000) whenever JD changes
  useEffect(() => {
    const timeInDays = getTimeInDays(currentJD, J2000_EPOCH);
    setSimulationTime(timeInDays);
  }, [currentJD]);

  // Advance JD smoothly each frame when playing
  const advanceDays = useCallback((daysDelta: number) => {
    // Skip advancing while tweening to a new external JD
    if (animatingRef.current) return;
    setCurrentJD(prev => prev + daysDelta * direction);
  }, [direction]);

  // Notify parent about time changes AFTER commit to avoid setState-during-render warnings.
  const onTimeUpdateRef = useRef(onTimeUpdate);
  useEffect(() => { onTimeUpdateRef.current = onTimeUpdate; }, [onTimeUpdate]);
  useEffect(() => {
    if (onTimeUpdateRef.current) {
      onTimeUpdateRef.current(currentJD, julianDayToDate(currentJD));
    }
  }, [currentJD]);

  const handleAsteroidClick = useCallback((neo: NEOData) => {
    // Open asteroid card -> close any planet card
    setOpenPlanet(null);
    setSelectedNeo(curr => (curr && String(curr.id) === String(neo.id) ? null : neo));
    onAsteroidClick?.(neo);
  }, [onAsteroidClick]);

  // Toggle planet info card; when opening a planet, close asteroid selection
  const handlePlanetToggle = useCallback((planetName: string) => {
    setSelectedNeo(null);
    setOpenPlanet(prev => (prev === planetName ? null : planetName));
  }, []);

  // If asteroids are hidden or the currently selected NEO is no longer in the provided list, clear selection
  useEffect(() => {
    if (!showAsteroids || neos.length === 0) {
      if (selectedNeo) setSelectedNeo(null);
      // keep only one card visible: if asteroids hide, leave planet card as-is
      return;
    }
    if (selectedNeo && !neos.some(n => String(n.id) === String(selectedNeo.id))) {
      setSelectedNeo(null);
    }
  }, [showAsteroids, neos, selectedNeo]);

  // Sync external selection id -> internal selectedNeo
  useEffect(() => {
    if (!selectedNeoId) {
      // Clear internal selection when external id is cleared
      setSelectedNeo(null)
      return
    }
    const match = neos.find(n => String(n.id) === String(selectedNeoId)) || null
    setSelectedNeo(match)
  }, [selectedNeoId, neos])

  if (minimal) {
    return (
      <Canvas
        style={{ width: '100%', height: '100%', display: 'block' }}
        camera={{ position: [0, 0, 5], fov: 50 }}
      >
        {/* @ts-ignore */}
        <color attach="background" args={[0x000000]} />
        <ambientLight intensity={0.6} />
        <pointLight position={[2, 3, 2]} intensity={1.2} />
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="#4da3ff" />
        </mesh>
      </Canvas>
    );
  }

  return (
    <Canvas 
      style={{ width: '100%', height: '100%', display: 'block' }}
      camera={{ 
        position: [0, 50, 150], 
        fov: 45,
        near: 0.1,
        far: 10000
      }}
      dpr={lowQuality ? 1 : dpr}
      gl={{ 
        antialias: !lowQuality,
        alpha: false,
        powerPreference: lowQuality ? 'low-power' : 'high-performance'
      }}
    >
  {/* Fondo negro base: si el fondo espacial está desactivado, queda negro; si está activado, la esfera suma por encima */}
  {/* @ts-ignore */}
  {!bgEnabled && <color attach="background" args={[0x000000]} />}
      {/* Smooth time advancement */}
      <TimeTicker
        isPlaying={timeState.isPlaying}
        rateDaysPerSecond={rateDaysPerSecond ?? (1 / 86400)}
        onAdvance={advanceDays}
      />
  {/* Fondo de Vía Láctea con fade in/out y rotación lenta */}
  <SpaceBackground enabled={bgEnabled} rotationSpeedRadPerSec={0.00015} fadeInMs={700} fadeOutMs={450} />
      
      {/* Iluminación fija a 300x (normalizada con base 170x) */}
      {(() => {
        // Reducimos ambient/hemisphere para que se note el terminador (día/noche)
        const norm = 300 / 170;
        return (
          <>
            <ambientLight intensity={0.015 * norm} color="#aab" />
            <hemisphereLight args={["#e0f0ff", "#222222", 0.03 * norm]} />
            {/* El Sol como fuente principal de luz */}
            <pointLight position={[0, 0, 0]} intensity={1400 * norm} distance={5000} decay={2} color="#ffffff" />
            <pointLight position={[600, 400, 600]} intensity={0.18 * norm} distance={0} decay={2} color="#cde" />
          </>
        );
      })()}

    {/* HDRI environment removed for stability */}
      
      {/* Se eliminan estrellas procedurales para evitar confusión con NEO/PHAs */}
      
      {/* Sol */}
      <Suspense fallback={(
        <mesh position={[0,0,0]}>
          <sphereGeometry args={[8, 32, 32]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>
      )}>
        <Planet
          name="Sun"
          texture={asset("textures/sun.jpg")}
          size={8}
          distance={0}
          rotation={25}
          retrograde={false}
          isSun
        />
      </Suspense>
      
      {/* Órbitas de los planetas */}
      {showOrbits && (
        <Suspense fallback={null}>
          {PLANETS.map((planet) => {
            const orbitColor = ORBIT_COLORS[planet.name] ?? '#88aaff';
            return (
              <OrbitPath
                key={`orbit-${planet.name}`}
                keplerianElements={{
                // a en UA aproximada: distancia visual estaba escalada; volvemos a UA aproximada
                a: planet.distance / 15, // mantener coherencia con visual
                e: planet.eccentricity || 0,
                i: planet.inclination || 0,
                Omega: planet.longitudeAscendingNode || 0,
                omega: planet.argumentPerihelion || 0,
                M: planet.meanAnomaly || 0,
                period: planet.orbitalPeriod || 365.25
              }}
                color={orbitColor}
                opacity={0.8}
                lineWidth={2}
                segments={orbitSegments}
              />
            );
          })}
        </Suspense>
      )}
      
      {/* Planetas */}
      <Suspense fallback={null}>
        {PLANETS.map((planet) => (
            <Planet 
            key={planet.name}
            {...planet} 
            simulationTime={simulationTime}
            haloEnabled={haloEnabled}
            segments={planetSegments}
            // pass UI intensity to each planet via prop using its directionalLight
            // intensity is handled inside Planet via its own light
            keplerianElements={{
              a: planet.distance / 15, // coherente con la órbita calculada arriba
              e: planet.eccentricity || 0,
              i: planet.inclination || 0,
              Omega: planet.longitudeAscendingNode || 0,
              omega: planet.argumentPerihelion || 0,
              M: planet.meanAnomaly || 0,
              period: planet.orbitalPeriod || 365.25
            }}
            infoOpen={openPlanet === planet.name}
            onToggleInfo={() => handlePlanetToggle(planet.name)}
          />
        ))}
      </Suspense>
      
      {/* Asteroides - todos, optimizados con Points */}
      {showAsteroids && (
        <Suspense fallback={null}>
          <AsteroidField
            neos={asteroids}
            time={simulationTime}
            onHover={onAsteroidHover}
            onClick={handleAsteroidClick}
            updateHz={asteroidUpdateHz}
            pointSize={pointSize}
            // maintain constant pixel size for points
            constantScreenSize={true}
            pointSizePx={3}
            autoScaleWithDistance={false}
            minPointSizePx={3}
            maxPointSizePx={3}
            // improve picking at long distances
            pickingRadiusWorld={0.7}
            selectedNeoId={selectedNeo?.id}
            highlightColor="#ffd700"
            glowEnabled={asteroidGlowEnabled}
            glowSizePx={6}
            glowOpacity={0.12}
          />
        </Suspense>
      )}

      {/* Órbita dorada del NEO/PHA seleccionado (solo si se muestran asteroides) */}
      {showAsteroids && selectedNeo && (
        <OrbitPath
          keplerianElements={{
            a: parseFloat(selectedNeo.a),
            e: parseFloat(selectedNeo.e),
            i: parseFloat(selectedNeo.i),
            Omega: parseFloat(selectedNeo.Omega),
            omega: parseFloat(selectedNeo.omega),
            M: parseFloat(selectedNeo.M),
            period: parseFloat(selectedNeo.period)
          }}
          color="#ffd700"
          opacity={0.95}
          lineWidth={3}
        />
      )}

      {/* Cartilla cerca del NEO/PHA seleccionado (solo si se muestran asteroides) */}
      {showAsteroids && selectedNeo && (() => {
        const a = parseFloat(selectedNeo.a);
        const e = parseFloat(selectedNeo.e);
        const inc = (parseFloat(selectedNeo.i) * Math.PI) / 180;
        const Om = (parseFloat(selectedNeo.Omega) * Math.PI) / 180;
        const om = (parseFloat(selectedNeo.omega) * Math.PI) / 180;
        const M0 = (parseFloat(selectedNeo.M) * Math.PI) / 180;
        const P = Math.max(1e-3, parseFloat(selectedNeo.period));
        const TWO_PI = Math.PI * 2;
        const n = TWO_PI / P;
        const M = M0 + n * simulationTime;
        const Mrad = ((M + Math.PI) % TWO_PI) - Math.PI;
        let E = e < 0.8 ? Mrad : Math.PI;
        for (let i = 0; i < 8; i++) {
          const f = E - e * Math.sin(E) - Mrad;
          const fp = 1 - e * Math.cos(E);
          const d = f / fp;
          E -= d;
          if (Math.abs(d) < 1e-7) break;
        }
        const cosE = Math.cos(E), sinE = Math.sin(E);
        const cosv = (cosE - e) / (1 - e * cosE);
        const sinv = Math.sqrt(1 - e * e) * sinE / (1 - e * cosE);
        const v = Math.atan2(sinv, cosv);
        const r = a * (1 - e * e) / (1 + e * Math.cos(v));
        const xPrime = r * Math.cos(v);
        const yPrime = r * Math.sin(v);
        const cosom = Math.cos(om), sinom = Math.sin(om);
        const x1 = xPrime * cosom - yPrime * sinom;
        const y1 = xPrime * sinom + yPrime * cosom;
        const cosi = Math.cos(inc), sini = Math.sin(inc);
        const y2 = y1 * cosi; const z2 = y1 * sini;
        const cosOm = Math.cos(Om), sinOm = Math.sin(Om);
        const x3 = x1 * cosOm - y2 * sinOm;
        const y3 = x1 * sinOm + y2 * cosOm;
        const z3 = z2;
        const pos: [number, number, number] = [x3 * AU_TO_UNITS, z3 * AU_TO_UNITS, -y3 * AU_TO_UNITS];
        const titleColor = selectedNeo.type === 'PHA' ? '#ff4444' : '#cccccc';
        const deg = (rad: number) => (rad * 180) / Math.PI;
        const Mdeg = ((deg(Mrad) % 360) + 360) % 360;
        const vdeg = ((deg(v) % 360) + 360) % 360;
        const fmt = (x: number) => (Number.isFinite(x) ? Number(x).toPrecision(3) : '—');
        const Pdays = Math.max(1e-6, parseFloat(selectedNeo.period));
        return (
          <Html position={pos} style={{ pointerEvents: 'none' }}>
            <div
              style={{
                background: 'rgba(0,0,0,0.85)',
                color: '#fff',
                padding: '12px 14px',
                borderRadius: 10,
                fontSize: 13,
                lineHeight: 1.45,
                width: 'max-content',
                border: `1px solid ${titleColor}88`,
                boxShadow: '0 2px 10px rgba(0,0,0,0.45)',
                backdropFilter: 'blur(4px)',
                WebkitBackdropFilter: 'blur(4px)',
                fontVariantNumeric: 'tabular-nums',
                transform: 'translate(16px, 12px)', // pequeño desplazamiento a la derecha y abajo
              }}
            >
                 <div style={{ fontWeight: 700, color: titleColor, marginBottom: 8, fontSize: 15 }}>
                {selectedNeo.name}
              </div>
                 <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', columnGap: 12, rowGap: 4 }}>
                   <span style={{ color: '#aaa' }}>ID</span>
                <span style={{ color: titleColor }}>{selectedNeo.id}</span>
                   <span style={{ color: '#bbb' }}>P (d)</span>
                <span>{fmt(Pdays)}</span>
                   <span style={{ color: '#bbb' }}>a (UA)</span>
                <span>{fmt(a)}</span>
                   <span style={{ color: '#bbb' }}>e</span>
                <span>{fmt(e)}</span>
                   <span style={{ color: '#bbb' }}>i (°)</span>
                <span>{fmt(deg(inc))}</span>
                   <span style={{ color: '#bbb' }}>Ω (°)</span>
                <span>{fmt(deg(Om))}</span>
                   <span style={{ color: '#bbb' }}>ω (°)</span>
                <span>{fmt(deg(om))}</span>
                   <span style={{ color: '#bbb' }}>M (°)</span>
                <span>{fmt(Mdeg)}</span>
                   <span style={{ color: '#bbb' }}>ν (°)</span>
                <span>{fmt(vdeg)}</span>
              </div>
            </div>
          </Html>
        );
      })()}
      
      {/* Controles mejorados para distancias grandes */}
      <OrbitControls 
        enablePan={true} 
        enableZoom={true} 
        enableRotate={true}
        autoRotate={false}
        maxDistance={5000}
        minDistance={10}
        maxPolarAngle={Math.PI}
        minPolarAngle={0}
        enableDamping={!lowQuality}
        dampingFactor={lowQuality ? 0 : 0.05}
      />
      {/* Post FX: Bloom y tone mapping (opt-in con ?bloom para evitar incompatibilidades) */}
      {/* Post FX removed for stability */}
    </Canvas>
  );
}
