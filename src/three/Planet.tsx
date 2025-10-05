import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import { TextureLoader, Mesh, BackSide, Group, Vector3, DirectionalLight, AdditiveBlending } from "three";
import { Html, Ring } from "@react-three/drei";
import { calculateHeliocentricPosition, JPL_PLANETS } from "../utils/jplOrbitalElements";
import { AU_TO_UNITS, J2000 } from "../utils/constants";
import { KeplerianElements } from "../utils/orbitalMath";

interface PlanetProps {
  name: string;
  texture: string;
  size: number;
  distance: number; // solo usado como fallback inicial
  rotation: number; // rotación sobre eje
  retrograde: boolean;
  isSun?: boolean;
  hasRings?: boolean;
  ringTexture?: string;
  simulationTime?: number;
  haloEnabled?: boolean;
  keplerianElements?: KeplerianElements;
  infoOpen?: boolean;
  onToggleInfo?: () => void;
  uiLightIntensity?: number;
  segments?: number; // sphere geometry segments for quality control
}

export default function Planet({
  name,
  texture,
  size,
  distance,
  rotation,
  retrograde,
  isSun = false,
  hasRings = false,
  ringTexture,
  simulationTime = 0,
  haloEnabled = true,
  keplerianElements,
  infoOpen = false,
  onToggleInfo,
  uiLightIntensity = 1.2,
  segments = 64,
}: PlanetProps) {
  const haloColor = (() => {
    switch (name) {
      case 'Mercury': return '#c2b280';
      case 'Venus': return '#f0d9a7';
      case 'Earth': return '#6ab7ff';
      case 'Mars': return '#ff6f61';
      case 'Jupiter': return '#ffd27a';
      case 'Saturn': return '#ffe0a3';
      case 'Uranus': return '#98e1ff';
      case 'Neptune': return '#8bb6ff';
      case 'Sun': return '#ffdd88';
      default: return '#9ecbff';
    }
  })();
  const groupRef = useRef<Mesh>(null);
  const planetRef = useRef<Mesh>(null);
  const ringRef = useRef<Mesh>(null);
  const ringGroupRef = useRef<Group>(null);
  const map = useLoader(TextureLoader, texture);
  // Tiempo simulado anterior para calcular dDays de forma estable
  const prevSimTimeRef = useRef<number>(simulationTime);
  const [showCard, setShowCard] = useState(false);
  // keep internal state synced with controlled prop
  useEffect(() => { setShowCard(infoOpen); }, [infoOpen]);
  
  // Obtener datos JPL para este planeta
  const jplPlanet = JPL_PLANETS.find(p => p.name === name);
  const lightRef = useRef<DirectionalLight>(null);

  // Rotación sobre su eje y órbita JPL realista
  useFrame(() => {
    if (planetRef.current && groupRef.current) {
      // Rotación sobre su eje basada en días simulados transcurridos (estable ante cambios de velocidad)
      const prev = prevSimTimeRef.current;
      let dDays = simulationTime - prev;
      prevSimTimeRef.current = simulationTime;
      if (!isFinite(dDays)) dDays = 0;
      if (rotation !== 0) {
        const rotPerDay = (2 * Math.PI) / Math.abs(rotation); // radianes por día
        const dir = retrograde ? -1 : 1;
        planetRef.current.rotation.y += dir * dDays * rotPerDay;
        // Hacer que los anillos (si existen) giren con la misma lógica/ritmo que el planeta
        if (ringRef.current && hasRings) {
          // Mantener la inclinación fija (ver JSX) y girar como un disco (alrededor de su eje normal)
          // Usamos el eje Z local del mesh (la inclinación está en el grupo padre).
          ringRef.current.rotation.z += dir * dDays * rotPerDay;
        }
      }
      
      if (!isSun && jplPlanet) {
        // Calcular posición usando elementos JPL reales
        const julianDay = J2000 + simulationTime; // J2000.0 + tiempo de simulación (días)
        const position = calculateHeliocentricPosition(jplPlanet, julianDay);
        
        // Posicionar el grupo (centro del planeta) en unidades de escena
        const scale = AU_TO_UNITS;
        groupRef.current.position.set(position.x * scale, position.z * scale, -position.y * scale);
        
        // Aplicar la misma posición a los anillos
        if (ringGroupRef.current && hasRings) {
          ringGroupRef.current.position.set(0, 0, 0); // centrado en el planeta en su grupo local
        }

        // Actualizar luz direccional: dirección desde el Sol (0,0,0) hacia el planeta
        if (lightRef.current) {
          const planetPos = groupRef.current.position.clone();
          const sunPos = new Vector3(0, 0, 0);
          const dirToSun = sunPos.clone().sub(planetPos).normalize(); // de planeta a Sol
          // Colocar la luz en el lado del Sol respecto al planeta
          const lightPos = planetPos.clone().add(dirToSun.multiplyScalar(10 * size));
          lightRef.current.position.copy(lightPos);
          // La luz debe apuntar al centro del planeta
          lightRef.current.target.position.set(planetPos.x, planetPos.y, planetPos.z);
          lightRef.current.target.updateMatrixWorld();
          lightRef.current.intensity = uiLightIntensity;
        }
      }
    }
  });

  // Asegurar que el target de la luz exista en la escena
  useEffect(() => {
    if (!isSun && lightRef.current && groupRef.current) {
      // Añadir el target al mismo parent (escena) para que tenga coordenadas globales válidas
      const parent = groupRef.current.parent;
      if (parent && !lightRef.current.target.parent) {
        parent.add(lightRef.current.target);
      }
    }
  }, [isSun]);

  return (
    <group ref={groupRef} position={[distance, 0, 0]}>
      <mesh ref={planetRef} onClick={() => (onToggleInfo ? onToggleInfo() : setShowCard(v => !v))}>
        <sphereGeometry args={[size, segments, segments]} />
        {isSun ? (
          <meshStandardMaterial map={map} emissive="#ffaa00" emissiveIntensity={1.6} />
        ) : (
          <meshStandardMaterial map={map} envMapIntensity={0.0} />
        )}
      </mesh>
      {/* Luz direccional local que simula la iluminación proveniente del Sol */}
      {!isSun && (
        <directionalLight
          ref={lightRef}
          intensity={1.2}
          color="#ffffff"
          castShadow={false}
          shadow-camera-near={0.1}
          shadow-camera-far={10000}
        />
      )}
      {/* Cartilla del planeta (toggle con click) */}
  {showCard && !isSun && keplerianElements && (
        (() => {
          // Posición local del planeta en el grupo (ya usamos groupRef para moverlo en AU)
          const pos: [number, number, number] = [0, 0, 0];
          const titleColor = haloColor;
          const fmt = (x: number) => (Number.isFinite(x) ? Number(x).toPrecision(3) : '—');
          const clamp360 = (d: number) => ((d % 360) + 360) % 360;
          return (
            <Html position={pos} style={{ pointerEvents: 'none' }}>
              <div
                style={{
                  background: 'rgba(0,0,0,0.85)',
                  color: '#fff',
                  padding: '10px 12px',
                  borderRadius: 10,
                  fontSize: 12,
                  lineHeight: 1.4,
                  width: 'max-content',
                  border: `1px solid ${titleColor}99`,
                  boxShadow: '0 2px 10px rgba(0,0,0,0.45)',
                  backdropFilter: 'blur(4px)',
                  WebkitBackdropFilter: 'blur(4px)',
                  fontVariantNumeric: 'tabular-nums',
                  transform: 'translate(14px, 10px)',
                }}
              >
                <div style={{ fontWeight: 700, color: titleColor, marginBottom: 6 }}>
                  {name}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', columnGap: 10, rowGap: 4 }}>
                  <span style={{ color: '#bbb' }}>P (d)</span>
                  <span>{fmt(keplerianElements.period ?? 0)}</span>
                  <span style={{ color: '#bbb' }}>a (UA)</span>
                  <span>{fmt(keplerianElements.a)}</span>
                  <span style={{ color: '#bbb' }}>e</span>
                  <span>{fmt(keplerianElements.e)}</span>
                  <span style={{ color: '#bbb' }}>i (°)</span>
                  <span>{fmt(keplerianElements.i)}</span>
                  <span style={{ color: '#bbb' }}>Ω (°)</span>
                  <span>{fmt(keplerianElements.Omega)}</span>
                  <span style={{ color: '#bbb' }}>ω (°)</span>
                  <span>{fmt(keplerianElements.omega)}</span>
                  <span style={{ color: '#bbb' }}>M (°)</span>
                  <span>{fmt(clamp360(keplerianElements.M))}</span>
                </div>
                {/* no hint text; controlled by click toggling */}
              </div>
            </Html>
          );
        })()
      )}
      {haloEnabled && !isSun && (
        <>
          <mesh renderOrder={-1}>
            <sphereGeometry args={[size * 4.0, 48, 48]} />
            <meshBasicMaterial
              color={haloColor}
              side={BackSide}
              depthWrite={false}
              transparent
              opacity={0.2}
              blending={AdditiveBlending}
            />
          </mesh>
          {/* Etiqueta con el nombre del planeta (siempre mirando a cámara) */}
          <Html position={[0, size * 1.6, 0]} center style={{ pointerEvents: 'none' }}>
            <div
              style={{
                background: 'rgba(0,0,0,0.6)',
                color: '#fff',
                padding: '2px 6px',
                borderRadius: 6,
                fontSize: 12,
                lineHeight: 1.2,
                border: `1px solid ${haloColor}55`,
                whiteSpace: 'nowrap',
              }}
            >
              {name}
            </div>
          </Html>
        </>
      )}
      
      {/* Anillos para Saturno - inclinación fija en el grupo y giro como un disco */}
      {hasRings && ringTexture && (
        <group ref={ringGroupRef} rotation={[Math.PI / 2, 0, 0]}>
          <mesh ref={ringRef}>
            <ringGeometry args={[size * 1.2, size * 2.5, 32]} />
            <meshStandardMaterial
              map={useLoader(TextureLoader, ringTexture)}
              transparent
              opacity={0.6}
              side={2} // DoubleSide
            />
          </mesh>
        </group>
      )}
    </group>
  );
}