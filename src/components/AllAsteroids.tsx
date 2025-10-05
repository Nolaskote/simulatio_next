import React, { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { NEOData } from '../utils/orbitalMath';
import { AU_TO_UNITS } from '../utils/constants';

interface AllAsteroidsProps {
  neos: NEOData[];
  time: number; // días simulados desde J2000
  onHover?: (neo: NEOData | undefined) => void;
  onClick?: (neo: NEOData) => void;
}

// Renderer de alto rendimiento para todos los NEO/PHA usando THREE.Points
export default function AllAsteroids({ neos, time, onHover, onClick }: AllAsteroidsProps) {
  const N = neos.length;
  const positions = useMemo(() => new Float32Array(N * 3), [N]);
  const colors = useMemo(() => new Float32Array(N * 3), [N]);
  const a = useMemo(() => new Float32Array(N), [N]);
  const e = useMemo(() => new Float32Array(N), [N]);
  const inc = useMemo(() => new Float32Array(N), [N]);
  const Omega = useMemo(() => new Float32Array(N), [N]);
  const omega = useMemo(() => new Float32Array(N), [N]);
  const M = useMemo(() => new Float32Array(N), [N]);
  const period = useMemo(() => new Float32Array(N), [N]);
  const pickIndexRef = useRef<number | null>(null);
  const lastSimTimeRef = useRef<number>(time);
  const cursorRef = useRef<number>(0);

  // Inicializar elementos y colores solo una vez
  useEffect(() => {
    for (let i = 0; i < N; i++) {
      const n = neos[i];
      a[i] = parseFloat(n.a);
      e[i] = parseFloat(n.e);
      inc[i] = (parseFloat(n.i) * Math.PI) / 180.0;
      Omega[i] = (parseFloat(n.Omega) * Math.PI) / 180.0;
      omega[i] = (parseFloat(n.omega) * Math.PI) / 180.0;
      M[i] = (parseFloat(n.M) * Math.PI) / 180.0; // en radianes
      period[i] = Math.max(1e-3, parseFloat(n.period));

      const base = i * 3;
      const c = n.type === 'PHA' ? 0xff3333 : 0xdddddd;
      const col = new THREE.Color(c);
      colors[base + 0] = col.r;
      colors[base + 1] = col.g;
      colors[base + 2] = col.b;
    }
    // Hacer una pasada inicial de posiciones para que no aparezcan en (0,0,0)
    lastSimTimeRef.current = time;
    initialComputeAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [N, neos]);

  const bufferRef = useRef<THREE.BufferGeometry>(null);
  const materialRef = useRef<THREE.PointsMaterial>(null);

  const TARGET_UPS = 10; // actualizaciones por segundo
  const CHUNK = Math.min(8000, Math.max(2000, Math.floor(N / 8))); // tamaño de bloque adaptativo
  const accumRef = useRef(0);

  // Solucionador rápido de Kepler en radianes, con pocas iteraciones
  const solveE = (Mrad: number, ecc: number): number => {
    let E = ecc < 0.8 ? Mrad : Math.PI;
    for (let k = 0; k < 12; k++) {
      const f = E - ecc * Math.sin(E) - Mrad;
      const fp = 1 - ecc * Math.cos(E);
      const d = f / fp;
      E -= d;
      if (Math.abs(d) < 1e-6) break;
    }
    return E;
  };

  const updateChunk = (start: number, count: number, dDays: number) => {
    const end = Math.min(N, start + count);
    for (let i = start; i < end; i++) {
      // Avance de anomalía media (radianes) según dDays
      const n = (2 * Math.PI) / period[i];
      const Mnew = M[i] + n * dDays;
      // Normalizar a [-pi, pi]
      let Mrad = Mnew;
      Mrad = ((Mrad + Math.PI) % (2 * Math.PI)) - Math.PI;
      M[i] = Mrad;

      const Ei = solveE(Mrad, e[i]);
      const cosE = Math.cos(Ei), sinE = Math.sin(Ei);
      const ecc = e[i];
      const cosv = (cosE - ecc) / (1 - ecc * cosE);
      const sinv = Math.sqrt(1 - ecc * ecc) * sinE / (1 - ecc * cosE);
      const v = Math.atan2(sinv, cosv);
      const r = a[i] * (1 - ecc * ecc) / (1 + ecc * Math.cos(v));

      const x_orb = r * Math.cos(v);
      const y_orb = r * Math.sin(v);

      const cw = Math.cos(omega[i]);
      const sw = Math.sin(omega[i]);
      const cO = Math.cos(Omega[i]);
      const sO = Math.sin(Omega[i]);
      const ci = Math.cos(inc[i]);
      const si = Math.sin(inc[i]);

      const x1 = x_orb * cw - y_orb * sw;
      const y1 = x_orb * sw + y_orb * cw;
      const z1 = 0;

      const x2 = x1;
      const y2 = y1 * ci - z1 * si;
      const z2 = y1 * si + z1 * ci;

      const x3 = x2 * cO - y2 * sO;
      const y3 = x2 * sO + y2 * cO;
      const z3 = z2;

      const base = i * 3;
      positions[base + 0] = x3 * AU_TO_UNITS;
      positions[base + 1] = z3 * AU_TO_UNITS; // ajustar ejes igual que planetas
      positions[base + 2] = -y3 * AU_TO_UNITS;
    }
  };

  const initialComputeAll = () => {
    // Inicial: computar todos en bloques para evitar congelón
    let idx = 0;
    const step = 10000;
    while (idx < N) {
      updateChunk(idx, step, 0); // dDays=0 usa M inicial
      idx += step;
    }
    if (bufferRef.current) {
      bufferRef.current.attributes.position.needsUpdate = true;
      bufferRef.current.computeBoundingSphere();
    }
  };

  useFrame((_, delta) => {
    accumRef.current += delta;
    const interval = 1 / TARGET_UPS;
    if (accumRef.current < interval) return;
    const d = accumRef.current;
    accumRef.current = 0;

    const prev = lastSimTimeRef.current;
    const dDays = time - prev;
    lastSimTimeRef.current = time;

    // Actualizar un bloque por tick
    const start = cursorRef.current;
    updateChunk(start, CHUNK, dDays);
    cursorRef.current = (start + CHUNK) % Math.max(1, N);

    if (bufferRef.current) {
      bufferRef.current.attributes.position.needsUpdate = true;
      // No computar bounding sphere cada tick para ahorrar; cada 10 ticks es suficiente
      if (start < CHUNK) bufferRef.current.computeBoundingSphere();
    }
  });

  // Picking simple por proximidad del cursor
  const handlePointerMove = (e: any) => {
    const idx = e.index as number | undefined;
    if (typeof idx === 'number') {
      pickIndexRef.current = idx;
      onHover?.(neos[idx]);
    } else {
      pickIndexRef.current = null;
      onHover?.(undefined);
    }
  };

  const handleClick = () => {
    if (pickIndexRef.current != null) {
      onClick?.(neos[pickIndexRef.current]);
    }
  };

  return (
    <points onPointerMove={handlePointerMove} onPointerOut={() => onHover?.(undefined)} onClick={handleClick}>
      <bufferGeometry ref={bufferRef}>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial ref={materialRef} vertexColors sizeAttenuation size={1.6} />
    </points>
  );
}
