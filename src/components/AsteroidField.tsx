import React, { useEffect, useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { NEOData } from '../utils/orbitalMath';
import { AU_TO_UNITS } from '../utils/constants';

type Props = {
  neos: NEOData[];
  time: number; // simulation time in days relative to J2000 baseline
  onHover?: (neo: NEOData | undefined) => void;
  onClick?: (neo: NEOData) => void;
  updateHz?: number;
  // World-space size (when constantScreenSize=false)
  pointSize?: number;
  // Pixel size (if constantScreenSize=true)
  pointSizePx?: number;
  // Auto-scale interactive point size with camera distance
  autoScaleWithDistance?: boolean;
  // Min/Max sizes for auto-scaling (pixels when constantScreenSize=true)
  minPointSizePx?: number;
  maxPointSizePx?: number;
  // Keep constant size on screen
  constantScreenSize?: boolean;
  // Picking radius for Points (in world units)
  pickingRadiusWorld?: number;
  selectedNeoId?: string | number;
  highlightColor?: string;
  // Optional visual glow layer (non-interactive)
  glowEnabled?: boolean;
  glowSizePx?: number;
  glowOpacity?: number;
};

// Position computations are offloaded to a Web Worker now for performance

export default function AsteroidField({ neos, time, onHover, onClick, updateHz = 12, pointSize = 1.8, pointSizePx = 8, autoScaleWithDistance = true, minPointSizePx = 5, maxPointSizePx = 9, constantScreenSize = true, pickingRadiusWorld = 0.7, selectedNeoId, highlightColor = '#ffd700', glowEnabled = false, glowSizePx = 12, glowOpacity = 0.25 }: Props) {
  const count = neos.length;
  const geomRef = useRef<THREE.BufferGeometry>(null);
  const pointsRef = useRef<THREE.Points>(null);
  const glowPointsRef = useRef<THREE.Points>(null);
  const workerRef = useRef<Worker | null>(null);
  const workerReadyRef = useRef(false);
  const busyRef = useRef(false);
  const materialRef = useRef<THREE.PointsMaterial>(null);
  const glowMaterialRef = useRef<THREE.PointsMaterial>(null);
  const glowGeomRef = useRef<THREE.BufferGeometry>(null);
  const { gl, raycaster, size } = useThree();

  // Elements stored in typed arrays for speed
  const { aArr, eArr, iArr, OmArr, omArr, M0Arr, PArr, colorArr, posArr } = useMemo(() => {
    const a = new Float32Array(count);
    const e = new Float32Array(count);
    const inc = new Float32Array(count);
    const Om = new Float32Array(count);
    const om = new Float32Array(count);
    const M0 = new Float32Array(count);
    const P = new Float32Array(count);
    const colors = new Float32Array(count * 3);
    const positions = new Float32Array(count * 3);

  const cPHA = new THREE.Color('#ff2222');
  const cNEO = new THREE.Color('#00e676'); // green for NEOs to stand out

    for (let i = 0; i < count; i++) {
      const neo = neos[i];
      a[i] = parseFloat(neo.a);
      e[i] = parseFloat(neo.e);
      inc[i] = (parseFloat(neo.i) * Math.PI) / 180;
      Om[i] = (parseFloat(neo.Omega) * Math.PI) / 180;
      om[i] = (parseFloat(neo.omega) * Math.PI) / 180;
      M0[i] = (parseFloat(neo.M) * Math.PI) / 180; // radians at baseline
      // Robust period: prefer dataset but clamp to realistic; fallback compute via Kepler from 'a'
      let pRaw = parseFloat(neo.period);
      const aAU = a[i];
      const pFromA = Number.isFinite(aAU) && aAU > 0 ? 365.25 * Math.sqrt(aAU * aAU * aAU) : NaN;
      if (!Number.isFinite(pRaw) || pRaw < 50 || pRaw > 200000) {
        pRaw = pFromA;
      }
      // Final clamp to avoid erratic motion
      P[i] = Math.min(200000, Math.max(50, Number.isFinite(pRaw) ? pRaw : 365.25));

      const c = neo.type === 'PHA' ? cPHA : cNEO;
      const idx = i * 3;
      colors[idx] = c.r;
      colors[idx + 1] = c.g;
      colors[idx + 2] = c.b;

      // init positions (will be updated on first frame)
      positions[idx] = 0;
      positions[idx + 1] = 0;
      positions[idx + 2] = 0;
    }

    return { aArr: a, eArr: e, iArr: inc, OmArr: Om, omArr: om, M0Arr: M0, PArr: P, colorArr: colors, posArr: positions };
  }, [neos, count]);

  // Set geometry attributes once
  useEffect(() => {
    const geom = geomRef.current!;
    geom.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colorArr, 3));
    // Large bounding sphere to avoid culling as positions stream in
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), AU_TO_UNITS * 1000);
    // Mirror on glow geometry if present
    if (glowGeomRef.current) {
      const g = glowGeomRef.current;
      g.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
      g.setAttribute('color', new THREE.BufferAttribute(colorArr, 3));
      g.boundingSphere = geom.boundingSphere?.clone() ?? null;
    }
  }, [posArr, colorArr]);

  // Throttled position updates via Web Worker
  const lastUpdate = useRef(0);
  const prevHighlightedIndex = useRef<number | null>(null);

  // Map id -> index for quick highlight lookup
  const idToIndex = useMemo(() => {
    const map = new Map<string, number>();
    for (let i = 0; i < count; i++) {
      map.set(String(neos[i].id), i);
    }
    return map;
  }, [neos, count]);

  // Improve picking distance for Points
  useEffect(() => {
    const params = (raycaster.params as unknown as { Points?: { threshold?: number } });
    params.Points = params.Points ?? {};
    params.Points.threshold = pickingRadiusWorld;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raycaster.params as any).Points = params.Points; // ensure assignment back
  }, [raycaster, pickingRadiusWorld]);

  useEffect(() => {
    // When time jumps a lot, allow next frame to refresh immediately
    lastUpdate.current = 0;
  }, [time]);

  // Setup / reinit worker when elements change
  useEffect(() => {
    // Clean up previous worker
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
      workerReadyRef.current = false;
    }
    if (count === 0) return;
    try {
      const worker = new Worker(new URL('../workers/asteroidWorker.ts', import.meta.url), { type: 'module' });
      workerRef.current = worker;
      worker.onmessage = (e: MessageEvent) => {
        const data = e.data;
        if (data?.type === 'ready') {
          workerReadyRef.current = true;
        } else if (data?.type === 'positions') {
          const geom = geomRef.current;
          if (!geom) return;
          const arr: Float32Array = new Float32Array(data.buffer);
          // Copy into our existing posArr buffer
          posArr.set(arr);
          const posAttr = geom.getAttribute('position') as THREE.BufferAttribute;
          posAttr.needsUpdate = true;
          // Also update glow geometry if exists
          if (glowGeomRef.current) {
            const gpos = glowGeomRef.current.getAttribute('position') as THREE.BufferAttribute | undefined;
            if (gpos) gpos.needsUpdate = true;
          }
          busyRef.current = false;
        }
      };
      // Initialize with element arrays (copy once)
      worker.postMessage({
        type: 'init',
        a: aArr,
        e: eArr,
        inc: iArr,
        Om: OmArr,
        om: omArr,
        M0: M0Arr,
        P: PArr,
        count,
      });
    } catch (err) {
      console.error('Failed to start asteroid worker', err);
    }
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [count, aArr, eArr, iArr, OmArr, omArr, M0Arr, PArr, posArr]);

  useFrame(() => {
    const now = performance.now();
    const minIntervalMs = 1000 / Math.max(1, updateHz);
    if (now - lastUpdate.current < minIntervalMs) return; // throttled updates
    lastUpdate.current = now;
    if (!workerRef.current || !workerReadyRef.current || busyRef.current) return;
    busyRef.current = true;
    workerRef.current.postMessage({ type: 'compute', time, scale: AU_TO_UNITS });
  });

  // Distance-based tuning for points (interactive), glow and picking
  useFrame(({ camera }) => {
    const d = camera.position.length();
    // thresholds chosen for this scene scale
    const near = 80; // start of glow fade-in
    const far = 500; // full glow
    const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
    const t = clamp((d - near) / (far - near), 0, 1);
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

    // Interactive points: smaller near, a poco más grandes lejos
    if (materialRef.current) {
      if (constantScreenSize && autoScaleWithDistance) {
        const targetPx = lerp(minPointSizePx, maxPointSizePx, t);
        const nextSize = targetPx * dpr;
        if (materialRef.current.size !== nextSize) {
          materialRef.current.size = nextSize;
          materialRef.current.needsUpdate = true;
        }
      }
    }

    // size in px grows slightly with distance, but remains subtle
    if (glowMaterialRef.current) {
      const minGlowPx = Math.max((pointSizePx ?? 7) + 1, 8);
      const targetSize = constantScreenSize ? lerp(minGlowPx, (glowSizePx ?? 12) * 0.9, t) : (pointSize * 1.3);
      if (glowMaterialRef.current.size !== targetSize) {
        glowMaterialRef.current.size = targetSize;
        glowMaterialRef.current.needsUpdate = true;
      }

      // Opacidad más contenida para no congestionar
      const targetOpacity = lerp(0.03, (glowOpacity ?? 0.18) * 0.9, t);
      if (glowMaterialRef.current.opacity !== targetOpacity) {
        glowMaterialRef.current.opacity = targetOpacity;
        glowMaterialRef.current.needsUpdate = true;
      }

      // Cambia blending con la distancia para evitar saturación acumulada
      const desiredBlending = t < 0.35 ? THREE.AdditiveBlending : THREE.NormalBlending;
      if (glowMaterialRef.current.blending !== desiredBlending) {
        glowMaterialRef.current.blending = desiredBlending;
        glowMaterialRef.current.needsUpdate = true;
      }
    }

    // Compute world-units-per-pixel and set picking threshold accordingly
    const persp = (camera as THREE.PerspectiveCamera);
    const fovRad = persp.isPerspectiveCamera ? (persp.fov * Math.PI) / 180 : Math.PI / 4;
    const viewportH = size?.height || gl.domElement.clientHeight || 800;
    const worldUnitsPerPixel = (2 * d * Math.tan(fovRad / 2)) / viewportH;

    // Current interactive pixel size (matches what we render)
    const currentInteractivePx = (constantScreenSize && materialRef.current)
      ? (materialRef.current.size / (gl.getPixelRatio ? gl.getPixelRatio() : 1))
      : (pointSizePx ?? 8);
  const extraMarginPx = 8; // enlarge hit area in pixels
    const computedWorld = worldUnitsPerPixel * (currentInteractivePx + extraMarginPx);
    const baseWorld = pickingRadiusWorld ?? 0.7;
    const thresholdWorld = Math.max(baseWorld, computedWorld);

    const params = (raycaster.params as unknown as { Points?: { threshold?: number } });
    params.Points = params.Points ?? {};
    if (params.Points.threshold !== thresholdWorld) {
      params.Points.threshold = thresholdWorld;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (raycaster.params as any).Points = params.Points;
    }

    // draw order: glow first, then interactive
    if (glowPointsRef.current) glowPointsRef.current.renderOrder = 0;
    if (pointsRef.current) pointsRef.current.renderOrder = 1;
  });

  // Apply highlight color to selected asteroid and revert previous selection
  useEffect(() => {
    const geom = geomRef.current;
    if (!geom) return;
    const colorAttr = geom.getAttribute('color') as THREE.BufferAttribute | undefined;
    if (!colorAttr) return;

    // Revert previous highlight, if any
    if (prevHighlightedIndex.current !== null) {
      const idx = prevHighlightedIndex.current;
      if (idx >= 0 && idx < count) {
        const neo = neos[idx];
  const base = neo.type === 'PHA' ? new THREE.Color('#ff2222') : new THREE.Color('#00e676');
        const cIdx = idx * 3;
        colorArr[cIdx] = base.r;
        colorArr[cIdx + 1] = base.g;
        colorArr[cIdx + 2] = base.b;
      }
    }

    // Apply new highlight
    let nextIndex: number | null = null;
    if (selectedNeoId !== undefined && selectedNeoId !== null) {
      const idx = idToIndex.get(String(selectedNeoId));
      if (idx !== undefined) {
        nextIndex = idx;
        const gold = new THREE.Color(highlightColor);
        const cIdx = idx * 3;
        colorArr[cIdx] = gold.r;
        colorArr[cIdx + 1] = gold.g;
        colorArr[cIdx + 2] = gold.b;
      }
    }
    prevHighlightedIndex.current = nextIndex;
    colorAttr.needsUpdate = true;
  }, [selectedNeoId, highlightColor, idToIndex, colorArr, neos, count]);

  const handlePointerMove = (e: any) => {
    if (typeof e.index === 'number') {
      const idx = e.index as number;
      onHover?.(neos[idx]);
    }
  };

  const handlePointerOut = () => onHover?.(undefined);
  const handleClick = (e: any) => {
    if (typeof e.index === 'number') onClick?.(neos[e.index as number]);
  };

  const dpr = gl.getPixelRatio ? gl.getPixelRatio() : 1;

  return (
    <group>
      {/* Interactive layer */}
      <points
        ref={pointsRef}
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
        onClick={handleClick}
      >
        <bufferGeometry ref={geomRef} />
        <pointsMaterial
          ref={materialRef}
          size={constantScreenSize ? (pointSizePx ?? 8) * dpr : pointSize}
          sizeAttenuation={!constantScreenSize}
          vertexColors
          depthWrite={false}
          transparent
          opacity={0.95}
          onBeforeCompile={(shader) => {
            shader.fragmentShader = shader.fragmentShader.replace(
              '#include <color_fragment>',
              `#include <color_fragment>
               // circular point sprite
               vec2 c = gl_PointCoord - vec2(0.5);
               float d = length(c);
               // smooth circular edge: alpha fades near radius
               float alpha = smoothstep(0.5, 0.45, d);
               if (d > 0.5) discard;
               gl_FragColor.a *= alpha;`
            );
          }}
        />
      </points>

      {/* Display-only glow layer */}
      {glowEnabled && (
        <points ref={glowPointsRef} frustumCulled={false} raycast={() => null as any}>
          <bufferGeometry ref={glowGeomRef} />
          <pointsMaterial
            ref={glowMaterialRef}
            size={constantScreenSize ? (glowSizePx ?? 12) * dpr : (pointSize * 1.4)}
            sizeAttenuation={!constantScreenSize}
            vertexColors
            depthWrite={false}
            depthTest={true}
            transparent
            opacity={glowOpacity ?? 0.15}
            blending={THREE.AdditiveBlending}
            onBeforeCompile={(shader) => {
              shader.fragmentShader = shader.fragmentShader.replace(
                '#include <color_fragment>',
                `#include <color_fragment>
                 vec2 c = gl_PointCoord - vec2(0.5);
                 float d = length(c);
                 float alpha = smoothstep(0.5, 0.40, d);
                 if (d > 0.5) discard;
                 gl_FragColor.a *= alpha;`
              );
            }}
          />
        </points>
      )}
    </group>
  );
}
