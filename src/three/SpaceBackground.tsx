import { useEffect, useRef } from 'react';
import { TextureLoader, SRGBColorSpace, BackSide, Mesh, MeshBasicMaterial } from 'three';
import { useFrame, useLoader, useThree } from '@react-three/fiber';
import { asset } from '../utils/asset';

export default function SpaceBackground({
  enabled = true,
  rotationSpeedRadPerSec = 0.00025,
  fadeInMs = 650,
  fadeOutMs = 500,
}: {
  enabled?: boolean;
  rotationSpeedRadPerSec?: number;
  fadeInMs?: number;
  fadeOutMs?: number;
} = {}) {
  // Follow the camera with a large BackSide sphere so the user is always inside it
  const { camera } = useThree();
  const radius = Math.max(1, (camera?.far ?? 10000) * 0.95);
  const tex = useLoader(TextureLoader, asset('textures/space.png'));
  useEffect(() => {
    if (tex) tex.colorSpace = SRGBColorSpace;
  }, [tex]);
  const meshRef = useRef<Mesh>(null);
  const matRef = useRef<MeshBasicMaterial>(null!);
  const brightnessRef = useRef<number>(enabled ? 1 : 0);
  const targetBrightnessRef = useRef<number>(enabled ? 1 : 0);
  const durationRef = useRef<number>(enabled ? fadeInMs : fadeOutMs);

  useEffect(() => {
    targetBrightnessRef.current = enabled ? 1 : 0;
    durationRef.current = enabled ? fadeInMs : fadeOutMs;
  }, [enabled, fadeInMs, fadeOutMs]);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.position.copy(camera.position);
      // Slow yaw rotation for subtle motion (delta-based)
      meshRef.current.rotation.y += rotationSpeedRadPerSec * delta;
    }
  });

  // Animate brightness toward target each frame (kept in opaque pass)
  useFrame((_, delta) => {
    const mat = matRef.current;
    if (!mat) return;
    const target = targetBrightnessRef.current;
    const duration = Math.max(1, durationRef.current) / 1000; // to seconds
    const diff = target - brightnessRef.current;
    if (Math.abs(diff) < 0.001) {
      brightnessRef.current = target;
      mat.color.setScalar(brightnessRef.current);
      return;
    }
    const step = Math.sign(diff) * Math.min(Math.abs(diff), delta / duration);
    brightnessRef.current = Math.max(0, Math.min(1, brightnessRef.current + step));
    mat.color.setScalar(brightnessRef.current);
  });

  return (
    <mesh ref={meshRef} renderOrder={-1000} frustumCulled={false}>
      <sphereGeometry args={[radius, 32, 32]} />
      <meshBasicMaterial
        ref={matRef}
        map={tex}
        side={BackSide}
        depthWrite={false}
        depthTest={false}
        toneMapped={false}
        // Keep this in the opaque pass; fade via color brightness
        color={'white'}
      />
    </mesh>
  );
}
