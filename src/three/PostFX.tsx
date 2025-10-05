import { useEffect, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

interface PostFXProps {
  exposure?: number;
  bloomIntensity?: number;
  bloomThreshold?: number;
  bloomRadius?: number;
}

export default function PostFX({ exposure = 1.2, bloomIntensity = 0.35, bloomThreshold = 0.85, bloomRadius = 0.6 }: PostFXProps) {
  const { gl, scene, camera, size } = useThree();

  const composer = useMemo(() => {
    try {
      const comp = new EffectComposer(gl);
      const renderPass = new RenderPass(scene, camera);
      const bloomPass = new UnrealBloomPass(new THREE.Vector2(size.width, size.height), bloomIntensity, bloomRadius, bloomThreshold);
      // Ensure final output goes to screen
      // @ts-ignore
      bloomPass.renderToScreen = true;
      comp.addPass(renderPass);
      comp.addPass(bloomPass);
      comp.setPixelRatio(Math.min(2, gl.getPixelRatio?.() || window.devicePixelRatio || 1));
      return comp;
    } catch (e) {
      console.warn('PostFX disabled:', e);
      return null as unknown as EffectComposer;
    }
  }, [gl, scene, camera, size, bloomIntensity, bloomRadius, bloomThreshold]);

  useEffect(() => {
    gl.toneMapping = THREE.ACESFilmicToneMapping;
    gl.toneMappingExposure = exposure;
    gl.outputColorSpace = THREE.SRGBColorSpace;
  }, [gl, exposure]);

  useEffect(() => {
    if (!composer) return;
    // Update size and pixel ratio when canvas size changes
    composer.setSize(size.width, size.height);
    // @ts-ignore
    composer.setPixelRatio?.(Math.min(2, gl.getPixelRatio?.() || window.devicePixelRatio || 1));
  }, [composer, size]);

  useFrame((_, delta) => {
    if (!composer) return;
    const prev = gl.autoClear;
    gl.autoClear = false;
    try {
      // @ts-ignore
      composer.render(delta);
    } finally {
      gl.autoClear = prev;
    }
  }, 1);

  return null;
}
