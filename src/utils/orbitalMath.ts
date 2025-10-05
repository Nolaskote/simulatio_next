// Utilidades para cálculos orbitales usando parámetros keplerianos
import * as THREE from 'three';

export interface KeplerianElements {
  a: number;      // Semi-eje mayor (UA)
  e: number;      // Excentricidad
  i: number;      // Inclinación (grados)
  Omega: number;  // Longitud del nodo ascendente (grados)
  omega: number;  // Argumento del perihelio (grados)
  M: number;      // Anomalía media (grados)
  period?: number; // Período orbital (días)
}

export interface NEOData {
  id: string;
  name: string;
  a: string;
  e: string;
  i: string;
  Omega: string;
  omega: string;
  M: string;
  period: string;
  type: 'NEO' | 'PHA';
}

// Convertir grados a radianes
export const degToRad = (degrees: number): number => degrees * (Math.PI / 180);

// Convertir radianes a grados
export const radToDeg = (radians: number): number => radians * (180 / Math.PI);

// Calcular anomalía excéntrica usando el método de Newton-Raphson
export const calculateEccentricAnomaly = (M: number, e: number, tolerance = 1e-6): number => {
  const M_rad = degToRad(M);
  let E = M_rad; // Aproximación inicial
  
  for (let i = 0; i < 100; i++) {
    const f = E - e * Math.sin(E) - M_rad;
    const fPrime = 1 - e * Math.cos(E);
    
    if (Math.abs(f) < tolerance) break;
    
    E = E - f / fPrime;
  }
  
  return E;
};

// Calcular posición cartesiana desde elementos keplerianos
export const keplerianToCartesian = (elements: KeplerianElements, time = 0): THREE.Vector3 => {
  const { a, e, i, Omega, omega, M } = elements;
  
  // Calcular anomalía media actual (considerando el tiempo)
  const currentM = M + (time * 360 / (elements.period || 365.25));
  
  // Calcular anomalía excéntrica
  const E = calculateEccentricAnomaly(currentM, e);
  
  // Calcular anomalía verdadera
  const cosv = (Math.cos(E) - e) / (1 - e * Math.cos(E));
  const sinv = Math.sqrt(1 - e * e) * Math.sin(E) / (1 - e * Math.cos(E));
  const v = Math.atan2(sinv, cosv);
  
  // Calcular distancia al foco
  const r = a * (1 - e * e) / (1 + e * Math.cos(v));
  
  // Convertir a coordenadas cartesianas en el plano orbital
  const x = r * Math.cos(v);
  const y = r * Math.sin(v);
  const z = 0;
  
  // Aplicar rotaciones para orientar la órbita
  const i_rad = degToRad(i);
  const Omega_rad = degToRad(Omega);
  const omega_rad = degToRad(omega);
  
  // Rotación alrededor del eje Z por omega (argumento del perihelio)
  const x1 = x * Math.cos(omega_rad) - y * Math.sin(omega_rad);
  const y1 = x * Math.sin(omega_rad) + y * Math.cos(omega_rad);
  const z1 = z;
  
  // Rotación alrededor del eje X por inclinación
  const x2 = x1;
  const y2 = y1 * Math.cos(i_rad) - z1 * Math.sin(i_rad);
  const z2 = y1 * Math.sin(i_rad) + z1 * Math.cos(i_rad);
  
  // Rotación alrededor del eje Z por Omega (longitud del nodo ascendente)
  const x3 = x2 * Math.cos(Omega_rad) - y2 * Math.sin(Omega_rad);
  const y3 = x2 * Math.sin(Omega_rad) + y2 * Math.cos(Omega_rad);
  const z3 = z2;
  
  return new THREE.Vector3(x3, z3, -y3); // Ajustar orientación para Three.js
};

// Calcular velocidad orbital (aproximada)
export const calculateOrbitalVelocity = (elements: KeplerianElements, position: THREE.Vector3): THREE.Vector3 => {
  const { a, e, M } = elements;
  const currentM = M;
  const E = calculateEccentricAnomaly(currentM, e);
  
  // Velocidad en el plano orbital
  const n = 2 * Math.PI / (elements.period || 365.25); // Movimiento medio
  const velocity = n * a / Math.sqrt(1 - e * e);
  
  // Dirección tangencial (aproximada)
  const tangent = new THREE.Vector3(-position.z, 0, position.x).normalize();
  
  return tangent.multiplyScalar(velocity * 0.1); // Escalar para visualización
};

// Determinar el tamaño del asteroide basado en su tipo y datos disponibles
export const estimateAsteroidSize = (neo: NEOData): number => {
  // Para la simulación, usaremos tamaños aproximados
  // En una implementación real, necesitarías datos de diámetro de la NASA
  if (neo.type === 'PHA') {
    return 0.01 + Math.random() * 0.02; // PHAs suelen ser más grandes
  }
  return 0.005 + Math.random() * 0.015; // NEOs más pequeños
};

// Determinar el color del asteroide basado en su tipo
export const getAsteroidColor = (neo: NEOData): string => {
  if (neo.type === 'PHA') {
    return '#ff4444'; // Rojo para PHAs (potencialmente peligrosos)
  }
  return '#888888'; // Gris para NEOs regulares
};
