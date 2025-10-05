// Elementos orbitales de JPL NASA para posiciones planetarias aproximadas
// Basado en: https://ssd.jpl.nasa.gov/planets/approx_pos.html
// Válido para el período 1800 AD - 2050 AD

export interface JPLPlanetData {
  name: string;
  a0: number;        // Semi-eje mayor (UA)
  adot: number;      // Tasa de cambio del semi-eje mayor (UA/siglo)
  e0: number;        // Excentricidad
  edot: number;      // Tasa de cambio de excentricidad (rad/siglo)
  I0: number;        // Inclinación (grados)
  Idot: number;      // Tasa de cambio de inclinación (grados/siglo)
  L0: number;        // Longitud media (grados)
  Ldot: number;      // Tasa de cambio de longitud media (grados/siglo)
  varpi0: number;    // Longitud del perihelio (grados)
  varpidot: number;  // Tasa de cambio de longitud del perihelio (grados/siglo)
  Omega0: number;    // Longitud del nodo ascendente (grados)
  Omegadot: number;  // Tasa de cambio de longitud del nodo ascendente (grados/siglo)
  b?: number;        // Término adicional para anomalía media
  c?: number;        // Término coseno adicional
  s?: number;        // Término seno adicional
  f?: number;        // Frecuencia adicional
}

// Elementos orbitales de JPL NASA (Tabla 1: 1800 AD - 2050 AD)
export const JPL_PLANETS: JPLPlanetData[] = [
  {
    name: "Mercury",
    a0: 0.38709927, adot: 0.00000037,
    e0: 0.20563593, edot: 0.00001906,
    I0: 7.00497902, Idot: -0.00594749,
    L0: 252.25032350, Ldot: 149472.67411175,
    varpi0: 77.45779628, varpidot: 0.16047689,
    Omega0: 48.33076593, Omegadot: -0.12534081
  },
  {
    name: "Venus",
    a0: 0.72333566, adot: 0.00000390,
    e0: 0.00677672, edot: -0.00004107,
    I0: 3.39467605, Idot: -0.00078890,
    L0: 181.97909950, Ldot: 58517.81538729,
    varpi0: 131.60246718, varpidot: 0.00268329,
    Omega0: 76.67984255, Omegadot: -0.27769418
  },
  {
    name: "Earth",
    a0: 1.00000261, adot: 0.00000562,
    e0: 0.01671123, edot: -0.00004392,
    I0: -0.00001531, Idot: -0.01294668,
    L0: 100.46457166, Ldot: 35999.37244981,
    varpi0: 102.93768193, varpidot: 0.32327364,
    Omega0: 0.0, Omegadot: 0.0
  },
  {
    name: "Mars",
    a0: 1.52371034, adot: 0.00001847,
    e0: 0.09339410, edot: 0.00007882,
    I0: 1.84969142, Idot: -0.00813131,
    L0: -4.55343205, Ldot: 19140.30268499,
    varpi0: -23.94362959, varpidot: 0.44441088,
    Omega0: 49.55953891, Omegadot: -0.29257343
  },
  {
    name: "Jupiter",
    a0: 5.20288700, adot: -0.00011607,
    e0: 0.04838624, edot: -0.00013253,
    I0: 1.30439695, Idot: -0.00183714,
    L0: 34.39644051, Ldot: 3034.74612775,
    varpi0: 14.72847983, varpidot: 0.21252668,
    Omega0: 100.47390909, Omegadot: 0.20469106,
    b: -0.00012452, c: 0.06064060, s: -0.35635438, f: 38.35125000
  },
  {
    name: "Saturn",
    a0: 9.53667594, adot: -0.00125060,
    e0: 0.05386179, edot: -0.00050991,
    I0: 2.48599187, Idot: 0.00193609,
    L0: 49.95424423, Ldot: 1222.49362201,
    varpi0: 92.59887831, varpidot: -0.41897216,
    Omega0: 113.66242448, Omegadot: -0.28867794,
    b: 0.00025899, c: -0.13434469, s: 0.87320147, f: 38.35125000
  },
  {
    name: "Uranus",
    a0: 19.18916464, adot: -0.00196176,
    e0: 0.04725744, edot: -0.00004397,
    I0: 0.77263783, Idot: -0.00242939,
    L0: 313.23810451, Ldot: 428.48202785,
    varpi0: 170.95427630, varpidot: 0.40805281,
    Omega0: 74.01692503, Omegadot: 0.04240589,
    b: 0.00058331, c: -0.97731848, s: 0.17689245, f: 7.67025000
  },
  {
    name: "Neptune",
    a0: 30.06992276, adot: 0.00026291,
    e0: 0.00859048, edot: 0.00005105,
    I0: 1.77004347, Idot: 0.00035372,
    L0: -55.12002969, Ldot: 218.45945325,
    varpi0: 44.96476227, varpidot: -0.32241464,
    Omega0: 131.78422574, Omegadot: -0.00508664,
    b: -0.00041348, c: 0.68346318, s: -0.10162547, f: 7.67025000
  }
];

// Convertir fecha juliana a siglos desde J2000.0
export const julianCenturiesFromJ2000 = (julianDay: number): number => {
  return (julianDay - 2451545.0) / 36525.0;
};

// Calcular elementos orbitales en un momento dado
export const calculateOrbitalElements = (planet: JPLPlanetData, julianDay: number) => {
  const T = julianCenturiesFromJ2000(julianDay);
  
  return {
    a: planet.a0 + planet.adot * T,
    e: planet.e0 + planet.edot * T,
    I: planet.I0 + planet.Idot * T,
    L: planet.L0 + planet.Ldot * T,
    varpi: planet.varpi0 + planet.varpidot * T,
    Omega: planet.Omega0 + planet.Omegadot * T
  };
};

// Calcular anomalía media con términos adicionales para planetas exteriores
export const calculateMeanAnomaly = (planet: JPLPlanetData, elements: any, T: number): number => {
  let M = elements.L - elements.varpi;
  
  // Agregar términos adicionales para Júpiter, Saturno, Urano y Neptuno
  if (planet.b && planet.c && planet.s && planet.f) {
    M += planet.b * T * T + planet.c * Math.cos(planet.f * T) + planet.s * Math.sin(planet.f * T);
  }
  
  // Normalizar a -180° ≤ M ≤ +180°
  while (M > 180) M -= 360;
  while (M < -180) M += 360;
  
  return M;
};

// Solucionar la ecuación de Kepler usando Newton-Raphson
export const solveKeplersEquation = (M: number, e: number, tolerance = 1e-10): number => {
  // M en grados -> radianes; e es adimensional
  const M_rad = (M % 360) * Math.PI / 180;

  // Aproximación inicial (buena para todas las excentricidades e < 0.9)
  let E = e < 0.8 ? M_rad : Math.PI;

  for (let i = 0; i < 50; i++) {
    const f = E - e * Math.sin(E) - M_rad;
    const fPrime = 1 - e * Math.cos(E);
    const delta = f / fPrime;
    E -= delta;
    if (Math.abs(delta) < tolerance) break;
  }

  return E; // radianes
};

// Calcular posición heliocéntrica usando elementos orbitales
export const calculateHeliocentricPosition = (planet: JPLPlanetData, julianDay: number) => {
  const T = julianCenturiesFromJ2000(julianDay);
  const elements = calculateOrbitalElements(planet, julianDay);
  const M = calculateMeanAnomaly(planet, elements, T);
  const E = solveKeplersEquation(M, elements.e);
  
  // Calcular anomalía verdadera
  const cosv = (Math.cos(E) - elements.e) / (1 - elements.e * Math.cos(E));
  const sinv = Math.sqrt(1 - elements.e * elements.e) * Math.sin(E) / (1 - elements.e * Math.cos(E));
  const v = Math.atan2(sinv, cosv);
  
  // Calcular distancia al foco
  const r = elements.a * (1 - elements.e * elements.e) / (1 + elements.e * Math.cos(v));
  
  // Coordenadas en el plano orbital
  const x_prime = r * Math.cos(v);
  const y_prime = r * Math.sin(v);
  const z_prime = 0;
  
  // Aplicar rotaciones para orientar la órbita
  const omega = elements.varpi - elements.Omega; // Argumento del perihelio
  const I_rad = elements.I * Math.PI / 180;
  const Omega_rad = elements.Omega * Math.PI / 180;
  const omega_rad = omega * Math.PI / 180;
  
  // Rotación por argumento del perihelio
  const x1 = x_prime * Math.cos(omega_rad) - y_prime * Math.sin(omega_rad);
  const y1 = x_prime * Math.sin(omega_rad) + y_prime * Math.cos(omega_rad);
  const z1 = z_prime;
  
  // Rotación por inclinación
  const x2 = x1;
  const y2 = y1 * Math.cos(I_rad) - z1 * Math.sin(I_rad);
  const z2 = y1 * Math.sin(I_rad) + z1 * Math.cos(I_rad);
  
  // Rotación por longitud del nodo ascendente
  const x3 = x2 * Math.cos(Omega_rad) - y2 * Math.sin(Omega_rad);
  const y3 = x2 * Math.sin(Omega_rad) + y2 * Math.cos(Omega_rad);
  const z3 = z2;
  
  return {
    x: x3,
    y: y3,
    z: z3,
    r: r,
    v: v * 180 / Math.PI // Anomalía verdadera en grados
  };
};
