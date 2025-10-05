// Web Worker to compute asteroid positions efficiently
// Receives orbital elements arrays and returns packed Float32Array positions

let a: Float32Array;
let eArr: Float32Array;
let inc: Float32Array;
let Om: Float32Array;
let om: Float32Array;
let M0: Float32Array;
let P: Float32Array;
let count = 0;

// Fast Newton-Raphson for eccentric anomaly (radians)
function solveE(Mrad: number, e: number): number {
  let E = e < 0.8 ? Mrad : Math.PI; // initial guess
  for (let i = 0; i < 5; i++) { // fixed iterations
    const f = E - e * Math.sin(E) - Mrad;
    const fp = 1 - e * Math.cos(E);
    const d = f / fp;
    E -= d;
    if (Math.abs(d) < 1e-7) break;
  }
  return E;
}

self.onmessage = (e: MessageEvent) => {
  const data: any = e.data;
  if (data?.type === 'init') {
    a = data.a;
    eArr = data.e;
    inc = data.inc;
    Om = data.Om;
    om = data.om;
    M0 = data.M0;
    P = data.P;
    count = data.count;
    (self as any).postMessage({ type: 'ready' });
  } else if (data?.type === 'compute') {
    const time: number = data.time;
    const scale: number = data.scale || 1;
    const pos = new Float32Array(count * 3);
    const TWO_PI = Math.PI * 2;
    for (let i = 0; i < count; i++) {
      const n = TWO_PI / P[i]; // mean motion [rad/day]
      const M = M0[i] + n * time; // radians
      let Mrad = ((M + Math.PI) % TWO_PI) - Math.PI;
      if (Mrad < -Math.PI) Mrad += TWO_PI;
      if (Mrad > Math.PI) Mrad -= TWO_PI;

      const ecc = eArr[i];
      const E = solveE(Mrad, ecc);
      const cosE = Math.cos(E);
      const sinE = Math.sin(E);
      const cosv = (cosE - ecc) / (1 - ecc * cosE);
      const sinv = Math.sqrt(1 - ecc * ecc) * sinE / (1 - ecc * cosE);
      const v = Math.atan2(sinv, cosv);
      const r = a[i] * (1 - ecc * ecc) / (1 + ecc * Math.cos(v));

      const xPrime = r * Math.cos(v);
      const yPrime = r * Math.sin(v);

      const omv = om[i];
      const Omv = Om[i];
      const incv = inc[i];

      const cosom = Math.cos(omv), sinom = Math.sin(omv);
      const x1 = xPrime * cosom - yPrime * sinom;
      const y1 = xPrime * sinom + yPrime * cosom;

      const cosi = Math.cos(incv), sini = Math.sin(incv);
      const y2 = y1 * cosi;
      const z2 = y1 * sini;

      const cosOm = Math.cos(Omv), sinOm = Math.sin(Omv);
      const x3 = x1 * cosOm - y2 * sinOm;
      const y3 = x1 * sinOm + y2 * cosOm;
      const z3 = z2;

      const idx = i * 3;
      pos[idx] = x3 * scale;
      pos[idx + 1] = z3 * scale; // swap to match scene axes
      pos[idx + 2] = -y3 * scale;
    }
    // Transfer positions back using transferable object to avoid copy
    (self as any).postMessage({ type: 'positions', buffer: pos.buffer }, [pos.buffer]);
  }
};
