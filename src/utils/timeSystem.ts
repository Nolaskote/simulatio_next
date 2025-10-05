// Sistema de tiempo para la simulación del sistema solar
// Basado en días julianos para precisión astronómica

export interface TimeState {
  julianDay: number;
  isPlaying: boolean;
  timeScale: number;
  currentDate: Date;
}

// Convertir fecha actual a día juliano
export const dateToJulianDay = (date: Date): number => {
  const year = date.getFullYear();
  const month = date.getMonth() + 1;
  const day = date.getDate();
  const hour = date.getHours();
  const minute = date.getMinutes();
  const second = date.getSeconds();
  
  // Algoritmo para convertir fecha gregoriana a día juliano
  let a = Math.floor((14 - month) / 12);
  let y = year + 4800 - a;
  let m = month + 12 * a - 3;
  
  let jdn = day + Math.floor((153 * m + 2) / 5) + 365 * y + Math.floor(y / 4) - Math.floor(y / 100) + Math.floor(y / 400) - 32045;
  
  // Agregar fracción del día
  let jd = jdn + (hour - 12) / 24 + minute / 1440 + second / 86400;
  
  return jd;
};

// Convertir día juliano a fecha
export const julianDayToDate = (jd: number): Date => {
  let jdn = Math.floor(jd + 0.5);
  let f = jd + 0.5 - jdn;
  
  let a = jdn + 32044;
  let b = Math.floor((4 * a + 3) / 146097);
  let c = a - Math.floor((b * 146097) / 4);
  
  let d = Math.floor((4 * c + 3) / 1461);
  let e = c - Math.floor((d * 1461) / 4);
  let m = Math.floor((5 * e + 2) / 153);
  
  let day = e - Math.floor((153 * m + 2) / 5) + 1;
  let month = m + 3 - 12 * Math.floor(m / 10);
  let year = b * 100 + d - 4800 + Math.floor(m / 10);
  
  let hour = Math.floor(f * 24);
  let minute = Math.floor((f * 24 - hour) * 60);
  let second = Math.floor(((f * 24 - hour) * 60 - minute) * 60);
  
  return new Date(year, month - 1, day, hour, minute, second);
};

// Calcular tiempo en días desde una fecha de referencia
export const getTimeInDays = (julianDay: number, referenceJD: number): number => {
  return julianDay - referenceJD;
};

// Fecha de referencia: 1 de enero de 2000, 12:00 UTC (J2000.0)
export const J2000_EPOCH = 2451545.0;

// Crear estado inicial del tiempo
export const createInitialTimeState = (): TimeState => {
  const now = new Date();
  const julianDay = dateToJulianDay(now);
  
  return {
    julianDay,
    isPlaying: true,
    timeScale: 1,
    currentDate: now
  };
};

// Formatear fecha para mostrar
export const formatDate = (date: Date): string => {
  return date.toLocaleDateString('es-ES', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

// Formatear día juliano
export const formatJulianDay = (jd: number): string => {
  return `JD ${jd.toFixed(2)}`;
};

// Calcular velocidad orbital realista para planetas
export const getPlanetTimeScale = (orbitalPeriod: number): number => {
  // Escalar el tiempo para que las órbitas sean visibles
  // 1 día real = varios días simulados
  return 365.25 / orbitalPeriod; // Tierra como referencia
};
