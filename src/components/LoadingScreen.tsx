import React, { useEffect, useRef, useState } from 'react';
import './loading.css';
import { asset } from '../utils/asset';

interface LoadingScreenProps {
  minDurationMs?: number; // default 5000
  ready: boolean; // when true, we can complete after min duration
  onComplete: () => void;
  assetsProgressPct?: number; // 0..100 from drei useProgress
}

export default function LoadingScreen({ minDurationMs = 5000, ready, onComplete, assetsProgressPct = 0 }: LoadingScreenProps) {
  const [progress, setProgress] = useState(0); // 0..1 time-based
  const startRef = useRef<number | null>(null);
  const completedRef = useRef(false);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    let raf = 0;
    const step = (ts: number) => {
      if (startRef.current === null) startRef.current = ts;
      const elapsed = ts - startRef.current;
      const base = Math.min(1, elapsed / minDurationMs); // time progress 0..1
      // Avanza hasta 0.99 si aún no está listo; cuando ready sea true y base === 1, sube a 1.0
      const target = ready ? base : Math.min(base, 0.99);
      setProgress(prev => (target > prev ? target : prev));

      if (!completedRef.current) {
        if (ready && base >= 1) {
          completedRef.current = true;
          setProgress(1);
          // Fade-out breve antes de completar
          setFading(true);
          setTimeout(onComplete, 350);
          return;
        }
        raf = requestAnimationFrame(step);
      }
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [ready, minDurationMs, onComplete]);

  const timePct = progress * 100;
  const combinedPctRaw = Math.max(timePct, assetsProgressPct || 0);
  const targetPct = ready ? combinedPctRaw : Math.min(99, combinedPctRaw);
  // Lock to a monotonic, non-decreasing percentage to avoid "titileo"
  const lastPctRef = useRef(0);
  const pct = Math.round(Math.max(lastPctRef.current, targetPct));
  lastPctRef.current = pct;

  return (
    <div className={`loader-root ${fading ? 'is-fading' : ''}`}>
  <div className="loader-bg" style={{ backgroundImage: `url(${asset('textures/meteoro_tierra.jpg')})` }} />
      <div className="loader-content">
        <div className="loader-ring" style={{ ['--p' as any]: `${pct}` }}>
          <div className="loader-logo">
            <img src={asset('textures/android-chrome-192x192.png')} alt="NASA Hackathon" />
          </div>
        </div>
        <div className="loader-percent">{pct}%</div>
        <div className="loader-text">Cargando todos los datos...</div>
      </div>
    </div>
  );
}
