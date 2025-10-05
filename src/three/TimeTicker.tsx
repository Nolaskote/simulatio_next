import { useFrame } from '@react-three/fiber';
import { useRef } from 'react';

interface TimeTickerProps {
  isPlaying: boolean;
  rateDaysPerSecond: number; // how many simulation days per real second
  onAdvance: (daysDelta: number) => void;
}

export default function TimeTicker({ isPlaying, rateDaysPerSecond, onAdvance }: TimeTickerProps) {
  const accum = useRef(0);

  useFrame((_, delta) => {
    if (!isPlaying || rateDaysPerSecond === 0) return;
    const daysDelta = delta * rateDaysPerSecond;
    accum.current += daysDelta;
    // Advance continuously; no fixed step needed for smoothness
    onAdvance(daysDelta);
  });

  return null;
}
