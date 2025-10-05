import React, { useMemo } from 'react';
import { Line } from '@react-three/drei';
import { keplerianToCartesian, KeplerianElements } from '../utils/orbitalMath';
import { AU_TO_UNITS } from '../utils/constants';

interface OrbitPathProps {
  keplerianElements: KeplerianElements;
  color?: string;
  opacity?: number;
  segments?: number;
  lineWidth?: number;
}

export default function OrbitPath({ 
  keplerianElements, 
  color = '#ffffff', 
  opacity = 0.3,
  segments = 100,
  lineWidth = 3
}: OrbitPathProps) {
  // Generar puntos de la órbita
  const orbitPoints = useMemo(() => {
    const points: [number, number, number][] = [];
    
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * 2 * Math.PI;
      const meanAnomaly = angle * 180 / Math.PI; // Convertir a grados
      
      const position = keplerianToCartesian({
        ...keplerianElements,
        M: meanAnomaly
      });
      
      const s = AU_TO_UNITS * (keplerianElements.a / (keplerianElements.a || 1));
      points.push([position.x * AU_TO_UNITS, position.y * AU_TO_UNITS, position.z * AU_TO_UNITS]);
    }
    
    return points;
  }, [keplerianElements, segments]);

  return (
    <Line
      points={orbitPoints}
      color={color}
      opacity={opacity}
      lineWidth={lineWidth}
      transparent
    />
  );
}
