import React from 'react';
import { NEOData } from '../utils/orbitalMath';
import './EyesOnAsteroids.css';

export default function AsteroidInfo({ neo }: { neo: NEOData | null }) {
  if (!neo) return null;
  return (
    <div className="panel panel--right">
      <h3 className={neo.type === 'PHA' ? 'title title--pha' : 'title'}>{neo.name}</h3>
      <p><strong>Tipo:</strong> {neo.type}</p>
      <p><strong>ID:</strong> {neo.id}</p>
      <p><strong>Semi-eje mayor:</strong> {parseFloat(neo.a).toFixed(3)} UA</p>
      <p><strong>Excentricidad:</strong> {parseFloat(neo.e).toFixed(3)}</p>
      <p><strong>Inclinación:</strong> {parseFloat(neo.i).toFixed(2)}°</p>
      <p><strong>Período orbital:</strong> {parseFloat(neo.period).toFixed(1)} días</p>
    </div>
  );
}
