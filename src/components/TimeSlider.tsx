import React, { useState, useRef, useEffect } from 'react';
import { TimeState, julianDayToDate, dateToJulianDay, formatDate } from '../utils/timeSystem';

interface TimeSliderProps {
  timeState: TimeState;
  onTimeChange: (newTimeState: TimeState) => void;
}

export default function TimeSlider({ timeState, onTimeChange }: TimeSliderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [currentTimeMode, setCurrentTimeMode] = useState<'hour' | 'day' | 'week' | 'month'>('day');
  const sliderRef = useRef<HTMLDivElement>(null);

  // Definir rangos de tiempo para cada modo
  const timeRanges = {
    hour: { range: 24, unit: 'horas', step: 0.25 }, // 24 horas en el futuro/pasado
    day: { range: 365, unit: 'días', step: 1 }, // 1 año en el futuro/pasado
    week: { range: 52, unit: 'semanas', step: 1 }, // 1 año en semanas
    month: { range: 24, unit: 'meses', step: 1 } // 2 años en meses
  };

  const currentRange = timeRanges[currentTimeMode];

  // Calcular posición del slider basada en el tiempo actual
  const getSliderPosition = () => {
    const now = new Date();
    const nowJD = dateToJulianDay(now);
    const timeDiff = timeState.julianDay - nowJD;
    
    // Convertir diferencia de tiempo según el modo actual
    let timeDiffInUnits: number;
    switch (currentTimeMode) {
      case 'hour':
        timeDiffInUnits = timeDiff * 24; // días a horas
        break;
      case 'day':
        timeDiffInUnits = timeDiff; // ya está en días
        break;
      case 'week':
        timeDiffInUnits = timeDiff / 7; // días a semanas
        break;
      case 'month':
        timeDiffInUnits = timeDiff / 30.44; // días a meses (promedio)
        break;
      default:
        timeDiffInUnits = timeDiff;
    }
    
    // Normalizar a rango de 0-100
    const normalizedPosition = Math.max(0, Math.min(100, (timeDiffInUnits + currentRange.range) / (2 * currentRange.range) * 100));
    return normalizedPosition;
  };

  const handleSliderChange = (clientX: number) => {
    if (!sliderRef.current) return;
    
    const rect = sliderRef.current.getBoundingClientRect();
    const position = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    
    // Convertir posición a diferencia de tiempo
    const timeDiffInUnits = (position / 100) * (2 * currentRange.range) - currentRange.range;
    
    let timeDiffInDays: number;
    switch (currentTimeMode) {
      case 'hour':
        timeDiffInDays = timeDiffInUnits / 24;
        break;
      case 'day':
        timeDiffInDays = timeDiffInUnits;
        break;
      case 'week':
        timeDiffInDays = timeDiffInUnits * 7;
        break;
      case 'month':
        timeDiffInDays = timeDiffInUnits * 30.44;
        break;
      default:
        timeDiffInDays = timeDiffInUnits;
    }
    
    const now = new Date();
    const nowJD = dateToJulianDay(now);
    const newJD = nowJD + timeDiffInDays;
    
    onTimeChange({
      ...timeState,
      julianDay: newJD,
      currentDate: julianDayToDate(newJD)
    });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    handleSliderChange(e.clientX);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging) {
      handleSliderChange(e.clientX);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const goToNow = () => {
    const now = new Date();
    onTimeChange({
      ...timeState,
      julianDay: dateToJulianDay(now),
      currentDate: now
    });
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging]);

  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      left: '50%',
      transform: 'translateX(-50%)',
      background: 'rgba(0, 0, 0, 0.8)',
      color: 'white',
      padding: '15px 20px',
      borderRadius: '10px',
      fontFamily: 'Arial, sans-serif',
      minWidth: '400px',
      textAlign: 'center'
    }}>
      <h3 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>Navegación Temporal</h3>
      
      {/* Selector de modo de tiempo */}
      <div style={{ marginBottom: '15px' }}>
        {(['hour', 'day', 'week', 'month'] as const).map((mode) => (
          <button
            key={mode}
            onClick={() => setCurrentTimeMode(mode)}
            style={{
              background: currentTimeMode === mode ? '#4CAF50' : '#333',
              color: 'white',
              border: 'none',
              padding: '5px 10px',
              margin: '0 2px',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            {mode === 'hour' ? 'Hora' : mode === 'day' ? 'Día' : mode === 'week' ? 'Semana' : 'Mes'}
          </button>
        ))}
      </div>
      
      {/* Slider de tiempo */}
      <div style={{ marginBottom: '15px' }}>
        <div
          ref={sliderRef}
          style={{
            width: '100%',
            height: '20px',
            background: 'linear-gradient(to right, #ff4444, #ffff00, #44ff44)',
            borderRadius: '10px',
            position: 'relative',
            cursor: 'pointer'
          }}
          onMouseDown={handleMouseDown}
        >
          <div
            style={{
              position: 'absolute',
              left: `${getSliderPosition()}%`,
              top: '50%',
              transform: 'translate(-50%, -50%)',
              width: '12px',
              height: '12px',
              background: 'white',
              borderRadius: '50%',
              border: '2px solid #333',
              cursor: 'grab'
            }}
          />
        </div>
        
        {/* Etiquetas del slider */}
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#aaa', marginTop: '5px' }}>
          <span>Pasado (-{currentRange.range} {currentRange.unit})</span>
          <span>Ahora</span>
          <span>Futuro (+{currentRange.range} {currentRange.unit})</span>
        </div>
      </div>
      
      {/* Información de tiempo */}
      <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '10px' }}>
        <p>{formatDate(timeState.currentDate)}</p>
        <p>Modo: {currentRange.unit} de simulación</p>
      </div>
      
      {/* Botón para ir al presente */}
      <button
        onClick={goToNow}
        style={{
          background: '#2196F3',
          color: 'white',
          border: 'none',
          padding: '8px 16px',
          borderRadius: '5px',
          cursor: 'pointer',
          fontSize: '12px'
        }}
      >
        🕐 Ir al Presente
      </button>
    </div>
  );
}
