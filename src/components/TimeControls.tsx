import React from 'react';
import { TimeState } from '../utils/timeSystem';

interface TimeControlsProps {
  timeState: TimeState;
  onTimeChange: (newTimeState: TimeState) => void;
  onDateChange: (date: Date) => void;
}

export default function TimeControls({ timeState, onTimeChange, onDateChange }: TimeControlsProps) {
  const handlePlayPause = () => {
    onTimeChange({
      ...timeState,
      isPlaying: !timeState.isPlaying
    });
  };

  const handleTimeScaleChange = (scale: number) => {
    onTimeChange({
      ...timeState,
      timeScale: scale
    });
  };

  const handleDateInput = (dateString: string) => {
    const newDate = new Date(dateString);
    if (!isNaN(newDate.getTime())) {
      onDateChange(newDate);
    }
  };

  const goToNow = () => {
    onDateChange(new Date());
  };

  return (
    <div style={{
      position: 'absolute',
      bottom: '20px',
      right: '20px',
      background: 'rgba(0, 0, 0, 0.8)',
      color: 'white',
      padding: '20px',
      borderRadius: '10px',
      fontFamily: 'Arial, sans-serif',
      minWidth: '300px'
    }}>
      <h3 style={{ margin: '0 0 15px 0' }}>Control de Tiempo</h3>
      
      {/* Fecha y hora actual */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
          Fecha y Hora:
        </label>
        <input
          type="datetime-local"
          value={timeState.currentDate.toISOString().slice(0, 16)}
          onChange={(e) => handleDateInput(e.target.value)}
          style={{
            width: '100%',
            padding: '5px',
            borderRadius: '3px',
            border: '1px solid #555',
            background: '#333',
            color: 'white'
          }}
        />
      </div>

      {/* Controles de reproducción */}
      <div style={{ marginBottom: '15px' }}>
        <button
          onClick={handlePlayPause}
          style={{
            background: timeState.isPlaying ? '#ff4444' : '#44ff44',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '5px',
            cursor: 'pointer',
            marginRight: '10px'
          }}
        >
          {timeState.isPlaying ? '⏸️ Pausar' : '▶️ Reproducir'}
        </button>
        
        <button
          onClick={goToNow}
          style={{
            background: '#4444ff',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          🕐 Ahora
        </button>
      </div>

      {/* Control de velocidad */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
          Velocidad: {timeState.timeScale}x
        </label>
        <input
          type="range"
          min="0.1"
          max="100"
          step="0.1"
          value={timeState.timeScale}
          onChange={(e) => handleTimeScaleChange(parseFloat(e.target.value))}
          style={{ width: '100%' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#aaa' }}>
          <span>0.1x</span>
          <span>1x</span>
          <span>10x</span>
          <span>100x</span>
        </div>
      </div>

      {/* Información del día juliano */}
      <div style={{ fontSize: '12px', color: '#aaa' }}>
        <p>Día Juliano: {timeState.julianDay.toFixed(2)}</p>
        <p>Estado: {timeState.isPlaying ? 'Reproduciendo' : 'Pausado'}</p>
      </div>
    </div>
  );
}
