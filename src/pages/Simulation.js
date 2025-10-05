// src/pages/Simulation.js o .tsx
import React from "react";
import SolarSystem from "../three/SolarSystem";

export default function Simulation() {
  return (
    <div style={{ width: "100vw", height: "100vh", background: "black" }}>
      <SolarSystem />
    </div>
  );
}