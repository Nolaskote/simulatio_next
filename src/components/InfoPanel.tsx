import React, { useState } from 'react'

type SectionProps = { title: string; children: React.ReactNode; defaultOpen?: boolean }

function Section({ title, children, defaultOpen = false }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="info-section">
      <button className="info-section__header" onClick={() => setOpen(o => !o)} aria-expanded={open}>
        <span className="info-section__chev" aria-hidden>
          {open ? <i className="fa-solid fa-chevron-down" /> : <i className="fa-solid fa-chevron-right" />}
        </span>
        <span>{title}</span>
      </button>
      {open && <div className="info-section__body">{children}</div>}
    </div>
  )
}

export default function InfoPanel({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  return (
    <div className={`info-panel ${visible ? 'is-visible' : 'is-hidden'}`} role="dialog" aria-label="Simulation information">
      <div className="info-panel__header">
        <div className="info-panel__title"><i className="fa-solid fa-circle-info" style={{ marginRight: 6 }} />Information</div>
        <button className="info-panel__close" onClick={onClose} aria-label="Close"><i className="fa-solid fa-xmark" /></button>
      </div>
      <div className="info-panel__content">
  <Section title="The 6 Keplerian elements" defaultOpen={false}>
          <ul>
            <li><strong>Semi-major axis (a):</strong> defines the size of the orbit; half the ellipse’s major axis.</li>
            <li><strong>Eccentricity (e):</strong> measures how elongated the ellipse is (0 is circular, close to 1 is very stretched).</li>
            <li><strong>Inclination (i):</strong> angle between the object’s orbital plane and the reference plane (e.g., the ecliptic).</li>
            <li><strong>Longitude of ascending node (Ω):</strong> locates the line of nodes (where the orbit crosses the reference plane) relative to a reference meridian.</li>
            <li><strong>Argument of perihelion (ω):</strong> angle from the ascending node to perihelion along the orbit.</li>
            <li><strong>Mean anomaly (M):</strong> parameter that increases uniformly with time and, together with e, lets you solve Kepler’s equation to get the position.</li>
          </ul>
          <p style={{ color: '#bbb', marginTop: 8 }}>
            With these six elements you can reconstruct the object’s position and trajectory at any instant using Kepler’s laws. In this simulation we also show two auxiliary parameters:
            <br />
            • <strong>v (°):</strong> the <em>true anomaly</em>, the angle between perihelion and the object’s current position along its orbit, in degrees.
            <br />
            • <strong>P (d):</strong> the <em>orbital period</em>, the time to complete one revolution around the Sun, in days.
          </p>
        </Section>

  <Section title="What is Julian Day (JD)?" defaultOpen={false}>
          <p><strong>Julian Day (JD)</strong> is an astronomical time scale that counts days (and fractions of a day) continuously from a conventional origin (noon on January 1, 4713 BCE, in the Julian calendar). It’s useful in astronomy because it avoids calendar ambiguities and makes precise intervals easy to measure.</p>
          <p>In this app you’ll see <strong>JD</strong> updating as the simulation advances: changing JD is equivalent to moving precisely into the past or the future.</p>
        </Section>

  <Section title="How does simulation speed work?" defaultOpen={false}>
          <ul>
            <li><strong>1s/s:</strong> 1 simulated second = 1 real second.</li>
            <li><strong>1min/s:</strong> 1 simulated minute = 1 real second.</li>
            <li><strong>1h/s:</strong> 1 simulated hour = 1 real second.</li>
            <li><strong>1d/s:</strong> 1 simulated day = 1 real second.</li>
            <li><strong>1w/s, 1m/s, 1y/s:</strong> 1 simulated week/month/year = 1 real second.</li>
          </ul>
          <p>You can toggle direction (past/future) and pause/resume. The scene updates orbital positions based on that time scale.</p>
        </Section>

  <Section title="Why NEO/PHA positions matter" defaultOpen={false}>
          <p><strong>NEOs</strong> (Near-Earth Objects) and <strong>PHAs</strong> (Potentially Hazardous Asteroids) are near-Earth objects whose orbits may approach Earth’s. Knowing their <strong>exact position</strong> and orbital evolution enables:</p>
          <ul>
            <li>Computing <strong>encounter probabilities</strong> and close-approach windows.</li>
            <li>Planning <strong>observations</strong> and new measurements to reduce uncertainties.</li>
            <li>Feeding models that evaluate <strong>impact risk</strong> and potential affected areas.</li>
            <li>Devising <strong>mitigation</strong> strategies (deflection, early warning) when needed.</li>
          </ul>
        </Section>
      </div>
    </div>
  )
}
