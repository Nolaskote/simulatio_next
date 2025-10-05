"""
Prepare dataset from CNEOS raw files into a unified CSV/JSON with SI units.

Inputs:
  public/data/cneos/raw/close_approaches.json (cad.api)
  public/data/cneos/raw/sentry_list.json (sentry.api full)

Outputs:
  data/processed/impact_dataset.csv
  data/processed/impact_dataset.json

Options:
  --augment N : generate N synthetic samples using simple Monte Carlo assumptions
"""

import argparse
import json
from pathlib import Path
import math
import random

import pandas as pd
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'public' / 'data' / 'cneos' / 'raw'
OUT = ROOT / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)


def load_close_approaches():
    p = RAW / 'close_approaches.json'
    if not p.exists():
        return pd.DataFrame()
    js = json.loads(p.read_text(encoding='utf-8'))
    fields = js.get('fields', [])
    rows = js.get('data', [])
    df = pd.DataFrame(rows, columns=fields)
    # Rename and cast
    df = df.rename(columns={
        'des': 'designation',
        'cd': 'close_approach_datetime',
        'dist': 'distance_au',
        'dist_min': 'distance_min_au',
        'v_rel': 'v_rel_kms',
        'h': 'H_mag',
    })
    for col in ['distance_au', 'distance_min_au', 'v_rel_kms', 'H_mag']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Convert AU->km (1 AU = 149,597,870.7 km)
    AU_KM = 149_597_870.7
    df['distance_km'] = df['distance_au'] * AU_KM
    df['distance_min_km'] = df['distance_min_au'] * AU_KM
    return df


def load_sentry():
    p = RAW / 'sentry_list.json'
    if not p.exists():
        return pd.DataFrame()
    js = json.loads(p.read_text(encoding='utf-8'))
    # Sentry returns list of objects with keys like: des, ps_cum (Palermo cumulative), ts_max (Torino max), ip (impact prob cumulative), v_inf, H, diameter, range, etc.
    data = js.get('data', []) or js.get('sentrydata', []) or []
    if not data:
        # Sometimes `data` is nested under `data` key already
        data = js.get('data', [])
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    if not isinstance(data, list):
        data = []
    df = pd.DataFrame(data)
    # Normalize columns if present
    rename = {
        'des': 'designation',
        'ip': 'impact_prob_cum',
        'ps_cum': 'palermo_cum',
        'ts_max': 'torino_max',
        'v_inf': 'v_inf_kms',
        'H': 'H_mag',
        'diameter': 'diameter_km',
    }
    df = df.rename(columns=rename)
    for c in ['impact_prob_cum', 'palermo_cum', 'torino_max', 'v_inf_kms', 'H_mag', 'diameter_km']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def estimate_moid_from_ca(df_ca: pd.DataFrame) -> pd.Series:
    # Simple proxy: min close approach distance per object (km) -> AU
    if df_ca.empty:
        return pd.Series(dtype=float)
    AU_KM = 149_597_870.7
    min_km = df_ca.groupby('designation')['distance_min_km'].min()
    return (min_km / AU_KM).rename('moid_au_est')


def merge_datasets(df_ca: pd.DataFrame, df_sentry: pd.DataFrame) -> pd.DataFrame:
    # Aggregate CA per designation
    if df_ca.empty and df_sentry.empty:
        return pd.DataFrame()
    agg = pd.DataFrame()
    if not df_ca.empty:
        g = df_ca.groupby('designation').agg({
            'distance_km': 'min',
            'distance_min_km': 'min',
            'v_rel_kms': 'median',
            'H_mag': 'median',
        })
        agg = g.rename(columns={
            'distance_km': 'min_distance_km',
            'distance_min_km': 'min_distance_min_km',
            'v_rel_kms': 'median_v_rel_kms',
            'H_mag': 'median_H_mag',
        })
        agg['observations'] = df_ca.groupby('designation').size()

    if not df_sentry.empty:
        agg = agg.join(df_sentry.set_index('designation'), how='outer', rsuffix='_sentry')

    # Derive moid estimate if missing
    if not df_ca.empty:
        agg = agg.join(estimate_moid_from_ca(df_ca), how='left')

    # SI normalization already OK (km, km/s). Fill defaults
    for col, val in [
        ('median_v_rel_kms', 20.0),
        ('median_H_mag', 22.0),
        ('diameter_km', None),
        ('impact_prob_cum', None),
    ]:
        if col not in agg.columns:
            agg[col] = val

    agg.reset_index(inplace=True)
    return agg


def augment_synthetic(df: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        moid_au_est = abs(random.gauss(0.05, 0.05))
        min_distance_min_km = max(1000.0, abs(random.gauss(1e6, 5e5)))
        median_v_rel_kms = max(5.0, min(40.0, random.gauss(20.0, 7.0)))
        median_H_mag = max(10.0, min(30.0, random.gauss(22.0, 3.0)))
        diameter_km = max(0.005, min(2.0, random.gauss(0.15, 0.2)))
        days_to_close_approach = random.randint(1, 365)
        # Synthetic impact probability loosely based on heuristic
        miss_ratio = max(0.0, min(1.0, (5_000_000.0 - min_distance_min_km) / 5_000_000.0))
        moid_ratio = max(0.0, min(1.0, (0.5 - min(0.5, moid_au_est)) / 0.5))
        z = 3*miss_ratio + 3*moid_ratio + 1*(median_v_rel_kms/40) + 2*(diameter_km/1) + 1.5*((35-median_H_mag)/20) + 1.5*(1-math.tanh(days_to_close_approach/365)) - 4
        ip = 1/(1+math.exp(-z))
        rows.append({
            'designation': f'SYN-{random.randint(100000,999999)}',
            'min_distance_min_km': min_distance_min_km,
            'min_distance_km': min_distance_min_km,  # proxy
            'median_v_rel_kms': median_v_rel_kms,
            'median_H_mag': median_H_mag,
            'diameter_km': diameter_km,
            'impact_prob_cum': ip,
            'moid_au_est': moid_au_est,
            'synthetic': True,
        })
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def estimate_diameter_from_H(H_mag: float, albedo: float = 0.14) -> float:
    """Estimate diameter (km) from absolute magnitude H and assumed albedo.
    D(km) = 1329 / sqrt(p) * 10^(-H/5)
    """
    if H_mag is None or not math.isfinite(H_mag):
        return float('nan')
    p = max(1e-3, min(1.0, albedo))
    return 1329.0 / math.sqrt(p) * (10 ** (-H_mag / 5.0))


def kinetic_energy_joules(diameter_km: float, velocity_kms: float, density_kg_m3: float = 2000.0) -> float:
    """Compute kinetic impact energy (J) assuming spherical body.
    mass = rho * (4/3)*pi*r^3, r = (diameter/2)
    v (m/s) from km/s
    Default density ~2000 kg/m^3 (≈ 2.0 g/cm^3) as an average bulk density for main-belt asteroids.
    """
    if not (math.isfinite(diameter_km) and math.isfinite(velocity_kms)):
        return float('nan')
    r_m = (max(0.0, diameter_km) * 1000.0) / 2.0
    volume_m3 = (4.0 / 3.0) * math.pi * (r_m ** 3)
    mass_kg = density_kg_m3 * volume_m3
    v_ms = max(0.0, velocity_kms) * 1000.0
    return 0.5 * mass_kg * (v_ms ** 2)


def energy_megatons_TNT(E_joules: float) -> float:
    """Convert Joules to megatons of TNT equivalent. 1 Mt = 4.184e15 J."""
    if not math.isfinite(E_joules):
        return float('nan')
    return E_joules / 4.184e15


def severity_score(E_mt: float, miss_km: Optional[float]) -> float:
    """Simple 0-1 severity index combining energy and miss distance.
    - Map energy with a smooth logistic on log10(E_mt+1).
    - Reduce severity with larger miss distances (cap at 5M km).
    Note: This is NOT a scientific scale; replace with a domain-specific metric.
    """
    if not math.isfinite(E_mt):
        return float('nan')
    # Energy component
    x = math.log10(max(1e-6, E_mt) + 1.0)  # 0 for tiny energies, grows slowly
    e_term = 1.0 / (1.0 + math.exp(-(x - 0.5) * 2.0))
    # Distance damping (closer -> higher). Use min distance if available.
    if miss_km is None or not math.isfinite(miss_km):
        d_term = 0.7  # unknown -> moderate
    else:
        d_ratio = max(0.0, min(1.0, (5_000_000.0 - miss_km) / 5_000_000.0))
        d_term = 0.3 + 0.7 * d_ratio
    return max(0.0, min(1.0, 0.2 + 0.8 * e_term * d_term))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--augment', type=int, default=0)
    args = ap.parse_args()

    df_ca = load_close_approaches()
    df_sentry = load_sentry()
    df = merge_datasets(df_ca, df_sentry)
    if args.augment > 0:
        df = augment_synthetic(df, args.augment)

    # Enrich: estimate missing diameter from H, compute kinetic energy and severity
    if not df.empty:
        # Choose diameter: prefer provided diameter_km, else estimate from H
        est_diam = df.get('diameter_km')
        if est_diam is None:
            df['diameter_km'] = float('nan')
        # Fill missing diameters
        df['diameter_est_km'] = df.apply(
            lambda r: r['diameter_km'] if (isinstance(r.get('diameter_km'), (float, int)) and math.isfinite(r['diameter_km']))
            else estimate_diameter_from_H(r.get('median_H_mag', float('nan'))), axis=1
        )
        # Velocity source
        df['velocity_kms_for_energy'] = df.get('median_v_rel_kms', pd.Series([float('nan')] * len(df)))
        # Energy
        df['energy_joules'] = df.apply(
            lambda r: kinetic_energy_joules(r.get('diameter_est_km', float('nan')), r.get('velocity_kms_for_energy', float('nan'))), axis=1
        )
        df['energy_mt_tnt'] = df['energy_joules'].apply(energy_megatons_TNT)
        # Severity (use min distance min as proxy of miss distance)
        miss = df.get('min_distance_min_km', df.get('min_distance_km', pd.Series([float('nan')] * len(df))))
        df['severity_score'] = [
            severity_score(mt, mk) for mt, mk in zip(df['energy_mt_tnt'].tolist(), miss.tolist())
        ]

    # Save processed
    out_csv = OUT / 'impact_dataset.csv'
    out_json = OUT / 'impact_dataset.json'
    df.to_csv(out_csv, index=False)
    out_json.write_text(df.to_json(orient='records', indent=2), encoding='utf-8')
    print(f'Saved {len(df)} rows -> {out_csv}')


if __name__ == '__main__':
    main()
