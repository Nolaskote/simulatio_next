Dataset pipeline
================

Sources
-------

- Sentry (Earth Impact Monitoring): https://cneos.jpl.nasa.gov/sentry/
- NEO Earth Close Approaches: https://cneos.jpl.nasa.gov/ca/
- APIs
  - Close Approaches: https://ssd-api.jpl.nasa.gov/doc/cad.html
  - Sentry list: https://ssd-api.jpl.nasa.gov/doc/sentry.html

Goals
-----

- Unificar datos de aproximaciones cercanas (distancia, velocidad, magnitud H) con métricas de riesgo (probabilidad acumulada, Palermo, Torino, diámetro).
- Convertir todo a unidades SI: distancias en km, velocidades en km/s, probabilidad [0,1].
- Producir un dataset tabular listo para entrenamiento.

Pipeline
--------

1) Descarga datos (Node):

```powershell
npm run data:fetch:cneos
```

Outputs en `public/data/cneos/raw/`:
- `close_approaches.json`
- `sentry_list.json`
- `sentry_removed.json`

2) Normalización y merge (Python):

```powershell
pip install pandas
python .\ml\prepare_dataset.py --augment 5000
```

Outputs en `data/processed/`:
- `impact_dataset.csv`
- `impact_dataset.json`

3) Entrenamiento (Python):

- Usa `impact_dataset.csv` para entrenar tu modelo. Alinea el preprocesado con el frontend (`src/ml/impactModel.ts`).
- Exporta a TFJS y colócalo en `public/models/impact_model/`.

Notas sobre incertidumbre
-------------------------

- Agrega columnas de incertidumbre (p.ej., rango de distancia, varianza de velocidad) si están disponibles.
- La recomendación del usuario sugiere Monte Carlo de órbitas: puedes expandir el dataset simulando múltiples realizaciones con ruido observacional.

Referencia
----------

- “Machine learning for the prediction of local asteroid damages” (paper adjunto). Usa sus variables y escalas como guía para las features y para la conversión de energías/daños locales si amplías el objetivo a severidad de impacto.
