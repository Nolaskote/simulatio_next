# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

## ML impact probability (TensorFlow.js)

This app uses the browser build of TensorFlow.js. We intentionally avoid `@tensorflow/tfjs-node` because it requires native binaries that fail to install on Node 22 without Visual Studio Build Tools. For browser compatibility and simpler deployment (GitHub Pages), we install:

- `@tensorflow/tfjs`

Optional: if you want faster CPU inference, you can add `@tensorflow/tfjs-backend-wasm` and the loader will try to enable it.

### Why `@tensorflow/tfjs-node` failed

You were seeing `node-pre-gyp`/`node-gyp` errors and missing prebuilt binaries for Node 22. Native TensorFlow Node bindings often require:

- A supported Node version with matching prebuilt binaries, or
- A full C++ toolchain on Windows (Visual Studio: Desktop development with C++), Python, and long build times.

Since this project is a Vite + React browser app and deployed to GitHub Pages, the browser build `@tensorflow/tfjs` is the correct choice.

### Where to place a trained model

If you have a trained Keras/TFJS model for impact probability, export it to TFJS format and place it at:

```
public/models/impact_model/model.json
public/models/impact_model/group1-shard1ofN.bin
...
```

At runtime, the helper `src/ml/impactModel.ts` will auto-load that model. If no model is present, it falls back to a documented heuristic placeholder.

### Using the predictor

```
import { predictImpactProbability } from './src/ml/impactModel'

const { probability, source } = await predictImpactProbability({
	moidAU: 0.02,
	missDistanceKm: 450000,
	relativeVelocityKmS: 17,
	diameterKm: 0.12,
	hMagnitude: 22.1,
	inclinationDeg: 5,
	daysToCloseApproach: 30,
})
```

Note: The heuristic is not scientific. Replace it by supplying a trained model.

### WASM backend setup (optional, faster CPU)

You installed `@tensorflow/tfjs-backend-wasm`. We configure the WASM path automatically using Vite's `BASE_URL` and copy `.wasm` binaries in `scripts/postbuild.mjs` to `dist/tfwasm/` for GitHub Pages.

- If you see runtime errors about missing `.wasm` files, ensure they exist in `public/tfwasm/` or that `postbuild` copied them from `node_modules/@tensorflow/tfjs-backend-wasm/dist`.

### Impact predictor UI

- Open the left panel and click “Predecir Impactos”.
- Select a NEO/PHA to see a probability and a demo landing marker on a 3D globe.
- Optional Google Maps view: set `VITE_GOOGLE_MAPS_API_KEY` in an `.env` file to enable the toggle in the predictor footer.

### Train and convert a demo model (Python)

We provide a demo training script (synthetic data): `ml/train_impact_model.py`.

1) Create and activate a Python env, then install deps:

```powershell
pip install tensorflow tensorflowjs numpy scikit-learn
```

2) Run training and export TFJS:

```powershell
python .\ml\train_impact_model.py
```

3) Copy the exported folder to serve it in the app:

```powershell
# Result is in .\model_tfjs
# Place it under public/models/impact_model
New-Item -ItemType Directory -Force -Path .\public\models\impact_model | Out-Null
Copy-Item -Recurse -Force .\model_tfjs\* .\public\models\impact_model\
```

At runtime, the app will load `/models/impact_model/model.json`.

## Entrenamiento real y conversión (recomendado en este repo)

Para entrenar desde el dataset procesado y evitar problemas típicos de Windows con el conversor de TFJS:

1) Usa el entorno virtual incluido y dependencias fijadas

```powershell
python -m venv .venv_ml
.venv_ml\Scripts\Activate.ps1
pip install -r ml\requirements-ml.txt
```

2) Entrena leyendo el CSV procesado (genera SavedModel + H5)

```powershell
python .\ml\train_from_dataset.py --input .\data\processed\impact_dataset.csv --output .\public\models\impact_model
```

Esto guardará:

- `public/models/impact_model_keras/saved_model/`
- `public/models/impact_model_keras/model.h5`

3) Conversión y deploy automáticos (GitHub Actions)

- No necesitas convertir localmente a TFJS en Windows. Al hacer push del archivo `public/models/impact_model_keras/model.h5` a la rama `main`, el workflow `Convert Keras model to TFJS`:
	- Convierte el H5 a TFJS en `public/models/impact_model/`.
	- Sube el TFJS como artefacto descargable.
	- Despliega la página a GitHub Pages (solo si el TFJS cambió).

Consejos y solución de problemas

- Si ves `ModuleNotFoundError: No module named 'numpy'` al entrenar, asegúrate de activar el entorno virtual antes de ejecutar Python:

```powershell
.venv_ml\Scripts\Activate.ps1
python .\ml\train_from_dataset.py --input .\data\processed\impact_dataset.csv --output .\public\models\impact_model
```

- Evita instalar `tensorflowjs` localmente en Windows; el conversor Python suele requerir binarios nativos (TF-DF) que fallan. Ya lo resuelve el workflow de GitHub Actions en Linux.

## Afinamiento (tuning), validación y prevención de overfitting

El script `ml/train_from_dataset.py` soporta:

- Red profunda (DNN) con capas configurables, Dropout y L2.
- Normalización dentro del modelo (se exporta junto al modelo).
- Selección automática de objetivo (`--target-type auto`) para cambiar pérdida y métricas:
	- `probability`: BCE + AUC.
	- `severity`: MSE + MAE/RMSE.
- EarlyStopping y ReduceLROnPlateau con paciencia y min_delta ajustables.
- Pesos de clase para clasificación desbalanceada.
- Validación cruzada K-Fold y búsqueda simple de hiperparámetros (grid pequeño).

### Ejemplos (PowerShell)

Entrenamiento profundo con normalización y early stopping:

```powershell
$py = ".\.venv_ml\Scripts\python.exe"; if (Test-Path $py) { $python=$py } else { $python="python" }

& $python .\ml\train_from_dataset.py `
	--input .\data\processed\impact_dataset.csv `
	--output .\public\models\impact_model `
	--model-type deep `
	--epochs 60 `
	--batch-size 256 `
	--learning-rate 5e-4 `
	--l2 5e-5 `
	--dropout 0.2 `
	--target-type auto `
	--early-stopping `
	--use-class-weights `
	--normalize-in-model on
```

Validación cruzada y pequeño grid de hiperparámetros (capas/LR/Dropout/L2):

```powershell
& $python .\ml\train_from_dataset.py `
	--input .\data\processed\impact_dataset.csv `
	--output .\public\models\impact_model `
	--model-type deep `
	--epochs 40 `
	--batch-size 256 `
	--target-type probability `
	--kfold 5 `
	--tune-hparams `
	--tune-epochs 12 `
	--grid-layers "256,256,128,128,64;256,128,64" `
	--grid-lr "0.001,0.0005,0.0001" `
	--grid-dropout "0.1,0.2,0.3" `
	--grid-l2 "0.0001,0.00005" `
	--early-stopping `
	--patience 6 `
	--min-delta 1e-5
```

Al finalizar, se generan también `metrics.json` (curvas ROC/PRC, AUC/PR-AUC, MAE/RMSE, dispersión/residual SE) y `metadata.json`. El CI copia estos archivos al folder TFJS junto con el modelo convertido.

### Curvas ROC/PRC y dispersión

Para problemas de probabilidad, se exportan:

- ROC (FPR/TPR y AUC)
- PRC (precisión/recuperación y AP-Score)

Y para ambos casos (probabilidad/severidad):

- Desviación estándar de residuos
- Error estándar de las predicciones (aprox. std/sqrt(n))

Puedes visualizar las curvas con cualquier notebook o con una página auxiliar si decides crearla.

### Prevención de overfitting

- EarlyStopping con `--patience` y `--min-delta`.
- Dropout (`--dropout`) y regularización L2 (`--l2`).
- Normalización incorporada en el modelo.
- K-Fold durante tuning para elegir hiperparámetros más robustos.

## Limitaciones del modelo y rango de validez

- Datos y features: el modelo fue entrenado con features agregadas (MOID estimado, distancia mínima, velocidad relativa, diámetro estimado, H, placeholders de inclinación y tiempo a acercamiento). Si en el futuro añades elementos orbitales precisos (e.g., inclinación real, excentricidad, MOID real), deberás reentrenar y versionar el modelo.
- Objetivo: si se usa `severity_score` como objetivo continuo, la interpretación es distinta a probabilidad de impacto; ajustamos la pérdida a MSE/MAE/RMSE pero no es un sustituto científico de un modelo físico.
- Calibración: se aplica Platt sobre la probabilidad modelada si se guarda `calibration.json`. Es una calibración simple y depende de la distribución del set de validación.
- Transferencia: generaliza bien dentro del rango observado en el dataset procesado. Fuera de rango (e.g., tamaños > 1 km si el entrenamiento capó a 1 km, o MOID/distancias muy superiores a los límites) las predicciones pueden carecer de sentido.
- Despliegue en navegador: el rendimiento y precisión pueden variar según el backend (WebGL/WASM/CPU) y dispositivo.

Rango de validez sugerido (según normalización actual y caps):

- MOID: 0–0.5 UA (cap)
- Distancia de paso (miss distance): 0–5,000,000 km (cap)
- Velocidad relativa: 0–40 km/s (cap)
- Diámetro estimado: 0–1 km (cap)
- Magnitud H: ~[15, 35]
- Inclinación: ~0–60° (placeholder actual)
- Días al acercamiento: |días| en escala anual (se normaliza con tanh)

Si cambias estos rangos o añades nuevas variables, actualiza la normalización, reentrena, y vuelve a convertir a TFJS.

## Modelo de daños locales (alineado con paper)

Incluimos `ml/train_damage_model.py` para replicar el enfoque de la literatura (por ejemplo, predicción de radios de daño y/o categorías de daño):

- Tareas soportadas:
	- `--task regression`: regresión multi-salida de radios (p. ej., R_1kPa, R_3kPa, R_20kPa).
	- `--task classification`: categorías de daño (multi-clase).
	- `--task multi`: tronco compartido con cabezas de regresión + clasificación.
- Transformación de objetivos de regresión: `--reg-transform {log10,log1p,identity}` (por defecto `log10`) para favorecer error relativo constante (~10%).
- Pérdida de regresión configurable: `--reg-loss {mae_log,mape,mse,mae}` (por defecto `mae_log`).
- Exporta SavedModel + H5 y JSONs de `metadata.json`/`metrics.json`.

Requisitos de dataset (CSV):

- Columnas de entrada `--features`: por ejemplo, diámetro_km, velocidad_km_s al impacto, densidad, ángulo de entrada, altura de estallido, tipo de objetivo (uno-hot), etc., según el paper/dataset que utilices.
- Columnas de salida de regresión `--reg-columns`: p. ej. `R_1kPa,R_3kPa,R_20kPa` (en km o m; define unidades y sé consistente).
- Columna de clase `--class-column` (enteros 0..K-1) si se usa clasificación o multi.

Ejemplos (PowerShell):

```powershell
$py = ".\.venv_ml\Scripts\python.exe"; if (Test-Path $py) { $python=$py } else { $python="python" }

# Regresión de radios (log10 + MAE en log):
& $python .\ml\train_damage_model.py `
	--input .\data\processed\damage_dataset.csv `
	--output .\public\models\damage_model `
	--features "diameter_km,velocity_kms,density_kg_m3,impact_angle_deg,target_soil_onehot,..." `
	--task regression `
	--reg-columns "R_1kPa,R_3kPa,R_20kPa" `
	--layers "256,256,128,128,64" `
	--reg-transform log10 `
	--reg-loss mae_log `
	--epochs 60 `
	--batch-size 256

# Clasificación de categorías de daño (multiclase):
& $python .\ml\train_damage_model.py `
	--input .\data\processed\damage_dataset.csv `
	--output .\public\models\damage_model `
	--features "diameter_km,velocity_kms,density_kg_m3,impact_angle_deg,target_soil_onehot,..." `
	--task classification `
	--class-column damage_class `
	--classes 5 `
	--layers "256,256,128,128,64" `
	--epochs 60 `
	--batch-size 256 `
	--use-class-weights
```

Nota: para igualar resultados reportados en papers (e.g., ~10% error en radios, ~98% accuracy en categorías), es clave contar con las mismas variables de entrada (física de impacto y condiciones del blanco), etiquetas bien definidas y el mismo preprocesamiento (log-scale para radios). Los datasets de acercamientos cercanos (CNEOS CAD/Sentry) no contienen variables locales de impacto; para este modelo de daños debes usar o construir un dataset específico de impactos con esas variables.
