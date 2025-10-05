"""
Train a TF model from processed dataset and export TFJS model into public/models/impact_model/.

Usage (PowerShell):
  pip install tensorflow tensorflowjs pandas numpy scikit-learn
  python .\ml\train_from_dataset.py --input .\data\processed\impact_dataset.csv --output .\public\models\impact_model

Features used (aligned with frontend src/ml/impactModel.ts):
  - moidAU (proxy: moid_au_est)
  - missDistanceKm (proxy: min_distance_min_km or min_distance_km)
  - relativeVelocityKmS (median_v_rel_kms)
  - diameterKm (diameter_km or diameter_est_km)
  - hMagnitude (median_H_mag)
  - inclinationDeg (not available -> 0)
  - daysToCloseApproach (not available -> 180)

Targets:
  - impact_prob_cum (if available). If missing, use severity_score as a surrogate (not scientific).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
# Prefer TensorFlow's bundled Keras; fallback to standalone keras if tf.keras is unavailable
try:
  from tensorflow import keras  # type: ignore
  _KERAS_SRC = 'tf.keras'
except Exception:
  import keras  # type: ignore
  _KERAS_SRC = 'keras'
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from typing import Optional, Dict, Tuple, List, Any


def build_shallow_model(input_dim: int, lr: float = 1e-3, norm_layer: Optional[keras.layers.Layer] = None):
  inp = keras.Input(shape=(input_dim,), dtype='float32')
  x = inp
  if norm_layer is not None:
    x = norm_layer(x)
  x = keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
  x = keras.layers.Dropout(0.1)(x)
  x = keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
  out = keras.layers.Dense(1, activation='sigmoid')(x)
  model = keras.Model(inp, out)
  # Compile will be configured externally to support different targets
  return model


def build_deep_model(input_dim: int, lr: float = 1e-3, l2: float = 1e-4, dropout: float = 0.2, norm_layer: Optional[keras.layers.Layer] = None, layers: Optional[List[int]] = None):
  """Deep DNN with BN, Dropout and L2 regularization. Compatible with TFJS conversion."""
  reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None
  inp = keras.Input(shape=(input_dim,), dtype='float32')
  x = inp
  if norm_layer is not None:
    x = norm_layer(x)
  for units in (layers or [256, 256, 128, 128, 64]):
    x = keras.layers.Dense(units, kernel_initializer='he_normal', kernel_regularizer=reg, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout)(x)
  out = keras.layers.Dense(1, activation='sigmoid')(x)
  model = keras.Model(inp, out)
  return model


def create_normalizer(X: np.ndarray) -> keras.layers.Layer:
  """Create and adapt a Keras Normalization layer on X."""
  norm = keras.layers.Normalization(axis=-1)
  norm.adapt(X)
  return norm


def compile_for_target(model: keras.Model, target_type: str, lr: float) -> Tuple[str, list]:
  """Compile model according to target type. Returns (loss_name, metrics)."""
  if target_type == 'probability':
    loss = 'binary_crossentropy'
    metrics = [keras.metrics.AUC(name='AUC')]
  elif target_type == 'severity':
    loss = 'mse'
    metrics = [keras.metrics.MeanAbsoluteError(name='MAE'), keras.metrics.RootMeanSquaredError(name='RMSE')]
  else:
    raise ValueError('target_type must be one of {probability,severity}')
  model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
  return loss, metrics


def parse_layers_spec(spec: str) -> List[List[int]]:
  """Parse layer specs like '256,256,128;256,128,64' into [[256,256,128],[256,128,64]]."""
  configs: List[List[int]] = []
  for chunk in spec.split(';'):
    nums = [int(x) for x in chunk.split(',') if x.strip()]
    if nums:
      configs.append(nums)
  return configs


def evaluate_model(model: keras.Model, X_val: np.ndarray, y_val: np.ndarray, target_type: str) -> Dict[str, Any]:
  ev = model.evaluate(X_val, y_val, verbose=0)
  # Map metrics to names
  if target_type == 'probability':
    # ev: [loss, AUC]
    return {'loss': float(ev[0]), 'AUC': float(ev[1]) if len(ev) > 1 else None}
  else:
    # ev: [loss(MSE), MAE, RMSE]
    out = {'loss': float(ev[0])}
    if len(ev) > 1:
      out['MAE'] = float(ev[1])
    if len(ev) > 2:
      out['RMSE'] = float(ev[2])
    return out


def kfold_score(X: np.ndarray, y: np.ndarray, args, target_type: str, layers: Optional[List[int]] = None, lr: float = 1e-3, l2val: float = 1e-4, dropout: float = 0.2) -> Dict[str, Any]:
  kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
  fold_metrics: List[Dict[str, Any]] = []
  f = 1
  for tr_idx, va_idx in kf.split(X):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    norm_cv = create_normalizer(X_tr) if args.normalize_in_model == 'on' else None
    mdl = build_deep_model(X.shape[1], lr=lr, l2=l2val, dropout=dropout, norm_layer=norm_cv, layers=layers) if args.model_type == 'deep' else build_shallow_model(X.shape[1], lr=lr, norm_layer=norm_cv)
    compile_for_target(mdl, target_type, lr=lr)
    cw = None
    if args.use_class_weights and target_type == 'probability':
      pos = float((y_tr > 0.5).sum())
      neg = float((y_tr <= 0.5).sum())
      if pos > 0:
        cw = {0: 1.0, 1: neg/pos}
    cb = []
    if args.early_stopping:
      cb = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, min_delta=args.min_delta, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1, args.patience//2), min_lr=1e-6)
      ]
    mdl.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=max(5, min(args.tune_epochs, args.epochs)), batch_size=args.batch_size, callbacks=cb, class_weight=cw, verbose=0)
    m = evaluate_model(mdl, X_va, y_va, target_type)
    fold_metrics.append(m)
    f += 1
  # Aggregate
  agg: Dict[str, Any] = {'folds': fold_metrics}
  # Compute mean/std for known keys
  keys = set().union(*[set(d.keys()) for d in fold_metrics])
  for k in keys:
    vals = [float(d[k]) for d in fold_metrics if d.get(k) is not None]
    if vals:
      agg[f'mean_{k}'] = float(np.mean(vals))
      agg[f'std_{k}'] = float(np.std(vals))
  return agg


def normalize_features(df: pd.DataFrame) -> np.ndarray:
  """Normalize features with robust fallbacks if columns are missing.
  Mirrors frontend normalization in src/ml/impactModel.ts, but tolerates
  datasets that don't include moid or miss-distance columns.
  """
  n = len(df)
  # MOID (AU) proxy
  moid_series = df.get('moid_au_est')
  if moid_series is None:
    moid_series = pd.Series([np.nan] * n)
  moidAU = moid_series.fillna(0.3).clip(lower=0, upper=0.5) / 0.5

  # Miss distance (km) proxy
  miss_series = df.get('min_distance_min_km')
  if miss_series is None:
    miss_series = df.get('min_distance_km')
  if miss_series is None:
    miss_series = pd.Series([np.nan] * n)
  missKm = miss_series.fillna(5_000_000).clip(lower=0, upper=5_000_000) / 5_000_000

  # Relative velocity (km/s)
  relv_series = df.get('median_v_rel_kms')
  if relv_series is None:
    relv_series = pd.Series([np.nan] * n)
  relV = relv_series.fillna(20).clip(lower=0, upper=40) / 40

  # Diameter (km)
  diam_series = df.get('diameter_km')
  if diam_series is None:
    diam_series = pd.Series([np.nan] * n)
  diam_est_series = df.get('diameter_est_km')
  if diam_est_series is None:
    diam_est_series = pd.Series([np.nan] * n)
  diam = diam_series.fillna(diam_est_series).fillna(0).clip(lower=0, upper=1.0) / 1.0

  # H magnitude -> normalized inverse scale
  H_series = df.get('median_H_mag')
  if H_series is None:
    H_series = pd.Series([22.0] * n)
  H = H_series.fillna(22).apply(lambda h: max(0.0, min(1.0, (35 - h) / 20)))

  # Placeholders for unavailable features
  inc = np.zeros(n)
  days = np.full(n, 180.0)
  daysN = 1 - np.tanh(np.abs(days) / 365.0)

  X = np.stack([moidAU.values, missKm.values, relV.values, diam.values, H.values, inc, daysN], axis=1).astype('float32')
  return X


def select_target(df: pd.DataFrame) -> np.ndarray:
  y = df.get('impact_prob_cum')
  if y is None or y.isna().all():
    # fallback to severity (not scientific but gives a continuous signal)
    y = df.get('severity_score')
  if y is None:
    raise RuntimeError('No suitable target column found (impact_prob_cum or severity_score).')
  y = y.fillna(0.0).clip(lower=0.0, upper=1.0).astype('float32').values.reshape(-1, 1)
  return y


def infer_target_type(y: np.ndarray) -> str:
  """Infer target type from labels: 'probability' if close to binary, else 'severity'."""
  unique = np.unique(np.round(y, 3))
  if unique.size <= 3 and set(np.round(unique, 0).tolist()).issubset({0.0, 1.0}):
    return 'probability'
  return 'severity'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--input', type=str, default=str(Path('data/processed/impact_dataset.csv')))
  ap.add_argument('--output', type=str, default=str(Path('public/models/impact_model')))
  ap.add_argument('--epochs', type=int, default=15)
  ap.add_argument('--batch-size', type=int, default=64)
  ap.add_argument('--model-type', type=str, choices=['shallow', 'deep'], default='deep')
  ap.add_argument('--learning-rate', type=float, default=1e-3)
  ap.add_argument('--l2', type=float, default=1e-4)
  ap.add_argument('--dropout', type=float, default=0.2)
  ap.add_argument('--early-stopping', action='store_true', help='Habilita EarlyStopping con restore_best_weights')
  ap.add_argument('--use-class-weights', action='store_true', help='Calcula class_weight si y es binaria (0/1)')
  ap.add_argument('--target-type', type=str, choices=['auto', 'probability', 'severity'], default='auto', help='Ajusta automáticamente la pérdida y métricas para probabilidad vs severidad')
  ap.add_argument('--kfold', type=int, default=1, help='Número de folds para validación cruzada (>=2 para activar)')
  ap.add_argument('--normalize-in-model', type=str, choices=['on', 'off'], default='on', help='Inserta capa Normalization en el modelo')
  ap.add_argument('--patience', type=int, default=5)
  ap.add_argument('--min-delta', type=float, default=0.0)
  ap.add_argument('--tune-hparams', action='store_true', help='Activa búsqueda simple de hiperparámetros (grid pequeño)')
  ap.add_argument('--tune-epochs', type=int, default=12, help='Épocas por configuración durante la búsqueda de hiperparámetros')
  ap.add_argument('--grid-layers', type=str, default='256,256,128,128,64;256,128,64', help='Capas candidatas separadas por ;, p.ej. "256,256,128;256,128,64"')
  ap.add_argument('--grid-lr', type=str, default='0.001,0.0005,0.0001')
  ap.add_argument('--grid-dropout', type=str, default='0.2,0.1,0.3')
  ap.add_argument('--grid-l2', type=str, default='0.0001,0.00005')
  args = ap.parse_args()

  df = pd.read_csv(args.input)
  # Filter rows with at least some core features
  df = df.copy()
  if df.empty:
    raise RuntimeError('Input dataset is empty.')

  X = normalize_features(df)
  y = select_target(df)
  Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

  # Normalization layer (in-model)
  norm_layer = create_normalizer(Xtr) if args.normalize_in_model == 'on' else None

  # Determinar tipo de objetivo
  if args.target_type == 'auto':
    target_type = infer_target_type(y)
  else:
    target_type = args.target_type

  # Búsqueda de hiperparámetros (opcional)
  best_params = {
    'layers': None,
    'lr': args.learning_rate,
    'dropout': args.dropout,
    'l2': args.l2
  }
  cv_report: Optional[Dict[str, Any]] = None
  if args.tune_hparams:
    layer_grids = parse_layers_spec(args.grid_layers)
    lrs = [float(x) for x in args.grid_lr.split(',') if x.strip()]
    drops = [float(x) for x in args.grid_dropout.split(',') if x.strip()]
    l2s = [float(x) for x in args.grid_l2.split(',') if x.strip()]
    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    print('Iniciando búsqueda de hiperparámetros (grid pequeño)...')
    # Usar train split completo para CV interna
    X_cv, y_cv = Xtr, ytr
    for layers in layer_grids:
      for lr in lrs:
        for dr in drops:
          for l2val in l2s:
            rep = kfold_score(X_cv, y_cv, args, target_type, layers=layers, lr=lr, l2val=l2val, dropout=dr)
            results.append(({ 'layers': layers, 'lr': lr, 'dropout': dr, 'l2': l2val }, rep))
            # Criterio de selección
    def score_of(rep: Dict[str, Any]) -> float:
      if target_type == 'probability':
        return float(rep.get('mean_AUC', 0.0))
      else:
        return -float(rep.get('mean_loss', 1e9))
    best = max(results, key=lambda tup: score_of(tup[1])) if results else None
    if best:
      best_params = best[0]
      cv_report = best[1]
      print('Mejores hiperparámetros encontrados:', best_params)

  # Elegir arquitectura con mejores hiperparámetros si hay
  layers_best = best_params['layers'] if best_params['layers'] is not None else None
  if args.model_type == 'deep':
    model = build_deep_model(X.shape[1], lr=best_params['lr'], l2=best_params['l2'], dropout=best_params['dropout'], norm_layer=norm_layer, layers=layers_best)
  else:
    model = build_shallow_model(X.shape[1], lr=best_params['lr'], norm_layer=norm_layer)

  # Compilar según objetivo
  loss_name, metrics = compile_for_target(model, target_type, lr=best_params['lr'])

  # Callbacks de entrenamiento
  callbacks = []
  if args.early_stopping:
    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, min_delta=args.min_delta, restore_best_weights=True))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1, args.patience//2), min_lr=1e-6))

  # class_weight opcional para etiquetas binarias
  class_weight: Optional[Dict[int, float]] = None
  if args.use_class_weights and target_type == 'probability':
    unique = np.unique(ytr)
    # Considerar binaria si sólo hay 0/1
    if unique.size <= 3 and set(np.round(unique, 0).tolist()).issubset({0.0, 1.0}):
      pos = float((ytr > 0.5).sum())
      neg = float((ytr <= 0.5).sum())
      if pos > 0:
        # balance simple: igualar pesos
        w_pos = neg / pos
        class_weight = {0: 1.0, 1: w_pos}

  # Validación cruzada opcional (sólo informe)
  if args.kfold and args.kfold >= 2:
    print(f"Running {args.kfold}-fold cross-validation (report only)...")
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    scores = []
    fold = 1
    for train_idx, val_idx in kf.split(X):
      X_tr, X_va = X[train_idx], X[val_idx]
      y_tr, y_va = y[train_idx], y[val_idx]
      norm_cv = create_normalizer(X_tr) if args.normalize_in_model == 'on' else None
      mdl = build_deep_model(X.shape[1], lr=args.learning_rate, l2=args.l2, dropout=args.dropout, norm_layer=norm_cv) if args.model_type == 'deep' else build_shallow_model(X.shape[1], lr=args.learning_rate, norm_layer=norm_cv)
      compile_for_target(mdl, target_type, lr=args.learning_rate)
      cw = None
      if args.use_class_weights and target_type == 'probability':
        pos = float((y_tr > 0.5).sum())
        neg = float((y_tr <= 0.5).sum())
        if pos > 0:
          cw = {0: 1.0, 1: neg/pos}
      hist = mdl.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=max(5, min(20, args.epochs)), batch_size=args.batch_size, callbacks=callbacks, class_weight=cw, verbose=0)
      eval_score = mdl.evaluate(X_va, y_va, verbose=0)
      scores.append(eval_score)
      print(f"Fold {fold} eval: {eval_score}")
      fold += 1
    # Promedio
    if scores:
      mean_scores = np.mean(np.stack(scores, axis=0), axis=0)
      print(f"CV mean eval: {mean_scores}")

  # Entrenamiento final
  model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, class_weight=class_weight)
  eval_out = model.evaluate(Xte, yte, verbose=0)
  print('Eval:', eval_out)

  # Métricas y curvas (según target)
  import json
  metrics_out: Dict[str, Any] = {
    'targetType': target_type,
    'eval': [float(x) for x in (eval_out if isinstance(eval_out, (list, tuple, np.ndarray)) else [eval_out])],
    'bestParams': best_params,
    'cvReport': cv_report
  }
  try:
    yhat = model.predict(Xte, verbose=0).reshape(-1)
    ytrue = yte.reshape(-1)
    if target_type == 'probability':
      fpr, tpr, _ = roc_curve((ytrue > 0.5).astype(int), yhat)
      prec, rec, _ = precision_recall_curve((ytrue > 0.5).astype(int), yhat)
      metrics_out['rocAUC'] = float(auc(fpr, tpr))
      metrics_out['prAUC'] = float(average_precision_score((ytrue > 0.5).astype(int), yhat))
      metrics_out['rocCurve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
      metrics_out['prCurve'] = {'precision': prec.tolist(), 'recall': rec.tolist()}
      residuals = (ytrue - yhat)
      metrics_out['residualStd'] = float(np.std(residuals))
      metrics_out['residualSE'] = float(np.std(residuals) / max(1, np.sqrt(len(residuals))))
    else:
      residuals = (ytrue - yhat)
      metrics_out['MAE'] = float(np.mean(np.abs(residuals)))
      metrics_out['RMSE'] = float(np.sqrt(np.mean(residuals**2)))
      metrics_out['residualStd'] = float(np.std(residuals))
      metrics_out['residualSE'] = float(np.std(residuals) / max(1, np.sqrt(len(residuals))))
  except Exception as e:
    print('Warning generating metrics/curves:', e)

  # Calibración opcional (sólo probabilidad)
  calib = None
  if target_type == 'probability':
    try:
      p = model.predict(Xte, verbose=0).reshape(-1, 1)
      # Logistic calibration on probabilities
      lr_clf = LogisticRegression(solver='lbfgs')
      lr_clf.fit(p, (yte > 0.5).ravel().astype(int))
      calib = {
        'type': 'platt',
        'coef': float(lr_clf.coef_.ravel()[0]),
        'intercept': float(lr_clf.intercept_.ravel()[0])
      }
      print('Fitted probability calibration (Platt).')
    except Exception as e:
      print('Calibration skipped due to error:', e)

  outdir = Path(args.output)
  outdir.mkdir(parents=True, exist_ok=True)
  # Save Keras formats for external conversion with Node CLI
  keras_dir = outdir.parent / 'impact_model_keras'
  keras_dir.mkdir(parents=True, exist_ok=True)
  # Save metadata
  metadata = {
    'targetType': target_type,
    'modelType': args.model_type,
    'normalizeInModel': (args.normalize_in_model == 'on'),
    'featureOrder': ['moidAU','missDistanceKm','relativeVelocityKmS','diameterKm','hMagnitude','inclinationDeg','daysToCloseApproach']
  }
  with open(keras_dir / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
  with open(keras_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_out, f, indent=2)
  # SavedModel (TF)
  model.save(keras_dir / 'saved_model')
  # H5 for converter convenience
  model.save(keras_dir / 'model.h5')
  # Save calibration (if any)
  if calib is not None:
    with open(keras_dir / 'calibration.json', 'w', encoding='utf-8') as f:
      json.dump(calib, f)
  # Also write metadata/calibration to TFJS folder (for local dev); CI conservará estos archivos junto al modelo convertido
  try:
    with open(outdir / 'metadata.json', 'w', encoding='utf-8') as f:
      json.dump(metadata, f, indent=2)
    with open(outdir / 'metrics.json', 'w', encoding='utf-8') as f:
      json.dump(metrics_out, f, indent=2)
    if calib is not None:
      with open(outdir / 'calibration.json', 'w', encoding='utf-8') as f:
        json.dump(calib, f)
  except Exception as e:
    print('Warning: could not write metadata/calibration to TFJS folder:', e)
  print(f'Saved Keras model to {keras_dir} (SavedModel + model.h5). Use the Node CLI to convert to TFJS.')


if __name__ == '__main__':
  main()
