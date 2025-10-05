"""
Train a local damage prediction model (aligned with literature) with options:
- Regression: multi-output radii (e.g., R_1kPa, R_3kPa, R_20kPa)
- Classification: damage category (multi-class)
- Multi-task: shared trunk with regression + classification heads

Exports Keras SavedModel + H5 and metadata/metrics JSONs.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
  from tensorflow import keras  # type: ignore
except Exception:
  import keras  # type: ignore

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, roc_auc_score, average_precision_score


def parse_layers(spec: str) -> List[int]:
  return [int(x) for x in spec.split(',') if x.strip()]


def build_trunk(input_dim: int, layers: List[int], dropout: float, l2val: float, norm: bool = True) -> Tuple[keras.Model, keras.layers.Layer]:
  reg = keras.regularizers.l2(l2val) if l2val and l2val > 0 else None
  inp = keras.Input(shape=(input_dim,), dtype='float32')
  x = inp
  if norm:
    norm_layer = keras.layers.Normalization(axis=-1)
  else:
    norm_layer = None
  if norm_layer is not None:
    x = norm_layer(x)
  for units in layers:
    x = keras.layers.Dense(units, kernel_initializer='he_normal', kernel_regularizer=reg, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout)(x)
  trunk = keras.Model(inp, x, name='trunk')
  return trunk, norm_layer


def build_model(input_dim: int, task: str, reg_outputs: int = 0, n_classes: int = 0, layers: List[int] = [256,256,128,128,64], dropout: float = 0.2, l2val: float = 1e-4, lr: float = 1e-3, norm: bool = True) -> Tuple[keras.Model, Dict[str, Any]]:
  trunk, norm_layer = build_trunk(input_dim, layers, dropout, l2val, norm=norm)
  x = trunk.output
  outputs = []
  losses: Dict[str, str] = {}
  metrics: Dict[str, List[Any]] = {}
  meta: Dict[str, Any] = {}

  if task in ('regression', 'multi') and reg_outputs > 0:
    y_reg = keras.layers.Dense(reg_outputs, name='y_reg')(x)
    outputs.append(y_reg)
    losses['y_reg'] = 'mse'
    metrics['y_reg'] = [keras.metrics.MeanAbsoluteError(name='MAE'), keras.metrics.RootMeanSquaredError(name='RMSE')]
    meta['regression'] = {'outputs': reg_outputs}

  if task in ('classification', 'multi') and n_classes > 0:
    y_clf = keras.layers.Dense(n_classes, activation='softmax', name='y_clf')(x)
    outputs.append(y_clf)
    losses['y_clf'] = 'sparse_categorical_crossentropy'
    metrics['y_clf'] = [keras.metrics.SparseCategoricalAccuracy(name='Acc')]
    meta['classification'] = {'classes': n_classes}

  if not outputs:
    raise ValueError('No outputs configured. Use --reg-outputs and/or --classes according to --task')

  model = keras.Model(trunk.input, outputs)
  model.compile(optimizer=keras.optimizers.Adam(lr), loss=losses, metrics=metrics)
  meta['layers'] = layers
  meta['dropout'] = dropout
  meta['l2'] = l2val
  meta['learningRate'] = lr
  meta['normalizeInModel'] = norm
  return model, meta


def make_regression_loss(name: str, reg_transform: str):
  # Options: mse, mae, mae_log (identical to mae when training on log targets), mape (on original scale if transform was applied)
  if name == 'mse':
    return 'mse'
  if name == 'mae':
    return 'mae'
  if name == 'mae_log':
    # When training on log10/log1p targets, this is equivalent to MAE in transformed space
    def loss(y_true, y_pred):
      return keras.backend.mean(keras.backend.abs(y_true - y_pred))
    return loss
  if name == 'mape':
    if reg_transform == 'log10':
      def loss(y_true, y_pred):
        y_t = keras.backend.maximum(keras.backend.pow(10.0, y_true), 1e-12)
        y_p = keras.backend.maximum(keras.backend.pow(10.0, y_pred), 1e-12)
        return keras.backend.mean(keras.backend.abs((y_t - y_p) / y_t))
      return loss
    if reg_transform == 'log1p':
      def loss(y_true, y_pred):
        y_t = keras.backend.maximum(keras.backend.expm1(y_true), 1e-12)
        y_p = keras.backend.maximum(keras.backend.expm1(y_pred), 1e-12)
        return keras.backend.mean(keras.backend.abs((y_t - y_p) / y_t))
      return loss
    # identity
    return 'mape'
  # default
  return 'mse'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--input', type=str, required=True, help='CSV con features y targets')
  ap.add_argument('--output', type=str, default=str(Path('public/models/damage_model')))
  ap.add_argument('--features', type=str, required=True, help='Columnas de entrada separadas por coma, en orden')
  ap.add_argument('--task', type=str, choices=['regression','classification','multi'], required=True)
  ap.add_argument('--reg-columns', type=str, default='', help='Columnas de salida de regresión (e.g., R_1kPa,R_3kPa,R_20kPa)')
  ap.add_argument('--class-column', type=str, default='', help='Columna de clase (enteros [0..K-1])')
  ap.add_argument('--classes', type=int, default=0, help='Número de clases (si no se puede inferir)')
  ap.add_argument('--layers', type=str, default='256,256,128,128,64')
  ap.add_argument('--dropout', type=float, default=0.2)
  ap.add_argument('--l2', type=float, default=1e-4)
  ap.add_argument('--learning-rate', type=float, default=1e-3)
  ap.add_argument('--epochs', type=int, default=60)
  ap.add_argument('--batch-size', type=int, default=256)
  ap.add_argument('--patience', type=int, default=6)
  ap.add_argument('--min-delta', type=float, default=1e-5)
  ap.add_argument('--kfold', type=int, default=0)
  ap.add_argument('--reg-transform', type=str, choices=['identity','log10','log1p'], default='log10', help='Transformación a aplicar sobre radios/targets de regresión')
  ap.add_argument('--epsilon', type=float, default=1e-6, help='Épsilon para evitar log(0) en log10')
  ap.add_argument('--use-class-weights', action='store_true', help='Pesa clases si la clasificación está desbalanceada')
  ap.add_argument('--reg-loss', type=str, choices=['mse','mae','mae_log','mape'], default='mae_log', help='Pérdida de regresión; mae_log y mape favorecen errores relativos ~10%')
  args = ap.parse_args()

  df = pd.read_csv(args.input)
  feat_cols = [c.strip() for c in args.features.split(',') if c.strip()]
  if not feat_cols:
    raise RuntimeError('Debes especificar --features con columnas de entrada')
  X = df[feat_cols].astype('float32').values

  y_reg = None
  y_clf = None
  reg_cols = [c.strip() for c in args.reg_columns.split(',') if c.strip()]
  if args.task in ('regression','multi') and reg_cols:
    y_reg_raw = df[reg_cols].astype('float32').values
    # Transform targets if requested
    if args.reg_transform == 'log10':
      y_reg = np.log10(np.maximum(y_reg_raw, args.epsilon)).astype('float32')
    elif args.reg_transform == 'log1p':
      y_reg = np.log1p(y_reg_raw).astype('float32')
    else:
      y_reg = y_reg_raw

  if args.task in ('classification','multi') and args.class_column:
    y_clf = df[args.class_column].astype('int64').values
    n_classes = int(df[args.class_column].max()) + 1 if args.classes <= 0 else args.classes
  else:
    n_classes = 0

  Xtr, Xte = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
  X_train, X_test = X[Xtr], X[Xte]

  data_train: Dict[str, Any] = {}
  data_test: Dict[str, Any] = {}
  if y_reg is not None:
    data_train['y_reg'] = y_reg[Xtr]
    data_test['y_reg'] = y_reg[Xte]
  if y_clf is not None:
    data_train['y_clf'] = y_clf[Xtr]
    data_test['y_clf'] = y_clf[Xte]

  model, meta = build_model(
    input_dim=X.shape[1],
    task=args.task,
    reg_outputs=(y_reg.shape[1] if y_reg is not None else 0),
    n_classes=n_classes,
    layers=parse_layers(args.layers),
    dropout=args.dropout,
    l2val=args.l2,
    lr=args.learning_rate,
    norm=True,
  )
  meta['regTransform'] = args.reg_transform
  meta['epsilon'] = args.epsilon
  # Override regression loss if applicable
  if args.task in ('regression','multi') and (y_reg is not None):
    reg_loss = make_regression_loss(args.reg_loss, args.reg_transform)
    # Re-compile to update regression loss
    losses = model.loss
    if isinstance(losses, dict) and 'y_reg' in losses:
      losses['y_reg'] = reg_loss
    else:
      losses = reg_loss
    model.compile(optimizer=model.optimizer, loss=losses, metrics=model.metrics)

  callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, min_delta=args.min_delta, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1, args.patience//2), min_lr=1e-6)
  ]

  class_weight = None
  if args.use_class_weights and args.task in ('classification','multi') and y_clf is not None:
    # simple inverse frequency weights
    unique, counts = np.unique(y_clf[Xtr], return_counts=True)
    total = counts.sum()
    cw = {int(k): float(total/(len(unique)*c)) for k, c in zip(unique, counts)}
    class_weight = {'y_clf': cw} if args.task == 'multi' else cw

  model.fit(X_train, data_train, validation_data=(X_test, data_test), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, class_weight=class_weight)

  # Eval and metrics
  eval_out = model.evaluate(X_test, data_test, verbose=0)
  metrics_out: Dict[str, Any] = {'eval': [float(x) for x in (eval_out if isinstance(eval_out, (list, tuple, np.ndarray)) else [eval_out])], 'task': args.task}
  # Additional: MAPE for regression, Accuracy for classification
  try:
    yhat = model.predict(X_test, verbose=0)
    if args.task == 'regression':
      yh = yhat if isinstance(yhat, np.ndarray) else yhat[0]
      # invert transform for metrics in original scale
      if args.reg_transform == 'log10':
        yh_inv = np.maximum(10**yh, 0.0)
        yt_inv = np.maximum(10**data_test['y_reg'], 0.0)
      elif args.reg_transform == 'log1p':
        yh_inv = np.expm1(yh)
        yt_inv = np.expm1(data_test['y_reg'])
      else:
        yh_inv = yh
        yt_inv = data_test['y_reg']
      mape = float(mean_absolute_percentage_error(yt_inv, yh_inv))
      metrics_out['MAPE'] = mape
    elif args.task == 'classification':
      yh = yhat if isinstance(yhat, np.ndarray) else yhat[0]
      pred = np.argmax(yh, axis=1)
      acc = float(accuracy_score(data_test['y_clf'], pred))
      metrics_out['Acc'] = acc
      # Multiclass ROC-AUC (ovr, macro) and PR-AUC (macro)
      try:
        metrics_out['rocAUC_macro'] = float(roc_auc_score(data_test['y_clf'], yh, multi_class='ovr', average='macro'))
        metrics_out['prAUC_macro'] = float(average_precision_score(pd.get_dummies(data_test['y_clf']).values, yh, average='macro'))
      except Exception:
        pass
    elif args.task == 'multi':
      yh_reg, yh_clf = yhat
      if args.reg_transform == 'log10':
        yh_inv = np.maximum(10**yh_reg, 0.0)
        yt_inv = np.maximum(10**data_test['y_reg'], 0.0)
      elif args.reg_transform == 'log1p':
        yh_inv = np.expm1(yh_reg)
        yt_inv = np.expm1(data_test['y_reg'])
      else:
        yh_inv = yh_reg
        yt_inv = data_test['y_reg']
      mape = float(mean_absolute_percentage_error(yt_inv, yh_inv))
      pred = np.argmax(yh_clf, axis=1)
      acc = float(accuracy_score(data_test['y_clf'], pred))
      metrics_out['MAPE'] = mape
      metrics_out['Acc'] = acc
      try:
        metrics_out['rocAUC_macro'] = float(roc_auc_score(data_test['y_clf'], yh_clf, multi_class='ovr', average='macro'))
        metrics_out['prAUC_macro'] = float(average_precision_score(pd.get_dummies(data_test['y_clf']).values, yh_clf, average='macro'))
      except Exception:
        pass
  except Exception as e:
    print('Metrics warning:', e)

  # Optional K-Fold reporting
  if args.kfold and args.kfold > 1:
    print(f"Running {args.kfold}-fold CV report for sanity...")
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    fold_scores: List[Dict[str, float]] = []
    for tr_idx, va_idx in kf.split(X):
      X_tr, X_va = X[tr_idx], X[va_idx]
      data_tr: Dict[str, Any] = {}
      data_va: Dict[str, Any] = {}
      if y_reg is not None:
        data_tr['y_reg'] = y_reg[tr_idx]
        data_va['y_reg'] = y_reg[va_idx]
      if y_clf is not None:
        data_tr['y_clf'] = y_clf[tr_idx]
        data_va['y_clf'] = y_clf[va_idx]
      mdl, _ = build_model(input_dim=X.shape[1], task=args.task, reg_outputs=(y_reg.shape[1] if y_reg is not None else 0), n_classes=n_classes, layers=parse_layers(args.layers), dropout=args.dropout, l2val=args.l2, lr=args.learning_rate, norm=True)
      cb = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=max(3, args.patience//2), restore_best_weights=True)
      ]
      mdl.fit(X_tr, data_tr, validation_data=(X_va, data_va), epochs=min(20, args.epochs), batch_size=args.batch_size, callbacks=cb, verbose=0)
      ev = mdl.evaluate(X_va, data_va, verbose=0)
      score = {'val_loss': float(ev[0] if isinstance(ev, (list, tuple, np.ndarray)) else ev)}
      fold_scores.append(score)
    metrics_out['cvFolds'] = fold_scores

  outdir = Path(args.output)
  outdir.mkdir(parents=True, exist_ok=True)
  keras_dir = outdir.parent / 'damage_model_keras'
  keras_dir.mkdir(parents=True, exist_ok=True)

  # Metadata
  import json
  metadata = {
    'task': args.task,
    'features': [*feat_cols],
    'regressionOutputs': reg_cols if reg_cols else None,
    'classColumn': args.class_column or None,
    'classes': n_classes,
    'model': meta,
  }
  with open(keras_dir / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
  with open(keras_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_out, f, indent=2)

  # Save model
  model.save(keras_dir / 'saved_model')
  model.save(keras_dir / 'model.h5')
  # Copy JSONs to TFJS folder for local use
  try:
    with open(outdir / 'metadata.json', 'w', encoding='utf-8') as f:
      json.dump(metadata, f, indent=2)
    with open(outdir / 'metrics.json', 'w', encoding='utf-8') as f:
      json.dump(metrics_out, f, indent=2)
  except Exception as e:
    print('Warning: could not write TFJS metadata/metrics:', e)

  print(f'Saved Keras model to {keras_dir} (SavedModel + model.h5). Use CI converter to produce TFJS at {outdir}.')


if __name__ == '__main__':
  main()
