"""
Demo: entrenamiento de un modelo simple de probabilidad de impacto.
IMPORTANTE: Este dataset es ficticio. Sustituir por datos reales (JPL/NASA) con features consistentes con src/ml/impactModel.ts.

Requisitos (instalar en un entorno Python 3.9+):
  pip install tensorflow tensorflowjs numpy scikit-learn

Salida:
  ./model_tfjs/model.json y shards .bin que podrás copiar a public/models/impact_model/
"""

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflowjs as tfjs


def gen_synthetic(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    # Features: [moidAU, missKm, relV, diamKm, Hnorm, inc, days]
    moidAU = rng.uniform(0.0, 0.5, size=(n, 1))
    missKm = rng.uniform(1e4, 5e6, size=(n, 1))
    relV = rng.uniform(5, 40, size=(n, 1))
    diam = rng.uniform(0.01, 1.0, size=(n, 1))
    H = rng.uniform(10, 30, size=(n, 1))
    inc = rng.uniform(0, 60, size=(n, 1))
    days = rng.integers(-365, 365, size=(n, 1))

    # Normalize similar to frontend
    moidN = np.clip(moidAU / 0.5, 0, 1)
    missN = np.clip(missKm / 5_000_000.0, 0, 1)
    relVN = np.clip(relV / 40.0, 0, 1)
    diamN = np.clip(diam / 1.0, 0, 1)
    HN = np.clip((35 - H) / 20.0, 0, 1)
    incN = np.clip(inc / 60.0, 0, 1)
    daysN = np.clip(1 - np.tanh(np.abs(days) / 365.0), 0, 1)

    X = np.concatenate([moidN, missN, relVN, diamN, HN, incN, daysN], axis=1).astype('float32')

    # Synthetic label via logistic of weighted sum (to match heuristic trend)
    z = 3.0*(1-moidN) + 3.0*(1-missN) + 1.0*relVN + 2.0*diamN + 1.5*HN + 0.5*(1-incN) + 1.5*daysN - 4.0
    p = 1/(1+np.exp(-z))
    y = (rng.uniform(size=(n, 1)) < p).astype('float32')
    return X, y


def build_model(input_dim=7):
    inp = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = tf.keras.layers.Dense(32, activation='relu')(inp)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['AUC'])
    return model


def main():
    X, y = gen_synthetic()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=123)
    model = build_model(X.shape[1])
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=10, batch_size=64)
    print('Eval:', model.evaluate(Xte, yte, verbose=0))
    # Save Keras then convert to TFJS
    model.save('impact_model_keras')
    tfjs.converters.save_keras_model(model, 'model_tfjs')
    print('Exported TFJS model to ./model_tfjs')


if __name__ == '__main__':
    main()
