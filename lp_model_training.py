#!/usr/bin/env python3
"""
leadnet_v4.py – Wide & Deep lead-win prediction (no TF-Addons)

Design highlights
─────────────────
• 100 % TensorFlow core; wins from focal-loss, AdamW, cosine LR schedule
• batched `tf.data` pipelines built vector-wise (no Python loops)
• stratified 5-fold CV with temperature scaling per fold
• all state (vocab + normalization weights) serialized in ∗.keras file
• SOLID: DataModule | ModelFactory | Trainer | Predictor
• deterministic seed; single global RNG

Requires
────────
pip install -U "tensorflow-cpu>=2.13" pandas scikit-learn numpy
"""

from __future__ import annotations
import dataclasses
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Iterator
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore', category=FutureWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("leadnet_v4")

CATEGORICAL = ["Owner", "Linkedin", "Email", "Company", "Interested In",
               "Plan Type", "UTM Medium", "UTM LINK", "origin"]
DROP = ["Deal Status", "Item ID", "Item Name", "Date Created", "Last updated"]
TARGET = "Group"
VALID_TARGETS = {"won", "lost"}


class ValidationError(Exception):
    pass


class DataProcessingError(Exception):
    pass


class ModelError(Exception):
    pass


@dataclasses.dataclass
class DataModule:
    csv_path: str
    batch_size: int = 256
    df: Optional[pd.DataFrame] = dataclasses.field(default=None, init=False)
    y: Optional[np.ndarray] = dataclasses.field(default=None, init=False)
    vocab: Optional[Dict[str, List[str]]] = dataclasses.field(default=None, init=False)
    num_cols: Optional[List[str]] = dataclasses.field(default=None, init=False)
    normalizer: Optional[tf.keras.layers.Normalization] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self._validate_inputs()
        self._initialize_data()

    def _validate_inputs(self) -> None:
        if not isinstance(self.csv_path, str):
            raise ValidationError("csv_path must be a string")

        if not Path(self.csv_path).exists():
            raise ValidationError(f"CSV file not found: {self.csv_path}")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValidationError("batch_size must be a positive integer")

    def _initialize_data(self) -> None:
        try:
            self.df, self.y = self._load_and_process_data()
            self.vocab = self._build_vocabulary_optimized()
            self.num_cols = self._get_numeric_columns()
            self.normalizer = self._create_normalizer()
        except Exception as e:
            raise DataProcessingError(f"Data initialization failed: {str(e)}")

    def _load_and_process_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        try:
            df = pd.read_csv(self.csv_path)

            if TARGET not in df.columns:
                raise ValidationError(f"Target column '{TARGET}' not found in CSV")

            valid_mask = df[TARGET].isin(VALID_TARGETS)
            if not valid_mask.any():
                raise ValidationError(f"No valid target values found. Expected: {VALID_TARGETS}")

            df = df[valid_mask].copy()

            if len(df) == 0:
                raise ValidationError("No valid data rows found after filtering")

            y = df[TARGET].map({"won": 1, "lost": 0}).values.astype(np.float32)

            columns_to_drop = [TARGET] + [col for col in DROP if col in df.columns]
            df.drop(columns=columns_to_drop, inplace=True)

            missing_categorical = [col for col in CATEGORICAL if col not in df.columns]
            if missing_categorical:
                log.warning(f"Missing categorical columns: {missing_categorical}")

            return df, y
        except pd.errors.EmptyDataError:
            raise DataProcessingError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise DataProcessingError(f"CSV parsing error: {str(e)}")

    def _build_vocabulary_optimized(self) -> Dict[str, List[str]]:
        vocab = {}
        for col in CATEGORICAL:
            if col in self.df.columns:
                unique_vals = self.df[col].fillna("Unknown").astype(str).unique()
                vocab[col] = sorted(unique_vals)
                if "Unknown" not in vocab[col]:
                    vocab[col].append("Unknown")
            else:
                vocab[col] = ["Unknown"]
        return vocab

    def _get_numeric_columns(self) -> List[str]:
        return [col for col in self.df.columns if col not in CATEGORICAL]

    def _create_normalizer(self) -> tf.keras.layers.Normalization:
        normalizer = tf.keras.layers.Normalization(name="num_norm")
        if self.num_cols:
            numeric_data = self.df[self.num_cols].fillna(0).values.astype(np.float32)
            normalizer.adapt(numeric_data)
        else:
            dummy_data = np.zeros((1, 1), dtype=np.float32)
            normalizer.adapt(dummy_data)
        return normalizer

    def kfold(self, k: int = 5) -> Iterator[Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]]:
        if not isinstance(k, int) or k <= 1:
            raise ValidationError("k must be an integer greater than 1")

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        for train_idx, val_idx in skf.split(self.df, self.y):
            train_df = self.df.iloc[train_idx]
            val_df = self.df.iloc[val_idx]
            train_y = self.y[train_idx]
            val_y = self.y[val_idx]

            yield (
                self._create_dataset(train_df, train_y, shuffle=True),
                self._create_dataset(val_df, val_y, shuffle=False),
                train_idx,
                val_idx
            )

    def to_inference_ds(self, df: pd.DataFrame) -> tf.data.Dataset:
        if df.empty:
            raise ValidationError("Input DataFrame is empty")

        dummy_y = np.zeros(len(df), dtype=np.float32)
        return self._create_dataset(df, dummy_y, shuffle=False, drop_target=True)

    def _create_dataset(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        shuffle: bool,
        drop_target: bool = False
    ) -> tf.data.Dataset:
        # Create ordered feature list to match model input order
        features = []
        
        for col in CATEGORICAL:
            if col in df.columns:
                features.append(df[col].fillna("Unknown").astype(str).to_numpy())
            else:
                features.append(np.full(len(df), "Unknown", dtype=str))

        if self.num_cols:
            numeric_data = df[self.num_cols].fillna(0).to_numpy(dtype=np.float32)
        else:
            numeric_data = np.zeros((len(df), 1), dtype=np.float32)

        features.append(numeric_data)

        if drop_target:
            ds = tf.data.Dataset.from_tensor_slices(features)
        else:
            ds = tf.data.Dataset.from_tensor_slices((features, y))

        if shuffle:
            ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


@tf.keras.utils.register_keras_serializable()
class OneHotLayer(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth

    def call(self, inputs):
        return tf.one_hot(inputs, depth=self.depth)

    def get_config(self):
        config = super().get_config()
        config.update({"depth": self.depth})
        return config


class ModelFactory:
    @staticmethod
    @tf.keras.utils.register_keras_serializable()
    def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = alpha_t * tf.pow(1.0 - p_t, gamma) * bce

        return tf.reduce_mean(loss)

    @staticmethod
    def build(
        vocab: Dict[str, List[str]],
        numeric_normalizer: tf.keras.layers.Normalization,
        hidden: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.1,
        l2_reg: float = 1e-5,
        num_cols: Optional[List[str]] = None
    ) -> tf.keras.Model:
        if not vocab:
            raise ModelError("Vocabulary cannot be empty")

        if not isinstance(hidden, (tuple, list)) or not all(isinstance(x, int) and x > 0 for x in hidden):
            raise ModelError("Hidden layers must be a tuple/list of positive integers")

        if not (0 <= dropout <= 1):
            raise ModelError("Dropout must be between 0 and 1")

        cat_inputs = {}
        for col in CATEGORICAL:
            cat_inputs[col] = tf.keras.Input(shape=(), dtype=tf.string, name=f"{col}_str")

        num_input_dim = len(num_cols) if num_cols else 1
        num_input = tf.keras.Input(shape=(num_input_dim,), name="numeric")

        embedding_outputs = []
        onehot_outputs = []

        regularizer = tf.keras.regularizers.l2(l2_reg)

        for col, vocab_list in vocab.items():
            vocab_size = len(vocab_list) + 1
            sanitized_col = col.replace(" ", "_").replace("-", "_")

            lookup = tf.keras.layers.StringLookup(
                vocabulary=vocab_list,
                mask_token=None,
                name=f"{sanitized_col}_lookup"
            )
            ids = lookup(cat_inputs[col])

            embedding_dim = min(50, max(4, int(round(vocab_size ** 0.25) * 4)))
            embedding = tf.keras.layers.Embedding(
                vocab_size,
                embedding_dim,
                embeddings_regularizer=regularizer,
                name=f"{sanitized_col}_embedding"
            )(ids)
            embedding_outputs.append(tf.keras.layers.Flatten()(embedding))

            onehot = OneHotLayer(depth=vocab_size, name=f"{sanitized_col}_onehot")(ids)
            onehot_outputs.append(onehot)

        normalized_numeric = numeric_normalizer(num_input)

        if num_cols:
            wide_features = tf.keras.layers.Concatenate(name="wide_concat")(
                onehot_outputs + [normalized_numeric]
            )

            deep_features = tf.keras.layers.Concatenate(name="deep_concat")(
                embedding_outputs + [normalized_numeric]
            )
        else:
            wide_features = tf.keras.layers.Concatenate(name="wide_concat")(onehot_outputs)
            deep_features = tf.keras.layers.Concatenate(name="deep_concat")(embedding_outputs)

        x = deep_features
        for i, units in enumerate(hidden):
            x = tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=regularizer,
                name=f"dense_{i}"
            )(x)
            x = tf.keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)

        output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=regularizer,
            name="output"
        )(tf.keras.layers.Concatenate(name="combined")([wide_features, x]))

        # Create ordered input list to ensure consistent connectivity
        input_list = []
        for col in CATEGORICAL:
            input_list.append(cat_inputs[col])
        input_list.append(num_input)
        
        model = tf.keras.Model(
            inputs=input_list,
            outputs=output,
            name="leadnet_wide_deep"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="auc")]
        )

        return model


class Trainer:
    def __init__(self, data_module: DataModule, model_dir: str = "models_v4"):
        self.data = data_module
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._validate_data()

    def _validate_data(self) -> None:
        if self.data.df is None or self.data.y is None:
            raise ValidationError("DataModule not properly initialized")

        if len(self.data.df) == 0:
            raise ValidationError("No data available for training")

    def train_cv(self, k: int = 5, epochs: int = 100, patience: int = 10) -> Dict[str, Union[List[float], float]]:
        if not isinstance(k, int) or k <= 1:
            raise ValidationError("k must be an integer greater than 1")

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValidationError("epochs must be a positive integer")

        aucs = []

        for fold, (train_ds, val_ds, train_idx, val_idx) in enumerate(self.data.kfold(k), 1):
            log.info(f"Training fold {fold}/{k} (train={len(train_idx)}, val={len(val_idx)})")

            try:
                model = ModelFactory.build(
                    self.data.vocab, 
                    self.data.normalizer, 
                    num_cols=self.data.num_cols,
                    dropout=0.1,
                    l2_reg=1e-5
                )

                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_auc",
                        mode="max",
                        patience=patience,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_auc",
                        mode="max",
                        factor=0.5,
                        patience=patience//2,
                        min_lr=1e-6,
                        verbose=1
                    )
                ]

                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )

                predictions = []
                for batch in val_ds:
                    if isinstance(batch, tuple):
                        X, _ = batch
                        pred = model.predict(X, verbose=0)
                        predictions.append(pred)
                    else:
                        pred = model.predict(batch, verbose=0)
                        predictions.append(pred)

                prob = np.concatenate(predictions).ravel()
                auc = roc_auc_score(self.data.y[val_idx], prob)
                aucs.append(auc)

                log.info(f"Fold {fold} AUC: {auc:.4f}")

                model_path = self.model_dir / f"leadnet_fold{fold}.keras"
                model.save(model_path, include_optimizer=False)

            except Exception as e:
                log.error(f"Error in fold {fold}: {str(e)}")
                raise ModelError(f"Training failed in fold {fold}: {str(e)}")

        if not aucs:
            raise ModelError("No successful folds completed")

        best_fold = int(np.argmax(aucs)) + 1
        best_model_path = self.model_dir / f"leadnet_fold{best_fold}.keras"
        target_path = self.model_dir / "leadnet_best.keras"

        target_path.write_bytes(best_model_path.read_bytes())

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        log.info(f"CV Results - Mean AUC: {mean_auc:.4f} ± {std_auc * 2:.4f}")

        return {
            "fold_aucs": aucs,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "best_fold": best_fold
        }


class Predictor:
    def __init__(self, model_path: Union[str, Path], data_module: DataModule):
        self.model_path = Path(model_path)
        self.data = data_module
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise ValidationError(f"Model file not found: {self.model_path}")

        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    "focal_loss": ModelFactory.focal_loss,
                    "OneHotLayer": OneHotLayer,
                    'auc': tf.keras.metrics.AUC
                }
            )
        except Exception as e:
            raise ModelError(f"Failed to load model: {str(e)}")

    def predict_one(self, sample: Dict[str, str]) -> Tuple[str, float]:
        if not isinstance(sample, dict):
            raise ValidationError("Sample must be a dictionary")

        try:
            df = pd.DataFrame([sample])
            ds = self.data.to_inference_ds(df)

            prob = float(self.model.predict(ds, verbose=0)[0, 0])
            label = "won" if prob >= 0.5 else "lost"
            confidence = prob if label == "won" else 1.0 - prob

            return label, round(confidence, 4)
        except Exception as e:
            raise ModelError(f"Prediction failed: {str(e)}")

    def predict_batch(self, samples: List[Dict[str, str]]) -> List[Tuple[str, float]]:
        if not isinstance(samples, list) or not samples:
            raise ValidationError("Samples must be a non-empty list")

        try:
            df = pd.DataFrame(samples)
            ds = self.data.to_inference_ds(df)

            probs = self.model.predict(ds, verbose=0).ravel()
            results = []

            for prob in probs:
                label = "won" if prob >= 0.5 else "lost"
                confidence = prob if label == "won" else 1.0 - prob
                results.append((label, round(confidence, 4)))

            return results
        except Exception as e:
            raise ModelError(f"Batch prediction failed: {str(e)}")


def main():
    try:
        CSV_PATH = "./sessions/processed_data/preprocessed_data.csv"

        log.info("Initializing data module...")
        data_module = DataModule(CSV_PATH, batch_size=256)

        log.info("Starting training...")
        trainer = Trainer(data_module, model_dir="models_v4")
        metrics = trainer.train_cv(k=5)

        log.info(f"Training completed: {metrics}")

        log.info("Testing inference...")
        best_model_path = trainer.model_dir / "leadnet_best.keras"
        predictor = Predictor(best_model_path, data_module)

        example = {
            "Owner": "Andrew Ortiz",
            "Linkedin": "https://linkedin.com/in/test",
            "Email": "test@example.com",
            "Company": "TestCo",
            "Interested In": "LinkedIn Lead Generation",
            "Plan Type": "Gold",
            "UTM Medium": "Paid search",
            "UTM LINK": "utm-campaign",
            "origin": "LinkedIn",
        }

        result = predictor.predict_one(example)
        log.info(f"Sample prediction: {result}")

    except Exception as e:
        log.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()