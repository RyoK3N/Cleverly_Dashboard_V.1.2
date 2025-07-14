import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing_pipe import DataPreprocessor

class TrainingPipeline:
    def __init__(self, model_path: str = 'model.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.label_encoders = {}
        self.feature_columns = None
        self.is_trained = False
        self.training_metrics = {}

    def _prepare_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
        df_features = df.copy()

        categorical_columns = ['Owner', 'Linkedin', 'Email', 'Company', 'Interested In', 
                             'Plan Type', 'UTM Medium', 'UTM LINK', 'origin']

        for col in categorical_columns:
            if col in df_features.columns:
                df_features[col] = df_features[col].fillna('Unknown').astype(str)
                if fit_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_features[col] = self.label_encoders[col].fit_transform(df_features[col])
                else:
                    if col in self.label_encoders:
                        unique_values = set(df_features[col])
                        known_values = set(self.label_encoders[col].classes_)
                        unknown_values = unique_values - known_values

                        if unknown_values:
                            unknown_class = 'Unknown'
                            if unknown_class not in known_values:
                                missing_classes = list(unknown_values) + [unknown_class]
                                extended_classes = np.concatenate([self.label_encoders[col].classes_, missing_classes])
                                self.label_encoders[col].classes_ = np.unique(extended_classes)

                            for unknown_val in unknown_values:
                                df_features[col] = df_features[col].replace(unknown_val, unknown_class)

                        df_features[col] = self.label_encoders[col].transform(df_features[col])
                    else:
                        df_features[col] = 0

        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)

        non_feature_cols = ['Group', 'Deal Status', 'Item ID', 'Item Name', 'Date Created', 'Last updated']
        for col in non_feature_cols:
            if col in df_features.columns:
                df_features = df_features.drop(col, axis=1)

        if fit_encoders:
            self.feature_columns = df_features.columns.tolist()

        return df_features

    def _build_model_pipeline(self, X: pd.DataFrame) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=min(10, X.shape[1]))),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))
        ])

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        print(f"Training data shape: {df.shape}")
        print(f"Group column values: {df['Group'].value_counts()}")

        X = self._prepare_features(df, fit_encoders=True)
        y = df['Group'].map({'won': 1, 'lost': 0})

        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts()}")

        if X.select_dtypes(include=['object']).shape[1] > 0:
            print("Warning: Found object columns after preprocessing:")
            print(X.select_dtypes(include=['object']).columns.tolist())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.pipeline = self._build_model_pipeline(X)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        print("Training model pipeline...")
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=skf, scoring='roc_auc')

        train_pred_proba = self.pipeline.predict_proba(X_train)[:, 1]
        train_score = roc_auc_score(y_train, train_pred_proba)
        test_score = roc_auc_score(y_test, y_pred_proba)
        overfitting_check = train_score - test_score

        self.training_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': test_score,
            'cv_mean_roc_auc': cv_scores.mean(),
            'cv_std_roc_auc': cv_scores.std(),
            'train_roc_auc': train_score,
            'overfitting_gap': overfitting_check,
            'n_features_used': X.shape[1],
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        self._log_training_results()

        self.is_trained = True
        self.save_model()

        return self.training_metrics

    def _log_training_results(self):
        metrics = self.training_metrics
        print(f"Model Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Train ROC-AUC: {metrics['train_roc_auc']:.4f}")
        print(f"Overfitting Gap: {metrics['overfitting_gap']:.4f}")
        print(f"Cross-validation ROC-AUC: {metrics['cv_mean_roc_auc']:.4f} (+/- {metrics['cv_std_roc_auc']*2:.4f})")
        print(f"Features Used: {metrics['n_features_used']}")

        if metrics['overfitting_gap'] > 0.1:
            print("WARNING: Potential overfitting detected!")
        else:
            print("Model appears to generalize well.")

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained and not self.load_model():
            raise ValueError("Model must be trained or loaded before making predictions")

        X = self._prepare_features(df, fit_encoders=False)

        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.feature_columns)

            for col in missing_cols:
                X[col] = 0

            X = X[self.feature_columns]

        predictions = self.pipeline.predict(X)
        prediction_probabilities = self.pipeline.predict_proba(X)[:, 1]

        return predictions, prediction_probabilities

    def predict_single(self, sample: Dict[str, Any]) -> Tuple[str, float]:
        df_sample = pd.DataFrame([sample])
        predictions, probabilities = self.predict(df_sample)

        result = 'won' if predictions[0] == 1 else 'lost'
        confidence = probabilities[0] if predictions[0] == 1 else 1 - probabilities[0]

        return result, confidence

    def save_model(self):
        model_data = {
            'pipeline': self.pipeline,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics
        }

        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else '.', exist_ok=True)

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.pipeline = model_data['pipeline']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            self.training_metrics = model_data.get('training_metrics', {})

            print(f"Model loaded from {self.model_path}")
            return True

        except FileNotFoundError:
            print(f"Model file {self.model_path} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def run_full_training_pipeline(self, won_csv: str, lost_csv: str) -> Dict[str, Any]:
        prep = DataPreprocessor()
        df_processed = prep.preprocess_pipeline(won_csv, lost_csv)
        return self.train(df_processed)

    def evaluate_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        X = self._prepare_features(df, fit_encoders=False)
        y = df['Group'].map({'won': 1, 'lost': 0})

        predictions, probabilities = self.predict(df)

        eval_metrics = {
            'accuracy': accuracy_score(y, predictions),
            'roc_auc': roc_auc_score(y, probabilities),
            'classification_report': classification_report(y, predictions),
            'confusion_matrix': confusion_matrix(y, predictions).tolist()
        }

        return eval_metrics


if __name__ == "__main__":
    prep = DataPreprocessor()
    df_processed = prep.preprocess_pipeline('sessions/sessions/won_20250714_140422.csv', 'sessions/sessions/lost_20250714_140422.csv')

    training_pipeline = TrainingPipeline()
    training_metrics = training_pipeline.train(df_processed)

    print("\nTesting prediction functionality:")
    sample_lead = {
        'Owner': 'Andrew Ortiz',
        'Linkedin': 'https://www.linkedin.com/in/test-profile/',
        'Email': 'test@example.com',
        'Company': 'Test Company',
        'Interested In': 'LinkedIn Lead Generation',
        'Plan Type': 'Gold',
        'UTM Medium': 'Paid search',
        'UTM LINK': 'test-campaign-link',
        'origin': 'LinkedIn'
    }

    prediction, confidence = training_pipeline.predict_single(sample_lead)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
