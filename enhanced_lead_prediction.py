import os
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from leadnet_core import DataModule, Trainer, Predictor
import logging

class DynamicPathManager:
    def __init__(self, sessions_dir: str = "./sessions/sessions"):
        self.sessions_dir = Path(sessions_dir)
        
    def get_latest_files(self) -> Dict[str, str]:
        files = {
            'won': None,
            'lost': None,
            'scheduled': None
        }
        
        for file_type in files.keys():
            pattern = f"{file_type}_*.csv"
            matches = list(self.sessions_dir.glob(pattern))
            if matches:
                latest = max(matches, key=lambda f: f.stat().st_mtime)
                files[file_type] = str(latest)
        
        return files
    
    def get_latest_timestamp(self) -> Optional[str]:
        csv_files = list(self.sessions_dir.glob("*.csv"))
        if not csv_files:
            return None
        
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        filename = latest_file.stem
        
        parts = filename.split('_')
        if len(parts) >= 3:
            return f"{parts[-2]}_{parts[-1]}"
        return None

class ProgressLogger:
    def __init__(self, log_file: str = "./logs/training_progress.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("LeadNetTraining")
    
    def log_progress(self, message: str, data: Dict = None):
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "data": data or {}
        }
        
        self.logger.info(f"{message} | {json.dumps(data) if data else ''}")
        
        progress_file = self.log_file.parent / "training_progress.json"
        try:
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            existing_data.append(progress_data)
            
            with open(progress_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write progress file: {e}")

class DataPreprocessor:
    UNNEEDED = {
        "Follow Up Tracker", "Notes", "Email Template #1", "Email Template #2",
        "Send Panda Doc?", "UTM Source", "UTM Campaign", "UTM Content",
        "Lead Source", "Channel FOR FUNNEL METRICS", "Subitems",
        "text_mkr7frmd", "Item ID", "Phone", "Last updated", "Sales Call Date",
        "Deal Status Date", "Date Created", "Item Name"
    }
    HIGH_NULL = 0.50

    def __init__(self, logger: ProgressLogger):
        self.logger = logger

    def load_and_clean(self, won_csv: str, lost_csv: str) -> pd.DataFrame:
        self.logger.log_progress("Loading training data", {
            "won_file": won_csv,
            "lost_file": lost_csv
        })
        
        won_df = pd.read_csv(won_csv)
        lost_df = pd.read_csv(lost_csv)
        
        self.logger.log_progress("Data loaded", {
            "won_samples": len(won_df),
            "lost_samples": len(lost_df)
        })
        
        df = pd.concat([won_df, lost_df], ignore_index=True)
        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        original_shape = df.shape
        
        hi_null_cols = df.columns[df.isna().mean() > self.HIGH_NULL]
        df = df.drop(columns=hi_null_cols, errors="ignore")
        df = df[df["Owner"].notna()]
        df = df.drop(columns=[c for c in self.UNNEEDED if c in df.columns], errors="ignore")
        
        self.logger.log_progress("Data cleaned", {
            "original_shape": original_shape,
            "cleaned_shape": df.shape,
            "dropped_columns": list(hi_null_cols)
        })
        
        return df

    def remove_null_utm_links(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)
        df = df[df['UTM LINK'].notna()]
        
        self.logger.log_progress("Removed null UTM links", {
            "original_samples": original_len,
            "remaining_samples": len(df)
        })
        
        return df

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.utils import resample
        
        won_data = df[df['Group'] == 'won']
        lost_data = df[df['Group'] == 'lost']
        
        min_samples = min(len(won_data), len(lost_data))
        
        if len(won_data) > min_samples:
            won_data = resample(won_data, replace=False, n_samples=min_samples, random_state=42)
        if len(lost_data) > min_samples:
            lost_data = resample(lost_data, replace=False, n_samples=min_samples, random_state=42)
        
        balanced_df = pd.concat([won_data, lost_data])
        
        self.logger.log_progress("Dataset balanced", {
            "won_samples": len(won_data),
            "lost_samples": len(lost_data),
            "total_samples": len(balanced_df)
        })
        
        return balanced_df

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing_before = df.isna().sum().sum()
        
        df['Linkedin'] = df['Linkedin'].fillna('Unknown')
        df['Company'] = df['Company'].fillna('Unknown')
        df['UTM Medium'] = df['UTM Medium'].fillna('Unknown')
        
        missing_after = df.isna().sum().sum()
        
        self.logger.log_progress("Missing values filled", {
            "missing_before": int(missing_before),
            "missing_after": int(missing_after)
        })
        
        return df

    def preprocess_pipeline(self, won_csv: str, lost_csv: str) -> pd.DataFrame:
        df = self.load_and_clean(won_csv, lost_csv)
        df = self.remove_null_utm_links(df)
        df = self.balance_dataset(df)
        df = self.fill_missing_values(df)
        
        return df

class EnhancedTrainer:
    def __init__(self, data_module: DataModule, model_dir: str = "models_weights", logger: ProgressLogger = None):
        self.data_module = data_module
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.logger = logger

    def train_with_progress(self, k: int = 5) -> Dict:
        trainer = Trainer(self.data_module, model_dir=str(self.model_dir))
        
        if self.logger:
            self.logger.log_progress("Starting cross-validation training", {
                "folds": k,
                "model_dir": str(self.model_dir)
            })
        
        metrics = trainer.train_cv(k=k)
        
        if self.logger:
            self.logger.log_progress("Training completed", {
                "metrics": metrics,
                "best_model": str(self.model_dir / "leadnet_best.keras")
            })
        
        return metrics

class LeadNetPipeline:
    def __init__(self, sessions_dir: str = "./sessions/sessions"):
        self.path_manager = DynamicPathManager(sessions_dir)
        self.logger = ProgressLogger()
        self.preprocessor = DataPreprocessor(self.logger)
        
    def train_model(self) -> Dict:
        try:
            files = self.path_manager.get_latest_files()
            
            if not files['won'] or not files['lost']:
                raise ValueError("Missing won or lost CSV files")
            
            self.logger.log_progress("Starting model training pipeline")
            
            df_processed = self.preprocessor.preprocess_pipeline(files['won'], files['lost'])
            
            save_path = './prepped_data/'
            os.makedirs(save_path, exist_ok=True)
            
            filename = 'preprocessed_data.csv'
            csv_path = os.path.join(save_path, filename)
            df_processed.to_csv(csv_path, index=False)
            
            self.logger.log_progress("Preprocessed data saved", {
                "file_path": csv_path,
                "samples": len(df_processed)
            })
            
            data_module = DataModule(csv_path, batch_size=256)
            trainer = EnhancedTrainer(data_module, logger=self.logger)
            
            metrics = trainer.train_with_progress(k=5)
            
            return {
                "status": "success",
                "metrics": metrics,
                "model_path": "./models_weights/leadnet_best.keras",
                "data_samples": len(df_processed)
            }
            
        except Exception as e:
            self.logger.log_progress(f"Training failed: {str(e)}", {"error": str(e)})
            return {
                "status": "error",
                "message": str(e)
            }
    
    def run_predictions(self) -> List[Dict]:
        try:
            files = self.path_manager.get_latest_files()
            
            if not files['scheduled']:
                raise ValueError("No scheduled CSV file found")
            
            model_path = "./models_weights/leadnet_best.keras"
            if not os.path.exists(model_path):
                raise ValueError("No trained model found. Please train a model first.")
            
            self.logger.log_progress("Starting predictions", {
                "scheduled_file": files['scheduled'],
                "model_path": model_path
            })
            
            csv_path = "./prepped_data/preprocessed_data.csv"
            if not os.path.exists(csv_path):
                raise ValueError("No preprocessed data found. Please train a model first.")
            
            data_module = DataModule(csv_path, batch_size=256)
            predictor = Predictor(model_path, data_module)
            
            df_scheduled = pd.read_csv(files['scheduled'])
            
            # Clean NaN values immediately after reading CSV
            df_scheduled = df_scheduled.fillna('Unknown')
            
            required_columns = ['Owner', 'Linkedin', 'Email', 'Company', 'Interested In', 'Plan Type', 'UTM Medium', 'UTM LINK', 'origin']
            
            available_columns = [col for col in required_columns if col in df_scheduled.columns]
            selected_columns = df_scheduled[available_columns]
            
            results = []
            for index, row in selected_columns.iterrows():
                sample_lead_dict = row.to_dict()
                
                # Ensure all required columns exist and clean any remaining NaN values
                for col in required_columns:
                    if col not in sample_lead_dict:
                        sample_lead_dict[col] = 'Unknown'
                    elif pd.isna(sample_lead_dict[col]) or sample_lead_dict[col] is None:
                        sample_lead_dict[col] = 'Unknown'
                    elif isinstance(sample_lead_dict[col], str) and sample_lead_dict[col].lower() in ['nan', 'none', '']:
                        sample_lead_dict[col] = 'Unknown'
                
                try:
                    prediction_result = predictor.predict_one(sample_lead_dict)
                    
                    results.append({
                        "index": index,
                        "lead_data": sample_lead_dict,
                        "prediction": prediction_result[0],
                        "confidence": prediction_result[1]
                    })
                except Exception as e:
                    self.logger.log_progress(f"Prediction failed for lead {index}: {str(e)}")
                    results.append({
                        "index": index,
                        "lead_data": sample_lead_dict,
                        "prediction": "Error",
                        "confidence": 0.0,
                        "error": str(e)
                    })
            
            self.logger.log_progress("Predictions completed", {
                "total_predictions": len(results),
                "successful_predictions": len([r for r in results if "error" not in r])
            })
            
            return results
            
        except Exception as e:
            self.logger.log_progress(f"Prediction failed: {str(e)}", {"error": str(e)})
            raise

if __name__ == "__main__":
    pipeline = LeadNetPipeline()
    
    print("Training model...")
    train_result = pipeline.train_model()
    print(f"Training result: {train_result}")
    
    if train_result["status"] == "success":
        print("Running predictions...")
        predictions = pipeline.run_predictions()
        print(f"Generated {len(predictions)} predictions")
        
        for pred in predictions[:5]:
            print(f"Lead {pred['index']}: {pred['prediction']} (confidence: {pred['confidence']})") 