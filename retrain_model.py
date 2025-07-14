
#!/usr/bin/env python3
"""
Retrain the model from scratch to fix connectivity issues
"""

import logging
from pathlib import Path
from lp_model_training import DataModule, Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("retrain")

def main():
    try:
        # Remove old models to start fresh
        models_dir = Path("models_v4")
        if models_dir.exists():
            for model_file in models_dir.glob("*.keras"):
                model_file.unlink()
                log.info(f"Removed old model: {model_file}")
        
        CSV_PATH = "./sessions/processed_data/preprocessed_data.csv"
        
        log.info("Initializing fresh data module...")
        data_module = DataModule(CSV_PATH, batch_size=256)
        
        log.info("Starting fresh training...")
        trainer = Trainer(data_module, model_dir="models_v4")
        metrics = trainer.train_cv(k=3, epochs=50, patience=8)  # Reduced for faster training
        
        log.info(f"Training completed successfully: {metrics}")
        
        # Test the new model
        from lp_model_training import Predictor
        best_model_path = trainer.model_dir / "leadnet_best.keras"
        predictor = Predictor(best_model_path, data_module)
        
        example = {
            "Owner": "Test Owner",
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
        log.info(f"Test prediction successful: {result}")
        
    except Exception as e:
        log.error(f"Retraining failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
