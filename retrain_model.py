
#!/usr/bin/env python3
"""
Retrain the leadnet model to fix input/output connectivity issues
"""

from lp_model_training import DataModule, Trainer
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    try:
        CSV_PATH = "./sessions/processed_data/preprocessed_data.csv"
        
        log.info("Initializing data module...")
        data_module = DataModule(CSV_PATH, batch_size=256)
        
        log.info("Starting training...")
        trainer = Trainer(data_module, model_dir="models_v4")
        metrics = trainer.train_cv(k=5, epochs=50, patience=8)
        
        log.info(f"Training completed: {metrics}")
        
    except Exception as e:
        log.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
